---
title: "Understanding R1-Zero-Like Training: A Critical Perspective"
type: source-summary
sources: [raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md]
related: [concepts/r1-zero-like-training.md, concepts/gspo-vs-grpo.md, concepts/rl-for-ad.md, concepts/chain-of-thought-for-ad.md, concepts/foundation-backbones-for-ad.md, sources/nord.md]
created: 2026-06-18
updated: 2026-06-18
confidence: high
---

## Overview

This paper dissects R1-Zero-like training by separating two factors that are often conflated:

1. **Base-model priors**: Qwen2.5-Math models already behave like strong math QA models, sometimes performing best with no prompt template at all.
2. **RL optimizer effects**: standard GRPO contains response-length and question-difficulty biases that can make longer reasoning look like an emergent capability even when it is partly an optimization artifact.

The paper introduces **Dr. GRPO** ("GRPO Done Right"), which removes both the response-length normalization and per-question reward-std normalization. In the authors' minimalist recipe, Qwen2.5-Math-7B + Dr. GRPO on MATH level 3-5 questions reaches **43.3% AIME 2024** and **51.4 average** across AIME24, AMC, MATH500, Minerva, and OlympiadBench.

**Project**: https://github.com/sail-sg/understand-r1-zero  
**Organizations**: Sea AI Lab, National University of Singapore, Singapore Management University

---

## Figures

![[raw/assets/x1 33.png]]

**Figure 1**: Dr. GRPO removes GRPO's length and std normalization terms, preventing incorrect responses from growing progressively longer.

![[raw/assets/x2 32.png]]

**Figure 2**: Oat-Zero-7B benchmark comparison using the paper's minimalist recipe.

![[raw/assets/x3 30.png]]

**Figure 3**: Base-model attributes: question-answering behavior, pass@8 exploration ability, and self-reflection counts.

![[raw/assets/x4 27.png]]

**Figure 4**: GRPO bias illustration. Length normalization and reward-std normalization reweight responses/questions relative to the unbiased group-centered advantage.

![[raw/assets/x5 25.png]]

**Figure 5**: GRPO vs. Dr. GRPO training dynamics and evaluation results.

![[raw/assets/x6 21.png]]

**Figure 6**: RL dynamics across template and question-set combinations.

![[raw/assets/x7 17.png]]

**Figure 7**: Domain-specific pretraining raises the RL ceiling for Llama-3.2-3B; GRPO length growth can mimic long-CoT emergence.

![[raw/assets/x8 12.png]]

**Figure 8**: Ablation over GRPO bias terms.

![[raw/assets/x9 6.png]]

**Figure 9**: Three independent GRPO vs. Dr. GRPO runs.

![[raw/assets/x10 7.png]]

**Figure 10**: Self-reflection keyword counts across 40,000 responses.

![[raw/assets/x11 4.png]]

**Figure 11**: False positives in keyword-based and LLM-based self-reflection detection.

![[raw/assets/x12 3.png]]

**Figure 12**: Keyword, LLM, and cross-validated self-reflection detection.

![[raw/assets/x13 6.png]]

**Figure 13**: DeepSeek-V3-Base examples that already exhibit "Aha moment" behavior before RL tuning.

![[raw/assets/x14 2.png]]

**Figure 14**: DeepSeek-V3-Base vs. DeepSeek-R1-Zero response categories across MATH difficulty levels.

![[raw/assets/x15 2.png]]

**Figure 15**: Self-reflection does not reliably imply higher inference-stage accuracy for DeepSeek-R1-Zero.

**Table 5 note**: the raw markdown references a Table 5 with average response lengths across response categories, but the table itself is not rendered in the local markdown. The surrounding text says response length increases substantially after R1-Zero training and that incorrect responses are longer on average than correct responses.

---

## Base-Model Findings

### Templates Construct Base Policies

The paper evaluates Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-7B, Llama-3.1-8B, DeepSeek-Math-7B, and DeepSeek-V3-Base-685B on 500 MATH training questions. The key diagnostic is whether the model answers the question or merely continues the prompt as text.

Findings:
- Llama and DeepSeek base models need a suitable template, especially the R1 template, to act like QA policies.
- Qwen2.5 models reach a 100% answering rate with **no template**.
- All tested base models have enough pass@8 exploration ability to produce some rewarding math trajectories, but Qwen2.5 models are strongest.
- DeepSeek-V3-Base has the lowest no-template answering rate, making it closest to a pure base model in this comparison.

### Qwen2.5-Math Performs Best Without Template

| Base model + Template | AIME24 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Math-1.5B |  |  |  |  |  |  |
| 4-shot | 10.0 | 28.9 | 25.2 | 11.4 | 24.0 | 19.9 |
| R1 template | 0.0 | 9.6 | 21.2 | 6.6 | 2.2 | 7.9 |
| Qwen template | 20.0 | 32.5 | 33.0 | 12.5 | 22.8 | 24.2 |
| No template | 16.7 | 43.4 | 61.8 | 15.1 | 28.4 | 33.1 |
| Qwen2.5-Math-7B |  |  |  |  |  |  |
| 4-shot | 10.0 | 42.2 | 45.0 | 13.2 | 31.4 | 28.4 |
| R1 template | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 |
| Qwen template | 16.7 | 38.6 | 50.6 | 9.9 | 16.6 | 26.5 |
| No template | 0.2 | 45.8 | 69.0 | 21.3 | 34.7 | 38.2 |

**Interpretation**: Qwen2.5-Math's pretraining likely includes concatenated question-answer text. That means "pure RL gains" in Qwen-based R1-Zero replications can be overstated if the baseline prompt/template suppresses a capability already present in the base model.

### Aha Moment Is Not Proof of RL Emergence

The paper finds self-reflection behavior before RL in nearly all tested base models, including DeepSeek-V3-Base. It uses keyword filtering plus GPT-4o-mini cross-validation because either method alone can produce false positives.

Important caveat: self-reflection during inference does **not** necessarily improve accuracy. In the DeepSeek-R1-Zero analysis, nearly half of responses with self-reflection do not outperform responses without self-reflection.

---

## GRPO Bias

The paper argues that standard GRPO is not an unbiased implementation of the PPO-style policy-gradient objective.

### Response-Level Length Bias

GRPO divides the loss by response length. For positive-advantage outputs, shorter correct responses receive larger updates. For negative-advantage outputs, longer incorrect responses are penalized less. This can encourage overlong incorrect reasoning.

### Question-Level Difficulty Bias

GRPO divides group-centered reward by the per-question reward standard deviation:

$$
\hat{A}_{i,t}^{GRPO} = \frac{R(q,o_i)-mean(R)}{std(R)}
$$

This reweights questions: very easy or very hard questions with low reward variance get high weight, while medium-difficulty high-variance questions get attenuated. This is the same mechanism that later matters for NoRD's AD-domain difficulty-bias analysis ([[sources/nord.md]]).

### Open-Source PPO Implementations

| Repository | Length-biased? | Note |
| --- | --- | --- |
| trl | Yes | Per-batch length normalization |
| OpenRLHF | Yes | Per-response length normalization |
| verl | Yes | Per-batch length normalization |
| SimpleRL-Zero | Yes | Inherits OpenRLHF-style loss |
| Open-Reasoner-Zero | Yes | Per-response normalization |

The authors argue the fix is to divide by a constant generation budget rather than the realized response length.

---

## Dr. GRPO

Dr. GRPO removes:

1. The per-response length normalization term.
2. The per-question reward-standard-deviation normalization term.

The effective advantage becomes:

$$
\hat{A}_{i,t}^{DrGRPO} = R(q,o_i)-mean(R)
$$

The reward is a minimal binary verifier:

$$
R(q,o)=
\begin{cases}
1 & \text{if } o \text{ contains the correct final answer to } q \\
0 & \text{otherwise}
\end{cases}
$$

Reported effects:
- Similar reward improvement trend to GRPO.
- Less uncontrolled response-length growth.
- Shorter incorrect responses.
- Better token efficiency.
- Statistically consistent improvements across three independent runs.

---

## Template and Question-Set Interaction

| Question set | # | Description |
| --- | --- | --- |
| ORZ | 57k | AIME + Numina-Math + Tulu3 MATH; diverse and large |
| MATH | 12k | High-school math competition questions |
| GSM | 8k | Simpler grade-school math questions |
| ASDiv | 2k | Basic algebra questions |

Findings:
- Templates strongly affect initial policy quality.
- RL can recover similar final performance if the question set has enough coverage.
- If the template mismatches the base model, RL gains may mostly reconstruct capability that the template destroyed.
- With Qwen-Math template, GSM-8K can nearly double harder benchmark accuracy despite being simpler and out of distribution, suggesting RL may reinforce useful behavior rather than inject new knowledge.

---

## Domain Pretraining

The Llama-3.2-3B experiments ask whether R1-Zero-like training works on a weak math base model. The answer is yes, but weakly:

- Vanilla Llama-3.2-3B improves under Dr. GRPO, but the gain is small.
- Llama-3.2-3B-FineMath improves more.
- Llama-3.2-3B-NuminaQA, continually pretrained on concatenated NuminaMath QA text, improves further after RL.

Conclusion: RL is not a substitute for domain knowledge in the base model; pretraining raises the RL ceiling.

---

## Benchmark Results

All compared models use a 3k generation budget unless noted. `*` means the best template, no template, is used for Qwen2.5-Math base models.

| Base model + Method | AIME24 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Math-1.5B | 20.0 | 32.5 | 33.0 | 12.5 | 22.8 | 24.2 |
| Qwen2.5-Math-1.5B* | 16.7 | 43.4 | 61.8 | 15.1 | 28.4 | 33.1 |
| Oat-Zero-1.5B | 20.0 | 53.0 | 74.2 | 25.7 | 37.6 | 42.1 |
| R1-Distill-Qwen-1.5B @ 3k | 2.5 | 21.7 | 52.2 | 16.3 | 17.3 | 22.0 |
| R1-Distill-Qwen-1.5B @ 8k | 20.0 | 49.4 | 77.4 | 25.0 | 35.8 | 41.5 |
| Qwen2.5-Math-1.5B-Instruct | 10.0 | 48.2 | 74.2 | 26.5 | 40.2 | 39.8 |
| Llama-3.2-3B | 0.0 | 2.4 | 6.4 | 6.3 | 1.3 | 3.3 |
| + RL w. Dr. GRPO | 3.3 | 7.2 | 10.0 | 11.0 | 2.2 | 6.8 |
| Llama-3.2-3B-FineMath | 0.0 | 3.6 | 18.4 | 5.9 | 2.2 | 6.0 |
| + RL w. Dr. GRPO | 3.3 | 10.8 | 38.0 | 12.9 | 9.0 | 14.8 |
| Llama-3.2-3B-NuminaQA | 0.0 | 0.0 | 0.6 | 0.0 | 0.1 | 0.14 |
| + RL w. Dr. GRPO (Oat-Zero-3B) | 6.7 | 18.1 | 50.0 | 14.3 | 14.7 | 20.7 |
| Llama-3.2-3B-Instruct | 6.7 | 15.7 | 38.8 | 11.8 | 12.6 | 17.1 |
| Qwen2.5-Math-7B | 16.7 | 38.6 | 50.6 | 9.9 | 16.6 | 26.5 |
| Qwen2.5-Math-7B* | 0.2 | 45.8 | 69.0 | 21.3 | 34.7 | 38.2 |
| SimpleRL-Zero-7B | 26.7 | 60.2 | 78.2 | 27.6 | 40.3 | 46.6 |
| PRIME-Zero-7B | 16.7 | 62.7 | 83.8 | 36.0 | 40.9 | 48.0 |
| OpenReasoner-Zero-7B @ 3k | 13.3 | 47.0 | 79.2 | 31.6 | 44.0 | 43.0 |
| OpenReasoner-Zero-7B @ 8k | 13.3 | 54.2 | 82.4 | 31.6 | 47.9 | 45.9 |
| Oat-Zero-7B | 43.3 | 62.7 | 80.0 | 30.1 | 41.0 | 51.4 |
| R1-Distill-Qwen-7B @ 3k | 10.0 | 26.2 | 60.1 | 23.0 | 23.1 | 28.5 |
| R1-Distill-Qwen-7B @ 8k | 33.3 | 68.4 | 88.1 | 35.9 | 47.7 | 54.7 |
| Qwen2.5-Math-7B-Instruct | 16.7 | 53.0 | 83.6 | 29.8 | 42.7 | 45.1 |

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Maximum response length | 3000 tokens |
| Sampling temperature | 1.0 |
| Top P, top K | 1.0, -1 |
| Responses per question | 8 |
| Optimizer | AdamW |
| Adam beta1, beta2 | 0.9, 0.95 |
| Weight decay | 0.0 |
| Gradient norm clipping | 1.0 |
| LR scheduler | Constant |
| Learning rate | 1e-6 |
| Inner proximal update epoch | 1 |
| KL loss coefficient | 0.0 |
| KL penalty coefficient | 0.0 |
| Policy clipping parameter | 0.2 |

---

## Relationship to AD Papers

This source is not an autonomous-driving paper, but it is methodologically important for the wiki:

- It supplies the original Dr. GRPO rationale behind NoRD's optimizer choice ([[sources/nord.md]]).
- It cautions that GRPO length growth should not automatically be read as emergent reasoning.
- It clarifies why reward-std normalization can suppress useful high-variance training cases, a pattern echoed in driving RL difficulty-bias analysis.
- It makes Qwen-family base-model priors a confound when interpreting "RL from base model" results in VLA papers that use Qwen backbones.

---

## Limitations

1. **Math/verifier domain**: the experiments use verifiable math rewards, so transfer to open-ended reasoning, multimodal VLA planning, or non-binary rewards is indirect.
2. **Causal claims about pretraining are partly inferential**: the Qwen concatenated-QA hypothesis is plausible from behavior but not proven from disclosed training data.
3. **Aha moment detection remains noisy**: even cross-validation of keyword and LLM detection can miss implicit self-reflection or retain false positives.
4. **Limited model families**: the base-model analysis centers on Qwen2.5, Llama, and DeepSeek; results may not generalize to newer backbones.
5. **No AD-specific validation**: Dr. GRPO is tested here on math; AD uses continuous, multimodal, simulator-based, and safety-gated rewards.
6. **Benchmark scope**: Oat-Zero's strong AIME24 score uses a specific 3k generation budget and Qwen2.5-Math context constraints; long-context comparisons need care.

---

## Key Takeaways

1. **Do not equate longer CoT with better reasoning**: GRPO's length bias can inflate incorrect responses.
2. **Prompt templates can destroy latent base-model capability**: RL gains may partly be recovery from template mismatch.
3. **Dr. GRPO is a simple optimizer correction**: remove response-length and reward-std normalization; use a group-centered reward baseline.
4. **Base-model priors matter**: Qwen2.5-Math is not a clean "blank base model" for measuring RL emergence.
5. **Self-reflection is not sufficient evidence of higher accuracy**: it can appear before RL and may not improve inference-stage correctness.
