---
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
type: source-summary
sources: [raw/papers/DAPO_ An Open-Source LLM Reinforcement Learning System at Scale.md]
related: [concepts/gspo-vs-grpo.md, concepts/r1-zero-like-training.md, concepts/discriminative-policy-optimization.md, concepts/rl-for-ad.md]
created: 2026-06-23
updated: 2026-06-23
confidence: high
---

# DAPO

DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) is an open-source large-scale reasoning-RL recipe built on GRPO and the `verl` framework. It addresses four practical failure modes observed when training Qwen2.5-32B from a base checkpoint: entropy collapse, zero-gradient prompt groups, unhealthy length dynamics, and noisy penalties for truncated responses.

The complete recipe reaches 50 AIME 2024 avg@32, compared with 30 for the authors' naive GRPO baseline and 47 for the cited DeepSeek-R1-Zero-Qwen-32B result, using roughly half the gradient-update steps reported for the latter.

## Key Takeaways

- **DAPO is a bundle, not only asymmetric clipping.** Clip-Higher, Dynamic Sampling, token-level loss, and overlong reward shaping interact with the data and distributed rollout system.
- **Exploration can be clipping-limited.** A symmetric upper ratio cap barely increases the absolute probability of low-probability positive-advantage tokens.
- **Same-reward groups waste gradient budget.** All-correct and all-wrong response groups have zero normalized advantages.
- **Loss reduction changes length incentives.** Sample-level averaging gives every response equal total weight; token-level averaging gives each token equal weight and therefore longer responses more total influence.
- **Truncation is not equivalent to incorrectness.** Hard punishment can label a valid but unfinished reasoning path as wrong; masking or soft length penalties are less noisy.
- **Training reward is not validation accuracy.** The authors observe stable training-reward growth even when validation quality does not improve.
- **Entropy has a healthy range.** Collapse removes exploration, but excessive entropy correlates with gibberish and repetition.

![Figure 1: DAPO reaches 50 AIME24 on Qwen2.5-32B with fewer update steps than the cited DeepSeek-R1-Zero result.](<../../raw/assets/x1 38.png>)

## Baseline Objective

DAPO retains GRPO's group-normalized reward advantage:

$$
\hat A_i=\frac{R_i-\operatorname{mean}(R)}{\operatorname{std}(R)}.
$$

It removes the frozen-reference KL penalty and changes loss clipping, group selection, loss reduction, and truncated-response reward treatment.

## Four Core Techniques

### 1. Clip-Higher

DAPO decouples the PPO ratio bounds:

$$
\operatorname{clip}(r,1-\varepsilon_{low},1+\varepsilon_{high}),
\qquad \varepsilon_{low}=0.2,\ \varepsilon_{high}=0.28.
$$

For positive advantages, a token with old probability 0.01 can rise only to 0.012 under a 1.2 ratio cap, while an already-likely token at 0.9 is effectively unconstrained because its nominal bound exceeds 1. The higher upper cap gives rare exploratory tokens more room to gain probability. The lower cap is kept at 0.2 to avoid suppressing probability mass too aggressively.

![Figure 2: AIME accuracy and policy entropy under clipping variants; asymmetric Clip-Higher slows entropy collapse.](<../../raw/assets/x2 36.png>)

![Figure 3: Mean probability of up-clipped tokens and the growing fraction of all-correct prompt groups.](<../../raw/assets/x4 30.png>)

### 2. Dynamic Sampling

Before each update, DAPO oversamples prompts and keeps only groups satisfying:

$$
0<\#\{\text{correct responses}\}<G.
$$

This guarantees non-zero group-relative advantages and a consistent number of effective prompts in every batch. Sampling cost becomes dynamic, but the synchronized rollout system is often dominated by long-tail generations, so extra short discarded groups may add little wall time.

The method deliberately removes all-correct and all-wrong groups. It improves gradient efficiency but shifts training toward questions of intermediate policy difficulty.

### 3. Token-Level Policy-Gradient Loss

GRPO normally averages tokens within each response, then averages responses. DAPO divides the total token loss by the total number of response tokens across the group:

$$
\frac{1}{\sum_i|o_i|}\sum_i\sum_t L_{i,t}.
$$

Each token receives equal weight regardless of response length. High-quality long reasoning gets more total gradient, and long gibberish can receive proportionally stronger penalty. The tradeoff is a deliberate sequence-length weighting: longer responses influence the update more.

![Figure 4: Token-level aggregation produces healthier entropy and response-length dynamics than sample-level loss.](<../../raw/assets/x6 23.png>)

### 4. Overlong Reward Shaping

Hardly assigning incorrect reward to every truncated response creates label noise: truncation may cut off otherwise valid reasoning. DAPO first proposes Overlong Filtering, which masks truncated samples, then Soft Overlong Punishment:

$$
R_{length}(y)=
\begin{cases}
0,&|y|\le L_{max}-L_{cache},\\
\frac{(L_{max}-L_{cache})-|y|}{L_{cache}},&L_{max}-L_{cache}<|y|\le L_{max},\\
-1,&|y|>L_{max}.
\end{cases}
$$

The experiments use expected maximum length 16,384 and a 4,096-token soft-punishment cache, for a 20,480-token generation cap.

![Figure 5: Overlong filtering stabilizes AIME performance and training dynamics relative to hard truncation punishment.](<../../raw/assets/x8 14.png>)

## Algorithm

1. Sample a prompt batch and snapshot the old policy.
2. Generate $G$ outputs per prompt and score them with a rule-based verifier.
3. Add only mixed-correctness prompt groups to a dynamic buffer; continue sampling until the target effective batch is full.
4. Compute group-normalized advantages.
5. Run multiple policy updates using asymmetric clipping and token-level loss reduction.
6. Apply overlong filtering or soft length shaping to truncated samples.

## Dataset Transformation

DAPO-Math-17K is built from web and competition problems. To reduce parser/reward errors, the authors transform answers into integers. For example, a problem whose original answer is $(a+\sqrt b)/c$ can be rewritten to request $a+b+c$. This makes exact rule-based verification easier, but changes the task wording and target distribution.

## Training Configuration

- Model: Qwen2.5-32B base.
- Framework: `verl` / HybridFlow.
- Optimizer: AdamW, constant learning rate $10^{-6}$, 20-rollout-step linear warmup.
- Prompt batch: 512.
- Responses per prompt: 16.
- Mini-batch: 512; 16 gradient updates per rollout step.
- Clip bounds: 0.2 lower, 0.28 upper.
- Generation: maximum 20,480 tokens, including 4,096-token soft penalty cache.
- AIME evaluation: avg@32, temperature 1.0, top-p 0.7.
- Reward: exact-equivalence rule, `+1` correct and `-1` incorrect, plus optional length shaping.
- Reference KL: removed.

![Figure 6: Dynamic sampling uses more generated instances but reaches the same or better performance in fewer updates and similar wall time.](<../../raw/assets/x10 10.png>)

## Table 1: Progressive DAPO Recipe

| Model / cumulative recipe | AIME24 avg@32 |
| --- | ---: |
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (DAPO) | **50** |

The increments are cumulative and order-dependent. This table does not identify isolated main effects or interactions; later gains depend on the earlier recipe already being enabled.

## Training Dynamics

The paper recommends monitoring:

- mean response length;
- validation accuracy;
- training reward;
- actor entropy;
- generation probability;
- clipping fractions;
- mixed/all-correct/all-wrong group ratios.

Training reward can keep rising while validation accuracy stagnates, so it is not sufficient for model selection. Slowly rising entropy can support exploration, but rapid entropy growth often signals random or repetitive generation.

![Figure 7: DAPO response length, reward, entropy, and generation-probability dynamics.](<../../raw/assets/x11 7.png>)

## Table 2: Reflective Behavior Case

The extracted table is a qualitative case rather than structured metrics.

| Field | Extracted content |
| --- | --- |
| Problem | Tetrahedron geometry with an orthocenter projection, a $30^\circ$ dihedral angle, and $SA=2$; return $k+m$ for volume $k/m$. |
| Early behavior | Long coordinate derivation without explicit checking/backtracking. |
| Later-RL behavior | The response interrupts itself—“wait a moment, let’s rethink”—and revises the geometric treatment of the dihedral angle. |
| Claimed observation | Reflection and backtracking patterns are rare initially and become more frequent during RL. |

This is illustrative evidence, not a controlled causal measurement of emergent reflection.

## Table 3: Supplementary Reflective Case

| Field | Extracted content |
| --- | --- |
| Problem | Aimeville set-counting problem over diamond rings, golf clubs, garden spades, and universally owned candy hearts; find residents owning all four. |
| Failure | The response obtains non-integer $a_4=54.75$, recognizes that the approach is inconsistent, and restarts with a different counting representation. |
| Reflective behavior | Explicitly states that the current inclusion-exclusion approach must be reconsidered. |
| Extraction caveat | The raw markdown includes a very long partial response but not a clean structured table or clearly extracted final answer. |

## Limitations

- **Math-only evidence.** No coding, multimodal, or autonomous-driving policy experiment validates transfer.
- **One main scale/model.** The headline and cumulative ablation center on Qwen2.5-32B base.
- **Cumulative ablation.** Technique increments are order-dependent and do not isolate interactions or independent contributions.
- **No seed variance.** AIME results are averaged over sampled responses, not repeated training runs.
- **Dynamic sampling changes the data distribution.** Easy and hard prompts are removed, emphasizing intermediate-difficulty questions and potentially neglecting frontier failures.
- **Potentially higher sampling cost.** The wall-time argument depends on synchronized rollout stragglers; other serving architectures may pay the full oversampling cost.
- **No KL constraint.** Removing reference KL enables capability movement but provides less protection against catastrophic policy drift or reward exploitation.
- **Length bias is redesigned, not removed.** Token-level loss gives longer sequences more total weight, which may encourage verbosity despite stronger long-gibberish penalties.
- **Reward shaping is task-specific.** The 16k/4k thresholds and integer-answer transformation are tailored to long-form math.
- **Exact-match verifier limits scope.** Complex continuous rewards, subjective preferences, and safety tradeoffs are not addressed.
- **Reflection evidence is anecdotal.** Two qualitative cases do not establish emergence frequency, novelty, or causal dependence on DAPO.

## Relevance to Autonomous Driving

- [[concepts/gspo-vs-grpo.md]] — canonical source for asymmetric clipping, dynamic sampling, token-level loss, and overlong shaping.
- [[concepts/r1-zero-like-training.md]] — demonstrates that scalable RL depends on systems and data details beyond the headline GRPO formula.
- [[concepts/discriminative-policy-optimization.md]] — DisCO later criticizes DAPO for retaining clipping and $\sqrt{p(1-p)}$ difficulty weighting.
- [[concepts/rl-for-ad.md]] — Clip-Higher and effective-group monitoring may transfer, but filtering all-failed driving groups can discard the most safety-critical cases.
