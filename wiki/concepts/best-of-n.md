---
title: Best-of-N Sampling
type: concept
sources: [raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md, raw/papers/From Representational Complementarity to Dual Systems_ Synergizing VLM and Vision-Only Backbones for End-to-End Driving.md]
related: [sources/autovla.md, sources/curious-vla.md, sources/drivevla-w0.md, sources/adathinkdrive.md, sources/nord.md, sources/vega.md, sources/dreameraD.md, sources/hybriddriveVLA.md, concepts/navsim-benchmark.md, concepts/rl-for-ad.md, concepts/dual-system-vla.md]
created: 2026-04-15
updated: 2026-04-19
confidence: high
---

## What It Is

Best-of-N (BoN) sampling generates N independent trajectory outputs from the same model under the same input — using different random seeds or temperatures — and selects the one with the highest score. In the AD literature, the score is typically NAVSIM's PDM simulator score (PDMS), making this an **oracle selection procedure**.

**Why it is not a deployment metric**: selecting the best output requires running the full NAVSIM PDM scorer on all N candidates. No such oracle exists at deployment time. BoN measures the model's *capability ceiling* — the performance achievable if a perfect lightweight selector existed. It is routinely reported alongside single-sample results because it reveals whether a model's failure mode is *generation quality* (hard ceiling) or *selection* (soft ceiling addressable by a learned ranker).

---

## Procedure

1. Run N forward passes through the same model on the same input, with different random seeds (or stochastic decoding)
2. For each output trajectory, compute PDMS using the NAVSIM PDM simulator
3. Report the score of the highest-PDMS trajectory

N is typically 4 or 6 in current AD papers. Comparing BoN scores across different N values is misleading.

---

## Results in the Wiki

### NAVSIM-v1 (PDMS)

| Method          | Single-sample PDMS | BoN PDMS  | N   | Gain              |
| --------------- | ------------------ | --------- | --- | ----------------- |
| **Curious-VLA** | 90.3               | **94.8**  | 6   | +4.5 (= human GT) |
| DriveVLA-W0★    | 90.2               | **93.0**  | 6   | +2.8              |
| AdaThinkDrive   | 90.3               | **93.0**  | 4   | +2.7              |
| NoRD            | 85.6               | **92.4**  | 6   | +6.8              |
| AutoVLA         | 89.11              | **92.12** | 6   | +3.01             |
| Vega            | 87.9               | **89.8**  | 6   | +1.9              |

★ DriveVLA-W0's "90.2" single-sample uses query-based expert with trajectory anchors (multi-candidate selection within the model); the underlying single-model output is 88.4 PDMS. BoN-6 adds oracle selection on top.

### NAVSIM-v2 (EPDMS)

| Method | Single-sample EPDMS | BoN EPDMS | N |
|--------|---------------------|-----------|---|
| Vega | 86.9 | **89.4** | 6 |

---

## Key Observations

### 1. NAVSIM-v1 is saturated at BoN-6

Curious-VLA BoN-6 (94.8 PDMS) matches the human ground-truth trajectory score on NAVSIM-v1. A model that can select among 6 samples already reaches human-level performance on this benchmark. This suggests **NAVSIM-v1 BoN-6 is a closed frontier**: further BoN-6 improvements are marginal, and future work should emphasize single-sample scores, NAVSIM-v2 extended metrics (EPDMS), or Navhard OOD evaluation.

### 2. BoN gain is inversely correlated with single-sample quality

NoRD gains +6.8 from BoN-6 despite a 85.6 single-sample baseline. AutoVLA gains only +3.0 from a 89.1 baseline. Weaker models generate more diverse outputs and are more likely to include one correct trajectory in N tries. Models already near the ceiling have smaller headroom. This means high BoN gain is not exclusively a sign of model quality — it can also reflect *high single-pass failure rate with output diversity*.

### 3. N matters and should be reported explicitly

AdaThinkDrive BoN-4 (93.0) and DriveVLA-W0 BoN-6 (93.0) achieve identical PDMS — but with N=4 vs. N=6. DriveVLA-W0 required 50% more inference to reach the same ceiling. Papers that omit N, or compare BoN-4 vs. BoN-6 results in the same table, conflate two different inference budgets.

### 4. Relationship to learned selectors: DreamerAD's vocabulary sampling

DreamerAD generates 256 candidate trajectories and selects the best via a learned AD-RM reward model trained on latent features — not PDMS oracle. This is a **deployable BoN variant**: the selector is an approximation but runs entirely from latent features without the PDM simulator. It achieves 87.7 EPDMS from a base of 85.1, a +2.6 gain. This is the only wiki method that closes the BoN gap in a deployment-feasible way. See [[sources/dreameraD.md]].

---

## Cross-Model BoN: Complementarity as Diversity Source

All BoN results above draw N samples from a **single model**. HybridDriveVLA ([[sources/hybriddriveVLA.md]]) introduces a complementary paradigm: draw one sample from each of two different backbone types (VLM + ViT) and select between them.

### Cross-Model Oracle Results (VLM + ViT in Same Architecture)

| Candidate Set | N | PDMS | Method |
|---|---|---|---|
| ViT-large samples only | 1 | 88.88 | Within-model |
| ViT-large samples only | 3 | 89.13 | Within-model BoN |
| ViT-large samples only | 6 | 89.32 | Within-model BoN |
| VLM samples only | 1 | 90.80 | Single model |
| VLM samples only | 3 | 91.57 | Within-model BoN |
| VLM samples only | 6 | 91.95 | Within-model BoN |
| VLM + ViT oracle | 2 | **93.58** | Cross-model oracle |
| VLM + ViT oracle | 6 (interp.) | **94.00** | Cross-model oracle + interpolations |

**Key insight**: cross-model diversity (93.58) vastly exceeds within-model sampling diversity (91.95 at BoN-6). Complementarity from different backbone inductive biases produces more useful diversity than stochastic sampling from a single model.

### Why Cross-Model Diversity Is Richer

- **Within-model BoN**: same representation, same training; stochastic decoding produces minor variations around the same solution
- **Cross-model BoN**: VLM is more aggressive (~66% of scenarios); ViT is more conservative; neither strictly contains the other; expert behavior often lies between them
- The two models win on completely different ~2–3% scenario subsets (neither is a subset of the other)

### HybridDriveVLA: Deployable Cross-Model Selector

HybridDriveVLA converts this oracle into a deployable system using a trajectory-level scorer:

1. Both VLM and ViT branches produce trajectories
2. 9 linear interpolations are added (τ_α = α·τ_ViT + (1-α)·τ_VLM)
3. A DrivoR-style scorer ranks all 11 candidates

**Result**: 92.10 PDMS — not oracle (93.58), but substantially better than any single-model result.

This is the second deployable BoN variant in the wiki (alongside DreamerAD). Key distinction:
- **DreamerAD**: within-model BoN (256 samples) using a learned latent reward model; +2.6 EPDMS
- **HybridDriveVLA**: cross-model BoN (2 branches + interpolations) using a trajectory scorer; +1.30 PDMS over VLM baseline

Both demonstrate that the oracle BoN gap can be partially closed by a learned selector without the PDM simulator.

---

## Implications for Benchmark Interpretation

When reading NAVSIM SOTA tables:
- A method that reports only BoN without a single-sample number is not directly comparable to single-sample results
- Curious-VLA BoN-6 (94.8) is the highest absolute PDMS in the wiki — but the most meaningful single-sample SOTA is FLARE RFT (91.4) and DriveFine (90.7)
- BoN results are most informative when paired with the single-sample result — the gap indicates how much a learned selector could theoretically recover

See [[concepts/navsim-benchmark.md]] for the full SOTA table with BoN and single-sample results labeled separately.
