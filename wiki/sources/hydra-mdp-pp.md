---
title: "Hydra-MDP++: Advancing End-to-End Driving via Expert-Guided Hydra-Distillation"
type: source-summary
sources: ["raw/papers/Hydra-MDP++_ Advancing End-to-End Driving via Expert-Guided Hydra-Distillation.md"]
related: [concepts/navsim-benchmark.md, concepts/selection-based-planning.md, concepts/teacher-pseudo-labels.md, concepts/perception-for-planning.md, concepts/inference-latency.md, concepts/evaluation-variance.md, concepts/pdm-lite.md, sources/drivesuprim.md, sources/recogdrive.md, sources/sgdrive.md, sources/wcog-vla.md, sources/wa-jepa.md, sources/da-wam.md, sources/auto-jepa.md, sources/diffusiondrive.md, sources/drivevla-w0.md, sources/metis.md]
created: 2026-09-27
updated: 2026-09-27
confidence: medium
---

**Paper**: Hydra-MDP++: Advancing End-to-End Driving via Expert-Guided Hydra-Distillation
**Authors**: Kailin Li, Zhenxin Li, Shiyi Lan, Yuan Xie, Zhizhong Zhang, Jiayi Liu, Zuxuan Wu, Zhiding Yu, Jose M. Alvarez
**Orgs**: NVIDIA, East China Normal University, Fudan University
**arXiv**: 2503.12820v1
**Code**: `github.com/NVlabs/Hydra-MDP`

---

## Why This Ingest Matters

Hydra-MDP++ was the wiki's **most-cited un-ingested method** (85 mentions). Two parts of the wiki depend on it:

1. **It is the ancestor of the selection-based planners.** The pattern is: score a fixed trajectory vocabulary with heads distilled from the benchmark's own simulator sub-scores, then pick the best candidate by a weighted cost. Planners built on it include [[sources/drivesuprim.md]], the GTRS family (same NVIDIA group), [[sources/da-wam.md]]'s factor heads and [[sources/auto-jepa.md]]'s CLOVER scorer. See [[concepts/selection-based-planning.md]].
2. **It introduced the extended metrics (TL, DDC, LK, EC) and named "EPDMS".** However, **its EPDMS is not the NAVSIM-v2 EPDMS** that the rest of the wiki uses. Several ingested papers copy its rows into official-v2 tables. See [The Two EPDMS Formulas](#two-epdms).

---

## Summary

The architecture is deliberately small:
- **Encoder**: a ResNet-34 (or V2-99) image backbone. Three front cameras are stitched to 256×1024, with two frames fused by a **temporal squeeze-and-excitation** module. The historical frame's gradient is detached. No LiDAR.
- **Decoder**: transformer layers over **k latent queries**, one per trajectory in a fixed vocabulary. The vocabulary is k-means centers of 700K nuPlan trajectories, 40 poses at 10 Hz over 4 s. The figure implies k = 8192; the text never states it.

**Training** has two losses:
- **Imitation**: cross-entropy against a softmax over the negative L2 distances between each vocabulary trajectory and the human log.
- **Hydra-distillation**: per-metric heads regress, with binary cross-entropy, the sub-scores obtained by **running the PDM simulator offline on every vocabulary trajectory in every training scene**, using privileged ground-truth perception.

**Inference** takes the lowest weighted cost:

$$\tilde f(T_i)=-\Big(k_{im}\log S^{im}_i+\sum_{m\in\text{penalties}}k_m\log S^m_i+k_w\log\sum_{w}\text{weight}_w S^w_i\Big)$$

The weights $k$ are chosen by grid search.

**Results**:
- **86.6 PDMS** with ResNet-34 (206 ms on a V100).
- **91.0 PDMS** with V2-99 (271 ms), on NAVSIM-v1 navtest.
- **80.6 / 84.1 "EPDMS"** under its own formula.

The paper also proposes four extended metrics: traffic-light compliance, driving-direction compliance, lane keeping and extended (cross-frame) comfort. They are used both as evaluation metrics and as extra distillation teachers.

---

## Method

![[hydra++_teaser.png|Three paradigms: rule-based planners on privileged perception, imitation-only neural planners, and Hydra-MDP++ distilling rule-based experts into an end-to-end planner]]

*Figure 1: Comparison of three paradigms for autonomous driving.*

![[hydra_fig.png|Hydra-MDP++ architecture: image backbone plus temporal SE perception network producing environment tokens; trajectory vocabulary embedded as queries, refined by transformer encoder/decoder layers; imitation head and multiple Hydra prediction heads distilled from offline PDM-simulator sub-scores; weighted-cost selection at inference]]

*Figure 2: The overall architecture of Hydra-MDP++.*

- **Environment tokens**: $F_{env}=\mathrm{Conv}(\mathrm{TemporalSE}(\mathrm{Concat}(F^{pre}_{img},F^{cur}_{img})))$.
- **Vocabulary queries**: $\mathcal V'_k=\mathrm{Transformer}(\mathrm{Mlp}(\mathcal V_k))+E$, where $E$ is the ego status. Then $\mathcal V''_k=\mathrm{Transformer}(Q{=}\mathcal V'_k,K,V{=}F_{env})$.
- **Losses**: $\mathcal L=\mathcal L_{im}+\mathcal L_{kd}$, with $\mathcal L_{kd}=-\sum_{m,i}\hat S^m_i\log S^m_i+(1-\hat S^m_i)\log(1-S^m_i)$ over every metric and every vocabulary entry. Extended comfort is excluded, because it needs the previous frame's prediction.
- The paper frames this as **"bridging rule-based and neural planners"**. Mechanically, it trains a classifier to predict the benchmark's own score components for a fixed trajectory set, and selects by a tuned combination of those predictions.

### The extended metrics as defined here

| Metric | Definition in this paper | Threshold |
|---|---|---|
| **TL** (traffic lights) | 0 if the ego crosses a crosswalk on red within 4 s | – |
| **DDC** (driving direction) | Projection of each step onto the nearest lane's positive direction must stay within $\tau_D$ | $\tau_D$ = 0.5 m |
| **LK** (lane keeping) | Minimum perpendicular distance to nearby lane segments must stay ≤ $\tau_D$ at every step | $\tau_D$ = 0.5 m |
| **EC** (extended comfort) | RMS difference between the current and previous frame's planned acceleration, jerk, yaw rate and yaw acceleration | 0.7 m/s², 0.5 m/s³, 0.1 rad/s, 0.1 rad/s² |

The introduction says the paper adds *three* metrics (TL, LK, EC); §3.4 and the formula use four, including DDC.

---

## The Two EPDMS Formulas {#two-epdms}

**This paper's EPDMS** (§3.4 and §4.1):

$$\mathrm{EPDMS}_{\text{Hydra++}}=\mathrm{NC}\cdot\mathrm{DAC}\cdot\mathrm{DDC}\cdot\mathrm{TL}\cdot\frac{5\,\mathrm{TTC}+2\,\mathrm{C}+5\,\mathrm{EP}+5\,\mathrm{LK}+5\,\mathrm{EC}}{22}$$

**The NAVSIM-v2 EPDMS used everywhere else in the wiki** (as restated by [[sources/suv.md]]):

$$\mathrm{EPDMS}_{\text{v2}}=\mathrm{NC}\cdot\mathrm{DAC}\cdot\mathrm{DDC}\cdot\mathrm{TLC}\cdot\frac{5\,\mathrm{TTC}+5\,\mathrm{EP}+2\,\mathrm{LK}+2\,\mathrm{HC}+2\,\mathrm{EC}}{16}$$

It also uses human-reference filtering; see [the evaluator correction](../concepts/navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols).

| | Hydra-MDP++ | NAVSIM-v2 |
|---|---|---|
| Comfort term | **C** (per-trajectory comfort), weight 2 | **HC** (history comfort), weight 2 |
| LK weight | **5** | 2 |
| EC weight | **5** | 2 |
| Denominator | 22 | 16 |
| LK definition | Within **0.5 m** of a lane segment → values of **65–70** | Official lane keeping → values of **92–98** |
| Test set | Navtest (the v1 split) | navtest v2 |

The formula is verified on this paper's own V2-99 row: 0.988 · 0.978 · 0.991 · 1 × (5·0.953 + 2·1 + 5·0.840 + 5·0.701 + 5·0.968)/22 = **84.1**, exactly as reported.

**Downstream consequence: a third "protocol" in the wiki's v2 tables.** Four rows from this paper's Table 2 recur across the wiki as if they were NAVSIM-v2 numbers:

| Row | Value | Carried by |
|---|---:|---|
| TransFuser | **77.8** | [[sources/recogdrive.md]] (Table 7), [[sources/sgdrive.md]], [[sources/wcog-vla.md]] |
| VADv2 | **76.6** | Same three |
| Hydra-MDP | **79.8** | ReCogDrive, SGDrive |
| Hydra-MDP++ (ResNet-34) | **80.6** | Same three |

- They sit beside rows evaluated with official NAVSIM-v2 (ARTEMIS 83.1, ReCogDrive 83.6, DiffusionDrive 84.3, all with LK ≈ 96).
- **The column those papers label "HC" and fill with 100 is this paper's "C" column.** ReCogDrive's Table 7 still heads it "C". SGDrive and WCog-VLA relabel it "HC".
- **This resolves an open puzzle on [[sources/wcog-vla.md]]**, which flagged TransFuser 77.8 as "a fourth distinct set" whose NC/DAC/EP/TTC match its own v1 row and whose LK (67.6) matches nothing else. Both follow from this formula: EPDMS computed on the v1 navtest with a strict LK.
- **The wiki's pre-fix list also carried "HydraMDP++ 84.1"** (via [[sources/wa-jepa.md]]'s partition). That is this paper's V2-99 own-formula number. On official NAVSIM-v2, Hydra-MDP++ is reported elsewhere at **81.4** (ResNet-34; LK 94.4, EC 70.9) and **85.6 / 85.1** (ViT-L), by [[sources/drivesuprim.md]], [[sources/da-wam.md]] and the shared baseline block.

---

## Results

### Table 1 — NAVSIM-v1 navtest (PDMS)

| Method | Inputs | Backbone | Latency (ms) | NC | DAC | EP | TTC | C | PDMS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PDM-Closed\* | GT perception | – | – | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 |
| TransFuser | Img+LiDAR | ResNet-34 | 221.2 | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| UniAD | Img | ResNet-34 | 555.6 (A100) | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 |
| PARA-Drive | Img | ResNet-34 | – | 97.9 | 92.4 | 79.3 | 93.0 | 99.8 | 84.0 |
| VADv2† | Img+LiDAR | ResNet-34 | – | 97.9 | 91.7 | 77.6 | 92.9 | 100 | 83.0 |
| **Hydra-MDP++** | Img | ResNet-34 | 206.2 | 97.6 | 96.0 | 80.4 | 93.1 | 100 | **86.6** |
| **Hydra-MDP++** | Img | V2-99 | 271.0 | 98.6 | 98.6 | 85.7 | 95.1 | 100 | **91.0** |

\* PDM-Closed uses ground-truth perception, and "limitations in the brake implementation" may cause extra collisions. † VADv2 is the authors' TransFuser-based reimplementation with a classification decoder. Latencies are on a V100 except UniAD (A100).

**VADv2† is "Hydra-MDP-𝒱8192" elsewhere.** Its submetrics (97.9 / 91.7 / 77.6 / 92.9 / 100 → 83.0) are the row that [[sources/sgdrive.md]] and [[sources/recogdrive.md]] label "Hydra-MDP-𝒱8192" / "Hydra-MDP". ReCogDrive's v1 table also labels the 86.5 row "Hydra-MDP++". Its submetrics (98.3 / 96.0 / 94.6 / 100 / 78.7) are **not** this paper's ResNet-34 row (97.6 / 96.0 / 80.4 / 93.1 → 86.6). SGDrive and WCog-VLA name it correctly: Hydra-MDP-𝒱8192-W-EP, the original Hydra-MDP.

### Table 2 — Navtest with this paper's extended metrics

| Method | Backbone | NC | DAC | EP | TTC | C | TL | DDC | LK | EC | EPDMS (own) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PDM-Closed\* | – | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 100 | 98.7 | 66.7 | 98.0 | 82.8 |
| TransFuser | ResNet-34 | 97.7 | 92.8 | 78.4 | 93.0 | 100 | 99.9 | 98.3 | 67.6 | 95.3 | 77.8 |
| VADv2† | ResNet-34 | 97.3 | 91.7 | 77.6 | 92.7 | 100 | 99.9 | 98.2 | 66.0 | 97.4 | 76.6 |
| Hydra-MDP | ResNet-34 | 97.5 | 96.3 | 80.1 | 93.0 | 100 | 99.9 | 98.3 | 65.5 | 97.4 | 79.8 |
| Hydra-MDP++ | ResNet-34 | 97.9 | 96.5 | 79.2 | 93.4 | 100 | 100 | 98.9 | 67.2 | 97.7 | 80.6 |
| Hydra-MDP++ | V2-99 | 98.8 | 97.8 | 84.0 | 95.3 | 100 | 100 | 99.1 | 70.1 | 96.8 | 84.1 |

This paper's TransFuser row has EP 78.4 / TTC 93.0. The downstream copies print EP 79.2 / TTC 92.8, i.e. TransFuser's *v1* values, with the same 77.8 aggregate. So the copies are not even verbatim.

### Tables 3 / 4 — ablations (ResNet-34)

| Weighted inference (W) | Temporal SE (TS) | Aux. perception (P) | PDMS | EPDMS (own) | EC |
|:-:|:-:|:-:|---:|---:|---:|
| – | – | – | 85.0 | 76.8 | 92.3 |
| ✓ | – | – | 86.5 | 79.8 | 92.3 |
| ✓ | ✓ | – | **86.6** | **80.6** | **97.7** |
| ✓ | ✓ | ✓ | 86.1 | 80.3 | 97.5 |

- **The weighted cost at inference is the main lever**: +1.5 PDMS and +3.0 own-EPDMS, mostly through DAC (92.0 → 96.0). Its weights are grid-searched, and the paper never says on which split.
- **Temporal SE is worth +0.1 PDMS but lifts EC from 92.3 to 97.7**, the cross-frame consistency metric.
- **Auxiliary perception supervision *hurts*** (−0.5 PDMS, −0.3 EPDMS). An early data point for [[concepts/perception-for-planning.md]]'s rule that perception helps only when the planner reads its output.
- Note that the PDMS rows (Table 3) and the EPDMS rows (Table 4) have different submetrics for the same W/TS/P settings. They are different models (see below).

### Table 5 — extended metrics as distillation teachers

| Model | Extended teachers | NC | DAC | EP | TTC | DDC | LK | EC | EPDMS (own) |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-34 | – | 97.6 | 96.0 | 80.4 | 93.1 | 97.5 | 65.5 | 97.4 | 79.5 |
| ResNet-34 | ✓ | 97.9 | 96.5 | 79.2 | 93.4 | 98.9 | 67.2 | 97.7 | 80.6 |
| V2-99 | – | 98.6 | 98.6 | 85.7 | 95.1 | 97.8 | 67.6 | 96.7 | 83.4 |
| V2-99 | ✓ | 98.8 | 97.8 | 84.0 | 95.3 | 99.1 | 70.1 | 96.8 | 84.1 |

**The two headline numbers come from two different models.**
- **The 91.0 PDMS model is the one trained *without* the extended teachers**: its NC/DAC/EP/TTC equal Table 5's first V2-99 row.
- **The 84.1 EPDMS model is trained *with* them**, and has lower DAC (97.8 vs 98.6) and EP (84.0 vs 85.7).
- The paper never reports the PDMS of the extended-teacher model, nor the EPDMS of the 91.0 model under the same table. Its claim that the extended teachers have a "negligible effect on the original metrics" is contradicted by its own V2-99 rows: EP −1.7, DAC −0.8.

![[visualization.png|Two scenes (right turn; dense straight driving) with planned (red) and ground-truth (green) trajectories, and the 8192-candidate vocabulary coloured by predicted NC, DAC, TTC, EP, LK and aggregated EPDMS scores]]

*Figure 3: Planned (red) vs. ground-truth (green) trajectories, and predicted scores over the 8,192 candidates for NC, DAC, TTC, EP, LK and the aggregate. Candidates scoring < 0.1 are omitted.*

---

## Positioning in the Wiki

- **v1 ladder.** 91.0 PDMS (V2-99, three cameras, no LiDAR, no VLM, no RL) sits beside WAM-Diff/ELF-VLA/DualDriveVLA 91.0 and above most 2025–26 VLAs at the time of their publication. [[sources/drivesuprim.md]] later builds on the same Hydra family to reach 93.5.
- **Selection lineage.** The mechanism later reappears in DriveSuprim, GTRS, the heads of [[sources/drivevla-w0.md]]'s anchor-based 90.2, and the scorer cohort that dominates navhard (see [[concepts/navhard-ood-evaluation.md#scorer-cohort]]): offline simulation of a fixed vocabulary, BCE heads per sub-score, and weighted selection. **The critique the wiki now attaches to that cohort starts here.** The distillation target is the benchmark's own scoring function, computed with ground-truth perception, so part of the gain is fitting the evaluator.
- **Privileged supervision.** In [[concepts/teacher-pseudo-labels.md]]'s terms, the teacher is the PDM simulator on ground-truth boxes and maps. The model sees only images at test time, but every training target was computed from privileged state.
- **Data-scale note.** The dataset description, "Navtrain and Navtest … contain 1192 and 136 scenarios", counts **logs**, not samples. That resolves [[sources/metis.md]]'s "navtrain subset (1,192 scenarios)": it is the full navtrain.

---

## Limitations

1. **Its "EPDMS" is not NAVSIM-v2's EPDMS.** It uses different weights (/22), C instead of HC, and a strict LK. Downstream papers copy its rows into official-v2 tables, and three relabel C as HC. See [above](#two-epdms).
2. **The headline 91.0 PDMS and 84.1 EPDMS are two different models** (without and with the extended teachers). The "negligible effect on original metrics" claim is contradicted by the V2-99 rows.
3. **The inference weights are grid-searched on an unstated split.** The weighted cost is the largest ablation effect (+1.5 PDMS), so test-set tuning would matter.
4. **The distillation targets are the benchmark's own sub-scores with ground-truth perception.** The method is partly a learned evaluator, which is the root of the scorer-cohort caveat.
5. **"Surpasses the rule-based teacher PDM-Closed" conflates two things.** The distillation teacher is the PDM *score* over the vocabulary, not the PDM-Closed *planner*, and PDM-Closed is flagged as brake-limited in the same table.
6. **Vocabulary size is never stated in the text** (8192 from the figure), and there is no vocabulary-size ablation. The V2-99 backbone's pretraining is not described.
7. **EC needs the previous frame's plan**, but how navtest (sampled frames) supplies it is not explained.
8. **Latency hardware is mixed** (UniAD on A100, the rest on V100). The 206–271 ms figures are not comparable with later RTX 4090 or H20 numbers.
9. **Single runs, no seeds**, and ablation effects as small as +0.1.
10. **Only navtest**: no navhard, and no official NAVSIM-v2 evaluation of its own.

---

## Key Cross-References

- [[concepts/navsim-benchmark.md]] — the [two EPDMS formulas](../concepts/navsim-benchmark.md#hydra-formula); a third convention lineage (the 77.8 / 76.6 / 79.8 / 80.6 rows); the 91.0 v1 row.
- [[concepts/selection-based-planning.md]] — the origin of vocabulary-scoring by distilled simulator heads.
- [[sources/wcog-vla.md]], [[sources/sgdrive.md]], [[sources/recogdrive.md]] — tables that carry this paper's own-formula rows as v2 numbers.
- [[sources/drivesuprim.md]] — the most successful descendant (93.5 PDMS).
- [[concepts/perception-for-planning.md]] — auxiliary perception −0.5 PDMS inside a selection planner.
