---
title: "Test-Time Trajectory Optimization for Autonomous Driving (TOAD)"
type: source-summary
sources: ["raw/papers/Test-Time Trajectory Optimization for Autonomous Driving.md"]
related: [sources/drivor.md, sources/clover.md, sources/world4scorer.md, sources/maneuverflow.md, sources/momworld.md, sources/drivereferee.md, sources/simscale.md, sources/drivefuture.md, sources/sparsedrivev2.md, sources/hydra-mdp-pp.md, sources/chainflow-vla.md, sources/idol.md, sources/reflectdrive.md, concepts/selection-based-planning.md, concepts/inference-time-safety.md, concepts/navhard-ood-evaluation.md, concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/best-of-n.md, concepts/evaluation-variance.md, concepts/inference-latency.md, concepts/data-driven-simulators.md, research-directions.md]
created: 2026-10-19
updated: 2026-10-19
confidence: medium
---

# Test-Time Trajectory Optimization for Autonomous Driving (TOAD)

**Paper**: Test-Time Trajectory Optimization for Autonomous Driving
**Authors**: Yihong Xu, Éloi Zablocki, Yuan Yin, Elias Ramzi, Ellington Kirby, Alexandre Boulch, Matthieu Cord (an "equal contribution" footnote is unattributed in the HTML)
**Orgs**: valeo.ai (Paris); Sorbonne Université, ISIR (Cord). **Six of the seven are [[sources/drivor.md]] authors.**
**arXiv**: 2606.07170v1 (5 June 2026)
**Code**: promised via valeoai.github.io/TOAD
**Source**: `raw/papers/Test-Time Trajectory Optimization for Autonomous Driving.md`. The clipping has the text, Tables 1–3 and 5–7, Figure 1, Figure 2(a), the first frame of Figure 5, and Figure 4(a) as a table; the Figure 3 timings were read from its inline SVG. Recovered for this page and added to `raw/assets/` (no existing file touched): Figure 2(b) from the arXiv HTML (`DrivorMPC_failure_TOAD.png`), Figure 5's eight frames from the HTML composed into two strips (`toad_evo1_strip.png`, `toad_evo2_strip.png`), and Figure 4(b)–(d), which the HTML lacks, cropped from page 8 of the arXiv PDF at 300 dpi (`toad_fig4.png`).

**Confidence: medium.** Six base planners and two benchmarks give one consistent result, every v1 row passes the closed-form check, and the search's cost is timed. Against that, several things are unstated:
- The "DrivoR" baseline rows are DrivoR's +134k-SimScale checkpoints.
- The paper never says which DrivoR checkpoint supplies the scorer used for all six planners.
- Hyperparameters were tuned on a 7-scene validation set, and every number is a single run.
- The navhard comparison table mixes evaluator versions.
- The HUGSIM protocol is not given.

---

## What It Is

**A planner's learned trajectory scorer, used as a reward and maximized at test time by the Cross-Entropy Method (CEM), starting from the planner's own choice.**

1. Any frozen base planner returns proposals $\mathcal T$ and its pick $\tau_\text{base}$.
2. Trajectories are mapped to controls (acceleration, yaw rate) through an inverse bicycle model. CEM then searches in control space, so every sample is smooth and dynamically feasible.
3. **Objective**: $J(u)=S(\mathrm{BM}(u,v_0);o,g)-\lambda_a\lVert u-u_\text{base}\rVert^2-\lambda_c C_\text{comf}(u)$. $S$ is **DrivoR's disentangled scorer** (frozen, the same one for every planner). The anchor term keeps the search in a trust region. $C_\text{comf}$ is the closed-form kinematic comfort limits, including the transition from the last executed control.
4. **Warm start**: $\mu_0=u_\text{base}$, $\sigma_0=\max(0.5\cdot\mathrm{std}(\mathcal U),\epsilon)$, so exploration scales with the planner's own spread. $M=64$ samples and $E=8$ elites per iteration. $K=5$ iterations for planners that generate proposals, $K=100$ for fixed-vocabulary ones (larger initial spread).
5. **Output**: the converged mean or $\tau_\text{base}$, whichever has the higher $S-\lambda_c C_\text{comf}$, so the result never scores below the base under the scorer.

No training and no new model. For planners other than DrivoR it does add DrivoR's ViT-S encoder and scorer.

**Results**

| Benchmark | Base → + TOAD | Notes |
|---|---|---|
| NAVSIM-v1 PDMS (navtest) | +0.1 to +2.1 over six planners; **DrivoR 94.6 → 94.7** | The "DrivoR 94.6" row is DrivoR + 134k SimScale scenes |
| NAVSIM-v2 navhard two-stage EPDMS | +1.1 to +15.1; **DrivoR 54.6 → 56.3**; iPad 34.7 → 49.8 | 54.6 is also the +134k SimScale model; spread across planners shrinks from 19.9 to 7.3 points |
| HUGSIM (zero-shot, DrivoR) | HD-Score up on 3 of 4 datasets (nuScenes 39.7 → 49.9), route completion down on 3 | No average, scenario set or commit |
| Search cost (A100, batch 1) | 1.9 ms (K = 5) to 20.4 ms (K = 100) | Plus ~100 ms for DrivoR's encoder on any non-DrivoR planner |

---

## Key Takeaways

- **One scorer, six planners, every one improved.** On navhard the gains run from +1.1 (ZTRS) to +15.1 (iPad), and the six planners land between 49.0 and 56.3, where they started between 34.7 and 54.6. On v1 the gains are +0.1 to +2.1 and come almost entirely from ego progress, since the safety columns are near saturation. On navhard they come from no-collision, drivable area and TTC, at some cost in progress.
- **The DrivoR baseline is the SimScale-augmented model.** "DrivoR 94.6" (99.1 / 99.2 / 96.9 / 100 / 91.6) and "DrivoR (ViT-S) 54.6" (stage closed forms 84.1 / 62.7) match DrivoR's rows trained with navval and **134k SimScale synthetic scenes** digit for digit. DrivoR without synthetic data is 93.7 on v1 (trainval) and 48.3 on navhard. TOAD never mentions SimScale. So its "new state of the art" 56.3 is a +134k-scene model plus search, and the scorer applied to the other five planners is probably that checkpoint's.
- **For weak planners, much of the gain is simply DrivoR's scorer.**
  - iPad: re-ranking its own proposals with DrivoR's scorer takes it from 34.7 to 45.6 navhard EPDMS, about 72% of TOAD's +15.1. Search adds the remaining 4.2.
  - Hydra-MDP: the same re-ranking hurts (37.3), while search reaches 49.7.
  - So "the gains come from the search itself, not from a stronger scorer" holds for Hydra-MDP and only partly for iPad.
  - A non-DrivoR planner also pays for DrivoR's encoder: iPad goes from 9.2 ms to 118.8 ms per sample.
- **Fixed-vocabulary scorers fail as rewards.** Searching with GTRS's scorer drops iPad to 23.9; with SparseDriveV2's, to 34.6 (Hydra-MDP: 38.1 and 29.6). These scorers also fail as re-rankers of other planners' proposals. SparseDriveV2's scorer re-ranking Hydra-MDP gives 9.5 EPDMS, with Stage-1 drivable-area compliance 52.9, a collapse that may partly be an interface mismatch rather than poor generalization. The paper's reading: a scorer trained only on its own vocabulary gets no supervision elsewhere, while DrivoR's is trained on freely decoded, gradient-stopped proposals.
- **The search over-optimizes its reward, visibly.**
  - On the 7-scene validation set, EPDMS peaks at $K=5$ iterations (48.5) and falls to 37.2 at $K=50$.
  - In Figure 5 the scorer's reward rises to 1.000 and 0.996 over 100 iterations, while the scene's true EPDMS moves 0.828 → 0.894 and 0.780 → 0.789.
  - The trust region is needed: without the anchor term EPDMS is 44.4, from a standard-normal start 42.7, against 48.5 for the full method.
- **"The set, not the scorer, is the bottleneck" fits weak planners, not the top of v1.** TOAD adds +0.1 to DrivoR on v1. On DrivoR's released 64-candidate pools the oracle is 99.33 against 93.69 selected ([[sources/world4scorer.md]], [[sources/clover.md]]), so a better candidate is usually already in the set. The navhard gain on DrivoR is larger (+1.7).
- **Near-free when the scorer is the planner's own.** For DrivoR the scorer reuses the planner's features, so the search adds 1.9 ms to a roughly 103 ms forward pass.

---

## Method

![[drivor-cem.png|Figure 1: Overview of TOAD. A frozen base planner outputs trajectory proposals and a selected anchor τ_base (left). Trajectories are mapped to controls by the inverse bicycle model (BM⁻¹). Then, a Cross-Entropy Method (CEM) loop runs for K steps in control space: sample controls, roll them out with the bicycle model (BM), score them under 'Scorer + reg' (the learned reward with comfort and anchor regularizers), refit the Gaussian to the highest-scoring elites. The scorer and BM are frozen.]]

**Figure 1**: Base planner → proposals → inverse bicycle model → controls (yaw rate, acceleration) → an initial CEM Gaussian. Each iteration samples control sequences, rolls them out through the bicycle model, scores them ("Scorer + reg"), keeps the elites and refits the Gaussian. After $K$ iterations it returns a trajectory.

### Objective and update (§3.2)

$$J(u)=S\big(\mathrm{BM}(u,v_0);o,g\big)-\lambda_a\,C_\text{anchor}(u)-\lambda_c\,C_\text{comf}(u),\qquad C_\text{anchor}=\lVert u-u_\text{base}\rVert^2$$

$$\mu_0=u_\text{base},\quad\sigma_0=\max(\beta\,\mathrm{std}(\mathcal U),\epsilon);\qquad \mu_k=\mathrm{mean}(\mathcal E_k),\ \sigma_k=\mathrm{std}(\mathcal E_k)$$

$$\tau^\star=\arg\max_{\tau\in\{\mathrm{BM}(\mu_K,v_0),\,\tau_\text{base}\}} S(\tau;o,g)-\lambda_c\,C_\text{comf}\big(\mathrm{BM}^{-1}(\tau,v_0)\big)$$

$C_\text{comf}$ accumulates squared violations of the standard limits on longitudinal and lateral acceleration, jerk, yaw rate and yaw acceleration, over the future controls and the last executed control.

| Hyperparameter | Value |
|---|---|
| Samples per iteration $M$ | 64 |
| Elites $E$ | $M/8=8$ |
| Iterations $K$ | 5 (generated proposals); 100 (vocabularies of more than 8k) |
| Initial-std scale $\beta$; floors | 0.5; 0.1 m/s², 0.025 rad/s |
| $\lambda_c$, $\lambda_a$ | 0.05, 0.5 |

Fixed across planners and benchmarks; tuned with iPad on warmup-two-stage (7 scenes).

---

## Results

### Table 5 (superset of Table 1): NAVSIM-v1 navtest

Residual = closed form − reported, computed here.

| Method | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| PDM-Closed | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 | −0.76 |
| Human driver | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 | −0.03 |
| Ego-status MLP | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 | |
| UniVLA | 96.9 | 91.1 | 91.7 | 96.7 | 76.8 | 81.7 | |
| DrivingGPT | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 | |
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 | |
| LTF | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 | |
| PARA-Drive | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | |
| DriveX-S | 97.5 | 94.0 | 93.0 | 100 | 79.7 | 84.5 | |
| World4Drive | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 | |
| DRAMA | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 | |
| VAD-v2 | 98.1 | 94.8 | 94.3 | 100 | 80.6 | 86.2 | |
| PRIX | 98.1 | 96.3 | 94.1 | 100 | 82.3 | 87.8 | |
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 | |
| DIVER | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 | |
| AutoVLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 | |
| DriveVLA-W0 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 | |
| ReCogDrive | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 | |
| Hydra-MDP++ | 98.6 | 98.6 | 95.1 | 100 | 85.7 | 91.0 | |
| Centaur | 99.5 | 98.9 | 98.0 | 100 | 85.9 | 92.6 | |
| DriveSuprim | 98.6 | 98.6 | 95.5 | 100 | 91.3 | 93.5 | |
| ZTRS (V2-99) | 98.8 | 99.5 | 95.1 | 100 | 74.7 | 86.9 | −0.96 |
| **+ TOAD** | 98.4 | 98.6 | 94.3 | 100 | 82.1 | **89.0** | −1.52 |
| GTRS (V2-99) | 99.0 | 98.9 | 95.9 | 100 | 82.8 | 90.4 | −1.18 |
| **+ TOAD** | 98.9 | 98.1 | 96.3 | 100 | 84.8 | **90.9** | −1.52 |
| Hydra-MDP | 98.5 | 98.7 | 94.9 | 100 | 85.5 | 90.9 | −1.62 |
| **+ TOAD** | 98.7 | 98.0 | 95.4 | 100 | 87.2 | **91.4** | −1.69 |
| iPad | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 | −1.68 |
| **+ TOAD** | 99.0 | 99.3 | 96.4 | 100 | 89.1 | **93.4** | −1.03 |
| RAP-DINO | 99.1 | 98.9 | 96.7 | 100 | 90.3 | 93.8 | −1.10 |
| **+ TOAD** | 99.0 | 99.2 | 96.6 | 100 | 90.1 | **93.9** | −1.13 |
| DrivoR ⚠ | 99.1 | 99.2 | 96.9 | 100 | 91.6 | 94.6 | −1.00 |
| **+ TOAD** | 99.0 | 99.3 | 96.8 | 100 | 91.8 | **94.7** | −1.06 |

The other rows are DrivoR's camera-only block, already checked on [[sources/drivor.md]]. ⚠ The DrivoR row is DrivoR + 134k SimScale ([audit](#table-audit)). The paper reports gains as relative percentages (+2.4% for ZTRS is +2.1 PDMS).

### Table 6 (superset of Table 2): navhard two-stage

Stage closed forms, their product and "product − reported" computed here. On DrivoR's paired table, pre-fix rows sit at −1.0 to +2.0 and corrected rows at −1.1 to −4.5 ([[concepts/navhard-ood-evaluation.md#evaluator-fix]]).

| Method | Stage-1 CF | Stage-2 CF | Product | Reported | Product − reported | Reads as |
|---|---:|---:|---:|---:|---:|---|
| PDM-Closed (privileged) | 82.0 | 63.7 | 52.2 | **56.6** | −4.36 | corrected-like; new value (pre-fix 51.3 elsewhere) |
| TransFuser | 67.8 | 34.8 | 23.6 | 23.1 | +0.49 | pre-fix (DrivoR's pre-fix LTF row) |
| DiffusionDrive | 64.6 | 40.1 | 25.9 | 24.2 | +1.72 | pre-fix-like |
| GuideFlow | 65.1 | 45.0 | 29.3 | 27.1 | +2.20 | pre-fix-like |
| Senna-E2E | 72.5 | 37.8 | 27.4 | 27.2 | +0.23 | pre-fix-like |
| MindDrive | 72.4 | 43.5 | 31.4 | 30.9 | +0.55 | pre-fix-like |
| World4Drive | 66.9 | 47.7 | 31.9 | 34.9 | −3.03 | corrected-like |
| SpanVLA | 81.7 | 50.7 | 41.4 | 40.1 | +1.31 | pre-fix-like |
| DriveSuprim | 79.5 | 53.8 | 42.7 | 42.1 | +0.65 | pre-fix-like |
| DiffVLA | 86.9 | 54.4 | 47.3 | 45.0 | +2.28 | pre-fix-like |
| GTRS-E | 83.0 | 62.4 | 51.9 | 49.4 | +2.46 | pre-fix-like |
| SimScale | 79.2 | 65.7 | 52.0 | 53.2 | −1.18 | corrected-like |
| DriveFuture | 90.1 | 58.3 | 52.5 | **55.5** | −3.00 | corrected-like |
| iPad | 74.7 | 42.8 | 31.9 | 34.7 | −2.77 | corrected-like |
| **+ TOAD** | 78.9 | 60.1 | 47.4 | **49.8** | −2.40 | |
| RAP-DINO (ViT-H) | 80.1 | 46.2 | 37.0 | 39.6 | −2.58 | corrected (DrivoR's paired row) |
| **+ TOAD** | 80.5 | 57.1 | 45.9 | **49.0** | −3.07 | |
| Hydra-MDP | 82.0 | 47.6 | 39.0 | 40.9 | −1.90 | corrected-like |
| **+ TOAD** | 79.0 | 59.4 | 46.9 | **49.7** | −2.77 | |
| GTRS (V2-99) ⚠ | 79.5 | 57.8 | 46.0 | 45.4 | +0.55 | DrivoR's suspect GTRS-Aug row |
| **+ TOAD** | 80.2 | 60.9 | 48.8 | **51.7** | −2.91 | |
| ZTRS (V2-99) | 78.5 | 59.3 | 46.6 | 48.1 | −1.51 | corrected (DrivoR's paired row) |
| **+ TOAD** | 79.9 | 58.9 | 47.0 | **49.2** | −2.16 | |
| DrivoR (ViT-S) ⚠ | 84.1 | 62.7 | 52.7 | 54.6 | −1.87 | DrivoR + 134k SimScale, corrected |
| **+ TOAD** | 83.5 | 65.4 | 54.6 | **56.3** | −1.65 | |

The full 19-sub-score rows are in the clipping's Table 6. The rows above DriveFuture are DriveFuture's leaderboard, carried with its evaluator mix ([audit](#table-audit)). What TOAD changes, Stage 2 (all six planners):

| Planner | S2 NC | S2 DAC | S2 TTC | S2 EP | S2 EC |
|---|---|---|---|---|---|
| iPad | 83.6 → 93.0 | 83.2 → 91.0 | 79.7 → 91.2 | 85.4 → 78.4 | 38.9 → 41.1 |
| RAP-DINO | 83.2 → 89.9 | 83.9 → 89.7 | 80.4 → 87.7 | 86.9 → 81.4 | 52.4 → 51.1 |
| Hydra-MDP | 85.2 → 91.4 | 82.5 → 90.1 | 82.5 → 88.9 | 85.2 → 80.1 | 47.7 → 45.2 |
| GTRS (V2-99) | 88.1 → 91.2 | 88.8 → 91.5 | 85.9 → 89.2 | 98.9 ⚠ → 75.7 | 56.9 → 54.3 |
| ZTRS | 91.1 → 92.3 | 90.4 → 90.2 | 89.8 → 90.4 | 63.6 → 67.8 | 66.1 → 57.3 |
| DrivoR | 92.3 → 93.6 | 91.6 → 94.0 | 90.6 → 92.0 | 75.7 → 74.1 | 44.7 → 47.3 |

### Table 3 / Table 7: what drives the gain (navhard)

| | iPad | Hydra-MDP |
|---|---:|---:|
| Base | 34.7 | 40.9 |
| + Smoothing through the bicycle model | 35.0 | 38.8 |
| + Re-scoring own proposals with DrivoR's scorer | **45.6** | 37.3 |
| + Re-scoring with GTRS's scorer | 34.5 | 40.3 |
| + Re-scoring with SparseDriveV2's scorer | 30.3 | **9.5** |
| + Search with GTRS's scorer | 23.9 | 38.1 |
| + Search with SparseDriveV2's scorer | 34.6 | 29.6 |
| **+ TOAD (search with DrivoR's scorer)** | **49.8** | **49.7** |

Notable sub-scores from Table 7:
- Re-scoring Hydra-MDP with DrivoR's scorer drops Stage-1 HC to 74.9 and EC to 8.0.
- SparseDriveV2's re-scoring of Hydra-MDP gives Stage-1 DAC 52.9 and Stage-2 DAC 51.2.
- Smoothing iPad raises EC (42.2 → 60.9 in Stage 1, 38.9 → 56.8 in Stage 2) but barely moves EPDMS.

### Table 4: HUGSIM, zero-shot (per dataset)

| Method | KITTI360 RC / HDS | nuScenes RC / HDS | Pandaset RC / HDS | Waymo RC / HDS | Mean HDS (unweighted, computed here) |
|---|---|---|---|---|---:|
| LTF | 19.4 / 8.0 | 50.1 / 35.1 | 51.3 / 28.2 | 38.3 / 23.8 | 23.8 |
| UniAD | 23.5 / 5.9 | 49.1 / 37.5 | 49.5 / 36.7 | 56.2 / 44.4 | 31.1 |
| VAD | 17.1 / 3.9 | 43.1 / 24.5 | 38.7 / 15.4 | 29.4 / 11.0 | 13.7 |
| GTRS-Dense | 29.5 / 19.5 | 59.3 / 47.4 | 56.0 / 37.2 | 52.5 / 41.0 | 36.3 |
| DrivoR | 28.8 / 17.7 | 54.2 / 39.7 | 58.3 / 35.1 | 56.5 / 44.2 | 34.2 |
| **DrivoR + TOAD** | 26.9 / 18.4 | 60.9 / **49.9** | 54.5 / 38.0 | 51.1 / 42.8 | 37.3 |

Full NC / DAC / TTC / Comfort columns are in the clipping. Comfort rises on every dataset (KITTI360 93.6 → 99.1, Pandaset 89.8 → 98.4). The unweighted means of LTF, VAD and UniAD come within 0.1–1.6 of DrivoR's 345-scenario averages (23.7, 13.4, 32.7), which suggests this is that set broken down by source dataset. **GTRS-Dense is above DrivoR on three of four datasets**, so on this table TOAD lifts DrivoR from below GTRS-Dense to just above it (37.3 against 36.3).

### Figure 3: per-sample cost (A100, batch 1; read from the SVG)

| Base planner | Base (ms) | Scorer backbone (ms) | TOAD search (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| DrivoR | ≈1.2 (shared backbone) | 101.4 | 1.9 | 104.5 |
| iPad | 9.2 | 107.7 | 1.9 | 118.8 |
| RAP | 676.7 | 101.2 | 4.6 | 782.5 |
| ZTRS | 108.1 | 100.9 | 19.4 | 228.4 |
| GTRS | 91.8 | 101.2 | 20.4 | 213.4 |
| Hydra-MDP | 181.1 | 100.7 | 19.1 | 300.9 |

### Figure 4: ablations with iPad on warmup-two-stage (7 scenes)

![[toad_fig4.png|Figure 4: Ablation study of TOAD with iPad on warmup-two-stage. (a) Component ablation; (b) per-iteration candidates M (K = 5, E = max(1, M/8)); (c) CEM iterations K (M = 64, E = 8); (d) number of elites E (M = 64, K = 5). Cropped from page 8 of the arXiv PDF; the HTML and the clipping lack panels (b)–(d).]]

| (a) Variant | EPDMS |
|---|---:|
| Vanilla iPad | 34.7 |
| Normal initialization (no warm start) | 42.7 |
| Trajectory space (no bicycle model) | 36.4 |
| Without scorer reward | 35.7 |
| Without comfort term | 46.9 |
| Without anchor term | 44.4 |
| **TOAD** | **48.5** |

| (b) $M$ | 2 | 4 | 8 | 16 | 32 | **64** |
|---|---:|---:|---:|---:|---:|---:|
| EPDMS | 38.5 | 36.2 | 36.6 | 45.4 | 45.3 | **48.5** |

| (c) $K$ | 2 | ≈3 | **5** | ≈7 | 10 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| EPDMS | 41.2 | 44.4 | **48.5** | 47.3 | 45.7 | 37.2 |

| (d) $E$ | 1 | 2 | 4 | **8** | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EPDMS | 44.1 | 40.7 | 46.2 | **48.5** | 43.5 | 45.6 | 34.9 |

Values read from the crop; the two $K$ points marked ≈ sit between labelled ticks. Adjacent settings differ by up to 8.8 points ($M$ 8 → 16), and every chosen value is the curve's maximum, on 7 scenes.

---

## Visualizations

![[DrivorMPC_success_TOAD.png|Figure 2(a): Success case. DrivoR's trajectory collides with the car ahead (EPDMS 0.0); DrivoR + TOAD finds a trajectory not in the original set that avoids it (EPDMS 0.75).]]

![[DrivorMPC_failure_TOAD.png|Figure 2(b): Failure case. DrivoR's trajectory stays on the road (EPDMS 0.79); DrivoR + TOAD goes off-road (EPDMS 0.0). Recovered from the arXiv HTML.]]

**Figure 2**: (a) A Las Vegas street: DrivoR's plan runs into the stopped car ahead (EPDMS 0.0); the searched plan swerves around it (0.75), a trajectory that was not among the proposals. (b) *(recovered from the arXiv HTML)* A curving road under a bridge: DrivoR's plan stays on the road (0.79); the searched one leaves it (0.0). The paper attributes such failures to unreliable visual cues.

![[iter_0000.png|Figure 5, first frame (iteration 0): the vanilla best trajectory (EPDMS 0.828), TOAD's trajectory (0.871, reward 0.131) and the human trajectory (0.818) at an intersection.]]

![[toad_evo1_strip.png|Figure 5, scene 1, iterations 0 / 25 / 50 / 100: reward 0.131 → 0.660 → 0.928 → 1.000, EPDMS 0.871 → 0.876 → 0.879 → 0.894 (vanilla 0.828, human 0.818). Composed from the arXiv HTML.]]

![[toad_evo2_strip.png|Figure 5, scene 2, iterations 0 / 25 / 50 / 100: reward 0.492 → 0.871 → 0.991 → 0.996, EPDMS 0.781 → 0.783 → 0.787 → 0.789 (vanilla 0.780, human 0.799). Composed from the arXiv HTML.]]

**Figure 5** *(the clipping has only the first frame; the other seven are from the arXiv HTML)*: Two scenes over 100 iterations, so a fixed-vocabulary planner's $K$. In the first, the CEM mean moves into the right-turn lane with more progress than the human. Its reward saturates at 1.000 while EPDMS rises 0.066 above the vanilla plan. In the second, a roundabout exit, the reward climbs from 0.492 to 0.996 and EPDMS moves by 0.009, staying below the human trajectory's 0.799. The scorer's reward and the evaluator agree in direction here but not in size.

---

## Table Audit {#table-audit}

1. **"DrivoR" is DrivoR + 134k SimScale in both tables.** v1 94.6 (99.1 / 99.2 / 96.9 / 100 / 91.6) and navhard 54.6 (stage closed forms 84.1 / 62.7) are DrivoR's +134k-SimScale rows exactly ([[sources/drivor.md]]). Without synthetic data DrivoR is 93.7 (trainval) / 48.3. TOAD's text never mentions SimScale. Which checkpoint provides the scorer used on the other five planners is not stated.
2. **The "GTRS (V2-99)" navhard base is DrivoR's suspect GTRS-Aug row**: 45.4, Stage-2 EP 98.9 (every other learned row has 53–88), product − reported +0.55 in a block of corrected rows. Its "+13.8%" is measured against that row. TOAD says it runs each planner's public checkpoint and released proposals, but its baseline row is copied.
3. **The navhard table mixes evaluators.** Nine of the thirteen rows taken from DriveFuture's leaderboard are pre-fix-like (+0.2 to +2.5), as the wiki found for that leaderboard ([[concepts/navhard-ood-evaluation.md#evaluator-fix]]). TOAD's own rows and its six bases are corrected-like. "Outperforming the strongest learned method (DriveFuture 55.5)" holds on one scale (+0.8), but GTRS-E 49.4, DiffVLA 45.0 and DriveSuprim 42.1 would each sit about 3 higher on it.
4. **PDM-Closed at 56.6 is new to the wiki.** SimScale and DrivoR print 51.3 under the pre-fix evaluator. This row's sub-scores give a product 4.4 below its total (corrected-like, at the band's edge), and its Stage-1 HC of 87.7 matches [[sources/idol.md]]'s copy rather than DrivoR's 97.7. "Only 0.3 behind the privileged PDM-Closed" depends on it; its source is not given.
5. **The ablation set.** Figure 4 says warmup-two-stage (7 scenes), yet "Vanilla iPad 34.7" equals iPad's navhard test score exactly, while the full method is 48.5 there and 49.8 on navhard. Either the baseline line was copied from the test table or the ablations ran on the test split.
6. **v1 comparison scope.** "DrivoR reaching 94.7 PDMS, the top entry" holds within Table 5. [[sources/chainflow-vla.md]] (94.8 on trainval, May 2026) and [[sources/clover.md]] (94.5) appeared earlier. Every v1 row passes the closed-form check.
7. **HUGSIM** has no scenario set, commit or average. The DrivoR and GTRS-Dense per-dataset rows are new; GTRS-Dense leads DrivoR on three of four datasets.
9. **A fourth Hydra-MDP v1 value** *(lint 2026-10-19)*: TOAD's "Hydra-MDP 90.9" (98.5 / 98.7 / 94.9 / 100 / 85.5) matches neither DrivoR's Hydra-MDP (V2-99) 90.3 nor Hydra-MDP-C 91.0, nor the 86.5 ResNet-34 row most tables carry. TOAD says it runs each planner's public checkpoint, so this may be its own evaluation; the backbone is not given.
8. **Small slips**: "Table 3 and Table 3 reports NAVSIM-v1 … and NAVSIM-v2" (Tables 1 and 2 are meant); "We splut". The table labels as "TransFuser" the row DrivoR's appendix calls LTF (23.1, pre-fix).

---

## Limitations

The paper's own: the search needs a base planner's warm start, and it relies on DrivoR's supervised scorer, whose training needs trajectory-level annotations.

The wiki adds:

1. **Synthetic data under the headline.** The 94.7 and 56.3 rest on DrivoR trained with 134k SimScale scenes, unstated. If the scorer applied to every planner is that checkpoint's, the large navhard gains for weak planners (+6 to +15, against +0.1 to +2.1 on navtest) may carry SimScale's 3DGS training into those planners. That would be the same navtest-small / navhard-large shape the wiki has recorded for SimScale itself ([[concepts/data-driven-simulators.md]]).
2. **Hyperparameters chosen on 7 scenes**, at curve maxima that swing by up to 9 points between neighbours, then fixed for six planners and three benchmarks.
3. **Single runs everywhere.** CEM is stochastic, yet no sampler-seed spread is reported. The v1 gains for the two strongest planners are +0.1.
4. **For non-DrivoR planners, TOAD is partly an ensemble**: it adds DrivoR's ViT-S encoder and scorer (~100 ms). iPad's gain is mostly re-ranking by that scorer.
5. **The reward is over-optimized** beyond a few iterations, so the trust region and $K$ do the work of keeping the search honest. The scorer never predicts NAVSIM-v2's extended comfort, lane keeping or traffic-light terms, which the comfort penalty only partly covers.
6. **Privileged supervision**: DrivoR's scorer learns the NAVSIM oracle's sub-scores, so TOAD maximizes a model of the benchmark's own metric at test time ([[concepts/selection-based-planning.md#top-of-leaderboard]]).
7. **HUGSIM evidence is mixed**: route completion falls on three of four datasets, Waymo's HD-Score falls, and there is no protocol statement.

---

## Relation to Other Wiki Pages

- [[sources/drivor.md]]: the same group. TOAD's scorer is DrivoR's gradient-stopped, sub-score scorer. Its "DrivoR" bases are DrivoR's +134k-SimScale rows, and its navhard block reuses DrivoR's corrected rows (RAP-DINO 39.6, ZTRS 48.1, and the suspect GTRS-Aug 45.4).
- [[sources/world4scorer.md]], [[sources/clover.md]]: the other two papers that improve DrivoR's output without changing its encoder, and both measure on its released 64-candidate pools. World4Scorer swaps the scorer (+0.50 at fixed candidates); CLOVER retrains generator and scorer (+0.74, almost all ranking); TOAD searches beyond the pool (+0.1 on v1, +1.7 on navhard over the +SimScale model). Their oracle figures (99.33 over DrivoR's pools) argue that the pool is not the bottleneck at the top of v1, against TOAD's conclusion.
- [[sources/maneuverflow.md]], [[sources/momworld.md]]: refine-after-scoring designs, both trained. ManeuverFlow refines its best 16 toward evaluator-best candidates and re-scores; MomWorld moves the selected plan by a residual flow. TOAD's refinement is untrained search against the scorer.
- [[sources/chainflow-vla.md]]: also uses a bicycle model in acceleration and yaw rate, inside its autoregressive proposal decoder, and is the 94.8 v1 entry TOAD's table omits.
- [[sources/drivereferee.md]], [[sources/reflectdrive.md]]: inference-time selection and repair, with an analytic verdict and with inpainting respectively ([[concepts/inference-time-safety.md]]).
- [[sources/simscale.md]]: the synthetic data under TOAD's DrivoR base.
- [[sources/drivefuture.md]]: the navhard leader TOAD passes (55.5 → 56.3), also a scored pipeline (GTRS-Dense over 100 proposals). TOAD's table copies DriveFuture's leaderboard, evaluator mix included.
- [[sources/sparsedrivev2.md]]: its scorer is one of the two that fail as a search reward (and as a re-ranker of foreign proposals).
- [[sources/hydra-mdp-pp.md]]: the vocabulary-scoring recipe behind Hydra-MDP, GTRS and ZTRS, all base planners here.
- Not ingested: **iPad**, **Hydra-MDP**, **GTRS / GTRS-Dense / GTRS-E**, **ZTRS**, **RAP** (the base planners), **Centaur** (test-time training), **UniAD**'s occupancy-based post-optimization, CEM and MPPI.

---

## Open Questions

- **Which DrivoR checkpoint is the scorer, and what does TOAD give with the 48.3-navhard model?** It would separate search from SimScale data.
- **How far can the search go before it exploits the scorer?** Report true EPDMS against scorer reward per iteration over a full split, and the sampler-seed spread of CEM.
- **Does iPad's gain survive with a scorer that has iPad's own capacity?** Re-ranking with DrivoR's scorer already gives most of it.
- **Do fixed-vocabulary scorers fail because of their training or their interface?** SparseDriveV2's 9.5 when re-ranking Hydra-MDP looks like a broken input convention as much as poor generalization.
- **Does TOAD help a planner trained with CLOVER's or World4Scorer's scorer?** Those scorers were trained on generated proposals too, and CLOVER's on a deliberately diverse pool.
