---
title: "SimScale: Learning to Drive via Real-World Simulation at Scale"
type: source-summary
sources: ["raw/papers/SimScale_ Learning to Drive via Real-World Simulation at Scale.md"]
related: [sources/simvta.md, sources/rad.md, concepts/data-driven-simulators.md, concepts/navhard-ood-evaluation.md, concepts/navsim-benchmark.md, concepts/selection-based-planning.md, concepts/diffusion-planner.md, concepts/teacher-pseudo-labels.md, concepts/rl-for-ad.md, concepts/evaluation-variance.md, concepts/world-model-for-ad.md, concepts/pdm-lite.md, sources/learning-to-drive-from-a-world-model.md, sources/senna2.md, sources/dreameraD.md, sources/diffusiondrive.md, sources/diffusiondrive-v2.md, sources/hydra-mdp-pp.md, sources/drivefuture.md, sources/momworld.md, sources/metis.md, sources/redrive.md, sources/drivevla-w0.md, research-directions.md]
created: 2026-10-06
updated: 2026-10-14
confidence: medium
---

# SimScale: Learning to Drive via Real-World Simulation at Scale

**Paper**: SimScale: Learning to Drive via Real-World Simulation at Scale
**Authors**: Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang, Yihang Qiu, Guang Li, Junli Wang, Yinfeng Gao, Zhang Zhang, Liang Wang, Hangjun Ye, Tieniu Tan, Long Chen, Hongyang Li
**Orgs**: MAIS, Institute of Automation, Chinese Academy of Sciences; OpenDriveLab at The University of Hong Kong; Xiaomi EV
**arXiv**: 2511.23369v3
**Code and data**: released (github.com/OpenDriveLab/SimScale; project page opendrivelab.com/SimScale)
**Source**: `raw/papers/SimScale_ Learning to Drive via Real-World Simulation at Scale.md`

---

## What It Is

The wiki's first paper whose contribution is **training data, not a planner**. SimScale takes NAVSIM's navtrain logs, reconstructs them as 3D Gaussian Splatting (3DGS) scenes, pushes the ego vehicle off its recorded path, lets the other cars react, asks a "pseudo-expert" what to do from the new state, and renders the result as multi-view camera video. The synthetic samples are mixed with the real ones and any end-to-end planner is trained on the mixture. Nothing changes at inference.

**The pipeline** (Figure 2, Algorithm 1), per training clip:

| Step | Window | What happens |
|---|---|---|
| 1. Perturb | $T \to T{+}H$ | Replace the ego trajectory with one from a 16,384-trajectory human vocabulary, within ±20 m longitudinal, ±2 m lateral and ±20° heading of the log; keep it only if EPDMS ≥ 0.8 and it neither collides, leaves the road nor renders badly |
| 2. React | both windows | Ego follows the trajectory through LQR; other agents follow IDM and respond to it |
| 3. Label | $T{+}H \to T{+}2H$ | A pseudo-expert plans from the perturbed state: **recovery** (retrieve the closest of 103,288 logged human trajectories) or **planner** (PDM-Closed with ground-truth state) |
| 4. Filter | $T{+}H \to T{+}2H$ | Keep the label only if every EPDMS sub-metric is 1, except ego progress, which must exceed 0.5 |
| 5. Render | $T \to T{+}2H$ | Multi-view images from the 3DGS engine (MTGS codebase); blocks with novel-view PSNR < 27 excluded |

Up to five non-overlapping sampling rounds give **147K recovery scenes and 237K planner scenes**, against about 100K real scenes.

**Results** (best variant per planner, navhard combined EPDMS; navtest EPDMS under the pre-fix evaluator):

| Planner | Paradigm | Params | navhard w/o → w/ sim | navtest w/o → w/ sim |
|---|---|---:|---|---|
| LTF | Regression | 56M | 24.4 → 30.2 (+5.8) | 81.5 → 84.4 (+2.9) |
| DiffusionDrive | Diffusion | 61M | 27.5 → 32.6 (+5.1) | 84.2 → 85.9 (+1.7) |
| GTRS-Dense (ResNet-34) | Vocabulary scoring | 67M | 38.3 → 46.9 (+8.6) | 82.3 → 84.6 (+2.3) |
| GTRS-Dense (V2-99) | Vocabulary scoring | 83M | 41.9 → 48.0 (+6.1) | 84.0 → 84.8 (+0.8) |
| GTRS-Dense (V2-99), three-model ensemble | Vocabulary scoring | 3 × 83M | 41.9 → **50.9** | – |

---

## Key Takeaways

- **Simulated off-log states with corrective labels help every planner tried, and help most on navhard.** All four planners gain 5–9 EPDMS on navhard and 1–3 on navtest. On navhard the gain is larger in Stage 2 (+5.6 to +7.6 per-stage EPDM) than in Stage 1 (+0.8 to +5.3). The direction is consistent: all ten simulation-trained rows (four planner configurations × two or three label types) beat their real-only baseline on navhard.
- **Training data alone breaks the navhard Stage-2 lane-keeping band for unscored planners.** LTF goes 47.9 → 58.3 and DiffusionDrive 46.4 → 60.8 in Stage-2 LK, with no scorer. The wiki had attributed the escape from the 45–50 band to candidate scoring ([[concepts/navhard-ood-evaluation.md#lk-correction]]).
- **A privileged planner that never sees pixels also loses lane keeping in Stage 2.** PDM-Closed, run on ground-truth state, drops from 99.3 to 73.7 LK (−25.6) and scores 51.3 combined. Learned planners trained on real data only lose 37–50 points of LK in the same table. So a sizeable part of the Stage-2 drop is in the scenarios themselves, not in the rendering.
- **The exploratory pseudo-expert scales better than the conservative one.** Recovery labels (retrieved human trajectories that return to the log) peak early and fall off; PDM-Closed labels keep improving longer. Recovery is better only at the smallest data size for V2-99.
- **For a vocabulary scorer, reward labels on simulated states beat expert trajectories, but only by +0.3 to +0.8**, and dropping expert supervision on the *real* data costs −0.7 and −3.1. The "reward is all you need" heading rests on margins within single-run noise.
- **Multimodal planners scale; the regressor does not.** Under planner labels LTF flattens at about a 1:1 sim:real ratio while DiffusionDrive keeps rising. The paper's explanation: more simulated states from the same real scene give several valid labels per scene, which a single-mode regressor averages.
- **Reactive agents and visual fidelity both matter, by about a point.** Reactive simulation beats non-reactive with fewer samples (+0.7 / +1.1 with 120K vs 141K); high-PSNR scenes beat low-PSNR ones in 8 of 8 matched cells (+0.1 to +1.6).
- **The paper's figures and its tables disagree.** Every GTRS-Dense table value is 0.5–1.9 EPDMS above the highest point plotted for the same configuration, and Figure 4 starts LTF at about 27.5 where Table 1 says 24.4. See [Reading the Results §1](#figure-table).
- **The 53.2 navhard "SimScale" row in the wiki does not come from this paper.** [[sources/drivefuture.md]]'s Table 1 carries SimScale (V2-99) at 53.2 with full sub-scores; the best number in this v3 paper is 50.9. See [§5](#provenance).

---

## Method

### Preliminary

A training sample spans $T{+}H$ steps: $T$ history frames in, $H$ future frames out. SimScale runs two consecutive simulations of length $H$ from each clip, so a synthetic sample spans $T{+}2H$.

### 3DGS simulation data engine

$\Phi(K_t, E_t, \{x_{i,t}, y_{i,t}, \theta_{i,t}\}_{i=1}^N)$ renders RGB from camera intrinsics $K_t$, extrinsics $E_t$ (from the ego pose and the ego-to-camera transform) and each non-ego vehicle's position and yaw.

- **Preprocessing** (following MTGS): exposure alignment across simultaneous views using projected LiDAR; LiDAR points split into background and per-vehicle groups by NAVSIM 3D boxes, used to initialize Gaussians.
- **Block-wise reconstruction**: each block is a spatio-temporal range; static background and movable vehicles are separate assets placed at requested poses. **Blocks with novel-view PSNR < 27 are dropped.**
- **Behaviour simulation is decoupled from sensor simulation**: trajectories are simulated first and rendered afterwards.

### Pseudo-expert scene simulation (Algorithm 1)

For each perturbation $a_\text{per} \in \mathcal{A}_\text{per}$ and each $t \in \{T, T{+}H\}$:
1. **Pseudo action**: $\tilde a_{t:t+H} = a_\text{per}$ at $t=T$, else $\pi_\text{exp}(\tilde o_t, \tilde s_t)$.
2. **Reactive state simulation**: $\tilde s_{t:t+H} = \mathcal T(\tilde a_{t:t+H}, \tilde s_t)$ (LQR ego, IDM agents).
3. **Sensor simulation**: $\tilde o_{t:t+H} = \Phi(\tilde s_{t:t+H})$.
4. **Reward filtering / labelling**: $\tilde r_{t:t+H} = f(\tilde a_{t:t+H}, \tilde s_{t:t+H})$; break if filtered.

**Trajectory perturbation.** Diversity comes from the clustered vocabulary; plausibility from the thresholds; uniform coverage from sparse sampling of endpoints on an interleaved grid ($\delta_\text{lon}=5$ m, $\delta_\text{lat}=0.5$ m). Infeasible perturbations are filtered non-reactively first; reactive checks run only on the sparse set.

**Recovery-based expert.** Each candidate trajectory of horizon $H$ is summarized as

$$\mathbf m = [\tilde v^x_t, \tilde v^y_t, \tilde\theta_t, \tilde x_{t+H-1}, \tilde y_{t+H-1}, \tilde\theta_{t+H-1}]$$

and the expert retrieves $\tilde a_{t:t+H} = \arg\min_{a\in\mathcal V_h}\|\mathbf m(a) - \mathbf m_r\|_1$, where $\mathbf m_r$ is built from the perturbed state and the **logged state at $T{+}2H$**. The result is human-like and steers back to the log; "all simulation trajectories successfully converge back to the human expert's final waypoint" (Figure 10).

**Planner-based expert.** PDM-Closed ([16]) with ground-truth states: $\tilde a_{t:t+H} = \mathbf P(\tilde s_{t:t+H})$. Rule- and cost-based, less human-like, more exploratory.

### Sim-real co-training

Batches are drawn at random from $\mathcal D \cup \mathcal D_\text{sim}$. Real data is kept fixed; simulation data is added in non-overlapping rounds.

- **Regression and diffusion planners** (imitation only):
$$\arg\min_\theta \mathbb E_{(a,o)\sim\mathcal D\cup\mathcal D_\text{sim}}\big[\mathcal L_\text{im}(a, \pi_\theta(\hat a\mid o))\big]$$
- **Scoring planner with pseudo-expert**: imitation plus reward-prediction loss on both sets, $\lambda\mathcal L_\text{im} + \mathcal L_r$, where $r$ distils EPDMS sub-scores.
- **Scoring planner, reward only on simulation**:
$$\arg\min_\theta \mathbb E_{\mathcal D}[\lambda\mathcal L_\text{im} + \mathcal L_r] + \mathbb E_{\mathcal D_\text{sim}}[\mathcal L_r]$$

The reward-only variant is the Hydra-MDP recipe ([[sources/hydra-mdp-pp.md]]) applied to simulated states: each vocabulary trajectory in each simulated scene gets sub-score labels, and no expert trajectory is imitated there. The paper calls it reward learning or offline RL; there is no policy-improvement loop.

### Planners

Official implementations of LTF, DiffusionDrive and GTRS-Dense, with two changes: images at 2048×512 and **LiDAR removed**, to match navhard. All trained from scratch on H20 GPUs, "to saturation", with identical settings with and without simulation data.

---

## Figures

*Figure 1 (teaser: the pipeline and the scaling trend across the three planner families) is linked to arXiv in the clipping and was not saved to `raw/assets/`.*

![[pipeline 2.png|Three-panel schematic. (a) Trajectory perturbation: from the real ego at time T, dashed purple sampled trajectories fan out to a simulated ego position at T+H beside the real ego. (b) Pseudo-expert trajectory generation from T+H to T+2H, shown twice: recovery-based, where the simulated ego heads back toward the real ego's path, and planner-based, where it follows its own planned route; grey other-agent boxes react to the simulated ego. Rendered camera frames are shown at T, T+H and T+2H for the real and simulated paths]]

*Figure 2: Pseudo-expert scene simulation. (a) Perturbation on $T \to T{+}H$; (b) reactive rollout and pseudo-expert labelling on $T{+}H \to T{+}2H$, recovery-based (b1) and planner-based (b2).*

![[data.png|Two bar-and-line charts of simulated data accumulated over five sampling rounds, with a horizontal line at the real-world data size of about 100K. (a) Recovery-based expert: cumulative 45K, 80K, 109K, 130K, 147K. (b) Planner-based expert: cumulative 65K, 120K, 167K, 205K, 237K. Per-round increments shrink in both]]

*Figure 3: Simulation data per sampling round. The recovery expert's stricter constraints accept fewer samples per round; the planner-based expert passes the real-data size after round 2.*

![[curve_scaling.png|Four scaling plots of navhard EPDMS against total samples on a log axis, each with fitted log-quadratic curves, shaded residual bands and marked inflection points. (a) LTF ResNet-34, 56M: recovery and planner curves both peak near 2×10^5 samples, planner higher, around 0.30. (b) DiffusionDrive ResNet-34, 61M: planner curve almost straight, rising to about 0.33; recovery curve peaks near 0.29. (c) GTRS-Dense ResNet-34, 67M: planner and reward-scoring curves rise to about 0.45 and flatten; recovery peaks near 0.42 and falls. (d) GTRS-Dense V2-99, 83M: reward and planner curves keep rising to about 0.47 and 0.46; recovery peaks early near 0.45 and falls. Fitted equations are printed under each panel]]

*Figure 4: Scaling with simulation data under different pseudo-experts. The marked "inflection point" is the vertex of the fitted parabola. Printed fits (EPDMS as a fraction, $x=\log N$):*

| Panel | Recovery | Planner | Reward only |
|---|---|---|---|
| (a) LTF | $-0.15x^2+1.62x-4.01$ | $-0.25x^2+2.69x-6.78$ | – |
| (b) DiffusionDrive | $-0.23x^2+2.44x-6.17$ | $-0.02x^2+0.33x-0.83$ | – |
| (c) GTRS-Dense R34 | $-0.62x^2+6.48x-16.63$ | $-0.38x^2+4.12x-10.73$ | $-0.39x^2+4.20x-10.93$ |
| (d) GTRS-Dense V2-99 | $-0.54x^2+5.74x-14.66$ | $-0.05x^2+0.64x-1.48$ | $-0.01x^2+1.47x-3.73$ |

*The base of the logarithm and the unit of $N$ are not stated, and none of the obvious choices reproduces the plotted curves from the printed coefficients, so the coefficients are recorded but not used here. Point values read from an enlarged copy of the figure are in [Reading the Results §1](#figure-table).*

![[vis_scene.png|Four simulated scenarios, each with a bird's-eye map showing a purple perturbed trajectory and a red pseudo-expert trajectory continuing from it, beside two rendered camera views. (a) Off-center lane drift, front and front-right views of a wide urban street. (b) Near collision, the ego very close to a dark SUV in the front view. (c) Leaving the drivable area, front and front-left views of a curved hotel driveway. (d) Cutting in, an orange SUV alongside in the front-right view]]

*Figure 5: Four simulated out-of-distribution scenes from navtrain: (a) off-centre lane drift, (b) near collision, (c) leaving the drivable area (called "departure" in the text), (d) cutting in. Purple: perturbed trajectory (used as history); red: pseudo-expert trajectory (supervision).*

![[position.png|Two block diagrams. (a) 3DGS-based online RL: a policy acts in a reconstructed world, receives a reward and an image, and is updated by RL repeatedly from policy K to K+1. (b) SimScale: a human state is perturbed N times in the world, a pseudo-expert supplies a trajectory, the trajectory and rendered image are stored N times, and the stored data trains any policy with IL or RL]]

*Figure 6: 3DGS-based online RL (e.g. RAD) against SimScale. Online RL collects states from the current policy; SimScale collects them from a fixed perturbation sampler and stores them for any planner.*

![[curve_scaling_react.png|Two scaling plots of navhard EPDMS for GTRS-Dense with reactive versus non-reactive simulation data. (a) ResNet-34: the non-reactive curve has two points, about 0.430 and 0.437, and flattens; the reactive curve rises to about 0.45. (b) V2-99: non-reactive points about 0.449 and 0.456, flattening; reactive rises to about 0.47]]

*Figure 7: Reactive against non-reactive simulation data, GTRS-Dense. Each non-reactive curve is fitted to its real-only point and two simulation points.*

![[data_statistic.png|(a) Histogram of EgoMLP EPDMS on simulated and real data: about 40 percent of simulated samples and about 32 percent of real ones score zero; averages 49.2 for simulation and 61.7 for real. (b) Bar chart of EgoMLP EPDMS over the 20 most frequent scenario tags in the simulated data, from about 0.61 for stationary_in_traffic down to about 0.41 for traversing_crosswalk, with the 49.2 average marked]]

*Figure 8: An ego-status-only planner (EgoMLP, trained on navtrain) scores lower on simulated states (average 49.2) than on real ones (61.7). Crosswalk and pedestrian tags are the hardest simulated scenarios.*

![[curve_scaling_real.png|Two line plots of navhard EPDMS against real data size at four points, only-real versus sim-real at a fixed ratio. (a) GTRS-Dense ResNet-34: only real about 0.333, 0.356, 0.377, 0.383; sim-real about 0.408, 0.416, 0.433, 0.447; gains labelled +22.4%, +16.7%, +14.9%, +16.7%. (b) V2-99: only real about 0.370, 0.390, 0.412, 0.419; sim-real about 0.414, 0.432, 0.450, 0.467; gains +12.1%, +10.8%, +9.2%, +11.5%]]

*Figure 9: Simulation gains at four real-data sizes (reward-only co-training, five rounds, simulation generated from the same scenes). The text gives the sizes as 10K, 20K, 50K and 100K; the axis is labelled 1×10⁵ to 10×10⁵.*

| Real data | R34 only real | R34 sim-real | Gain | V2-99 only real | V2-99 sim-real | Gain |
|---|---:|---:|---:|---:|---:|---:|
| 10K | ≈33.3 | ≈40.8 | +22.4% | ≈37.0 | ≈41.4 | +12.1% |
| 20K | ≈35.6 | ≈41.6 | +16.7% | ≈39.0 | ≈43.2 | +10.8% |
| 50K | ≈37.7 | ≈43.3 | +14.9% | ≈41.2 | ≈45.0 | +9.2% |
| 100K | ≈38.3 | ≈44.7 | +16.7% | ≈41.9 | ≈46.7 | +11.5% |

*Values read from the figure; percentages as labelled. The 100K endpoints (≈44.7 and ≈46.7) are the full-navtrain reward-only runs that Table 1 reports as 46.9 and 48.0.*

![[vis_expert_recovery_1.png|Qualitative results of the recovery-based expert with real and simulation data: bird's-eye trajectories and rendered views in which the simulated trajectories converge back to the human expert's final waypoint]]

*Figure 10: Recovery-based expert. All simulated trajectories return to the human's final waypoint.*

![[vis_expert_planner_1.png|Qualitative results of the planner-based expert at a Y-branch intersection, with real and simulated trajectories and rendered views]]

*Figure 11: Planner-based expert, Y-branch intersection.*

![[vis_expert_planner_2.png|Qualitative results of the planner-based expert on a dense urban avenue, with real and simulated trajectories and rendered views]]

*Figure 12: Planner-based expert, dense urban avenue.*

![[vis_expert_planner_3.png|Qualitative results of the planner-based expert on a narrow local road, with real and simulated trajectories and rendered views]]

*Figure 13: Planner-based expert, narrow local road. Per the text, planner-based trajectories vary more and cover more of the scene than recovery ones.*

![[vis_scene_add.png|Additional simulated out-of-distribution scenes from navtrain with perturbed and pseudo-expert trajectories and rendered camera views]]

*Figure 14: Additional simulated scenes.*

---

## Tables

### Table 7 (superset of Table 1): navhard, two-stage

PDM-Closed uses ground-truth symbolic input; all others use cameras. S = per-stage EPDM score. **Bold method rows are the ones Table 1 reports.** The last column is computed here: the product of the two reported stage scores.

| Method | Backbone | Sim. | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | S | EPDMS | S1×S2 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PDM-Closed** | – | – | 1 | 94.4 | 98.8 | 100 | 99.5 | 100 | 93.5 | 99.3 | 87.7 | 36.0 | – | **51.3** | – |
| | | | 2 | 88.1 | 90.6 | 96.3 | 98.5 | 100 | 83.1 | 73.7 | 91.5 | 25.4 | – | | |
| **LTF** | ResNet-34 | w/o | 1 | 97.3 | 80.2 | 97.8 | 99.3 | 83.4 | 96.2 | 92.9 | 97.8 | 71.1 | 61.3 | **24.4** | 24.0 |
| | | | 2 | 79.4 | 69.0 | 85.6 | 98.5 | 83.8 | 76.7 | 47.9 | 97.0 | 70.6 | 39.2 | | |
| LTF | ResNet-34 | recovery | 1 | 96.4 | 78.4 | 98.9 | 99.8 | 80.5 | 96.2 | 92.7 | 97.6 | 78.2 | 60.7 | 29.8 | 28.1 |
| | | | 2 | 88.9 | 71.1 | 91.8 | 99.0 | 77.3 | 85.5 | 53.8 | 96.8 | 47.5 | 46.3 | | |
| **LTF** | ResNet-34 | planner | 1 | 96.1 | 85.3 | 99.4 | 99.3 | 84.7 | 94.7 | 93.6 | 97.6 | 77.3 | 66.3 | **30.2** | 29.7 |
| | | | 2 | 85.5 | 66.9 | 91.6 | 99.1 | 93.0 | 81.1 | 58.3 | 95.1 | 42.9 | 44.8 | | |
| **DiffusionDrive** | ResNet-34 | w/o | 1 | 96.8 | 86.0 | 98.8 | 99.3 | 84.0 | 95.8 | 96.7 | 97.6 | 79.6 | 66.7 | **27.5** | 27.0 |
| | | | 2 | 80.1 | 72.8 | 84.4 | 98.4 | 85.9 | 76.6 | 46.4 | 96.3 | 72.8 | 40.5 | | |
| DiffusionDrive | ResNet-34 | recovery | 1 | 97.2 | 88.4 | 99.1 | 99.8 | 83.9 | 96.0 | 96.7 | 97.6 | 76.9 | 69.4 | 30.4 | 28.9 |
| | | | 2 | 82.4 | 67.7 | 89.1 | 98.6 | 89.0 | 77.6 | 53.8 | 95.2 | 46.8 | 41.7 | | |
| **DiffusionDrive** | ResNet-34 | planner | 1 | 97.2 ‡ | 88.7 | 99.3 | 99.3 | 82.8 | 96.9 | 98.0 | 97.3 | 59.6 | 67.5 | **32.6** | 31.6 |
| | | | 2 | 82.4 ‡ | 72.1 | 92.9 | 98.5 | 92.1 | 80.6 | 60.8 | 95.4 | 31.9 | 46.8 | | |
| **GTRS-Dense** | ResNet-34 | w/o | 1 | 99.3 | 96.6 | 99.6 | 100 | 57.4 | 99.5 | 92.6 | 89.5 | 16.4 | 67.1 | **38.3** | 37.4 |
| | | | 2 | 92.8 | 88.6 | 95.5 | 99.4 | 55.9 | 91.3 | 55.7 | 91.1 | 35.7 | 55.8 | | |
| GTRS-Dense | ResNet-34 | recovery | 1 | 97.6 | 94.2 | 99.3 | 99.6 | 76.7 | 97.6 | 94.9 | 97.1 | 37.8 | 69.8 | 43.0 | 41.4 |
| | | | 2 | 92.0 | 88.1 | 94.3 | 98.6 | 83.6 | 89.8 | 58.1 | 91.4 | 25.9 | 59.3 | | |
| GTRS-Dense | ResNet-34 | planner | 1 | 98.7 | 94.2 | 99.7 | 100 | 74.3 | 98.2 | 95.8 | 96.7 | 34.7 | 70.8 | 46.1 | 45.2 |
| | | | 2 | 94.2 | 92.1 | 95.0 | 99.0 | 79.9 | 91.8 | 58.9 | 91.0 | 32.9 | 63.9 | | |
| **GTRS-Dense** | ResNet-34 | reward | 1 | 97.6 | 96.4 | 99.3 | 100 | 75.7 | 97.8 | 93.3 | 97.3 | 32.9 | 72.4 | **46.9** § | 45.9 |
| | | | 2 | 94.3 | 92.7 | 95.1 | 99.5 | 80.2 | 91.5 | 56.2 | 90.6 | 28.3 | 63.4 | | |
| **GTRS-Dense** | V2-99 | w/o | 1 | 98.9 | 94.9 | 99.1 | 100 | 76.1 | 98.4 | 93.8 | 94.9 | 37.8 | 70.4 | **41.9** | 41.2 |
| | | | 2 | 89.9 | 90.5 | 94.1 | 99.3 | 77.6 | 88.5 | 56.0 | 92.0 | 30.2 | 58.5 | | |
| GTRS-Dense | V2-99 | recovery | 1 | 99.1 | 98.2 | 99.4 | 100 | 71.9 | 99.1 | 95.6 | 95.8 | 32.4 | 73.4 | 46.4 | 46.3 |
| | | | 2 | 95.0 | 90.8 | 94.6 | 99.4 | 75.5 | 93.5 | 57.4 | 93.7 | 36.8 | 63.1 | | |
| GTRS-Dense | V2-99 | planner | 1 | 99.3 | 97.1 | 99.9 | 100 | 67.2 | 99.3 | 94.0 | 94.4 | 23.6 | 71.7 | 47.7 | 47.0 |
| | | | 2 | 95.6 | 91.9 | 95.0 | 98.9 | 76.7 | 93.7 | 61.9 | 90.8 | 38.0 | 65.5 | | |
| **GTRS-Dense** | V2-99 | reward | 1 | 99.3 | 97.6 | 99.4 | 100 | 71.5 | 99.6 | 96.0 | 95.8 | 30.7 | 72.5 | **48.0** | 47.4 |
| | | | 2 | 94.9 | 94.3 | 94.5 | 99.3 | 79.0 | 92.6 | 57.8 | 93.4 | 30.9 | 65.4 | | |

‡ Table 1 prints this row's NC as **97.4 / 86.4**; Table 7 prints 97.2 / 82.4, which are the recovery row's values. The closed form of the Stage-2 sub-scores is 44.2 with 86.4 and 42.1 with 82.4, against a reported 46.8; Table 1's value fits the pattern of the other rows (closed form 1–4 below the reported stage score), so Table 7's NC is probably copied from the row above.
§ Table 3 prints the same configuration, with the same stage scores, as **45.9**.

The reported combined score is 0.1 to 1.7 above the product of the stage scores on every row, consistent with the wiki's reading that the combined score multiplies stages per scene before averaging ([[concepts/navhard-ood-evaluation.md#does-the-combined-score-really-multiply-the-stages]]). Table 1 marks the gains as ↑24%, ↑19%, ↑22% and ↑15%, and the per-stage gains as +5.0/+5.6, +0.8/+6.3, +5.3/+7.6 and +2.1/+6.9.

### Table 8 (superset of Table 2): navtest

**Bold rows are the ones Table 2 reports.** The last column is computed here: the closed form of the printed sub-scores minus the reported EPDMS ([[concepts/navsim-benchmark.md#residual-sign]]).

| Method | Backbone | Sim. | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS | Residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Human Agent** | – | – | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | **90.3** | +4.1 |
| **LTF** | ResNet-34 | w/o | 97.7 | 94.0 | 99.3 | 99.8 | 87.2 | 96.7 | 95.5 | 98.3 | 82.9 | **81.5** | +2.3 |
| LTF | ResNet-34 | recovery | 97.9 | 95.1 | 99.5 | 99.8 | 87.7 | 97.2 | 97.2 | 98.3 | 87.1 | 83.6 | +2.5 |
| **LTF** | ResNet-34 | planner | 98.3 | 95.6 | 99.6 | 99.8 | 87.1 | 97.5 | 97.2 | 98.3 | 88.2 | **84.4** | +2.6 |
| **DiffusionDrive** | ResNet-34 | w/o | 98.4 | 95.5 | 99.5 | 99.8 | 87.5 | 97.5 | 96.9 | 98.4 | 87.7 | **84.2** | +2.8 |
| DiffusionDrive | ResNet-34 | recovery | 98.4 | 96.7 | 99.6 | 99.8 | 87.6 | 97.5 | 97.5 | 98.3 | 87.1 | 85.4 | +2.8 |
| **DiffusionDrive** | ResNet-34 | planner | 98.5 | 97.1 | 99.6 | 99.8 | 87.4 | 97.8 | 98.0 | 98.3 | 87.5 | **85.9** | +2.9 |
| **GTRS-Dense** | ResNet-34 | w/o | 97.6 | 97.5 | 99.0 | 99.9 | 87.9 | 97.0 | 95.9 | 97.5 | 55.9 | **82.3** | +1.4 |
| GTRS-Dense | ResNet-34 | recovery | 98.2 | 97.9 | 99.5 | 99.9 | 87.2 | 97.8 | 96.3 | 97.6 | 56.5 | 83.4 | +1.8 |
| GTRS-Dense | ResNet-34 | planner | 97.8 | 98.3 | 99.5 | 99.9 | 89.2 | 97.3 | 97.3 | 96.9 | 55.1 | 84.0 | +1.5 |
| **GTRS-Dense** | ResNet-34 | reward | 98.4 | 98.8 | 99.4 | 99.9 | 88.0 | 98.0 | 96.5 | 97.6 | 58.0 | **84.6** | +1.9 |
| **GTRS-Dense** | V2-99 | w/o | 97.6 | 98.5 | 99.5 | 99.9 | 89.5 | 97.2 | 96.8 | 97.2 | 57.2 | **84.0** | +1.8 |
| GTRS-Dense | V2-99 | recovery | 98.3 | 98.7 | 99.4 | 99.9 | 88.9 | 97.9 | 97.1 | 97.6 | 55.9 | 84.5 | +1.9 |
| GTRS-Dense | V2-99 | planner | 98.7 | 99.0 | 99.5 | 99.8 | 86.8 | 98.3 | 95.9 | 97.9 | 58.5 | 84.5 | +2.2 |
| **GTRS-Dense** | V2-99 | reward | 97.6 | 98.9 | 99.6 | 99.9 | 89.9 | 97.1 | 97.2 | 97.7 | 58.7 | **84.8** | +1.8 |

**Every row's residual is +1.4 to +2.9, the pre-fix signature, and the Human Agent is the pre-fix 90.3.** The evaluator is not named in the paper.

### Table 3: Reward-only supervision, GTRS-Dense, navhard

| Backbone | Real | Real expert | Sim. | Sim. expert | S1 | S2 | EPDMS |
|---|:-:|:-:|:-:|:-:|---:|---:|---:|
| ResNet-34 | ✓ | ✓ | | | 67.1 | 55.8 | 38.3 |
| ResNet-34 | ✓ | | | | 66.8 | 54.9 | 37.6 |
| ResNet-34 | ✓ | ✓ | ✓ | ✓ | 70.8 | 63.9 | 46.1 |
| ResNet-34 | ✓ | ✓ | ✓ | | 72.4 | 63.4 | **45.9** § |
| V2-99 | ✓ | ✓ | | | 70.4 | 58.5 | 41.9 |
| V2-99 | ✓ | | | | 68.1 | 55.6 | 38.8 |
| V2-99 | ✓ | ✓ | ✓ | ✓ | 71.7 | 65.5 | 47.7 |
| V2-99 | ✓ | ✓ | ✓ | | 72.5 | 65.4 | 48.0 |

### Table 4: Reactive vs non-reactive simulation, GTRS-Dense reward-only, navhard

| Type | Rounds | Sim. samples | ResNet-34 | V2-99 |
|---|---:|---:|---:|---:|
| Non-reactive | 2 | 141K | 43.7 | 45.6 |
| Reactive | 2 | 120K | 44.4 | 46.7 |
| Reactive | 3 | 167K | 45.0 | 47.9 |

Non-reactive sampling accepts 21K more samples (fewer collisions) and scores lower.

### Table 5: Curation configuration

| Term | Notation | Value |
|---|---|---|
| *3DGS data engine* | | |
| PSNR threshold (blocks excluded below) | PSNR | < 27 |
| *Trajectory perturbation* | | |
| Clustered human-trajectory vocabulary | $\mathcal V_c$ | 16,384 |
| Relative heading range | $\Delta\theta$ | ±20° |
| Longitudinal shift range | $r_\text{lon}$ | ±20 m |
| Lateral shift range | $r_\text{lat}$ | ±2 m |
| Longitudinal sampling step | $\delta_\text{lon}$ | 5 m |
| Lateral sampling step | $\delta_\text{lat}$ | 0.5 m |
| Reward filter | $R^\text{per}_\text{EPDMS}$ | ≥ 0.8 |
| *Pseudo-expert trajectory generation* | | |
| Reward filter | $R^\text{exp}_\text{EPDMS}$ | $S_m = 1\ \forall m \ne \text{EP}$; $S_\text{EP} > 0.5$ |
| Recovery: full human-trajectory vocabulary | $\mathcal V_h$ | 103,288 |
| Planner: privileged planner | $\mathbf P$ | PDM-Closed |

### Table 6: Models and training

| | LTF | DiffusionDrive | GTRS-Dense |
|---|---|---|---|
| Sensors | 3 cameras | 3 cameras | 3 cameras |
| Resolution | 2048×512 | 2048×512 | 2048×512 |
| Horizon / frequency | 4 s / 2 Hz | 4 s / 2 Hz | 4 s / 10 Hz |
| Backbone | ResNet-34 | ResNet-34 | ResNet-34 / V2-99 |
| Parameters | 56M | 61M | 67M / 83M |
| Auxiliary tasks | Detection, segmentation | Detection, segmentation | None |
| GPUs | 8 × H20 | 8 × H20 | 32 × H20 |
| Epochs | 100 | 100 | 50 |
| Total batch size | 512 | 512 | 352 |
| Initial LR | 2.83×10⁻⁴ | 6×10⁻⁴ | 4×10⁻⁴ |
| Schedule / optimizer | Constant / Adam | Cosine / AdamW | Cosine / AdamW |

GTRS-Dense trains on 16,384 vocabulary trajectories and infers over 8,192 on navhard and 16,384 on navtest. DiffusionDrive uses 20 anchors.

### Table 9: Multi-expert ensemble, GTRS-Dense, navhard

Predicted sub-scores of each vocabulary trajectory are averaged across models.

| Backbone | Sim. | Expert | S1 | S2 | EPDMS |
|---|:-:|---|---:|---:|---:|
| ResNet-34 | | | 67.1 | 55.8 | 38.3 |
| ResNet-34 | ✓ | recovery | 69.8 | 59.3 | 43.0 |
| ResNet-34 | ✓ | planner | 70.8 | 63.9 | 46.1 |
| ResNet-34 | ✓ | reward | 72.4 | 63.4 | 46.9 |
| ResNet-34 | ✓ | ensemble of the three | 72.8 | 65.1 | 47.7 |
| V2-99 | | | 70.4 | 58.5 | 41.9 |
| V2-99 | ✓ | recovery | 73.4 | 63.1 | 46.4 |
| V2-99 | ✓ | planner | 71.7 | 65.5 | 47.7 |
| V2-99 | ✓ | reward | 72.5 | 65.4 | 48.0 |
| V2-99 | ✓ | ensemble of the three | 74.8 | 67.6 | **50.9** |
| Both backbones | ✓ | ensemble of all six | 75.4 | 66.4 | 50.4 |

### Table 10: Visual fidelity, GTRS-Dense, navhard

About 10K samples per round from blocks with PSNR < 27, against matched random samples from PSNR ≥ 27.

| Backbone | PSNR | Pseudo-expert R1 | Pseudo-expert R2 | Reward-only R1 | Reward-only R2 |
|---|---|---:|---:|---:|---:|
| ResNet-34 | < 27 | 41.0 | 41.1 | 40.8 | 41.6 |
| ResNet-34 | ≥ 27 | 41.9 | 41.2 | 41.3 | 42.9 |
| V2-99 | < 27 | 44.4 | 44.5 | 43.7 | 44.8 |
| V2-99 | ≥ 27 | 45.4 | 46.1 | 44.9 | 46.3 |

High fidelity wins all eight cells, by +0.1 to +1.6.

### Results given only in the text

| Quantity | Value |
|---|---|
| navhard split | 244 Stage-1 scenarios, 4,164 Stage-2 synthetic scenarios |
| navtest split | 12,146 scenarios |
| Real training set | navtrain, "100K interactive real-world scenarios" (103,288 human trajectories in the retrieval vocabulary) |
| Simulated scenes | 147K recovery, 237K planner |
| Ensemble of all six models | "no additional EPDMS gains" (50.4) |

---

## Reading the Results

### 1. The figures and the tables do not agree {#figure-table}

Values read from an enlarged copy of Figure 4 (precision about ±0.2), against Table 7:

| Planner | Labels | Real-only point in Fig. 4 | Table 7 w/o | Highest plotted point | Five-round point | Table 7 |
|---|---|---:|---:|---:|---:|---:|
| LTF | recovery | ≈27.5 | **24.4** | ≈29.3 | ≈28.4 | 29.8 |
| LTF | planner | ≈27.5 | **24.4** | ≈30.2 | ≈30.2 | 30.2 |
| DiffusionDrive | recovery | ≈27.5 | 27.5 | ≈30.5 | ≈30.0 | 30.4 |
| DiffusionDrive | planner | ≈27.5 | 27.5 | ≈32.8 | ≈31.8 | 32.6 |
| GTRS-Dense R34 | recovery | 38.3 | 38.3 | ≈42.5 | ≈40.8 | **43.0** |
| GTRS-Dense R34 | planner | 38.3 | 38.3 | ≈44.8 | ≈44.3 | **46.1** |
| GTRS-Dense R34 | reward | 38.3 | 38.3 | ≈45.0 | ≈44.7 | **46.9** |
| GTRS-Dense V2-99 | recovery | 41.9 | 41.9 | ≈45.9 | ≈44.4 | **46.4** |
| GTRS-Dense V2-99 | planner | 41.9 | 41.9 | ≈46.5 | ≈45.6 | **47.7** |
| GTRS-Dense V2-99 | reward | 41.9 | 41.9 | ≈47.2 | ≈46.7 | **48.0** |

- **For all six GTRS-Dense curves, the table value is 0.5–1.9 above every plotted point.** For LTF and DiffusionDrive the table equals the plotted maximum within 0.5.
- **The figures agree with each other.** Figure 9's full-navtrain endpoints (≈44.7, ≈46.7) are Figure 4's five-round reward points, and Table 4's reactive ResNet-34 value (45.0) is Figure 4(c)'s highest reward point.
- **Three numbers circulate for one configuration** (ResNet-34, reward-only, full data): 46.9 (Tables 1, 7, 9), 45.9 (Table 3) and about 45.0 at best in the figures. The abstract's "up to +8.6 EPDMS on navhard" is 46.9 − 38.3; on the plotted runs it is about +6.7.
- **LTF's curve starts at about 27.5, DiffusionDrive's real-only value, not at Table 1's 24.4.** If the plotted point is SimScale's own real-only LTF, LTF gains about +2.7, not +5.8. If it is a plotting slip, the LTF curve has no anchor.
- The paper does not say whether tables and figures come from different checkpoints, evaluation runs or splits. Table 1 is captioned "navhard Leaderboard", which could mean an official server, but the text describes navhard as the 244/4,164-scenario split without distinguishing.

### 2. Baseline provenance {#baseline-provenance}

Table 7's LTF and DiffusionDrive "w/o" rows are **digit-identical across all 18 sub-scores and both stage scores** to the rows [[sources/metis.md]] labels "copied from GTRS" (and that [[sources/redrive.md]] copies from Metis). Either SimScale's real-only baselines are GTRS's published numbers, or GTRS's numbers come from the same runs. SimScale states that every model "follows the same training strategy, whether using only the navtrain split or augmented with simulation data", which would make them its own runs. Figure 4's 27.5 starting point for LTF (§1) fits the first reading for LTF.

GTRS-Dense was retrained: its V2-99 baseline is 41.9 here against the published 41.7 carried on [[concepts/navhard-ood-evaluation.md#scorer-cohort]], with different sub-scores.

### 3. What the navhard gain is made of {#navhard-gain}

| | Stage-1 gain (per-stage EPDM) | Stage-2 gain | navtest gain |
|---|---:|---:|---:|
| LTF | +5.0 | +5.6 | +2.9 |
| DiffusionDrive | +0.8 | +6.3 | +1.7 |
| GTRS-Dense R34 | +5.3 | +7.6 | +2.3 |
| GTRS-Dense V2-99 | +2.1 | +6.9 | +0.8 |

- **Stage 2 gains most in every planner.** Stage 2 re-renders the scene from 3DGS at a displaced ego pose; navtest uses only real images.
- **Two explanations fit, and the paper runs no experiment that separates them.** (a) The policy has learned to recover from off-log states, which Stage 2 is built from. (b) The policy has seen thousands of 3DGS renders and is less disturbed by 3DGS appearance in Stage 2. Table 10 shows rendering quality matters by up to 1.6 points. A control that renders the *unperturbed* logged trajectory through the same engine, with the human label, would separate (a) from (b).
- **The renderer is close to the benchmark's.** SimScale's engine is the MTGS codebase; The reference list gives MTGS and the navhard benchmark paper (Cao et al., *Pseudo-Simulation for Autonomous Driving*) both with authors "T. Li" and "H. Li", matching SimScale authors Tianyu Li and Hongyang Li of OpenDriveLab (the list gives initials only). Whether Stage 2 is rendered with MTGS is not stated in this paper. If it is, explanation (b) becomes a train–test renderer match.
- **The mechanism the wiki worries about runs in this direction.** [[sources/learning-to-drive-from-a-world-model.md]] observed that rendering artifacts that grow with the offset from the log can be read by a policy as a cue for the corrective label. SimScale's labels are corrective by construction (recovery labels point back to the log), so the same shortcut is available. No probe is reported.

### 4. Stage-2 lane keeping {#stage2-lk}

| Method | Scores candidates? | Stage-1 LK | Stage-2 LK | Drop |
|---|:-:|---:|---:|---:|
| PDM-Closed (ground-truth state, no images) | rule-based | 99.3 | **73.7** | −25.6 |
| LTF, real only | No | 92.9 | 47.9 | −45.0 |
| LTF + planner labels | No | 93.6 | **58.3** | −35.3 |
| DiffusionDrive, real only | No | 96.7 | 46.4 | −50.3 |
| DiffusionDrive + planner labels | No | 98.0 | **60.8** | −37.2 |
| GTRS-Dense V2-99, real only | Yes | 93.8 | 56.0 | −37.8 |
| GTRS-Dense V2-99 + planner labels | Yes | 94.0 | 61.9 | −32.1 |

- **Unscored planners leave the 45–50 band with training data alone.** The wiki's previous correction read the band as broken by candidate scoring ([[concepts/navhard-ood-evaluation.md#lk-correction]]). Here two unscored planners reach 58.3 and 60.8, level with or above DriveFuture's scored 58.3.
- **PDM-Closed sees no rendering and still loses 25.6 points.** Its NC (94.4 → 88.1), DAC (98.8 → 90.6) and TTC (93.5 → 83.1) also fall. Stage 2 is harder in its scenarios, not only in its images. That bounds the rendering share of the learned planners' 35–50-point drop to the remainder, about 10–25 points, which may be rendering, perception or planning.
- PDM-Closed is a centreline follower, so its lane keeping is a generous reference. It is still the first number in the wiki that measures the Stage-2 drop without pixels.

### 5. The circulating 53.2 {#provenance}

[[sources/drivefuture.md]] lists "SimScale, V2-99" at 53.2 with Stage-1 sub-scores 99.6 / 99.1 / 99.9 / 100.0 / 69.6 / 99.6 / 95.8 / 95.6 / 28.4 and Stage-2 94.5 / 94.2 / 95.8 / 99.2 / 75.8 / 92.8 / 60.1 / 96.1 / 43.2.
- No row in this paper (v3) has those sub-scores. Its best is the three-model ensemble at 50.9 (sub-scores not printed); its best single model is 48.0.
- The row is internally plausible: the product of its closed-form stage scores is 52.0, and SimScale's own rows sit 0.1–1.7 above that product.
- Its source is probably a leaderboard submission or another version of the paper; the wiki cannot tell. The cohort table on [[concepts/navhard-ood-evaluation.md#scorer-cohort]] keeps 53.2 with this note.

### 6. Reward-only labels and the benchmark metric {#reward-only}

- **The margin over expert labels is small**: +0.8 (ResNet-34; −0.2 if Table 3's 45.9 is right) and +0.3 (V2-99) on navhard; +0.6 and +0.3 on navtest. Navhard has 244 Stage-1 scenes, an SE of roughly a point ([[concepts/evaluation-variance.md]]).
- **Dropping the expert on real data hurts**: −0.7 (ResNet-34) and −3.1 (V2-99). The scorer needs imitation on real data and does better without it on simulated data.
- **Every label is filtered or computed by the benchmark's own metric.** Perturbations need EPDMS ≥ 0.8; pseudo-expert trajectories need every sub-metric at 1; reward labels are EPDMS sub-scores. The evaluation is the same metric on NAVSIM splits only. This is the scorer-cohort pattern ([[concepts/selection-based-planning.md#top-of-leaderboard]]) extended to new states, and it carries the same untested question: does it transfer to a benchmark whose metric was not learned?

### 7. Scaling claims {#scaling}

- Each curve is fitted with three parameters to five or six points; the non-reactive curves in Figure 7 to three.
- **Adjacent rounds reverse by up to 2.7 EPDMS.** DiffusionDrive recovery reads about 30.5 → 27.8 → 28.6 → 30.0 over rounds 2–5; LTF planner 30.2 → 29.3 → 30.2 over rounds 3–5. Each point is a separately trained model with 7–33% more total data than the one before. These reversals are the wiki's first look at run-to-run scatter on navhard, and they are as large as several of the differences the fits are used to argue.
- **"LTF degrades past 1:1" rests on one point** (round 4, ≈29.3); round 5 is back at ≈30.2. **"DiffusionDrive scales linearly" rests mostly on rounds 4 and 5** (≈32.8, ≈31.8).
- **No extrapolation is tested.** The largest sim:real ratio is about 2.3:1.
- **Real-data scaling (Figure 9)**: the absolute gain from simulation does not shrink between 10K and 100K real scenes (ResNet-34 about +7.5 → +6.4; V2-99 about +4.4 → +4.8). The relative gain is largest at 10K for ResNet-34 (+22.4%).

### 8. Comfort is inherited from the teacher {#comfort}

PDM-Closed scores EC 36.0 / 25.4. Students trained on its labels lose Stage-2 extended comfort: LTF 70.6 → 42.9, DiffusionDrive 72.8 → 31.9. The paper names this as a limitation of a rule-based teacher (E.1). On navtest the effect is absent or reversed (LTF EC 82.9 → 88.2), so it appears where the teacher's trajectories differ most from the log.

---

## Relationships

- **[[sources/learning-to-drive-from-a-world-model.md]]**: the other paper in the wiki that trains a policy on off-log states with corrective labels. openpilot is on-policy (states come from the current policy), uses reprojection or a learned world model, and labels with a future-anchored learned teacher; it is tested in MetaDrive and deployed. SimScale's states come from a fixed perturbation sampler, are rendered by 3DGS with reacting agents, labelled by PDM-Closed or retrieval, and tested on NAVSIM. Neither compares against the other's state sampler or teacher.
- **[[sources/senna2.md]]** and [[sources/rad.md]]: RL in 3DGS reconstructions (RAD with log-replayed agents). Figure 6 positions SimScale against this "3DGS-based online RL". The trade: SimScale's data works for any planner and can be reused; online RL follows the policy's own state distribution.
- **[[sources/dreameraD.md]]**: off-log states from a latent world model instead of a renderer, with a learned reward and GRPO.
- **[[sources/simvta.md]]** *(2026-10-14)*: the same lab (SimScale's first author is its project lead) uses a 3DGS engine (World Engine) on navtrain in a different role. The renders are video-loss targets for a world-action model during GRPO, rendered along the current policy's own executed paths, and the policy never sees them as input. Its one sampler comparison, renders cached once from the initial policy against renders refreshed each round (90.4 vs 91.3 PDMS), is the nearest thing in the wiki to the fixed-versus-on-policy test this paper's limitations call for.
- **[[sources/hydra-mdp-pp.md]]**: the scorer-distillation recipe; reward-only co-training is that recipe on simulated states.
- **[[sources/drivefuture.md]]** and **[[sources/momworld.md]]**: both build on GTRS-Dense, the planner that benefits most here; DriveFuture's navhard table is where the 53.2 SimScale row comes from.
- **[[sources/diffusiondrive.md]]** and **[[sources/diffusiondrive-v2.md]]**: DiffusionDrive is one of the three planners. DiffusionDriveV2's diagnosis (imitation supervises one anchor per scene) is the counterpart of SimScale's finding that a multimodal head absorbs several labels per scene.
- **[[sources/metis.md]]**, **[[sources/redrive.md]]**: carry SimScale's LTF and DiffusionDrive baseline rows as "copied from GTRS" (§2).
- **[[sources/drivevla-w0.md]]**: also a data-scaling paper (world models amplify real-data scaling); SimScale scales synthetic data at fixed real data.

---

## Limitations

**Evidence**

1. **Tables and figures disagree** for GTRS-Dense (tables 0.5–1.9 higher than any plotted point) and for LTF's real-only point (27.5 against 24.4). The headline +8.6 is the table value; the plotted runs give about +6.7.
2. **Baseline provenance is unclear** for LTF and DiffusionDrive; their rows equal rows another paper attributes to GTRS.
3. **Renderer domain and recovery are confounded.** No control renders unperturbed log trajectories; the training renderer and the navhard Stage-2 renderer may share a codebase.
4. **NAVSIM only**, and every label is filtered or computed with the benchmark's metric. No Bench2Drive, HUGSIM, nuScenes or real-world test.
5. **Single runs** on a 244-scene split; several claims rest on sub-point margins (reward-only vs planner labels, ensemble on ResNet-34 +0.8, reactive vs non-reactive +0.7).
6. **Scaling fits** have three parameters for five or six noisy points, and run-to-run reversals reach 2.7 EPDMS.
7. **navtest uses the pre-fix evaluator** (residuals +1.4 to +2.9; Human Agent 90.3), unstated in the paper.
8. **Text and tables disagree in places**: "LTF and DiffusionDrive ... gains exceeding 20%" (DiffusionDrive is 19%, and GTRS-Dense ResNet-34's +22% is above it, against "weaker baseline models benefit the most"); Figure 9's axis is labelled 1×10⁵ to 10×10⁵ for 10K–100K; Table 7's DiffusionDrive planner NC duplicates the recovery row; Table 3 prints 45.9 for Table 1's 46.9.

**Method**

9. **Not on-policy.** Perturbations are drawn from a fixed vocabulary around the log, one step deep; the policy never generates its own states (the paper lists this, E.1).
10. **The planner-based teacher is privileged and rule-based** (ground-truth state, poor comfort: EC 36.0 / 25.4 on navhard), and its comfort deficit passes to students.
11. **Other agents follow IDM**, which limits interaction diversity (E.2).
12. **Cost is not reported**: reconstruction, rendering of 384K multi-view clips, reactive simulation, and 16,384-trajectory sub-score labels per simulated state for GTRS-Dense.
13. **Camera-only planners at 2048×512.** LiDAR was removed from LTF and DiffusionDrive, so the baselines are not their published configurations.
14. The ensemble triples inference cost.

**Source conversion**

15. Figure 1 is not in `raw/assets/`. All other figures (2–14) are embedded; Figure 4 and Figure 9 values are read from enlarged images. All ten tables are reproduced (Tables 1 and 2 as marked rows of Tables 7 and 8, with the one cell where Tables 1 and 7 differ). The reference list is truncated in the clipping after reference 103.

---

## Key Cross-References

- [[concepts/data-driven-simulators.md]] — 3DGS reconstruction as a training-data generator; pseudo-expert labels as a supervision source; a fixed perturbation sampler against on-policy rollouts.
- [[concepts/navhard-ood-evaluation.md]] — the navhard effect, the Stage-2 LK result, PDM-Closed's 51.3, the 53.2 provenance and the LTF/DiffusionDrive baseline rows.
- [[concepts/navsim-benchmark.md]] — a fully pre-fix navtest table.
- [[concepts/selection-based-planning.md]] — reward-only labels on simulated states; a sub-score-averaging ensemble.
- [[concepts/diffusion-planner.md]] — multimodal heads scale with diverse labels; a regressor does not.
- [[concepts/teacher-pseudo-labels.md]] — PDM-Closed and log retrieval as teachers; a student inherits the teacher's comfort.
- [[concepts/rl-for-ad.md]] — "offline RL" by sub-score regression, and the comparison with 3DGS online RL.
- [[concepts/evaluation-variance.md]] — run-to-run reversals along a scaling axis on navhard.
