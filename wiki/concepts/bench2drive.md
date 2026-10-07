---
title: Bench2Drive Benchmark
type: concept
sources: ["raw/papers/Outcome-Grounded World Modeling for Autonomous Driving.md", "raw/papers/Raw2Drive_ Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving(in CARLA v2).md", "raw/papers/Think2Drive_ Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2).md", "raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md", raw/papers/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md, raw/papers/DynVLA_ Learning World Dynamics for Action Reasoning in Autonomous Driving.md, raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md, raw/papers/DeepSight_ Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving.md]
related: [sources/world4scorer.md, sources/raw2drive.md, sources/think2drive.md, sources/momworld.md, sources/orion.md, sources/linkvla.md, sources/autovla.md, sources/automot.md, sources/unidrivevla.md, sources/dynvla.md, sources/drive-jepa.md, sources/deepsight.md, concepts/navsim-benchmark.md, concepts/dual-system-vla.md, concepts/diffusion-planner.md, concepts/pdm-lite.md, research-directions.md]
created: 2026-04-15
updated: 2026-10-06
confidence: high
---

## What It Is

Bench2Drive is a **closed-loop autonomous driving benchmark** built on CARLA V2. Unlike NAVSIM's non-reactive simulator, Bench2Drive uses fully interactive simulation where surrounding agents respond to ego vehicle behavior. Models are evaluated on pre-defined routes across multiple CARLA towns (Town02, Town05) with varying traffic density and scenario complexity.

---

## Key Contrast with NAVSIM

| Aspect | NAVSIM | Bench2Drive |
|--------|--------|-------------|
| Simulator | Non-reactive (pre-recorded replay) | Interactive (CARLA V2 agents) |
| Agent behavior | Fixed regardless of ego | React to ego vehicle actions |
| Primary sensor | 8 cameras + LiDAR (real-world data) | Camera-only (simulation) |
| Primary metric | PDMS (safety sub-metrics) | DS + SR (route completion) |
| RL training usage | Common (GRPO via PDM simulator) | Less common |
| Data domain | Real-world (nuPlan/OpenScene) | Simulation (CARLA) |

The interactive simulation is both Bench2Drive's strength (tests reactive behavior) and limitation (sim-to-real gap; no sensor noise; simplified dynamics).

---

## Metrics

### Driving Score (DS) — primary
Composite metric combining route completion with infraction penalties:
- Route completion percentage × infraction multipliers (collision, red light, wrong lane, agent blocking)
- Collision sets DS to 0 for that route segment
- Range: 0–100; higher is better

### Success Rate (SR)
Percentage of routes completed without any major infraction. More binary than DS but directly measures end-to-end planning robustness.

### Secondary metrics (reported inconsistently across papers)

| Metric | Meaning |
|--------|---------|
| Efficiency | Route completion speed relative to baseline |
| Comfort | Jerk and acceleration smoothness |
| Multi-Ability Mean | Average across 5 scenario types (see below) |

---

## Scenario Types (Multi-Ability)

Five standardized scenario categories; reported separately by ORION and LinkVLA:

| Scenario | Description |
|----------|------------|
| Merging | Lane merge under opposing traffic |
| Overtake | Passing a slower vehicle |
| Brake | Emergency stop in response to obstacle |
| Give-Way | Yielding to crossing or oncoming agents |
| Traffic-Sign | Recognizing and obeying signs (stop, yield, speed limit) |

---

## SOTA Progression

| Method | DS ↑ | SR (%) ↑ | Efficiency | Comfort | Notes |
|--------|------|----------|-----------|---------|-------|
| UniAD-Base | 45.81 | 16.36 | 129.21 | 43.58 | Multi-task E2E, no VLM |
| VAD | 42.35 | 15.00 | 157.94 | 46.01 | — |
| DriveAdapter | 64.22 | 33.08 | 70.22 | 16.01 | — |
| DriveTransformer | 63.46 | 35.01 | 100.64 | 20.78 | — |
| Drive-JEPA | 64.52 | 36.82 | 157.85 | 30.24 | V-JEPA + MTD proposal planner; improves over iPad/DriveTransformer but is far below current VLA leaders |
| Raw2Drive | 71.36 | 50.24 | 214.17 | 22.42 | **The only camera-input RL entry**: model-based RL in a camera world model aligned to a privileged one (Think2Drive), with privileged reward heads and a privileged policy as initialization; its table stops at DriveAdapter 64.22 and omits ORION. See [[sources/raw2drive.md]] |
| World4Scorer | 73.27 | 46.36 | – | – | **A NAVSIM generator–scorer adapted to CARLA** (94.0 PDMS on NAVSIM): adds a route point 20 m ahead, uses CARLA's ground-truth ego pose / velocity / acceleration (the reference agents estimate them from GNSS and IMU), keeps turn commands through intersections and re-ranks by route agreement at weight 4σ; none of these ablated. Trained on the 1000-clip base set. Its table stops at "ReCogDrive 71.36", a value identical to Raw2Drive's DS that the wiki's ReCogDrive page does not record. SR is below Raw2Drive's. See [[sources/world4scorer.md#bench2drive]] |
| Hydra-NeXt *(not ingested)* | 73.86 | 50.00 | 197.76 | 20.68 | From [[sources/momworld.md]]'s table |
| MomWorld | 74.07 | 50.00 | 198.87 | 20.43 | Momentum latent world model; base planner unstated and the row is Hydra-NeXt's within about a point on most columns; see [below](#momworld) |
| ORION | 77.74 | 54.62 | 151.48 | 17.38 | VLM + VAE planner; +14.28 DS vs. prior SOTA at time |
| AutoVLA | 78.84 | 57.73 | 146.93 | 39.33 | Physical codebook; higher comfort than ORION |
| UniDriveVLA | 78.37 | — | — | — | Best result without PDM-Lite oracle; MoT 3-expert |
| SimLingo | 85.07 | 67.27 | 259.23 | 33.67 | Fast MLP head (34ms) |
| AutoMoT | 87.34 | — | — | — | Frozen Qwen3-VL-4B + 1.6B AE; async MoT (7.6× speedup) |
| DeepSight (Think2Drive) | 86.23 | 71.36 | 201.71 | 16.11 | Parallel 5-frame DINOv3 latent world model in BEV + adaptive CoT; strongest Think2Drive-protocol VLM in its own table (see expert caveat below) |
| HiP-AD *(not ingested)* | 86.77 | 69.09 | 203.12 | 19.36 | Non-VLM; from [[sources/momworld.md]]'s table |
| DynVLA | 88.34 | 72.73 | — | — | Dynamics CoT with ego/environment VQ dynamics tokens; below LinkVLA, above AutoMoT; table omits LinkVLA |
| SparseDriveV2 *(not ingested)* | 89.15 | 70.00 | 199.84 | 18.32 | Non-VLM scorer ("scoring is all you need"); from [[sources/momworld.md]]'s table |
| **LinkVLA** | **91.01** | **74.55** | **255.84** | **34.62** | **Shared codebook + C2F; current SOTA** |
| *Think2Drive (privileged expert)* | *91.85* | *85.41* | *269.14* | *25.97* | *Reference, not a leaderboard entry: ground-truth BEV input; generated the base-set training data. From [[sources/raw2drive.md]]'s and [[sources/dynvla.md]]'s tables; see [below](#expert) and [[sources/think2drive.md]]* |
| *PDM-Lite (privileged expert)* | *97.02* | *92.27* | – | – | *Reference: rule-based; the alternative expert-data protocol. From [[sources/dynvla.md]]'s table* |

**PDM-Lite caveat**: PDM-Lite is a privileged oracle planner (uses ground-truth waypoints or HD map access) that some Bench2Drive methods use as a fallback or auxiliary module. UniDriveVLA (78.37) is explicitly noted as the best result *without* PDM-Lite. Methods that use PDM-Lite score higher but are not fairly comparable to methods that do not. The precise PDM-Lite usage for each method above is not always disclosed.

**Expert-data protocol caveat (Think2Drive vs. PDM-Lite)**: separate from inference-time privileged fallback, papers also differ in which *expert planner generated their training data*. Bench2Drive's official base set uses **Think2Drive** expert data; some methods (SimLingo, DiffusionDrive, AutoMoT) train on stronger **PDM-Lite** expert data, which raises DS. [[sources/deepsight.md]] deliberately restricts its comparison to the Think2Drive protocol and marks PDM-Lite rows (SimLingo 85.94, DiffusionDrive 77.68) as reference-only, reporting +7.39 DS over AutoVLA (78.84) as its headline. **Attribution is inconsistent across papers**: DeepSight labels AutoVLA as Think2Drive, while AutoMoT's table labels AutoVLA as PDM-Lite. Treat cross-paper DS deltas as fragile unless both papers use the same expert-data protocol. DeepSight's own table omits LinkVLA (91.01), DynVLA (88.34), and AutoMoT (87.34), so its "SOTA" is comparison-scope-limited to the Think2Drive VLM set.

### The Expert Behind the Base Set: Think2Drive {#expert}

*(2026-10-06)* [[sources/think2drive.md]] is the planner whose driving became Bench2Drive's official training data ("officially generated using the Think2Drive paradigm", [[sources/unidrivevla.md]]).

- **What it is.** A privileged planner: a 128×128×34 ground-truth BEV rasterization in, 30 discrete throttle/steer/brake actions out. It is trained by model-based RL: a DreamerV3 world model learns CARLA's transitions, rewards and terminations, and the planner learns from 15-step imagined rollouts. Three days on one A6000.
- **How good the teacher is.** 91.85 DS / 85.41 SR on Bench2Drive ([[sources/dynvla.md]]'s table), against PDM-Lite's 97.02 / 92.27. In its own paper: 83.8 DS on its 390-route single-scenario benchmark, and 98.6% route completion on the 7–10 km CARLA v2 routes. Its per-scenario success rates range from 0.61 (AccidentTwoWays) to 1.0, mean 0.84, and three of its four weakest scenarios are merges into or across traffic flows.
- **How close the students are.** The best Think2Drive-protocol student on this page, [[sources/deepsight.md]], is 5.6 DS below the teacher (86.23). LinkVLA (91.01, protocol undisclosed) is 0.84 below it. The leaders are approaching the expert whose demonstrations they imitate, which raises the question of whether the base set bounds the leaderboard. PDM-Lite-protocol methods learn from a teacher 5.2 DS stronger.
- **The students' weakest abilities only partly match the teacher's.** Merging and give-way are the lowest multi-ability scores for every method tabulated here. Merging is also the teacher's lowest (81.27), but **give-way is not: the teacher scores 90.00 there** against 0–50 for every student in the wiki ([[sources/raw2drive.md]]'s Table 4 gives Think2Drive's profile: Merging 81.27, Overtaking 83.92, Emergency brake 90.24, Give way 90.00, Traffic sign 87.67; mean 86.26). The give-way gap is not inherited from the expert's data.
- **Coupling a student more tightly to the expert did not close the gap.** [[sources/raw2drive.md]] trains a camera policy with Think2Drive's own world model as latent target, its reward heads and its policy as initialization, and scores 71.36, 20.5 below the teacher. Imitation students of the same expert's data come within 5.6.
- **Imitation from cameras starts far below the teacher.** In Think2Drive's own experiment, TCP trained on 200K of its frames scores 36.4 DS against the teacher's 83.8 on the same routes.

**Why Bench2Drive's DS is not CARLA v2's DS.** CARLA's Driving Score multiplies a penalty for every infraction along a route, so the same per-kilometre infraction rate scores lower on a longer route. Think2Drive scores 56.8 DS on the long official v2 routes at 98.6% completion, and 91.85 on Bench2Drive. Its paper proposes a "Weighted Driving Score" that divides infraction counts by scenario density (91.7 on the long routes); no other paper here uses it.

---

## Multi-Ability Breakdown (LinkVLA vs. ORION vs. SimLingo)

| Scenario | SimLingo | ORION | **LinkVLA** |
|----------|---------|-------|------------|
| Merging | 53.75 | 25.00 | **60.00** |
| Overtake | 68.89 | 71.11 | **80.00** |
| Brake | 81.67 | 78.33 | **93.33** |
| Give-Way | 50.00 | 30.00 | **50.00** |
| Traffic-Sign | 82.11 | 69.15 | **83.68** |
| **Mean** | 67.28 | 54.72 | **73.40** |

Braking (+11.7 vs. SimLingo) and overtaking (+11.1) drive LinkVLA's gains. ORION's give-way (30.00) is the weakest result — the VAE planner may not handle reactive yielding well.

**Raw2Drive multi-ability** ([[sources/raw2drive.md]] Table 4): Merging 43.35, Overtaking 51.11, Emergency Brake 60.00, Give Way 50.00, Traffic Sign 62.26, **Mean 53.34**, level with ORION (54.72) overall and well above it on merging (43.35 vs 25.00).

**DeepSight multi-ability** (Table 2, vs. ORION as its strongest baseline): Merging 60.00, Overtaking 91.11, Emergency Brake 78.33, Give Way 50.00, Traffic Sign 71.58, **Mean 70.20** (+15.48 over ORION 54.72). Its overtaking (91.11) and merging (60.00) match or exceed LinkVLA's, which the paper attributes to long-horizon BEV spatial modeling of multi-vehicle interactions; give-way (50.00) ties LinkVLA. Note DeepSight uses Think2Drive expert data whereas LinkVLA's expert protocol is undisclosed in the wiki, so the per-scenario comparison is indicative, not head-to-head.

### MomWorld: a Result That Tracks Its Base Planner {#momworld}

[[sources/momworld.md]] reports **74.07 DS / 50.00 SR**, with mean ability 55.07.

| | DS | SR | Effi. | Comf. | Merge | Overtake | Em. Brake | Give Way | Traffic Sign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hydra-NeXt | 73.86 | 50.00 | 197.76 | 20.68 | 40.00 | 64.44 | 61.67 | 50.00 | 50.00 |
| MomWorld | 74.07 | 50.00 | 198.87 | 20.43 | 45.63 | 64.21 | 61.71 | 50.00 | 54.10 |

- The paper does not say which planner MomWorld is built on for Bench2Drive. Its NAVSIM model is built on GTRS-Dense and its nuScenes model on MomAD.
- Success rate is identical to Hydra-NeXt. DS differs by 0.21. Only Merging (+5.6) and Traffic Sign (+4.1) move by more than a point.
- The same group's other momentum methods run on a SparseDrive base and score 47.91 (MomAD), 49.21 (DIVER) and 51.55 (GraphWorld). **The 22–26 DS between those rows and MomWorld is the base planner, not the method.**
- The printed mean ability (55.07) is not the mean of the five printed abilities (55.13).

**Three rows in this wiki disagree with MomWorld's table.**

| Method | Earlier wiki value | MomWorld's table | Note |
|---|---|---|---|
| MomAD | 44.54 DS / 16.71 SR ([[sources/orion.md]], [[sources/deepsight.md]]) | 47.91 / 18.11 | The earlier value is the row MomWorld assigns to **SparseDrive**. MomWorld shares MomAD's first author, so its attribution is the better-sourced one |
| SimLingo | 85.07 DS (this page), 85.94 ([[sources/deepsight.md]]) | 86.02 | Three values for one method; SR 67.27 agrees everywhere |
| SparseDrive | — | 44.54 / 16.71 | See the MomAD row |

Two non-VLM entries in that table are near the top of this page. **SparseDriveV2 (89.15 DS)** is above every ingested method except LinkVLA; **HiP-AD (86.77)** is above all but LinkVLA, DynVLA and AutoMoT. Neither is ingested.

---

## Relationship to NAVSIM

Most NAVSIM SOTA methods — FLARE (91.4 PDMS), DriveFine (90.7), WAM-Flow (90.3), Curious-VLA (90.3) — **do not evaluate on Bench2Drive**. Most Bench2Drive methods — ORION, LinkVLA, AutoMoT — **do not evaluate on NAVSIM**. Cross-benchmark comparison is generally not possible.

The two benchmarks test complementary capabilities:
- **NAVSIM**: diverse real-world-like traffic; reward-shaped safety metrics; non-reactive agents
- **Bench2Drive**: interactive CARLA simulation; explicit scenario types; closed-loop agent reactions

AutoVLA and DynVLA are the wiki methods with published results on both. AutoVLA is competitive but not SOTA on either (89.11 PDMS, 78.84 DS). DynVLA is stronger on both (91.7 PDMS, 88.34 DS), but still trails LinkVLA on Bench2Drive and DriveSuprim/HybridDriveVLA on NAVSIM. This reinforces that strong cross-benchmark performance is possible, but SOTA on one benchmark still does not imply SOTA on the other.

Drive-JEPA ([[sources/drive-jepa.md]]) also reports both NAVSIM and Bench2Drive, but with a different profile: very strong NAVSIM-v1 (93.3 PDMS) and moderate Bench2Drive (64.52 DS / 36.82 SR). Its Bench2Drive value is evidence that MTD improves over iPad (60.52 DS), not evidence of closed-loop frontier performance; LinkVLA (91.01), DynVLA (88.34), and AutoMoT (87.34) remain much stronger on this benchmark.

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- Does Bench2Drive's interactive simulation better predict real-world safety than NAVSIM's non-reactive scoring?
- Does LinkVLA's 91.01 DS (on a 1B backbone) scale further with a 7B backbone, and would DynVLA's dynamics-token CoT close the remaining gap if combined with LinkVLA-style action tokenization?
- Are the GRPO-optimized NAVSIM methods (FLARE, DriveFine) competitive on Bench2Drive's interactive scenarios, or does their non-reactive training regime limit reactive behavior?
- Can PDM-Lite usage be standardized across papers, or should the benchmark track PDM-Lite and non-PDM-Lite leaderboards separately?
- **What do SparseDriveV2 (89.15 DS) and HiP-AD (86.77) do?** Both are non-VLM planners that match the VLA leaders here, and both are known to this wiki only through [[sources/momworld.md]]'s table.
- **Which Bench2Drive row is MomAD's?** Two ingested tables give it SparseDrive's numbers.
- **Does the expert bound the leaderboard?** The best Think2Drive-protocol student is 5.6 DS below its 91.85-DS teacher and LinkVLA (protocol undisclosed) is 0.84 below it. Whether any student trained only on the base set can pass the expert, and whether PDM-Lite-protocol gains are simply the stronger teacher, is untested. See [above](#expert).
