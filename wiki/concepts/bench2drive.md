---
title: Bench2Drive Benchmark
type: concept
sources: [raw/papers/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md]
related: [sources/orion.md, sources/linkvla.md, sources/autovla.md, sources/automot.md, sources/unidrivevla.md, concepts/navsim-benchmark.md, concepts/dual-system-vla.md, concepts/diffusion-planner.md]
created: 2026-04-15
updated: 2026-04-15
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
| ORION | 77.74 | 54.62 | 151.48 | 17.38 | VLM + VAE planner; +14.28 DS vs. prior SOTA at time |
| AutoVLA | 78.84 | 57.73 | 146.93 | 39.33 | Physical codebook; higher comfort than ORION |
| UniDriveVLA | 78.37 | — | — | — | Best result without PDM-Lite oracle; MoT 3-expert |
| SimLingo | 85.07 | 67.27 | 259.23 | 33.67 | Fast MLP head (34ms) |
| AutoMoT | 87.34 | — | — | — | Frozen Qwen3-VL-4B + 1.6B AE; async MoT (7.6× speedup) |
| **LinkVLA** | **91.01** | **74.55** | **255.84** | **34.62** | **Shared codebook + C2F; current SOTA** |

**PDM-Lite caveat**: PDM-Lite is a privileged oracle planner (uses ground-truth waypoints or HD map access) that some Bench2Drive methods use as a fallback or auxiliary module. UniDriveVLA (78.37) is explicitly noted as the best result *without* PDM-Lite. Methods that use PDM-Lite score higher but are not fairly comparable to methods that do not. The precise PDM-Lite usage for each method above is not always disclosed.

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

---

## Relationship to NAVSIM

Most NAVSIM SOTA methods — FLARE (91.4 PDMS), DriveFine (90.7), WAM-Flow (90.3), Curious-VLA (90.3) — **do not evaluate on Bench2Drive**. Most Bench2Drive methods — ORION, LinkVLA, AutoMoT — **do not evaluate on NAVSIM**. Cross-benchmark comparison is generally not possible.

The two benchmarks test complementary capabilities:
- **NAVSIM**: diverse real-world-like traffic; reward-shaped safety metrics; non-reactive agents
- **Bench2Drive**: interactive CARLA simulation; explicit scenario types; closed-loop agent reactions

AutoVLA is the only wiki method with published results on both (89.11 PDMS, 78.84 DS). It is competitive but not SOTA on either, which suggests the two benchmarks test different skills and SOTA on one does not imply SOTA on the other.

---

## Open Questions

- Does Bench2Drive's interactive simulation better predict real-world safety than NAVSIM's non-reactive scoring?
- Does LinkVLA's 91.01 DS (on a 1B backbone) scale further with a 7B backbone?
- Are the GRPO-optimized NAVSIM methods (FLARE, DriveFine) competitive on Bench2Drive's interactive scenarios, or does their non-reactive training regime limit reactive behavior?
- Can PDM-Lite usage be standardized across papers, or should the benchmark track PDM-Lite and non-PDM-Lite leaderboards separately?
