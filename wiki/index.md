# Wiki Index

Master catalog of all wiki pages. Updated on every ingest.

---

## Sources

| Page                                      | Description                                                                                                                                           |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ReCogDrive](sources/recogdrive.md)       | VLM + diffusion planner + RL for end-to-end AD; PDMS 89.6 on NAVSIM-v1                                                                                |
| [WAM-Flow](sources/wam-flow.md)           | Discrete flow matching VLA for AD; PDMS 90.3 SOTA on NAVSIM-v1; 1 camera only                                                                         |
| [UniUGP](sources/uniugp.md)               | Unified VLA + world model; CoT + video generation + FM trajectory; SOTA nuScenes FID/FVD and DriveLM                                                  |
| [Senna-2](sources/senna2.md)              | Dual-system VLM + E2E alignment; 3DGS HRL; +19.3% consistency F1; EPDMS 86.6 SOTA on NAVSIM-v2                                                        |
| [ReflectDrive](sources/reflectdrive.md)   | Masked discrete diffusion + gradient-free reflective inference; goal-conditioned NMS + safety anchor inpainting; claims >AutoVLA on NAVSIM-v1         |
| [Reasoning-VLA](sources/reasoning-vla.md) | Learnable action queries (1-step parallel); unified 8-dataset corpus; GT-based GRPO; 91.7 PDMS claimed (comparison scope limited); 61× faster than AR |

---

## Concepts

| Page | Description |
|------|-------------|
| [Discrete Flow Matching](concepts/discrete-flow-matching.md) | DFM over token spaces via CTMC; parallel bidirectional generation; geometry-aware Gibbs paths; metric-aligned numerical tokenizer |
| [Diffusion-Based Trajectory Planner](concepts/diffusion-planner.md) | DDPM/DiT applied to continuous trajectory generation; MoT coupling; DFM and FM comparisons |
| [Reinforcement Learning for Autonomous Driving](concepts/rl-for-ad.md) | RL approaches in AD; GRPO applied to diffusion and DFM policies; sim-assisted RL |
| [VLM Domain Adaptation for Autonomous Driving](concepts/vlm-domain-adaptation.md) | Adapting general VLMs to driving via data curation, SFT, CoT integration; multi-stage training |
| [NAVSIM Benchmark](concepts/navsim-benchmark.md) | Planning benchmark, PDMS/EPDMS metrics, non-reactive simulator; current SOTA |
| [World Models for Autonomous Driving](concepts/world-model-for-ad.md) | Video generation as visual causal learning; integration with VLA planners; FID/FVD metrics |
| [Dual-System VLA](concepts/dual-system-vla.md) | VLM for decisions + E2E for trajectory; decision adapter; kinematic mapping; consistency alignment |
| [Inference-Time Safety](concepts/inference-time-safety.md) | Gradient-free safety correction at inference; discrete token search + inpainting-as-repair; taxonomy vs. guidance/RL/anchors |

---

## Entities

*(none yet)*
