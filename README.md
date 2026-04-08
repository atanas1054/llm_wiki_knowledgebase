# LLM-Powered AD Research Wiki

A structured knowledge base for end-to-end autonomous driving research, built by ingesting academic papers with an LLM assistant (Claude). Each paper is read, discussed, and distilled into cross-linked wiki pages covering source summaries and reusable concept notes.

## Structure

```
raw/
  papers/      # Source papers (markdown, converted from arXiv HTML)
  assets/      # Figures extracted from papers
wiki/
  sources/     # One page per paper — summary, figures, tables, limitations
  concepts/    # Cross-paper concept notes (updated as new papers are ingested)
  index.md     # Master catalog of all wiki pages
  log.md       # Append-only ingest log
CLAUDE.md      # Workflow instructions for the LLM assistant
```

## Papers Ingested (22)

| Paper | Org | Key Contribution | Benchmark |
|-------|-----|-----------------|-----------|
| [ReCogDrive](wiki/sources/recogdrive.md) | — | VLM + DiT diffusion planner + GRPO RL | 89.6 PDMS NAVSIM-v1 |
| [WAM-Flow](wiki/sources/wam-flow.md) | — | Discrete flow matching (CTMC) + GRPO; 1 camera | 90.3 PDMS NAVSIM-v1 |
| [UniUGP](wiki/sources/uniugp.md) | — | Unified VLA + world model; MoT; SOTA nuScenes FID/FVD | nuScenes + DriveLM |
| [Senna-2](wiki/sources/senna2.md) | — | Dual-system VLM+E2E consistency alignment; 3DGS HRL | 86.6 EPDMS NAVSIM-v2 |
| [ReflectDrive](wiki/sources/reflectdrive.md) | — | Masked discrete diffusion + gradient-free reflective inference | NAVSIM-v1 |
| [Reasoning-VLA](wiki/sources/reasoning-vla.md) | — | Learnable action queries (1-step parallel); unified 8-dataset GRPO | 91.7 PDMS (claimed) |
| [Percept-WAM](wiki/sources/percept-wam.md) | — | World-PV/BEV tokens unify perception + planning; IoU-aware confidence | 90.2 PDMS NAVSIM-v1 |
| [ORION](wiki/sources/orion.md) | — | VLM + VAE generative planner; reasoning-action latent alignment | 77.74 DS Bench2Drive |
| [LinkVLA](wiki/sources/linkvla.md) | — | Shared language-action codebook; C2F 2-pass decoder; 48ms | 91.01 DS Bench2Drive SOTA |
| [HERMES](wiki/sources/hermes.md) | — | Offline VLM annotation → risk-aware student; no VLM at inference | 6.81 RFS WOD-E2E |
| [DriveFine](wiki/sources/drivefine.md) | — | Block-MoE masked diffusion + hybrid offline/online RL | 90.7 PDMS NAVSIM-v1 |
| [Curious-VLA](wiki/sources/curious-vla.md) | — | Narrow Policy diagnosis; FTE + ADAS + SDR; BoN-6 matches human GT | 90.3 PDMS / 94.8 BoN-6 |
| [AutoVLA](wiki/sources/autovla.md) | — | K-Disk physical action codebook; adaptive CoT via length-penalty GRPO | 89.11 PDMS NAVSIM-v1 |
| [AutoMoT](wiki/sources/automot.md) | — | Frozen VLM + trained action expert; async KV cache (7.6× speedup) | 87.34 DS Bench2Drive |
| [AutoDrive-R²](wiki/sources/autodrive-r2.md) | Alibaba AMAP | 4-step CoT with self-reflection + physics GRPO; 6K samples beats 103K | 0.19m L2 nuScenes |
| [Alpamayo-R1](wiki/sources/alpamayo-r1.md) | NVIDIA | Cosmos-Reason + CoC dataset (700K) + FM action expert; 3-reward GRPO | 99ms; internal evals |
| [AdaThinkDrive](wiki/sources/adathinkdrive.md) | Xiaomi EV | Adaptive Think Reward: mode-comparison GRPO learns when to reason | 90.3 PDMS / 93.0 BoN-4 |
| [FutureSightDrive](wiki/sources/futuresightdrive.md) | Xi'an Jiaotong + Alibaba Amap | Visual ST-CoT: VQ-VAE AR future frame as planning intermediate; vocabulary expansion | 85.1 PDMS NAVSIM / 0.96m L2 nuScenes |
| [DriveDreamer-Policy](wiki/sources/drivedreamer-policy.md) | GigaAI + U of Toronto | Causal depth→video→action WAM; geometry-grounded; 3 FM generators | 89.2 PDMS NAVSIM-v1 / 88.7 EPDMS NAVSIM-v2 |
| [DriveVLA-W0](wiki/sources/drivevla-w0.md) | CASIA + Yinwang | Supervision deficit → world model self-supervision; AR/diffusion WM; MoE action expert; scaling law | 90.2★ PDMS (anchors) / 93.0 BoN-6 NAVSIM-v1 |
| [UniDriveVLA](wiki/sources/unidrivevla.md) | HUST + Xiaomi EV + U Macau | MoT 3-expert (und/per/act) + masked joint attention; sparse 5-task perception; 3-stage progressive training | 78.37 DS Bench2Drive (best w/o PDM-Lite); 0.51m L2 nuScenes no-ego |
| [FLARE](wiki/sources/flare.md) | OpenDriveLab + Li Auto | Annotation-free DINOv2 future feature prediction; BC-GRPO; 1 camera | 86.9 PDMS SFT / 91.4 PDMS RFT (single-sample) NAVSIM-v1 |

## Concept Pages (9)

| Concept | Description |
|---------|-------------|
| [Diffusion-Based Trajectory Planner](wiki/concepts/diffusion-planner.md) | Continuous diffusion, DFM, masked diffusion, FM action expert, and learnable-query paradigms compared |
| [Discrete Flow Matching](wiki/concepts/discrete-flow-matching.md) | CTMC-based DFM theory; WAM-Flow vs. masked diffusion vs. continuous FM |
| [RL for Autonomous Driving](wiki/concepts/rl-for-ad.md) | GRPO variants: simulator, GT-based, hierarchical (3DGS), adaptive Think, LRM-as-critic |
| [VLM Domain Adaptation](wiki/concepts/vlm-domain-adaptation.md) | Data curation, CoT integration, dual-mode SFT, frozen VLM, Physical AI pre-training |
| [NAVSIM Benchmark](wiki/concepts/navsim-benchmark.md) | PDMS/EPDMS metrics; SOTA table with caveats; Navhard OOD results |
| [World Models for AD](wiki/concepts/world-model-for-ad.md) | Video generation as causal learning; UniUGP generation expert; FID/FVD |
| [Dual-System VLA](wiki/concepts/dual-system-vla.md) | VLM decisions + E2E trajectory; consistency alignment; async KV cache |
| [Inference-Time Safety](wiki/concepts/inference-time-safety.md) | Gradient-free safety correction; inpainting-as-repair pattern |
| [Perception-Enhanced Planning](wiki/concepts/perception-for-planning.md) | World-PV/BEV tokens; grid-conditioned AR detection; IoU-aware confidence |

## Workflow

The ingest workflow is defined in `CLAUDE.md`. Each paper goes through:
1. Read source + figures
2. Discuss key takeaways and limitations
3. Create/update `wiki/sources/<paper>.md` with embedded figures and full tables
4. Create/update relevant `wiki/concepts/` pages
5. Update `wiki/index.md` and append to `wiki/log.md`
