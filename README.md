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

## Papers Ingested (33)

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
| [AutoDrive-R²](wiki/sources/autodrive-r2.md) | Alibaba AMAP | 4-step CoT with self-reflection + physics GRPO; 6K samples | 0.19m L2 nuScenes |
| [Alpamayo-R1](wiki/sources/alpamayo-r1.md) | NVIDIA | Cosmos-Reason + CoC dataset (700K) + FM action expert; 3-reward GRPO | 99ms; internal evals |
| [AdaThinkDrive](wiki/sources/adathinkdrive.md) | Xiaomi EV | Adaptive Think Reward: mode-comparison GRPO learns when to reason | 90.3 PDMS / 93.0 BoN-4 |
| [FutureSightDrive](wiki/sources/futuresightdrive.md) | Xi'an Jiaotong + Alibaba Amap | Visual ST-CoT: VQ-VAE AR future frame as planning intermediate | 85.1 PDMS NAVSIM / 0.96m L2 nuScenes |
| [DriveDreamer-Policy](wiki/sources/drivedreamer-policy.md) | GigaAI + U of Toronto | Causal depth→video→action WAM; geometry-grounded; 3 FM generators | 89.2 PDMS NAVSIM-v1 / 88.7 EPDMS NAVSIM-v2 |
| [DriveVLA-W0](wiki/sources/drivevla-w0.md) | CASIA + Yinwang | Supervision deficit → world model self-supervision; scaling law | 90.2★ PDMS / 93.0 BoN-6 NAVSIM-v1 |
| [UniDriveVLA](wiki/sources/unidrivevla.md) | HUST + Xiaomi EV | MoT 3-expert (und/per/act); sparse 5-task perception | 78.37 DS Bench2Drive; 0.51m L2 nuScenes |
| [FLARE](wiki/sources/flare.md) | OpenDriveLab + Li Auto | Annotation-free DINOv2 future feature prediction; BC-GRPO | 91.4 PDMS RFT NAVSIM-v1 |
| [DreamerAD](wiki/sources/dreameraD.md) | Chongqing Chang'an | Latent world model RL; shortcut forcing (80× speedup); latent reward model | 87.7 EPDMS NAVSIM-v2 |
| [Vega](wiki/sources/vega.md) | Tsinghua + GigaAI | Instruction-conditioned AR+Diffusion; InstructScene 100K; NL instruction following | 86.9 EPDMS / 89.4 BoN-6 NAVSIM-v2 |
| [NoRD](wiki/sources/nord.md) | Applied Intuition + TAMU + UCB | Reasoning-free VLA; Dr. GRPO fixes difficulty bias; data efficiency | 85.6 PDMS / 92.4 BoN-6 NAVSIM-v1 |
| [DiffusionDrive](wiki/sources/diffusiondrive.md) | HUST + Horizon Robotics | Truncated diffusion (20 anchors, 2 steps); cascade decoder; 45 FPS | 88.1 PDMS NAVSIM-v1 (non-VLM baseline) |
| [DiffusionDriveV2](wiki/sources/diffusiondrive-v2.md) | HUST + Horizon Robotics | Intra/Inter-Anchor GRPO + multiplicative noise on truncated diffusion | 91.2 PDMS NAVSIM-v1 / 85.5 EPDMS NAVSIM-v2 |
| [Epona](wiki/sources/epona.md) | Horizon Robotics + Tsinghua + PKU | AR+Diffusion WM (MST+DiTs, 2.5B); chain-of-forward training; backbone for DreamerAD | FVD 82.8 NuScenes; 86.2 PDMS NAVSIM-v1 |
| [HybridDriveVLA / DualDriveVLA](wiki/sources/hybriddriveVLA.md) | — | 3-RQ complementarity analysis (CKA/SAE); VLM+ViT dual-branch + style-axis interpolation; fast–slow deployment | 92.10 PDMS NAVSIM-v1; 91.0 PDMS @ 3.2× throughput |
| [WAM-Diff](wiki/sources/wam-diff.md) | Fudan + Yinwang | Masked diffusion + LoRA MoE (64 experts) + GSPO (sequence-level RL); reverse-causal decoding | 91.0 PDMS NAVSIM-v1 / 89.7 EPDMS NAVSIM-v2 |

## Concept Pages (13)

| Concept | Description |
|---------|-------------|
| [Diffusion-Based Trajectory Planner](wiki/concepts/diffusion-planner.md) | Continuous diffusion, DFM, masked diffusion, FM action expert, and learnable-query paradigms compared |
| [Discrete Flow Matching](wiki/concepts/discrete-flow-matching.md) | CTMC-based DFM theory; WAM-Flow vs. masked diffusion vs. continuous FM |
| [RL for Autonomous Driving](wiki/concepts/rl-for-ad.md) | GRPO variants: simulator, GT-based, hierarchical (3DGS), adaptive Think, LRM-as-critic, Dr. GRPO |
| [VLM Domain Adaptation](wiki/concepts/vlm-domain-adaptation.md) | Data curation, CoT integration, dual-mode SFT, frozen VLM, reasoning-free adaptation |
| [NAVSIM Benchmark](wiki/concepts/navsim-benchmark.md) | PDMS/EPDMS metrics; full SOTA table with caveats; Navhard OOD results |
| [World Models for AD](wiki/concepts/world-model-for-ad.md) | 10 architecture patterns from cascaded generation to latent RL reward sources |
| [Dual-System VLA](wiki/concepts/dual-system-vla.md) | VLM decisions + E2E trajectory; consistency alignment; async KV cache; MoT paradigm; complementarity + fast–slow deployment |
| [Inference-Time Safety](wiki/concepts/inference-time-safety.md) | Gradient-free safety correction; inpainting-as-repair; DriveFine block-MoE contrast |
| [Perception-Enhanced Planning](wiki/concepts/perception-for-planning.md) | World-PV/BEV tokens; grid-conditioned AR detection; IoU-aware confidence; sparse MoT |
| [Best-of-N Sampling](wiki/concepts/best-of-n.md) | Oracle trajectory selection; NAVSIM-v1 saturated at BoN-6 (94.8 = human GT); deployable variants |
| [Bench2Drive Benchmark](wiki/concepts/bench2drive.md) | CARLA V2 closed-loop; interactive agents; DS + SR metrics; SOTA LinkVLA 91.01 DS |
| [Chain-of-Thought for AD](wiki/concepts/chain-of-thought-for-ad.md) | Text/visual/self-reflection CoT types; annotation methods; adaptive CoT; NoRD challenge |
| [Mixture of Experts for AD](wiki/concepts/mixture-of-experts.md) | 4 patterns: sparse LoRA MoE, block-level task MoE, MoT (frozen+trained), side expert; RL routing instability; catastrophic forgetting |

## Workflow

The ingest workflow is defined in `CLAUDE.md`. Each paper goes through:
1. Read source + figures
2. Discuss key takeaways and limitations
3. Create/update `wiki/sources/<paper>.md` with embedded figures and full tables
4. Create/update relevant `wiki/concepts/` pages
5. Update `wiki/index.md` and append to `wiki/log.md`
