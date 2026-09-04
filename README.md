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

## At a Glance

- **67 papers** ingested into `wiki/sources/`, **30 concept pages** in `wiki/concepts/`
- Dominant benchmark is **NAVSIM** (v1 PDMS / v2 EPDMS); also Bench2Drive, nuPlan, nuScenes, WOD-E2E, HUGSIM, PhysicalAI-AV
- Current NAVSIM-v1 leaders (single-pass, non-BoN): **CLEAR 93.7 = DA-WAM 93.7** > DriveSuprim 93.5 > Drive-JEPA 93.3 > **WCog-VLA 92.9 (2B)** > HybridDriveVLA 92.1 (ensemble) > WA-JEPA 91.8 > DynVLA 91.7 > SimWAM 91.5
- Highest NAVSIM-v2 EPDMS: **WA-JEPA 91.7** — but **v2 EPDMS is not currently comparable across papers.** Two papers score Transfuser at 76.7 and 84.0 from *identical* submetrics. See [Evaluator Drift](wiki/concepts/navsim-benchmark.md). Best-of-6 ceiling: **Curious-VLA 94.8 PDMS** (= human GT)
- Reactive/OOD leader: **GeoWAM 36.6 EPDMS navhard** — navhard is far from saturated and Stage 2 collapses every method's lane keeping to ~48
- Closed-loop HUGSIM leader: **WA-JEPA 0.4462 HD-Score** zero-shot — on a scenario set and controller version that make earlier HUGSIM numbers in this wiki incomparable
- **Baseline rows are not commensurable by default.** Five ingested NAVSIM-v2 tables provably mix evaluator conventions, and the failure modes now include *non-headline configurations* cited unmarked (BrainWAM lists DynVLA’s SFT-only score against its published 91.7), *cross-version submetric contamination* (WCog-VLA’s v2 TransFuser row reuses its own v1 submetrics), and *oracle Best-of-N scores in single-sample tables* (WCog-VLA lists AutoVLA at its 92.12 BoN rather than 89.11). Check submetrics and source configuration before comparing any two rows.
- Every leaderboard claim in this wiki carries a **comparison-scope caveat** — most papers omit the actual frontier from their tables. See [NAVSIM Benchmark](wiki/concepts/navsim-benchmark.md).

## Papers Ingested (67)

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
| [DriveSuprim](wiki/sources/drivesuprim.md) | Fudan + NVIDIA | Non-VLM selection-based; 8192-vocab + coarse-to-fine (→256) + rotation aug + EMA self-distill | **93.5 PDMS NAVSIM-v1** (highest non-BoN) / 87.1 EPDMS NAVSIM-v2 |
| [DriveVA](wiki/sources/driveva.md) | U. Twente | Wan2.2-TI2V-5B video backbone; single DiT over joint [video latents ‖ action tokens]; zero-shot cross-dataset | 90.9 PDMS NAVSIM-v1; −78.9% L2 nuScenes (zero-shot) |
| [ExploreVLA](wiki/sources/explorevla.md) | — | Show-o (Phi-1.5 + MAGVIT-v2); dense RGB+depth world model SFT; safety-gated entropy exploration reward (GRPO) | 90.4 PDMS / 93.7 BoN-6 NAVSIM-v1; 88.8 EPDMS NAVSIM-v2 |
| [ELF-VLA](wiki/sources/elf-vla.md) | Tsinghua + University of Macau + Beijing Jiaotong | Teacher-diagnosed persistent failures; feedback-guided refinement re-injected into GRPO with policy shaping | 91.0 PDMS NAVSIM-v1; 87.1 EPDMS NAVSIM-v2 |
| [DynVLA](wiki/sources/dynvla.md) | — | Dynamics CoT: compact ego/environment dynamics tokens before action tokens; Dynamics Tokenizer + SFT + GRPO RFT | 91.7 PDMS NAVSIM-v1; 88.34 DS Bench2Drive |
| [SpanVLA](wiki/sources/spanvla.md) | UCLA + Motional + Northeastern | Sparse-KV action bridge + flow-matching action expert from historical initialization; GRPO with negative-recovery samples | 90.3 PDMS NAVSIM-v1; 86.4 EPDMS NAVSIM-v2; 40.1 navhard |
| [OneDrive](wiki/sources/onedrive.md) | — | Single causal VLM decoder unifies AR text, parallel perception queries, and planning queries; pretrained attention transfers better than FFNs | 0.28 L2 / 0.18 collision nuScenes; 86.8 PDMS NAVSIM-v1; 156ms NAVSIM latency |
| [OneVL](wiki/sources/onevl.md) | Xiaomi | Latent CoT with dual auxiliary decoders: language explanation plus future-frame world-model supervision, discarded at inference via prefill | 88.84 PDMS NAVSIM-v1 at 4.46s; 0.24s MLP variant at 86.83 PDMS |
| [HAD](wiki/sources/had.md) | Fudan + NVIDIA | Hierarchical diffusion planner with polar trajectory expansion, metric-decoupled RL, and offline reward retrieval | 90.2 PDMS NAVSIM-v1; 88.6 EPDMS NAVSIM-v2; 47.5 RC / 30.8 HDS HUGSIM |
| [Latent-WAM](wiki/sources/latent-wam.md) | Chongqing Chang'an + collaborators | Spatial-aware compressed latent world states with WorldMirror geometric distillation and causal latent dynamics prediction | 89.3 EPDMS NAVSIM-v2; 45.9 RC / 28.9 HDS HUGSIM zero-shot |
| [Drive-JEPA](wiki/sources/drive-jepa.md) | — | V-JEPA driving-video pretraining plus proposal-centric multimodal trajectory distillation and momentum-aware selection | 93.3 PDMS NAVSIM-v1; 87.8 EPDMS NAVSIM-v2; 64.52 DS Bench2Drive |
| [FeaXDrive](wiki/sources/feaxdrive.md) | Tongji + NTU | Trajectory-centric diffusion planning with curvature regularization, drivable-area guidance, and feasibility-aware GRPO | 90.0 PDMS NAVSIM-v1; 2.40% curvature violation |
| [Policy World Model](wiki/sources/policy-world-model.md) | Dalian University of Technology | Show-o-based policy world model with action-free future video forecasting, 28-token frames, and collaborative state-action prediction | 88.1 PDMS NAVSIM-v1; 0.41 L2 / 0.04 collision nuScenes w/ ego |
| [CLEAR](wiki/sources/clear.md) | Tsinghua | Drive-JEPA encoder + Qwen hidden-state adaptive scheduler/scorer + single-step VAE latent drift | **93.7 PDMS NAVSIM-v1** (highest non-BoN); 88.6 EPDMS NAVSIM-v2 |
| [DeepSight](wiki/sources/deepsight.md) | — (ICML) | Parallel 5-frame DINOv3 latent prediction in BEV via World Queries + adaptive CoT; +3.57% latency vs. FSDrive's +60.71% | 86.23 DS / 71.36 SR Bench2Drive (Think2Drive) |
| [DriveWAM](wiki/sources/drivewam.md) | CUHK-Shenzhen + Didi Chuxing | Wan2.2-TI2V-5B as policy core; chunked AR video→action inverse dynamics; frozen VLM chunk guidance + selective KV memory (12× cheaper at 300s) | 90.1 PDMS NAVSIM-v1; 0.83 ADE@4s PhysicalAI-AV |
| [SimWAM](wiki/sources/simwam.md) | HUST + Dongfeng | Isolated attention mask makes future-video prediction training-time-only; Flow-GRPO SDE + LoRA RL; swappable video prior (1.3B ≈ 5B) | **91.5 PDMS NAVSIM-v1** (highest WAM) at 518ms; 0.04% zero-shot nuScenes collision |
| [SGDrive](wiki/sources/sgdrive.md) | Li Auto + Fudan + Tongji + Surrey | Scene-agent-goal ⟨world⟩ queries (occupancy + safety-critical boxes + 4s goal, at t and t+n) + block-wise anti-leakage mask + DiT planner | 87.4 PDMS SFT / 91.1 RFT NAVSIM-v1; 86.2 EPDMS NAVSIM-v2 |
| [DriveLaW](wiki/sources/drivelaw.md) | HUST + Xiaomi EV | Chained generation→planning: Video DiT mid-denoising latents are the planning state; noise reinjection; controlled representation comparison (video > VLM > BEV) | 89.1 PDMS NAVSIM-v1 (no RL); **FID 4.6 nuScenes** (best in wiki) |
| [How Can Driving World Models Do Counterfactual Prediction?](wiki/sources/driving-wm-counterfactuals.md) | Purdue + Bosch CAI | Action-conditioned generation is rung-2, not counterfactual: it discards the factual continuation; 186-case CARLA benchmark with matched counterfactual ground truth; training-free evidence transport as a constructive check | 0.38 / 0.31 recovered fraction for Vista / DrivingWorld → 0.70 / 0.67 |
| [Auto-JEPA](wiki/sources/auto-jepa.md) | Tsinghua | JEPA prediction of the future **ego-trajectory** latent, used as a retrieval key over 110,335 recorded trajectories; frozen V-JEPA 2, no perception labels, no trajectory generator | 91.3 PDMS NAVSIM-v1; 85.6 EPDMS matched-protocol |
| [WA-JEPA](wiki/sources/wa-jepa.md) | Afari Intelligent Drive + UESTC | Rebuilds V-JEPA for planning: future masking, flow-matched latent futures, joint scene-action MMDiT; regression future-prediction shown *worse than none* | 91.7 EPDMS NAVSIM-v2 (corrected); 0.4462 HD-Score HUGSIM |
| [GeoWAM](wiki/sources/geowam.md) | Uber AV Labs + Case Western | Forecasts dense metric point maps instead of pixels; geometry-conditioned deterministic action head; annotation-free 3D structure | 36.6 EPDMS navhard (SOTA); 90.2 navtest (protocol-incomparable) |
| [DA-WAM](wiki/sources/da-wam.md) | HKUST-GZ + Leapmotor | One future latent per candidate, scored against its own; live JEPA supervision during planning; retrieved hard negatives | 93.7 PDMS NAVSIM-v1 (ties CLEAR) |
| [ForeSight](wiki/sources/foresight.md) | Fudan + Imperial + Surrey | Frozen 2.5B Epona as the planner's *primary* visual encoder; WM-QFormer + state queries + factorized attention; 900 ms (870 ms is the WM) | 89.3 PDMS NAVSIM-v1 |
| [BrainWAM](wiki/sources/brainwam.md) | CASIA + Li Auto | VLA and WAM branches coordinated through 8 action tokens (CAB + CIF); identifies modality competition in raw-token fusion | 89.5 PDMS v1 / 89.6 EPDMS v2 |
| [Adaptive-WAM](wiki/sources/adaptive-wam.md) | AIR Tsinghua + USTC + Beihang | Quality-routed early exit from intermediate video-DiT blocks; separates noise level from readout depth | 90.8 PDMS / 89.9 EPDMS at 170 ms |
| [GeoWorldAD](wiki/sources/geoworldad.md) | NTU + Xiaomi EV + Zhejiang | Ego-aligned multi-scale geometry + latent future-depth tokens; second independent geometry WAM | 91.0 PDMS / 90.4 EPDMS |
| [WCog-VLA](wiki/sources/wcog-vla.md) | Tongji + NTU | Joint multi-agent trajectory diffusion + agent tokens in a 2B VLM + Stackelberg Game-CoT + DiffGRPO | 92.9 PDMS NAVSIM-v1 |
| [PaIR-Drive](wiki/sources/pair-drive.md) | — | Parallel IL and GRPO residual-refinement branches; intention-conditioned trajectory tree + RWM selection; reusable across base planners | 91.2 PDMS / 87.9 EPDMS single-plan; 94.0 / 89.6 BoN-6 |
| [DIAL](wiki/sources/dial.md) | — | Eight-intent CFG expands continuous-flow proposal support; multi-intent GRPO preserves preference contrast | WOD-E2E held RFS 7.696→8.211; BoN-128 ceiling 9.14 |
| [PlannerRFT](wiki/sources/plannerrft.md) | — | Diffusion-planner RFT with PPO-learned lateral/longitudinal exploration, GRPO survival reward, and nuMax acceleration | 72.21 Test14-hard / 85.80 Test14-random reactive nuPlan |
| [Plan-R1](wiki/sources/plan-r1.md) | — | Trajectory planning as motion-token language modeling; dual-model reactive rollout; VD-GRPO fixes variance downweighting of unsafe groups | 90.04 reactive Test14-random nuPlan |
| [DAPO](wiki/sources/dapo.md) | ByteDance Seed + Tsinghua | Open-source 32B reasoning-RL system: Clip-Higher, dynamic sampling, token-level loss, overlong shaping (methodological reference) | Qwen2.5-32B AIME24 avg@32 30→50 |
| [DisCO](wiki/sources/disco.md) | — | Binary-reward discriminative RL replacing GRPO weighting/clipping; DRO hard negatives + KL constraint (methodological reference) | 1.5B six-task math avg 0.533 vs. GRPO 0.457 |
| [Understanding R1-Zero-Like Training](wiki/sources/understanding-r1-zero-like-training.md) | Sea AI Lab + NUS + SMU | Critical analysis of R1-Zero RL: template/pretraining confounds, GRPO length + difficulty bias, Dr. GRPO (methodological reference) | Oat-Zero-7B 43.3 AIME24 |
| [All Roads Lead to Rome](wiki/sources/all-roads-lead-to-rome.md) | ANU + Shanghai AI Lab + GE Research | GRPO diversity collapse in VLM reasoning; base models retain broader parallel reasoning; MUPO multi-group optimization (methodological reference) | MUPO-Thinker-7B 51.6/58.8 math acc@1/4 |

The last four entries are **methodological references** — LLM/VLM reasoning-RL papers ingested for their optimizer analysis rather than for driving results. They inform [RL for AD](wiki/concepts/rl-for-ad.md), [GSPO vs. GRPO](wiki/concepts/gspo-vs-grpo.md), and [R1-Zero-Like Training](wiki/concepts/r1-zero-like-training.md).

## Concept Pages (30)

| Concept | Description |
|---------|-------------|
| [Diffusion-Based Trajectory Planner](wiki/concepts/diffusion-planner.md) | Continuous diffusion, DFM, masked diffusion, FM action expert, and learnable-query paradigms compared |
| [Discrete Flow Matching](wiki/concepts/discrete-flow-matching.md) | CTMC-based DFM theory; WAM-Flow vs. masked diffusion vs. continuous FM |
| [RL for Autonomous Driving](wiki/concepts/rl-for-ad.md) | GRPO variants: simulator, GT-based, hierarchical (3DGS), adaptive Think, LRM-as-critic, Dr. GRPO, entropy-based exploration |
| [VLM Domain Adaptation](wiki/concepts/vlm-domain-adaptation.md) | Data curation, CoT integration, dual-mode SFT, frozen VLM, reasoning-free adaptation |
| [NAVSIM Benchmark](wiki/concepts/navsim-benchmark.md) | PDMS/EPDMS metrics; full SOTA table with caveats; Navhard OOD results |
| [World Models for AD](wiki/concepts/world-model-for-ad.md) | World-model patterns from video generation to latent status prediction, dynamics tokens, and entropy rewards |
| [Dual-System VLA](wiki/concepts/dual-system-vla.md) | VLM decisions + E2E trajectory; consistency alignment; async KV cache; MoT paradigm; complementarity + fast–slow deployment |
| [Inference-Time Safety](wiki/concepts/inference-time-safety.md) | Gradient-free safety correction; inpainting-as-repair; DriveFine block-MoE contrast |
| [Perception-Enhanced Planning](wiki/concepts/perception-for-planning.md) | World-PV/BEV tokens; grid-conditioned AR detection; IoU-aware confidence; sparse MoT |
| [Best-of-N Sampling](wiki/concepts/best-of-n.md) | Oracle trajectory selection; NAVSIM-v1 saturated at BoN-6 (94.8 = human GT); deployable variants |
| [Bench2Drive Benchmark](wiki/concepts/bench2drive.md) | CARLA V2 closed-loop; interactive agents; DS + SR metrics; SOTA LinkVLA 91.01 DS |
| [Chain-of-Thought for AD](wiki/concepts/chain-of-thought-for-ad.md) | Text/visual/self-reflection CoT types; annotation methods; adaptive CoT; NoRD challenge |
| [Mixture of Experts for AD](wiki/concepts/mixture-of-experts.md) | 4 patterns: sparse LoRA MoE, block-level task MoE, MoT (frozen+trained), side expert; RL routing instability; catastrophic forgetting |
| [Selection-Based Planning](wiki/concepts/selection-based-planning.md) | Fixed-vocabulary trajectory scoring; coarse-to-fine filtering; oracle ceiling 98.7 PDMS; hard-negative / directional bias failure modes |
| [Action Tokenization and Codebooks](wiki/concepts/action-tokenization.md) | Discrete action vocabularies, learned codebooks, continuous action experts, and tokenizer tradeoffs |
| [GSPO vs. GRPO](wiki/concepts/gspo-vs-grpo.md) | Sequence-level RL for MoE/masked-diffusion policies vs. token/group-level GRPO recipes |
| [PDM-Lite](wiki/concepts/pdm-lite.md) | Privileged oracle planner/fallback caveat for Bench2Drive comparisons |
| [nuScenes and Waymo Evaluations](wiki/concepts/nuscenes-waymo-evals.md) | Open-loop L2/collision/RFS metrics and cross-benchmark caveats |
| [Foundation Backbones for AD](wiki/concepts/foundation-backbones-for-ad.md) | Qwen, InternVL, Cosmos, Wan, Show-o, and other backbone choices in driving VLAs |
| [Navhard and OOD Evaluation](wiki/concepts/navhard-ood-evaluation.md) | NAVSIM-v2 navhard, distribution-shift scoring, and OOD caveats |
| [HUGSIM Benchmark](wiki/concepts/hugsim-benchmark.md) | Closed-loop interactive benchmark with route completion and HD-Score; HAD reports 47.5 RC / 30.8 HDS |
| [nuPlan Benchmark](wiki/concepts/nuplan-benchmark.md) | Reactive/non-reactive closed-loop evaluation, Val14/Test14 splits, scorer caveats, nuMax training acceleration |
| [PhysicalAI-AV Benchmark](wiki/concepts/physicalai-av-benchmark.md) | NVIDIA's 1,700h / 306K-clip real-world open-loop benchmark (ADE/FDE); no shared test protocol yet |
| [Adaptive Routing](wiki/concepts/adaptive-routing.md) | Scene-conditioned candidate budget and diversity control; LLM hidden states choose `(alpha, N)` and score trajectories |
| [Intent-Conditioned Planning](wiki/concepts/intent-conditioned-planning.md) | Discrete maneuver variables for multimodal proposals; intent-CFG, intent-balanced GRPO, ontology requirements |
| [Parallel Imitation and RL](wiki/concepts/parallel-il-rl.md) | Separate IL and RL parameter spaces; reusable residual proposal policies; modularity conditions and reference-shift caveats |
| [Discriminative Policy Optimization](wiki/concepts/discriminative-policy-optimization.md) | Positive/negative rollout scoring, GRPO difficulty-weight analysis, hard-negative DRO, and conditions for driving transfer |
| [R1-Zero-Like Training](wiki/concepts/r1-zero-like-training.md) | RL directly on base models; template/base-prior confounds; Dr./VD-GRPO corrections and their relevance to AD GRPO claims |
| [Divergent Thinking in VLMs](wiki/concepts/divergent-thinking-in-vlms.md) | Reasoning-strategy diversity, MUPO, acc@k scaling, and why correlated samples limit parallel test-time gains |
| [Counterfactual Prediction](wiki/concepts/counterfactual-prediction.md) | Pearl's ladder applied to driving; four senses of "counterfactual" in AD; abduction as the missing step; matched-ground-truth benchmarking and the recovered-fraction metric |

## Open Threads

Questions the wiki has surfaced but not resolved, in rough order of how much they'd change the picture:

1. **Does test-time future imagination help at all?** Three independent results now say no. SimWAM's mask ablation removes the action expert's access to future tokens and loses nothing (isolated 90.3 vs bidirectional 90.2). DriveLaW, an imagine-then-act method, finds that *earlier* denoising latents plan better and that nearly-clean generated futures **collapse the policy** (t=1 → 89.1, t=10 → 23.2 PDMS). And the counterfactual escape hatch — "imagination must matter for evaluating alternative maneuvers" — is now partly closed from the other side: on a CARLA benchmark with matched counterfactual ground truth, action-conditioned generation scores a recovered fraction of **0.38 (Vista) / 0.31 (DrivingWorld)**, i.e. closer to a replay where the event never happened than to the true counterfactual, because direct prediction never conditions on the factual continuation. The video generator is clearly valuable as a representation learner — DriveLaW's video latents beat VLM hidden states by 2.6 PDMS and BEV by 5.0 under a fixed planner — but running it forward to a clean future at decision time remains unsupported. [Auto-JEPA](wiki/sources/auto-jepa.md) suggests the question has been posed on the wrong axis: it *does* predict at inference and the prediction carries the entire system (remove it and PDMS falls 91.3 → 52.6), but what it predicts is the future **ego trajectory** latent, not a world state. The generalization that survives all five papers is therefore **future-prediction objectives are valuable; instantiated future world states at decision time are not**. [DA-WAM](wiki/sources/da-wam.md) then supplied the variable everyone was missing — **how many futures**. Its matched ablation: shared-across-candidates 92.81 < **no future at all 93.31** < one future per candidate 93.46. A *shared* future is worse than not modelling the future, because an averaged future cannot tell the scorer which candidate causes the hazard (ego progress collapses 91.36 → 88.68 while NC/TTC rise). That retro-explains SimWAM and DriveLaW, which both tested shared futures. The positive half is thin: +0.15 PDMS, single run, and the predicted horizon is only 0.5 s. [WA-JEPA](wiki/sources/wa-jepa.md) adds a third axis: the objective's *form* matters as much as its presence — on multi-view scene latents, deterministic regression scores **worse than no future prediction at all** (90.7 vs 91.1 EPDMS) while flow matching on the same target reaches 91.7, because regression collapses to a temporal mean. That puts DeepSight, FLARE, and Latent-WAM's deterministic objectives under a question none of them has answered. [ForeSight](wiki/sources/foresight.md) is the newest and most committed member of the imagine-then-act camp, and it prices the paradigm: bolting a frozen 2.5B foundation world model onto a working planner with vanilla attention buys **+0.3 PDMS** (86.8 → 87.1), and the finished system costs **900 ms with 870 ms of it in the world model** for 89.3 PDMS — against SimWAM's 91.5 at 518 ms with generation removed at inference. Its shared-future configuration is DA-WAM's (c), so the two matched measurements of that setup (−0.50 and +0.3) now straddle zero. **ForeSight also supplies the first dissent**: its denoising sweep finds more denoising monotonically better (25 steps 88.0 → 100 steps 89.3), the opposite direction to DriveLaW's. The two vary different parameters — extraction point versus total schedule length — and ForeSight never reports its extraction step, so the disagreement is unresolved rather than settled either way. Still open for long-horizon rollout and reactive interaction. **BrainWAM then breaks the tie in DriveLaW’s favour**: with decoupled video/action timesteps, one video denoising step of three recovers 89.3 of an achievable 89.5 PDMS at 475 ms, so two of three papers that measured denoising depth now say the planner wants an early, barely-formed latent. **BrainWAM also supplies the wiki’s first negative VLM-fusion result** — mixing VLM, video-generator, and action tokens in one attention pool scores 87.8, below its own WAM-only branch at 88.1, which it diagnoses as modality competition between a clean pretrained stream and a still-denoising one. **[Adaptive-WAM](wiki/sources/adaptive-wam.md) then shows the question was mis-posed.** It separates the *video noise level* from the *DiT readout depth* — axes every prior paper varied together — and finds noise level worth **<=0.15 PDMS** across five indices of a 40-step schedule while readout depth is worth **4.80**, with the **mid-network block beating the full-depth exit**. Nobody had reported which layer they read from. It also prices the alternative on identical hardware: **170 ms to plan from an intermediate feature versus 13.22 s to synthesize the future that feature encodes.** And its adaptation ladder (frozen 84.20 / separately-tuned-then-cached 84.95 / joint LoRA 90.62 / full fine-tuning 90.64) argues that a generative prior must be adapted jointly with the action objective rather than used off the shelf. **[GeoWorldAD](wiki/sources/geoworldad.md) then reopens the shared-future half of the question.** Every prior measurement of a *shared* generated future used a photometric or feature-space target and landed near or below zero (DA-WAM −0.50, SimWAM ~0, ForeSight +0.3). GeoWorldAD’s shared future is **geometric** — latent tokens supervised by future depth — and it measures **+1.7 PDMS / +2.8 EPDMS**, with ego progress up 3.3 and safety flat, the exact mirror of DA-WAM’s progress collapse. The confound is that its baseline has 32K planner steps against the full model’s 96K, so the mechanism is not separated from 3× the training. The wiki’s rule is now scoped: *a shared **photometric** future is useless to harmful*, and the geometric case is open. GeoWorldAD also measures something no backbone comparison here had varied: **an anchor-frame geometry foundation model beats a from-scratch planner by only 0.6 PDMS while lowering NC/DAC/TTC, and re-expressing its output in per-timestep ego frames — pure re-parameterization — is worth +2.5.** See [World Models for AD](wiki/concepts/world-model-for-ad.md#test-time-imagination) and [Counterfactual Prediction](wiki/concepts/counterfactual-prediction.md).
2. **Is NAVSIM-v1 saturated?** Best-of-6 reaches 94.8 = the human ground-truth score (Curious-VLA). If oracle selection already matches the logged human, single-sample gains above ~93 may be measuring selection quality rather than driving quality. See [Best-of-N Sampling](wiki/concepts/best-of-n.md).
3. **Are video-prior gains about scale or objective?** SimWAM shows video-prior scale barely matters (Wan2.1-1.3B 90.2 ≈ Wan2.2-5B 90.3) while DriveWAM shows dropping the video objective is catastrophic. Together they point at the training signal, not the backbone — but no paper has tested this across architectures.
4. **Do PDMS gains transfer to closed-loop?** Most 90+ PDMS methods report no Bench2Drive, HUGSIM, or navhard result. Stage-2 navhard remains near 40 EPDMS for everything measured.
5. **Is navtest measuring the wrong thing for world models?** [GeoWAM](wiki/sources/geowam.md) adds future-geometry forecasting to DVGT-2 and gains **+0.6 EPDMS on navtest but +4.9 on navhard** — the same change worth eight times more under the reactive protocol, exactly as a world-model thesis predicts. Every world-model paper in the wiki optimizes navtest; only two report navhard at all. If the asymmetry replicates, the field has been evaluating anticipation on a benchmark where anticipation barely pays.
6. **Comparison-scope inflation is systemic.** Nearly every ingested paper claims SOTA against a table that omits the actual frontier. The wiki tracks this per-paper, but it makes cross-paper ranking unreliable in principle.

## Known Gaps

Methods cited frequently across ingested papers but **not yet ingested** (mention counts refreshed at the 2026-09-02 lint, `wiki/log.md` excluded):

| Method | Mentions | Why it matters |
|---|---:|---|
| **Hydra-MDP / ++** | 85 | Most-cited un-ingested method by a wide margin; underpins the EPDMS / Hydra-MDP++ metric definitions used throughout the wiki |
| **DVGT-2** | 56 | [GeoWAM](wiki/sources/geowam.md)'s encoder, point-head initialization, *and* strongest baseline on both navtest (89.6) and navhard (31.7) — GeoWAM's own contribution over it is only +0.6 / +4.9 |
| **Vista** | 72 | One of two frozen backbones evaluated in [How Can Driving World Models Do Counterfactual Prediction?](wiki/sources/driving-wm-counterfactuals.md) |
| **DrivingWorld** | 40 | The other one |
| **DrivoR** | 40 | 93.1 PDMS in DA-WAM's table; also GeoWAM's strongest HUGSIM baseline |
| **CLOVER** | 24 | [Auto-JEPA](wiki/sources/auto-jepa.md) initializes its scorer from CLOVER's released checkpoint, and that scorer contributes +3.7 of Auto-JEPA's 91.3 PDMS |
| **SparseDriveV2** | 23 | 92.0 PDMS / 90.1 corrected EPDMS |
| **WorldRFT** | 23 | Recurring NAVSIM-v2 baseline |
| **SimLingo** | 26 | Recurring closed-loop baseline |
| **iPad** | 29 | 91.7 PDMS |
| **EponaV2** | 17 | 36.1 navhard EPDMS — second only to GeoWAM, and it uses RL supervision |
| **DriveFuture** | 13 | 89.9 corrected EPDMS; DA-WAM's closest cited relative |
| **CoWorld-VLA** | 11 | 90.0 corrected EPDMS; shares four authors with WA-JEPA |
| **Discrete-WAM** | 10 | 90.4 corrected EPDMS |
| **Centaur** | 10 | 92.6 PDMS |
| **NavFormer** / **LEAD (LTFv6)** | 8 / 8 | 34.1 / 31.9 navhard EPDMS, both RL-supervised |

Also cited but not ingested: WoTE (25), WorldMirror (23), LAW (21), GoalFlow (18), VGGT (16), Doe-1 (14), DrivingGPT (14), VaVAM (12), World4Drive (10), Drive-WM (10), OccWorld (9), GTRS (9), DIVER (9), ColaVLA (8), DriveWorld (8), SeerDrive (6), UniWorldVLA (5), VGGT-World (5), ZTRS (4), ResAD (3), IDOL / LCDrive / BeyondDrive / MapAnything / CUT3R / DUSt3R (2 each).

**Highest-value next ingests**: DVGT-2 (an attribution gap in a current result), CLOVER (same), and Hydra-MDP (a definitional dependency for the whole EPDMS discussion).

## Workflow

The ingest workflow is defined in `CLAUDE.md`. Each paper goes through:
1. Read source + figures
2. Discuss key takeaways and limitations
3. Create/update `wiki/sources/<paper>.md` with embedded figures and full tables
4. Create/update relevant `wiki/concepts/` pages
5. Update `wiki/index.md` and append to `wiki/log.md`
