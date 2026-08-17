---
title: "DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/DeepSight_ Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/bench2drive.md, concepts/chain-of-thought-for-ad.md, concepts/foundation-backbones-for-ad.md, concepts/pdm-lite.md, sources/futuresightdrive.md, sources/dynvla.md, sources/flare.md, sources/adathinkdrive.md, sources/orion.md, sources/autovla.md]
created: 2026-07-01
updated: 2026-07-01
confidence: high
---

**Paper**: DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving
**Authors**: Lingjun Zhang, Changjie Wu, Linzhe Shi, Jiangyang Li, Jiaxin Liu, Lei Yang, Hang Zhang, Mu Xu, Hong Wang
**Org**: not stated in source markdown (venue tag: Machine Learning, ICML)
**arXiv**: 2605.10564v1
**Code**: https://github.com/hotdogcheesewhite/DeepSight

---

## Summary

DeepSight is a unified generative-understanding VLA for closed-loop driving whose central idea is **parallel prediction of latent semantic features for five consecutive future frames** in the bird's-eye-view (BEV) space, produced in a single forward pass via learnable **World Queries**. Instead of autoregressively generating future *pixels* (VAE-codebook world models such as [[sources/futuresightdrive.md]]), DeepSight regresses the DINOv3 features of future BEV frames with an MSE loss — capturing planning-relevant semantics and spatial layout while avoiding the cost and short-sightedness of pixel generation. A second component, an **adaptive Chain-of-Thought** module, lets the model decide per-scene whether to emit text reasoning (for long-tail cases such as emergency-vehicle yielding, construction zones, traffic signs) or a placeholder token, keeping inference cheap.

On Bench2Drive (CARLA V2 closed-loop, Think2Drive expert protocol), DeepSight reports **84.52 DS / 65.91% SR without CoT** and **86.23 DS / 71.36% SR with adaptive CoT**, plus best open-loop nuScenes L2/collision in its comparison table. The world model contributes the bulk of the gain; adaptive CoT is a smaller complementary boost.

---

## Core Idea: Latent States Prediction vs. Pixel World Models

![[x1 39.png|Paradigms of different unified world models]]

**Figure 1**: (a) VLMs that explicitly output codebook tokens for a single future frame cannot support long-horizon prediction — this "short-sightedness" hinders trajectory planning. (b) DeepSight predicts future **multi-frame latent features**, enabling long-sighted planning. (c) DeepSight leads most Bench2Drive metrics vs. E2E methods.

The paper argues an ideal driving world model needs four human-like capabilities: precise **semantic understanding**, accurate **spatial localization**, **long-horizon motion modeling**, and **rapid response**. Prior unified world models fall short because they (1) predict codebook/texture tokens that prioritize appearance over semantics, (2) predict only short-term futures (~0.5s), and (3) model only forward-view rather than surrounding agents.

---

## Architecture

![[x2 37.png|DeepSight pipeline: long-term driving-world model + adaptive CoT]]

**Figure 2**: Two modules. (a) **Long-term driving-world model** aligns DINOv3 features extracted from future multi-frame RGB in BEV space during training. (b) **Adaptive CoT module** integrates external knowledge to enhance reasoning in long-tail cases.

**Backbone**: Qwen2.5-VL-3B (vision encoder frozen, LLM fully fine-tuned).

### Unified Outputs

The model $M_\text{uni}$ jointly produces, in a single forward pass:
- Future latent features $\mathbf{F}=[f_0,f_1,f_2,f_3,f_4]$, each $f_k\in\mathbb{R}^{h_\text{bev}\times w_\text{bev}\times d_\text{bev}}$ at time $t+k\cdot\Delta t$ ($\Delta t=0.5$s → 2s horizon)
- Adaptive CoT text $T_\text{cot}$
- Trajectory waypoints $\mathbf{P}_t=\{p_1,\dots,p_n\}$, $p_i=(x_i,y_i)$

$$\mathbf{F},\,T_\text{cot},\,\mathbf{P}_t = M_\text{uni}(\mathbf{I}_t,\,\mathbf{I}_{t-\tau},\,T_\text{target},\,T_\text{ego},\,\mathbf{Q}_\text{world})$$

Inputs: current multi-view images $\mathbf{I}_t$ (spatial modeling), historical frames $\mathbf{I}_{t-\tau}$ ($\tau=1,2,3,4$; motion dynamics), ego state $T_\text{ego}$, target point $T_\text{target}$, and learnable **World Queries** $\mathbf{Q}_\text{world}=[q_0,\dots,q_4]$.

### Driving-World Model (parallel latent prediction)

World Queries enable **parallel inference of motion states across all five future time steps in one forward pass** — the key to long-horizon modeling without autoregressive drift or per-frame generation cost.

**Ground-truth construction**: DINOv3 (frozen, ViT-L/16) is the semantic feature extractor. Targets are $f_i = \phi_\text{dino}(I_i^\text{bev})$, where $I_i^\text{bev}$ is a **BEV-rendered image or semantic segmentation map**. The model focuses on characterizing the semantic and spatial distribution of the environment rather than pixel texture.

### Adaptive CoT

After observing inputs and modeling future state $\mathbf{F}$, the model autonomously decides whether to activate CoT:

$$T_\text{cot} = M_\text{uni}(\mathbf{I}_t,\,\mathbf{I}_{t'},\,T_\text{target},\,T_\text{ego},\,\mathbf{Q}_\text{world}\mid\mathbf{F})$$

If activated, it emits structured reasoning text; otherwise a placeholder token $T_\text{cot}^{\emptyset}$ minimizes overhead. Across 220 routes the mechanism triggered in **less than 30% of frames**.

### Unified Training Loss

Trajectory waypoints are quantized to discrete **action tokens** by their pixel-space BEV-grid coordinates ($p_i\to t_i\in\{1,\dots,K\}$), so trajectory and CoT share a tokenized space:

$$L=\lambda_\text{traj}L_\text{traj}+\lambda_\text{cot}L_\text{cot}+\lambda_\text{world}L_\text{world}$$

- $L_\text{traj}=\text{CE}$ over trajectory tokens
- $L_\text{cot}=\text{CE}$ over CoT text tokens
- $L_\text{world}=\text{MSE}$ between predicted latents $\mathbf{F}$ and DINOv3 ground-truth $\mathbf{F}^\text{gt}$

Weights: $\lambda_\text{cot}=\lambda_\text{traj}=1.0$; $\lambda_\text{world}$ tuned (best 1.0, see Table 10).

### Inference

Prefill $\mathcal{X}=(\mathbf{I}_{t-\tau},\mathbf{I}_t,T_\text{ego},T_\text{target},\mathbf{Q}_\text{world})$; then:

$$p(\mathbf{P}_t,T_\text{cot},\mathbf{F}\mid\mathcal{X}) = p(\mathbf{F}\mid\mathcal{X})\cdot p(T_\text{cot}\mid\mathcal{X},\mathbf{F})\cdot p(\mathbf{P}_t\mid\mathcal{X},\mathbf{F},T_\text{cot})$$

All three outputs in one unified forward pass; no external generative model required. Parallel latent decoding + adaptive CoT give negligible overhead vs. a native VLM.

---

## Adaptive CoT Annotation Pipeline

Because no standardized adaptive-CoT dataset exists for Bench2Drive, the authors built an automated labeling pipeline based on **Qwen3-VL-235B**, synthesizing **~1.3M structured annotations** (to be open-sourced). Three components: (1) scene-complexity assessment, (2) complexity-based external-knowledge retrieval, (3) driving-behavior determination.

![[x4 31.png|A complete sample of the annotation dataset]]

**Figure 4**: A complete annotation sample with reasoning steps.

![[x5 28.png|Prompt for CoT annotation by Qwen3-VL-235B]]

**Figure 5**: The annotation prompt.

Two post-processing stages: a **Format Filter** enforcing the three-part structure (infer current action from history → decide whether complex decision-making is required → summarize) and filtering mismatches (judged simple but reasoned; judged complex but no reasoning); and a **Classify** step that keeps only the summary reasoning for "complex" scenes and assembles the true trajectory for "simple" scenes — so only the key reasoning part is distilled into DeepSight.

---

## Results

### Bench2Drive Closed-Loop (Table 1, 220 routes, base set)

\* = expert feature distillation. Red = gains over latest SOTA under the same Think2Drive protocol. **Gray methods use a different (PDM-Lite) expert distribution and are reference-only.**

| Method | Paradigm | Expert | DS ↑ | SR (%) ↑ | Efficiency ↑ | Comfort ↑ |
|---|---|---|---:|---:|---:|---:|
| TCP* | E2E | Think2Drive | 40.70 | 15.00 | 54.26 | 47.80 |
| TCP-traj* | E2E | Think2Drive | 59.90 | 30.00 | 76.54 | 18.08 |
| ThinkTwice* | E2E | Think2Drive | 62.44 | 31.23 | 69.33 | 16.22 |
| DriveAdapter* | E2E | Think2Drive | 64.22 | 33.08 | 70.22 | 16.01 |
| VAD | E2E | Think2Drive | 42.35 | 15.00 | 157.94 | 46.01 |
| GenAD | E2E | Think2Drive | 44.81 | 15.90 | — | — |
| MomAD | E2E | Think2Drive | 44.54 | 16.71 | 170.21 | 48.63 |
| DriveTrans | E2E | Think2Drive | 63.46 | 35.01 | 100.64 | 20.78 |
| ReasonPlan | VLM | Think2Drive | 64.01 | 34.55 | 180.64 | 25.63 |
| ORION | VLM | Think2Drive | 77.74 | 54.62 | 151.48 | 17.38 |
| AutoVLA | VLM | Think2Drive | 78.84 | 57.73 | 146.93 | 39.33 |
| *DiffusionDrive* (gray) | E2E | PDM-Lite | 77.68 | 52.72 | — | — |
| *SimLingo* (gray) | VLM | PDM-Lite | 85.94 | 66.82 | 244.18 | 25.49 |
| **DeepSight w/o adaptive CoT** | VLM | Think2Drive | **84.52** (+5.68) | **65.91** (+8.81) | 198.80 (+18.16) | 14.25 |
| **DeepSight** | VLM | Think2Drive | **86.23** (+7.39) | **71.36** (+13.63) | 201.71 (+21.07) | 16.11 |

Gains (red) are computed vs. **AutoVLA** (78.84 DS / 57.73 SR), the strongest prior Think2Drive method. DeepSight's efficiency (201.71) is the highest in the table. Comfort (16.11) reflects the usual agility–smoothness trade-off; the paper notes it could be improved via post-smoothing / controller tuning.

### Multi-Ability (Table 2, %)

| Method | Paradigm | Merging | Overtaking | Emerg. Brake | Give Way | Traffic Sign | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| ReasonPlan | VLM | 37.50 | 26.67 | 33.30 | 40.00 | 45.76 | 36.66 |
| DriveTrans | VLM | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| ORION | VLM | 25.00 | 71.11 | 78.33 | 30.00 | 69.15 | 54.72 |
| **DeepSight** | VLM | **60.00** | **91.11** | 78.33 | **50.00** | **71.58** | **70.20** (+15.48) |

Largest gains in overtaking (91.11) and merging (60.00) — the paper attributes this to strong long-horizon BEV spatial modeling of multi-vehicle causal relationships.

### Open-Loop nuScenes (Table 11, with ego status)

\* = reproduced with the same base model and settings as DeepSight.

| Method | L2 1s | L2 2s | L2 3s | **Avg L2 ↓** | Col 1s | Col 2s | Col 3s | **Avg Col ↓** |
|---|---|---|---|---|---|---|---|---|
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.18 | 0.43 | 0.23 |
| LAW | 0.26 | 0.57 | 1.01 | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| World4Drive | 0.23 | 0.47 | 0.81 | 0.50 | 0.02 | 0.12 | 0.33 | 0.16 |
| FSDrive* | 0.27 | 0.33 | 0.56 | 0.35 | 0.07 | 0.10 | 0.24 | 0.14 |
| **DeepSight** | **0.16** | **0.31** | **0.52** | **0.33** | **0.02** | **0.07** | 0.27 | **0.12** |

Note: §4.2 separately states "our L2 error is reduced to 0.58" (a different, likely no-ego protocol); Table 11's 0.33 avg is under the with-ego protocol used for the VAD/LAW/World4Drive comparison. The FSDrive comparison here is DeepSight's **own reproduction** (0.35 avg), not FSDrive's published no-ego 0.96 avg — different protocol, not directly comparable to the FSDrive paper.

---

## Ablations

### Explicit Reconstruction vs. Latent Semantic; Short vs. Long Horizon (Table 3, Dev 10 routes)

| ID | Type | Frame | RC ↑ | IS ↑ | DS ↑ |
|---|---|---|---:|---:|---:|
| 1 | VAE | One | 47.56 | 0.64 | 27.75 |
| 2 | DINOv3 | One | 90.49 | 0.83 | 74.79 |
| 3 | VAE | Five | 27.02 | 0.66 | 14.66 |
| 4 | DINOv3 | Five | 95.95 | 0.89 | 86.57 |

- **Latent semantic ≫ explicit VAE reconstruction**: DINOv3 vs. VAE single-frame → **+47.04 DS** (ID 2 vs. 1).
- **VAE cannot model long horizon**: extending VAE to five frames *drops* DS by **−13.09** (ID 1→3).
- **Latent world modeling benefits from long horizon**: DINOv3 single→five frames → **+11.78 DS** (ID 2→4).

### Forward-View vs. BEV (Table 4, Dev 10 routes)

| ID | Type | RC ↑ | IS ↑ | DS ↑ |
|---|---|---:|---:|---:|
| 1 | Front view | 89.47 | 0.87 | 77.77 |
| 2 | BEV view | 95.95 | 0.89 | 86.57 |

BEV spatial modeling adds **+8.8 DS** over front-view — the front view lacks surrounding-agent modeling needed for safe long-horizon planning.

### World Model vs. Adaptive CoT (Table 5, 220 routes)

| ID | WM | ADA-CoT | DS ↑ | SR ↑ | Eff ↑ |
|---|---|---|---:|---:|---:|
| 1 | ✗ | ✗ | 58.16 | 28.18 | 190.76 |
| 2 | ✗ | ✓ | 69.87 | 42.27 | 187.39 |
| 3 | ✓ | ✗ | 84.52 | 65.91 | 198.80 |
| 4 | ✓ | ✓ | **86.23** | **71.36** | 201.71 |

The **world model is the dominant contributor** (+26.36 DS from ID 1→3). Adaptive CoT alone (ID 2, 69.87) helps far less than the world model alone (ID 3, 84.52); its incremental value on top of WM is +1.71 DS / +5.45 SR (ID 3→4). The two are complementary, but the paper is explicit that CoT alone has "inherent limitations" in AD.

### Inference Overhead (Table 6)

| Method | WM | CoT | Add. Time % ↓ |
|---|---|---|---:|
| Qwen2.5-VL | ✗ | ✗ | 0 |
| FSDrive | ✓ | ✗ | +60.71 |
| DeepSight | ✓ | ✗ | +3.57 |
| DeepSight | ✓ | ✓ | +7.69 (+4.12) |

Parallel latent prediction adds only **+3.57%** latency vs. a native VLM — an order of magnitude cheaper than FSDrive's autoregressive pixel CoT (+60.71%). Conditional CoT activation adds only +4.12% on average.

### λ_world Sensitivity (Table 10, Dev 10 routes)

| λ_world | RC ↑ | IS ↑ | DS ↑ |
|---|---:|---:|---:|
| 0.5 | 88.23 | 0.89 | 78.93 |
| **1.0** | **95.95** | **0.89** | **86.57** |
| 2.0 | 89.12 | 0.89 | 79.82 |

Non-monotonic: performance peaks at 1.0 and drops at 2.0 (over-weighting the ambiguous world-modeling objective degrades planning), but even 2.0 beats minimal world modeling.

---

## Implementation Details

- **Hardware**: 64× NVIDIA H20 GPUs (96 GB each).
- **Main training**: lr $2\times10^{-5}$, batch 128, 2 epochs on Bench2Drive. (Appendix B SFT config: lr $2\times10^{-4}$, batch 64, 2 epochs.)
- **Bench2Drive**: 10,000 training segments (~150 m each); 220 official short routes across 44 scenarios (5 routes each).
- **Backbone params** (Tables 7–9): Qwen2.5-VL vision encoder — 32 layers, hidden 1280, 16 heads, patch 14; LLM — 36 layers, hidden 2048, 2 KV heads, head size 128; DINOv3-ViT-L/16 — 300M params, hidden 1024, pretrained on LVD-1689M, patch 16.

### Qualitative

![[x3 33.png|Qualitative closed-loop results with CoT output]]

**Figure 3**: Three scenarios; first three columns are consecutive BEV frames (red box = ego), fourth is the critical PV image, fifth is the model's CoT analyzing construction zones, emergency vehicles, and traffic signs.

![[x6 24.png|Qualitative results on Bench2Drive closed-loop set]]

**Figure 6**: Adverse weather (wet-road reflections), jaywalking pedestrians, narrow-road overtaking by borrowing the opposite lane.

![[x7 19.png|DeepSight world-prediction visualization]]

**Figure 7**: Predicted future DINO features preserve road geometry and track agent/ego motion across frames without the feature blurring common to pixel world models.

---

## Limitations

1. **Not the overall wiki Bench2Drive frontier.** DeepSight's Table 1 omits [[sources/linkvla.md]] (91.01 DS), [[sources/dynvla.md]] (88.34 DS), and [[sources/automot.md]] (87.34 DS, PDM-Lite) — all higher DS in the wiki. The SOTA claim is scoped to the **Think2Drive expert protocol** and its comparison set (strongest there: AutoVLA 78.84). Under Think2Drive the +7.39 DS jump is large and honest, but "SOTA on Bench2Drive" is comparison-scope-limited. See [[concepts/bench2drive.md]] and [[concepts/pdm-lite.md]].
2. **Expert-attribution inconsistency across papers.** DeepSight labels AutoVLA as *Think2Drive*; [[sources/automot.md]]'s table labels AutoVLA as *PDM-Lite*. The literature does not consistently attribute expert-data distributions, so cross-paper DS comparisons remain fragile.
3. **Privileged training targets.** Ground-truth latents require **BEV-rendered images or semantic segmentation maps** run through DINOv3 — an annotation/rendering dependency at training time (not used at inference, but still a supervision cost beyond raw video).
4. **Benchmark breadth.** Closed-loop only on Bench2Drive; open-loop only on nuScenes. No NAVSIM / NAVSIM-v2 / EPDMS, no HUGSIM, no Waymo — cannot place it on the wiki's dominant NAVSIM leaderboard.
5. **nuScenes protocol ambiguity.** Table 11 uses ego status (BEV-Planner caveat: ego status inflates L2); the text's separate 0.58 L2 figure is unreconciled with the table's 0.33.
6. **Low comfort** (14.25 w/o CoT, 16.11 with) — among the lowest in Table 1; DeepSight's policy is agile/aggressive.
7. **Compute.** 64× H20 GPUs; the authors flag high VLM complexity for real-time scaling as the main future-work bottleneck.
8. **No RL/RFT.** SFT-only; unlike most NAVSIM-era peers there is no GRPO reward optimization.

---

## Key Cross-References

- **World-model pattern**: [[concepts/world-model-for-ad.md]] — DeepSight adds parallel multi-frame **DINOv3 latent-feature** prediction in BEV; contrast with FLARE (single-frame DINOv2 auxiliary loss), FSDrive (VQ-VAE pixel CoT), and Latent-WAM (latent world-status, no VLM).
- **Bench2Drive standing & expert protocol**: [[concepts/bench2drive.md]], [[concepts/pdm-lite.md]] — Think2Drive vs. PDM-Lite *expert-data* distinction.
- **Adaptive CoT**: [[concepts/chain-of-thought-for-ad.md]] — scene-gated reasoning like [[sources/adathinkdrive.md]], here with a Qwen3-VL-235B annotation pipeline.
- **Backbone roles**: [[concepts/foundation-backbones-for-ad.md]] — DINOv3-ViT-L/16 as a training-time semantic-feature *target* extractor (removed at inference).
- **Direct baseline**: [[sources/futuresightdrive.md]] — the VAE/pixel world model DeepSight ablates against and beats on efficiency (+3.57% vs. +60.71% latency).
