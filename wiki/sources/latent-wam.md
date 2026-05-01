---
title: "Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/perception-for-planning.md, concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/foundation-backbones-for-ad.md, sources/epona.md, sources/drivevla-w0.md, sources/dreameraD.md, sources/driveva.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

# Latent-WAM

**Paper**: Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving  
**arXiv**: https://arxiv.org/html/2603.24581v1  
**Authors**: Linbo Wang, Yupeng Zheng, Qiang Chen, Shiwei Li, Yichen Zhang, Zebin Xing, Qichao Zhang, Xiang Li, Deheng Qian, Pengxuan Yang, Yihang Dong, Ce Hao, Xiaoqing Ye, Junyu Han, Yifeng Pan, Dongbin Zhao  
**Type**: Perception-free latent world-model planner using spatial compression, geometric distillation, and causal latent dynamics prediction.

## Core Idea

Latent-WAM targets a weakness in prior world-model planners: pixel/video generation is expensive, while earlier latent world models are often too weak spatially and dynamically. The paper compresses multi-view images into a compact latent world state and trains that latent state to be useful for planning.

The method has three parts:

- **Spatial-Aware Compressive World Encoder (SCWE)**: DINOv2-Base image encoder plus 16 learnable scene queries per camera view. It compresses multi-view patch tokens into compact scene tokens.
- **Geometric distillation**: frozen WorldMirror/VGGT-style geometric features supervise the DINO patch tokens through cosine feature alignment. The geometry model is training-only and cached offline.
- **Dynamic Latent World Model (DLWM)**: a causal Transformer predicts future world status tokens from historical scene tokens and ego status, using teacher-forced frame-wise attention, 3D-RoPE, and ego status supervision.

At inference, only SCWE and the trajectory decoder are used. The dynamic world model, EMA target encoder, and geometric foundation model are training-time representation shapers, not deployed modules.

## Figures

![[x1 30.png|Figure 1: NAVSIM-v2 EPDMS vs. training data scale. Latent-WAM is shown as a compact 104M model with 89.3 EPDMS using far less data than Epona and Drive-JEPA.]]

![[x2 28.png|Figure 2: Latent-WAM architecture. SCWE compresses image patch tokens into scene tokens with geometric distillation; DLWM predicts future latent world status; a trajectory decoder plans from the current world status and driving command.]]

![[x3 28.png|Figure 3: Teacher-forcing attention mask. Tokens attend bidirectionally within a frame block and causally across previous frame blocks.]]

![[x4 26.png|Figure 4: NAVSIM trajectory comparison against Epona and World4Drive. Yellow is prediction and green is human trajectory.]]

![[x5 24.png|Figure 5: Scene-token attention maps. Geometric distillation focuses attention on lane markings, drivable areas, and intent-relevant spatial structure.]]

![[latent-wam-x6.png|Figure 6: Supplementary NAVSIM trajectory visualizations across additional scenes.]]

![[latent-wam-x7.png|Figure 7: Supplementary going-forward attention maps across consecutive frames, comparing with and without geometric distillation.]]

![[latent-wam-x8.png|Figure 8: Supplementary lane-changing attention maps across consecutive frames.]]

![[latent-wam-x9.png|Figure 9: Supplementary left-turn attention maps across consecutive frames.]]

![[latent-wam-x10.png|Figure 10: Supplementary right-turn attention maps across consecutive frames.]]

![[latent-wam-x11.png|Figure 11: HUGSIM trajectory planning visualization on nuScenes-derived scenes.]]

![[latent-wam-x12.png|Figure 12: HUGSIM trajectory planning visualization on KITTI-360-derived scenes.]]

![[latent-wam-x13.png|Figure 13: Additional HUGSIM trajectory planning visualization on KITTI-360.]]

![[latent-wam-x14.png|Figure 14: HUGSIM trajectory planning visualization on Waymo-derived scenes.]]

![[latent-wam-x15.png|Figure 15: HUGSIM trajectory planning visualization on PandaSet-derived scenes.]]

## Method Details

### Spatial-Aware Compressive World Encoder

Input is 3 camera views across 4 frames. Each image is resized/cropped to 224x448 and encoded by DINOv2-Base. A set of 16 scene queries is concatenated with image patch tokens and processed by the encoder, producing compact scene tokens with latent dimension 256.

The key design choice is to preserve planning-relevant spatial information while compressing the large image-token set. Without compression, long-horizon latent prediction is expensive; with naive compression, spatial grounding degrades. Latent-WAM makes compression viable by distilling dense geometric features into the image patch tokens.

### Geometric Alignment

The teacher is WorldMirror, described as a feed-forward geometric foundation model built on VGGT. It receives multi-view images plus camera intrinsics/extrinsics and outputs patch-level geometric features with dimension 2048. The SCWE patch tokens are projected into the same dimension and trained with normalized cosine similarity:

$$
\mathcal{L}_{align}=1-\cos(\mathrm{LN}(\phi(\hat{X})), \mathrm{LN}(f_g(I)))
$$

The teacher is frozen and its features are precomputed. This avoids runtime geometry-model cost during both training loops and inference.

### Dynamic Latent World Model

Per-camera scene tokens are aggregated with ego status tokens into a frame-wise world status:

$$
S_{world}\in\mathbb{R}^{T\times(M\times N+1)\times D_l}
$$

where the extra token is ego status. DLWM is a 4-layer causal Transformer decoder with 8 heads and 3D-RoPE over time, camera index, and token index. It predicts future world status tokens from historical world status tokens. The target encoder is an EMA copy of SCWE, giving stable latent targets.

Ego status prediction adds supervised command, velocity, and acceleration losses, forcing the future latent world state to carry motion-relevant information rather than only visual reconstruction information.

### Trajectory Decoder

A lightweight Transformer trajectory decoder consumes current world status and learnable trajectory queries. It decodes K candidate trajectories, each with poses $(x,y,\theta)$ in ego coordinates, and selects the candidate corresponding to the driving command.

### Training and Inference

Total loss:

$$
\mathcal{L}=\mathcal{L}_{traj}+0.1\mathcal{L}_{align}+0.2\mathcal{L}_{wm}+0.1\mathcal{L}_{ego}
$$

Training uses 32 A100 GPUs, batch size 512, AdamW, 100 epochs, BF16, and takes about two days. The inference model has 104M parameters. Training additionally uses an EMA encoder, bringing the training-time total to 191M parameters, but only 104M are trainable.

## Main Results

### Table 1: NAVSIM-v2

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ReCogDrive | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| WorldRFT | 97.8 | 96.5 | 99.5 | 99.8 | 88.5 | 97.0 | 97.4 | 98.1 | 69.1 | 86.7 |
| Drive-JEPA dagger | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | 87.8 |
| World4Drive | 97.8 | 96.3 | 99.4 | 99.8 | 88.3 | 97.1 | 97.7 | 98.0 | 53.9 | 84.8 |
| Epona | 97.1 | 95.7 | 99.3 | 99.7 | 88.6 | 96.3 | 97.0 | 98.0 | 67.8 | 85.1 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **Latent-WAM** | **98.1** | **97.3** | **99.6** | **99.8** | **87.7** | **97.3** | **97.6** | **98.1** | **87.3** | **89.3** |

The paper marks Drive-JEPA as perception-annotation-dependent. Latent-WAM is classified as perception-free because it does not require perception labels or inference-time perception modules, though it does use a frozen geometric foundation model as a training teacher.

### Table 2: HUGSIM Zero-Shot Closed-Loop

| Method | RC Easy | RC Medium | RC Hard | RC Extreme | RC Avg | HDS Easy | HDS Medium | HDS Hard | HDS Extreme | HDS Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UniAD | 58.6 | 41.2 | 40.4 | 26.0 | 40.6 | 48.7 | 29.5 | 27.3 | 14.3 | 28.9 |
| VAD | 38.7 | 27.0 | 25.5 | 23.0 | 27.9 | 24.3 | 9.9 | 10.4 | 8.2 | 12.3 |
| LTF | 68.4 | 40.7 | 36.9 | 25.5 | 41.4 | 52.8 | 24.6 | 19.8 | 8.1 | 24.8 |
| GTRS-Dense | 64.2 | 50.0 | 20.7 | 22.3 | 38.0 | 55.5 | 39.0 | 11.7 | 14.3 | 28.6 |
| **Latent-WAM** | **84.2** | **42.5** | **30.6** | **35.5** | **45.9** | **72.5** | **24.0** | **12.2** | **18.1** | **28.9** |

Latent-WAM is trained only on NAVSIM and evaluated zero-shot on HUGSIM. It ranks first on average RC and ties UniAD on average HD-Score, but the split profile is uneven: Easy is very strong, Medium/Hard HDS are lower than some baselines, and Extreme RC is comparatively strong.

## Ablations

### Table 3: Component Ablation

| Compression | Geometry | World Model | Ego Status | EPDMS | Delta |
| --- | --- | --- | --- | ---: | ---: |
| no | no | no | no | 87.9 | 0.0 |
| yes | no | no | no | 87.7 | -0.2 |
| yes | no | yes | no | 88.0 | +0.1 |
| yes | no | yes | yes | 88.3 | +0.4 |
| yes | yes | no | no | 88.6 | +0.7 |
| yes | yes | yes | no | 89.0 | +1.1 |
| yes | yes | yes | yes | 89.3 | +1.4 |

Compression alone slightly hurts because it discards patch detail. Geometry recovers and improves spatial grounding; world modeling and ego-status supervision add complementary gains.

### Table 4: Geometry Injection

| Geometric information | EPDMS |
| --- | ---: |
| No geometric feature | 88.3 |
| Concatenation | 88.0 |
| Distillation | 89.3 |

Directly concatenating frozen geometry features hurts, while distillation into the trainable vision backbone works. The paper argues frozen features are misaligned with the planning objective if used as raw inputs.

### Table 5: Vision Backbone for Distillation

| DINO backbone | EPDMS |
| --- | ---: |
| Small | 86.3 |
| Base | 89.3 |
| Small-LoRA | 84.7 |
| Base-LoRA | 68.5 |

Full fine-tuning of DINOv2-Base is critical. LoRA is unstable for this high-dimensional geometric distillation target, with Base-LoRA collapsing hardest.

### Table 6: World-Model Prediction Temporal Stride

| Prediction temporal stride | EPDMS |
| --- | ---: |
| 0 -> 8 | 88.4 |
| -3 -> 0 -> 4 -> 8 | 89.3 |
| -3 -> -2 -> -1 -> 0 -> 2 -> 4 -> 6 -> 8 | 89.1 |

Moderate stride works best. Predicting only the final frame underuses temporal supervision; denser stride adds overhead and redundant adjacent-frame targets.

### Table 7: Per-Module Inference Latency

| Module | Parameters | Memory usage | Latency |
| --- | ---: | ---: | ---: |
| World Encoder | 86.6M | - | 100 ms |
| Trajectory Decoder | 8.4M | - | 6 ms |
| All modules | 104M | 1.1 GB | 107 ms |

The deployed model is lightweight relative to large VLM/video-backbone planners. Most latency is the world encoder.

## Relationship to Other Wiki Papers

- **Epona**: Epona uses explicit AR+diffusion world modeling and can generate long-horizon video. Latent-WAM instead predicts compact future latent world status and never decodes pixels at inference.
- **DriveVLA-W0**: both use training-time world modeling to improve planning representations. Latent-WAM is more compact and reports NAVSIM-v2 89.3 EPDMS vs. DriveVLA-W0 86.1 in the paper's table.
- **DreamerAD**: DreamerAD uses a latent world model as an RL reward source. Latent-WAM uses latent future prediction only as representation learning; no RL stage is reported.
- **DriveVA**: DriveVA couples future video latents and action tokens inside one DiT. Latent-WAM avoids video generation entirely and is much smaller.
- **Percept-WAM / Perception-Enhanced Planning**: Latent-WAM does not train explicit detection/BEV heads, but geometric distillation plays a similar role: force spatially accurate representations before trajectory decoding.
- **HAD**: both report HUGSIM, but HAD is a hierarchical diffusion/RL planner while Latent-WAM is a latent world-state representation learner.

## Limitations

- **No NAVSIM-v1 PDMS result**: the paper focuses on NAVSIM-v2 EPDMS and HUGSIM.
- **No RL/RFT stage**: the strong 89.3 EPDMS is SFT/self-supervised training only; the paper does not test whether GRPO, MDPO, or latent-reward RL adds further gains.
- **Geometry teacher dependence**: WorldMirror features are cached and not used at inference, but the training recipe still depends on a frozen geometric foundation model and camera calibration.
- **LoRA failure is severe but unexplained mechanistically**: Base-LoRA drops to 68.5 EPDMS, suggesting instability that may matter for scaling or low-memory training.
- **HUGSIM score is mixed**: average RC is best in the table, but HDS only ties UniAD and Hard/Medium HDS are weaker than some baselines.
- **Comparison scope is narrow relative to the full wiki**: NAVSIM-v2 table excludes WAM-Diff, ExploreVLA, DriveDreamer-Policy, DreamerAD, Vega BoN-6, HAD, and several VLA/RL methods already tracked in this wiki.

