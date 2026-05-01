---
title: "Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving"
type: source-summary
sources: [raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md]
related: [concepts/world-model-for-ad.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/bench2drive.md, concepts/foundation-backbones-for-ad.md, sources/latent-wam.md, sources/epona.md, sources/drivesuprim.md, sources/diffusiondrive.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

# Drive-JEPA

Drive-JEPA combines Video Joint-Embedding Predictive Architecture (V-JEPA) pretraining with a proposal-centric end-to-end driving planner. The paper's central claim is that V-JEPA-style latent predictive video pretraining gives a ViT encoder planning-relevant dynamics representations, while multimodal trajectory distillation (MTD) counters the single-human-trajectory supervision bottleneck in imitation learning.

**Source**: `raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md`  
**arXiv**: https://arxiv.org/html/2601.22032v1  
**Code**: https://github.com/linhanwang/Drive-JEPA  
**Authors**: Linhan Wang, Zichong Yang, Chen Bai, Guoxiang Zhang, Xiaotong Liu, Xiaoyin Zheng, Xiao-Xiao Long, Chang-Tien Lu, Cheng Lu

## Key Takeaways

- V-JEPA pretraining is directly useful for driving: a simple perception-free transformer decoder on top of the pretrained ViT-L reaches 89.0 PDMS on NAVSIM-v1, +3.0 over Epona's 86.2 in the paper's perception-free comparison.
- Drive-JEPA's full planner uses front-view history images only: `I_t` and `I_{t-1}` resized to 512 x 256, plus ego history/status features and command embeddings.
- The planner is proposal-centric rather than pure regression: 32 online continuous trajectory proposals are iteratively refined with Waypoint-anchored Deformable Attention (WADA), then scored for final selection.
- MTD builds an 8192-trajectory vocabulary, scores candidate trajectories with a NAVSIM-v2-style simulator, and uses high-scoring trajectories above threshold 0.95 as pseudo-teachers during proposal training.
- MTD improves diversity but hurts temporal comfort; the momentum-aware selector explicitly mixes the learned EPDMS-style score with a comfort score against the previously selected trajectory.
- Main reported results: 93.3 PDMS on NAVSIM-v1, 87.8 EPDMS on NAVSIM-v2, and 64.52 DS / 36.82 SR on Bench2Drive.

## Method

### Driving Video Pretraining

Drive-JEPA initializes from V-JEPA 2 and adapts it to driving videos. The authors curate 208 hours of driving video from CoVLA, DrivingDojo, and OpenScene, keep front-view clips, resize images to 512 x 256, sample 8-frame clips at 2 Hz, and train with a JEPA objective: masked context representations predict target latent representations without reconstructing pixels.

This makes Drive-JEPA closer to [[sources/latent-wam.md]] and FLARE-style feature prediction than to pixel-decoding world models such as [[sources/epona.md]]. It is a latent predictive pretraining signal that shapes the encoder, not an inference-time video generator.

### Perception-Free Baseline

The perception-free variant feeds current and previous front-view images into the pretrained ViT encoder, adds a simple transformer decoder with learnable waypoint queries, and trains with MSE against ground-truth waypoints. No BEV map, object detection, tracking, or LiDAR annotation is used. This setup is the basis for the Table 1/Table 7 representation comparisons.

### Proposal-Centric Planner

The full Drive-JEPA planner predicts `N_p=32` trajectory proposals. Initial BEV proposals are learned and then refined for `L` iterations. Each proposal contains waypoint queries, and each refinement block combines spatial cross-attention, self-attention, an MLP, and WADA feature sampling around the current predicted waypoints.

This differs from fixed-vocabulary selection methods such as [[sources/drivesuprim.md]]: Drive-JEPA uses an offline 8192 vocabulary to obtain pseudo-teacher targets, but deployed proposals remain continuous online predictions.

### Multimodal Trajectory Distillation

MTD addresses imitation-learning mode collapse. The authors cluster more than 100k training trajectories into 8192 centers, run a rule-based simulator over the vocabulary for each training scene, and select high-quality pseudo-teacher trajectories. Appendix B sets the simulated EPDMS threshold to 0.95; if more than `N_pseudo` pass the threshold, the method uniformly samples from the high-quality subset.

The trajectory loss supervises proposals against both the human trajectory and the pseudo-teacher set. In ablations, using pseudo-teachers improves over `N_pseudo=0`, but the best count is shallow: `N_pseudo=1` and `N_pseudo=4` both reach 87.8 EPDMS, while `N_pseudo=8` drops to 87.5.

### Momentum-Aware Trajectory Selection

The selector predicts proposal scores from pooled proposal-query features with BCE supervision derived from simulator EPDMS labels. Because MTD increases multimodality and can create frame-to-frame jitter, the final score is recalibrated with a comfort term comparing each proposal to the previous selected trajectory:

```text
S <- (7 S + S_c) / 8
```

This mechanism is important: in Table 5, adding MTD without momentum-aware selection increases diversity to 40% but collapses EC to 47.9; adding momentum-aware selection keeps diversity at 40%, restores EC to 84.8, and reaches 87.8 EPDMS.

## Figures

![[teaser_v2.png]]

**Figure 1.** Perception-free comparison plots pretraining data scale vs. PDMS, showing Drive-JEPA at 208h/89.0 PDMS above LAW, World4Drive, and Epona. The perception-based panel plots PDMS vs. EPDMS for ResNet34 and ViT/L variants of iPad, DriveSuprim, HydraMDP++, and Drive-JEPA.

![[x1 31.png]]

**Figure 2.** Architecture overview. The top row shows V-JEPA driving-video pretraining and MTD. The bottom row shows Drive-JEPA's planner: history-image encoding, ego/history embeddings, waypoint-anchored proposal generation, and momentum-aware selection.

![[demo_proposals.png]]

**Figure 3.** BEV proposal comparison. Without MTD, proposals collapse near the human mode; with MTD, proposals cover multiple plausible intersection maneuvers.

![[x2 29.png]]

**Figure 4.** Qualitative trajectory comparison across pedestrian avoidance, intersections, stop-sign, traffic-light, roundabout, and low-speed scenes. The figure overlays human, Drive-JEPA, iPad, and Transfuser trajectories in front-view images and BEV.

![[val_score_epoch_groups.png]]

**Figure 5.** Validation curve showing higher PDM score over training epochs when MTD is used.

![[more_mtd_demo.png]]

**Figure 6.** Additional BEV proposal visualizations. MTD produces broader multimodal proposal distributions across intersections and curved-road scenes.

## Tables

### Table 1: Perception-Free Planners

| Method | Encoder size | Data scale | PDMS |
| --- | ---: | ---: | ---: |
| LAW | 21M | about 20h | 83.8 |
| World4Drive | 21M | about 20h | 85.1 |
| Epona | 1.1B | 128h | 86.1 |
| Ours | 307M | 208h | 89.0 |

### Table 2: NAVSIM-v1

| Method | Backbone | Inputs | NC | DAC | EP | C | TTC | PDMS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LAW | ResNet34 | C & L | 97.4 | 93.3 | 78.8 | 100 | 91.9 | 83.8 |
| World4Drive | ResNet34 | C & L | 97.4 | 94.3 | 79.9 | 100 | 92.8 | 85.1 |
| Epona | ViT/G | Camera | 97.9 | 95.1 | 80.4 | 99.9 | 93.8 | 86.2 |
| Ours | ViT/L | Camera | 98.7 | 96.2 | 82.9 | 100 | 95.5 | 89.0 |
| Transfuser | ResNet34 | C & L | 97.7 | 92.8 | 79.2 | 100 | 92.8 | 84.0 |
| HydraMDP | ResNet34 | C & L | 98.3 | 96.0 | 78.7 | 100 | 94.6 | 86.5 |
| HydraMDP++ | ResNet34 | C & L | 97.6 | 96.0 | 80.4 | 100 | 93.1 | 86.6 |
| DiffusionDrive | ResNet34 | C & L | 98.2 | 96.2 | 82.2 | 100 | 94.7 | 88.1 |
| GoalFlow | ResNet34 | C & L | 98.4 | 98.3 | 85.0 | 100 | 94.6 | 90.3 |
| DriveDPO | ResNet34 | C & L | 98.5 | 98.1 | 84.3 | 100 | 94.8 | 90.0 |
| DriveSuprim | ResNet34 | Camera | 97.8 | 97.3 | 86.7 | 100 | 93.6 | 89.9 |
| iPad | ResNet34 | Camera | 98.4 | 97.9 | 87.4 | 99.9 | 94.9 | 91.1 |
| Drive-JEPA (Ours) | ResNet34 | Camera | 98.2 | 98.0 | 88.8 | 99.9 | 94.2 | 91.5 |
| Hydra-MDP | ViT/L | C & L | 98.4 | 97.7 | 85.0 | 100 | 94.5 | 89.9 |
| iPad | ViT/L | Camera | 99.2 | 97.4 | 87.8 | 99.7 | 96.3 | 91.7 |
| DriveSuprim | ViT/L | Camera | 98.6 | 98.6 | 91.3 | 100 | 95.5 | 93.5 |
| Drive-JEPA (Ours) | ViT/L | Camera | 99.1 | 98.2 | 90.8 | 99.9 | 95.9 | 93.3 |

### Table 3: NAVSIM-v2

| Method | Backbone | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transfuser | ResNet34 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ | ResNet34 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | ResNet34 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| iPad | ResNet34 | 98.7 | 97.8 | 99.1 | 99.8 | 84.0 | 98.0 | 96.0 | 98.0 | 68.2 | 84.1 |
| Drive-JEPA (Ours) | ResNet34 | 98.8 | 97.4 | 99.0 | 99.8 | 83.5 | 98.0 | 96.2 | 98.1 | 85.6 | 85.4 |
| HydraMDP++ | ViT/L | 98.5 | 98.5 | 99.5 | 99.7 | 87.4 | 97.9 | 95.8 | 98.2 | 75.7 | 85.6 |
| iPad | ViT/L | 98.7 | 98.0 | 98.9 | 99.8 | 86.6 | 98.3 | 97.2 | 98.3 | 74.6 | 85.8 |
| DriveSuprim | ViT/L | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | 87.1 |
| Drive-JEPA (Ours) | ViT/L | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | 87.8 |

### Table 4: Bench2Drive

| Method | Efficiency | Comfort | SR | DS |
| --- | ---: | ---: | ---: | ---: |
| AD-MLP | 48.45 | 22.63 | 0.00 | 18.05 |
| UniAD | 129.21 | 43.58 | 16.36 | 45.81 |
| VAD | 157.94 | 46.01 | 15.00 | 42.35 |
| TCP | 76.54 | 18.08 | 30.00 | 59.90 |
| DriveDPO | 166.80 | 26.79 | 30.62 | 62.02 |
| iPad | 153.83 | 35.51 | 33.18 | 60.52 |
| DriveTransformer | 100.64 | 20.78 | 35.01 | 63.46 |
| Ours | 157.85 | 30.24 | 36.82 | 64.52 |

### Table 5: Module Ablation on NAVSIM-v2

`M1`: V-JEPA 2 checkpoint. `M2`: driving-video pretraining. `M3`: MTD. `M4`: momentum-aware trajectory selection.

| M1 | M2 | M3 | M4 | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | Diversity | EPDMS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no | no | no | no | 98.7 | 97.8 | 99.1 | 99.8 | 84.0 | 98.0 | 96.0 | 98.0 | 68.2 | 25% | 84.1 |
| yes | no | no | no | 98.7 | 98.0 | 98.9 | 99.8 | 86.6 | 98.3 | 97.2 | 98.3 | 74.6 | 21% | 85.8 |
| no | yes | no | no | 98.3 | 98.1 | 99.1 | 99.9 | 89.1 | 97.7 | 97.7 | 98.1 | 69.7 | 24% | 86.1 |
| no | yes | yes | no | 98.5 | 98.6 | 99.1 | 99.8 | 89.1 | 97.9 | 97.6 | 97.8 | 47.9 | 40% | 84.5 |
| no | yes | yes | yes | 98.4 | 98.6 | 99.1 | 99.8 | 88.4 | 97.8 | 97.6 | 97.9 | 84.8 | 40% | 87.8 |

### Table 6: Number of Pseudo-Teacher Trajectories

| `N_pseudo` | 0 | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EPDMS | 87.2 | 87.8 | 87.7 | 87.8 | 87.5 |

### Table 7: Vision Pretraining Comparison

| Vision encoder | Size | PDMS |
| --- | --- | ---: |
| Epona | ViT/G | 86.2 |
| ImageNet | ResNet34 | 76.0 |
| DepthAnything | ViT/L | - |
| MAE | ViT/L | - |
| DINOv2 | ViT/L | 76.1 |
| SigLIP | ViT/L | 83.4 |
| V-JEPA 2 | ViT/L | 86.1 |
| Ours | ViT/L | 89.0 |

### Table 8: Input Image Resolution

| Method | Transfuser | HydraMDP++ | DriveSuprim | GoalFlow | iPad | Ours |
| --- | --- | --- | --- | --- | --- | --- |
| Input image resolution | 1024 x 256 | 1024 x 256 | 1024 x 256 | 1024 x 256 | 4 x 768 x 432 | 2 x 512 x 256 |

## Relationships

- **DriveSuprim**: both use an 8192 trajectory vocabulary and NAVSIM-style scoring, but DriveSuprim is a fixed-vocabulary selector at inference while Drive-JEPA uses the vocabulary only for pseudo-teacher distillation into continuous online proposals.
- **Epona**: Drive-JEPA outperforms Epona in the paper's perception-free V-JEPA table, but the world-model style differs: Epona uses autoregressive diffusion future-state modeling; Drive-JEPA uses JEPA latent prediction without pixel decoding.
- **Latent-WAM**: both are non-VLM world-model-style methods that avoid inference-time video generation. Latent-WAM predicts future latent world status with geometric distillation; Drive-JEPA pretrains a latent predictive video encoder and then uses simulator-distilled proposal supervision.
- **DiffusionDrive / HAD**: Drive-JEPA is not a diffusion planner. It uses iterative proposal refinement and selection, with simulator-derived pseudo-teachers, while diffusion methods generate trajectories through denoising/refinement objectives.
- **Bench2Drive VLA methods**: Drive-JEPA reports 64.52 DS, above older non-VLM and proposal-centric baselines like iPad but far below current wiki leaders such as LinkVLA and DynVLA.

## Limitations

- The pretraining recipe is not lightweight: 208 hours of curated video, V-JEPA 2 initialization, 8 H800 GPUs, and a 3-day pretraining stage.
- The paper's NAVSIM SOTA claim is comparison-scope sensitive. In this wiki, DriveSuprim reports 93.5 PDMS on NAVSIM-v1 vs. Drive-JEPA's 93.3, and NAVSIM-v2 has higher entries such as WAM-Diff 89.7, Vega BoN-6 89.4, and Latent-WAM 89.3.
- MTD relies on the NAVSIM-v2 rule simulator and an offline 8192 trajectory vocabulary; pseudo-teachers may encode simulator bias, especially because the simulator was originally designed for evaluation.
- MTD alone increases diversity but severely hurts EC in Table 5. The final result depends on the momentum-aware selector to repair frame-to-frame comfort.
- Bench2Drive performance is modest relative to current VLA methods: 64.52 DS is useful evidence for MTD over iPad, but it is not competitive with LinkVLA, DynVLA, or AutoMoT.
- The paper does not provide HUGSIM, nuScenes open-loop, Waymo, or navhard evaluations, so robustness beyond NAVSIM and Bench2Drive remains unclear.
