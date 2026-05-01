---
title: "From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction"
type: source-summary
sources: [raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/nuscenes-waymo-evals.md, concepts/foundation-backbones-for-ad.md, sources/drivevla-w0.md, sources/driveva.md, sources/futuresightdrive.md, sources/drivedreamer-policy.md, sources/latent-wam.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

# From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction

## Citation

**Authors**: Zhida Zhao, Talas Fu, Yifan Wang, Lijun Wang, Huchuan Lu  
**Affiliation**: Dalian University of Technology  
**arXiv**: 2510.19654v1  
**Code**: https://github.com/6550Zhao/Policy-World-Model

## Key Takeaways

- **Policy World Model (PWM)** unifies future-state video forecasting and trajectory planning in one autoregressive transformer, then uses forecasted future frames as planning rationales.
- The key design is **action-free future forecasting**: PWM learns world dynamics from unlabeled driving video, rolls out plausible future frames before action prediction, and then predicts the action from observed frames, ego/navigation state, text, and predicted future frame tokens.
- PWM trades photorealistic video detail for planning efficiency using a **context-guided tokenizer** that compresses each frame to **28 low-resolution tokens** while retaining first-frame high-resolution context.
- A **Dynamic Focal Loss (DFL)** upweights tokens that change across adjacent frames, improving both visual forecasting and downstream safety metrics.
- The empirical signature is safety-oriented: on nuScenes, PWM reports the best collision rate in its table, especially with ego status (**0.04% average collision**) while its average L2 is not the best. On NAVSIM-v1, it reaches **88.1 PDMS** with only a front camera, matching DiffusionDrive's Camera+LiDAR score in the paper's comparison.

## Method

PWM starts from Show-o and trains three capabilities in one token space:

1. **Tokenizer training**: a frozen high-resolution branch encodes the first frame, while a trainable low-resolution branch encodes each 128x224 frame into a 4x7 token grid. Cross-attention transfers context from the high-resolution branch so each future frame can be represented by only 28 tokens.
2. **Action-free video pretraining**: OpenDV-YouTube video clips are converted to token sequences and trained with next-frame prediction. Tokens within a frame use bidirectional attention, while frames are autoregressive over time.
3. **Planning fine-tuning**: the model consumes observed image tokens plus ego/navigation state, generates a textual scene/action description and future frame tokens, then predicts the current/future trajectory action tokens.

This makes PWM different from action-conditioned world models: future video is generated **before** the action is known, so it can serve as anticipatory context rather than only as a consequence checker.

## Figures

**Figure 1: World-model paradigms.** The raw markdown preserves only the caption, not a local extracted image. It contrasts conventional decoupled video world models, unified-but-independent generation/planning models, and PWM's policy world model where planning conditions on learned future-state forecasts.

**Figure 2: PWM architecture and context-guided tokenizer.**

![[x2 31.png]]

The model uses current/history image tokens plus ego/navigation state, generates description and future state tokens, and predicts action tokens. The tokenizer uses a frozen high-resolution first-frame branch to guide a low-resolution 28-token-per-frame branch.

**Figure 3: Action-free video world modeling and parallel prediction.**

![[x3 29.png]]

PWM pretrains from action-free video sequences and switches from token-by-token generation to next-frame token generation, with bidirectional attention within each frame.

**Figure 4: NAVSIM qualitative future forecasting and BEV planning visualization.**

![[frame_2.jpg]]

The paper uses sampled future frames plus BEV trajectory comparisons to show that predicted future visual context can change the chosen trajectory.

**Figure 5: Planning comparison with and without future-frame prediction.**

![[frame_10.jpg]]

Qualitative examples compare the planning result without future frames against PWM with explicit future-frame prediction.

**Appendix Figure 1: Input sequence details.**

![[x13 5.png]]

This figure specifies how historical frame tokens, ego/navigation text, generated text, future frame tokens, and action tokens are ordered for the unified autoregressive transformer.

**Appendix Figure 2: DFL qualitative comparison.**

![[frame_2 1.jpg]]

Rows compare ground-truth future frames, predictions without DFL, and predictions with DFL at sampled future timesteps.

**Appendix Figure 3: UMAP projection of temporal latent representations.**

![[x14 1.png]]

The temporal latent trajectory is smooth in 2D UMAP space, which the authors interpret as evidence that PWM learns coherent temporal dynamics across predicted driving frames.

**Appendix qualitative frame with caption missing in raw markdown.**

![[frame_10 1.jpg]]

The raw markdown only says "Refer to caption"; this local asset is preserved for completeness.

**Appendix Figures 4-5: Additional planning comparison visualizations.**

![[frame_10 2.jpg]]

The appendix says Figures 4 and 5 compare planning with and without future-frame prediction: left-side predicted future frames, right-side BEV planning outcomes.

**Appendix Figure 6: More DFL visual comparisons.**

![[frame_2 2.jpg]]

Additional rows compare ground truth, prediction without DFL, and prediction with DFL.

## Main Results

### Table 1: nuScenes validation planning

Lower is better for L2 and collision. The dagger rows use ego status.

| Method | Ego | L2 1s | L2 2s | L2 3s | L2 Avg | Col 1s | Col 2s | Col 3s | Col Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ST-P3 | No | 1.59 | 2.64 | 3.73 | 2.65 | 0.69 | 3.62 | 8.39 | 4.23 |
| UniAD | No | 0.59 | 1.01 | 1.48 | 1.03 | 0.16 | 0.51 | 1.64 | 0.77 |
| VAD-Base | No | 0.69 | 1.22 | 1.83 | 1.25 | 0.06 | 0.68 | 2.52 | 1.09 |
| BEV-Planner | No | 0.30 | 0.52 | 0.83 | 0.55 | 0.10 | 0.37 | 1.30 | 0.59 |
| Omni-Q | No | 1.15 | 1.96 | 2.84 | 1.98 | 0.80 | 3.12 | 7.46 | 3.79 |
| LAW | No | 0.24 | 0.46 | 0.76 | 0.49 | 0.08 | 0.10 | 0.39 | 0.19 |
| Drive-OccWorld | No | 0.25 | 0.44 | 0.72 | 0.47 | 0.03 | 0.08 | 0.22 | 0.11 |
| **PWM** | No | 0.41 | 0.75 | 1.17 | 0.78 | **0.01** | **0.01** | **0.18** | **0.07** |
| UniAD | Yes | 0.20 | 0.42 | 0.75 | 0.46 | 0.02 | 0.25 | 0.84 | 0.37 |
| VAD-Base | Yes | 0.17 | 0.34 | 0.60 | 0.37 | 0.04 | 0.27 | 0.67 | 0.33 |
| BEV-Planner | Yes | 0.16 | 0.32 | 0.57 | 0.35 | 0.00 | 0.29 | 0.73 | 0.34 |
| OccWorld-D | Yes | 0.39 | 0.73 | 1.18 | 0.77 | 0.11 | 0.19 | 0.67 | 0.32 |
| Omni-Q | Yes | 0.14 | 0.29 | 0.55 | 0.33 | 0.00 | 0.13 | 0.78 | 0.30 |
| RDA-Driver | Yes | 0.17 | 0.37 | 0.69 | 0.41 | 0.01 | 0.05 | 0.26 | 0.11 |
| DiffusionDrive | Yes | 0.27 | 0.54 | 0.90 | 0.57 | 0.03 | 0.05 | 0.16 | 0.08 |
| **PWM** | Yes | 0.20 | 0.38 | 0.65 | 0.41 | 0.01 | **0.02** | **0.09** | **0.04** |

PWM is not the best L2 method with ego status, but it has the lowest average collision rate in the paper's table.

### Table 2: NAVSIM NavTest comparison

| Method | Input | NC | DAC | EP | TTC | Comfort | PDMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Human | - | 100.0 | 100.0 | 87.5 | 100.0 | 99.9 | 94.8 |
| Constant Velocity | - | 69.9 | 58.8 | 49.3 | 49.3 | 100.0 | 21.6 |
| Ego Status MLP | - | 93.0 | 77.3 | 62.8 | 83.6 | 100.0 | 65.6 |
| VADv2 | C&L | 97.2 | 89.1 | 76.0 | 91.6 | 100.0 | 80.9 |
| TransFuser | C&L | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 |
| DRAMA | C&L | 98.0 | 93.1 | 80.1 | 94.8 | 100.0 | 85.5 |
| Hydra-MDP | C&L | 98.3 | 96.0 | 78.7 | 94.6 | 100.0 | 86.5 |
| DiffusionDrive | C&L | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| UniAD | C | 97.8 | 91.9 | 78.8 | 92.9 | 100.0 | 83.4 |
| LTF | C | 97.4 | 92.8 | 79.0 | 92.4 | 100.0 | 83.8 |
| PARA-Drive | C | 97.9 | 92.4 | 79.3 | 93.0 | 99.8 | 84.0 |
| LAW | C | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| DrivingGPT | SC | **98.9** | 90.7 | 79.7 | 94.9 | 95.6 | 82.4 |
| **PWM** | SC | 98.6 | 95.9 | 81.8 | **95.4** | **100.0** | **88.1** |

PWM matches DiffusionDrive's PDMS while using a single front camera rather than Camera+LiDAR in this comparison. In the broader wiki, 88.1 is no longer a frontier NAVSIM-v1 score.

### Table 3: Impact of world modeling and DFL

**nuScenes validation split**

| Pre-train | DFL-p | DFL-f | LPIPS | PSNR | FVD | Avg L2 | Avg Col |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| No | No | No | 0.27 | 21.07 | 826.15 | 3.34 | 1.51 |
| Yes | No | No | 0.24 | 22.16 | 239.13 | 2.29 | 1.05 |
| Yes | No | Yes | 0.24 | 22.24 | 96.53 | 1.23 | 0.56 |
| Yes | Yes | No | 0.22 | 22.88 | 96.99 | 1.04 | 0.26 |
| **Yes** | **Yes** | **Yes** | **0.22** | **23.07** | **67.13** | **0.78** | **0.07** |

**NAVSIM**

| Pre-train | DFL-p | DFL-f | LPIPS | PSNR | FVD | NC | DAC | EP | TTC | Comfort | PDMS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No | No | No | 0.27 | 19.90 | 431.47 | 97.3 | 89.7 | 69.1 | 92.2 | 100.0 | 77.8 |
| Yes | No | No | 0.25 | 20.71 | 199.79 | 97.7 | 90.8 | 72.6 | 93.7 | 99.9 | 80.7 |
| Yes | No | Yes | 0.24 | 21.06 | 114.85 | 98.3 | 94.5 | 73.4 | 95.1 | 100.0 | 83.5 |
| Yes | Yes | No | 0.23 | 21.22 | 110.95 | 98.3 | 94.5 | 80.4 | 94.4 | 100.0 | 86.3 |
| **Yes** | **Yes** | **Yes** | **0.23** | **21.57** | **85.95** | **98.6** | **95.9** | **81.8** | **95.4** | **100.0** | **88.1** |

Pretraining and DFL are both material. The biggest jump comes from action-free video pretraining, but DFL sharply improves FVD and planning safety.

### Table 4: Forecast horizon ablation

| Forecast frames | nuScenes Avg L2 | nuScenes Avg Col | nuScenes Latency | NAVSIM NC | NAVSIM DAC | NAVSIM EP | NAVSIM TTC | NAVSIM Comfort | NAVSIM PDMS | NAVSIM Latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.80 | 0.13 | 0.88s | 98.0 | 95.1 | **82.4** | 94.1 | 99.9 | 87.3 | 0.57s |
| 5 | 0.82 | 0.10 | 1.01s (+0.13) | 98.4 | 95.4 | 81.5 | 94.8 | 100.0 | 87.7 | 0.69s (+0.12) |
| **10** | **0.78** | **0.07** | 1.13s (+0.25) | 98.6 | **95.9** | 81.8 | **95.4** | **100.0** | **88.1** | 0.85s (+0.28) |
| 15 | 0.80 | 0.09 | 1.26s (+0.38) | **98.7** | 95.8 | 81.4 | **95.4** | **100.0** | 88.0 | 0.97s (+0.40) |

Ten forecasted frames are the best trade-off. Future frames improve collision/TTC/DAC but slightly reduce ego progress versus no forecasting on NAVSIM.

### Table 5: Pretraining data scale on nuScenes

| OpenDV pretraining data | 10 forecast frames Avg L2 | 10 forecast frames Avg Col | 0 forecast frames Avg L2 | 0 forecast frames Avg Col |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 2.95 | 1.26 | 1.62 | 0.90 |
| 50% | 0.85 | 0.14 | 0.88 | 0.21 |
| **100%** | **0.78** | **0.07** | **0.80** | **0.13** |

The no-pretraining row is especially important: explicit future forecasting is harmful without sufficient world-model pretraining, because low-quality forecasts corrupt planning.

### Table 6: DFL during pretraining

| DFL-p | LPIPS | PSNR | FVD |
| --- | ---: | ---: | ---: |
| No | 0.23 | 20.29 | 211.26 |
| **Yes** | **0.22** | **20.32** | **118.07** |

DFL improves FVD during action-free pretraining even before downstream planning fine-tuning.

## Appendix Tables

### Appendix Table 1: Dynamic Focal Loss beta ablation

Alpha is fixed at 1.0.

| Beta | LPIPS | PSNR | FVD | Avg L2 | Avg Col |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.23 | 22.69 | **65.07** | 0.82 | 0.12 |
| **0.4** | **0.22** | 23.07 | 67.13 | **0.78** | **0.07** |
| 0.7 | **0.22** | **23.11** | 71.84 | 0.84 | 0.09 |
| 1.0 | **0.22** | 22.88 | 96.99 | 1.04 | 0.26 |
| 2.0 | **0.22** | 22.65 | 93.83 | 1.27 | 0.27 |

The planning optimum is beta=0.4, not the lowest FVD beta=0.1. Treat DFL as planning-coupled weighting, not only a visual metric optimizer.

### Appendix Table 2: Textual generation task on nuScenes

| Text task type | LPIPS | PSNR | FVD | Avg L2 | Avg Col |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | **0.22** | **23.12** | **65.45** | **0.77** | 0.09 |
| Scene | **0.22** | 22.97 | 67.52 | **0.77** | 0.08 |
| Action | **0.22** | 23.07 | 67.13 | 0.78 | **0.07** |

Text generation has limited impact in this setup. Action descriptions slightly reduce collision, but visual forecasting and DFL carry the main gains.

## Training and Data

- **OpenDV-YouTube**: 1,747 hours of front-camera video at 10 Hz, spanning 244 cities. BLIP-2 provides text descriptions.
- **Tokenizer**: high-resolution branch processes 256x448 into 448 tokens and remains frozen; low-resolution branch processes 128x224 into 28 tokens; both codebooks use size 8192.
- **Tokenizer training**: 800K steps on 6 A800 GPUs, AdamW, learning rate 2e-4.
- **PWM pretraining**: 300K steps, learning rate 1e-4, with CC12M and FineWeb auxiliary data to retain captioning/text ability.
- **nuScenes fine-tuning**: 1 second history, 11 predicted frames, 6 waypoints over 3 seconds, 16 epochs, 2 A800 GPUs, batch size 8, learning rate 3e-5.
- **NAVSIM fine-tuning**: 2 seconds history, 10 predicted frames, 8 waypoints over 4 seconds, 20 epochs, batch size 14.
- **Input modality**: front camera only; no data augmentation reported.

## Relationships

- **Versus DriveVLA-W0**: both use unlabeled video/world-model supervision, but DriveVLA-W0 uses world modeling as a training-time representation signal, while PWM forecasts future frames at inference before predicting actions.
- **Versus FutureSightDrive**: both use future visual states as planning intermediates. FSDrive generates a structured visual CoT frame with lane/divider overlays; PWM rolls out compact video tokens directly from action-free world modeling.
- **Versus DriveDreamer-Policy**: DDP uses causal depth -> video -> action flow-matching generators with explicit geometry; PWM uses a single autoregressive transformer and no depth modality.
- **Versus DriveVA**: DriveVA couples video latents and action tokens in a single diffusion target from a large video backbone and emphasizes zero-shot transfer. PWM is lighter in token count and explicitly forecasts future video as a planning rationale, but its NAVSIM score is lower in the broader wiki.
- **Versus Latent-WAM and Drive-JEPA**: those methods avoid decoded future frames at inference; PWM pays inference-time forecasting cost for explicit visual anticipation.
- **Versus OneVL**: OneVL trains future-frame latent decoders and discards them at inference. PWM keeps the forecasting path active during inference.

## Limitations

- **Front-camera only**: robustness may degrade under poor visibility, occlusion, or side/rear hazards that require multi-view context.
- **Short horizon**: the paper focuses on short forecasting/planning windows, limiting long-horizon route-level applicability.
- **NAVSIM-v1 only**: no NAVSIM-v2/EPDMS, navhard, Bench2Drive, HUGSIM, or Waymo closed-loop result is reported.
- **Not a current NAVSIM frontier result**: 88.1 PDMS was strong against the paper's comparison set, but many later wiki methods exceed 90 PDMS.
- **Forecast quality dependency**: Table 5 shows that explicit forecasting can hurt when the world model is undertrained.
- **Latency trade-off**: the best 10-frame horizon adds 0.28s on NAVSIM versus no forecasting; 15 frames add 0.40s with no meaningful PDMS gain.
- **Conservative planning bias**: future frames improve collision/TTC/DAC but slightly reduce NAVSIM ego progress versus no forecasting.
- **Large pretraining requirement**: the method relies on 1,747 hours of OpenDV video and substantial A800 training.

## Bottom Line

PWM is most important as evidence that **inference-time, action-free future video forecasting can directly improve planning safety**, not as a current leaderboard method. Its strongest contribution is the mechanism: use an efficient 28-token frame representation, pretrain on unlabeled driving video, generate future states before action prediction, and let the policy condition on those predicted states as multimodal rationales.
