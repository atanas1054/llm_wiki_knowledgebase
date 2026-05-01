---
title: "HAD: Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving"
type: source-summary
sources: [raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md]
related: [concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/selection-based-planning.md, concepts/navhard-ood-evaluation.md, concepts/hugsim-benchmark.md, sources/diffusiondrive.md, sources/diffusiondrive-v2.md, sources/drivesuprim.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

# HAD

**Paper**: HAD: Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving  
**arXiv**: https://arxiv.org/html/2604.03581v1  
**Authors**: Wenhao Yao, Xinglong Sun, Zhenxin Li, Shiyi Lan, Zi Wang, Jose M. Alvarez, Zuxuan Wu  
**Type**: Non-VLM end-to-end planner using Transfuser-style camera/LiDAR features, hierarchical diffusion, and metric-decoupled RL.

## Core Idea

HAD addresses two problems in non-VLM end-to-end planning:

- **Large action-space selection is hard**: selecting directly from all candidate trajectories makes the policy rank too many plausible and implausible options at once.
- **Standard diffusion perturbations damage trajectories**: Gaussian noise can create zigzag or kinematically unrealistic samples.
- **Single scalar RL rewards are too coarse**: optimizing an aggregate PDMS/EPDMS reward hides which driving metric needs improvement and can create reward hacking.

The solution is a **Hierarchical Diffusion Policy**:

1. **Driving Intention Establishment**: denoise and score 20 coarse trajectory anchors, then select top-K intentions.
2. **Structure-Preserved Trajectory Expansion**: expand each selected anchor in polar coordinates using radial scaling and angular offsets.
3. **Local Trajectory Refinement**: refine the expanded local candidates and score them with distance, safety-metric, and metric-decoupled RL heads.

For RL, HAD introduces **Metric-Decoupled Policy Optimization (MDPO)**. Instead of a single reward head, it learns per-metric RL logits for NAVSIM metrics such as NC, DAC, DDC, TLC, EP, TTC, LK, and HC. Rewards are obtained through **Offline Trajectory Reward Retrieval**: precompute metric scores for an 8192-trajectory vocabulary and use nearest-neighbor retrieval during training instead of running online simulation for every rollout.

## Figures

![[x1 29.png|Figure 1: Prior planners search the full action space and use online scalar rewards; HAD narrows the search through hierarchical diffusion and retrieves metric-decoupled rewards from an offline cache.]]

![[x2 27.png|Figure 2: HAD overview. Sensor input is encoded by a Transfuser-style environment encoder; trajectory input goes through intention decoding, top-K filtering, trajectory expansion, refinement, MDPO heads, and final scoring.]]

![[x3 27.png|Figure 3: Online simulation reward computation vs. offline reward retrieval. HAD queries the nearest trajectory in an offline vocabulary reward cache.]]

![[x4 25.png|Figure 4: Trajectory expansion comparison. Random noise damages trajectory structure; XY expansion under-explores local regions; polar expansion preserves shape and covers the local maneuver region.]]

![[x5 23.png|Figure 5: Synthesized images from NavHard Stage 2; visible 3DGS artifacts create sensor-noise stress for Transfuser-style BEV features.]]

![[x6 20.png|Figure 6: NAVSIM qualitative comparison vs. Hydra-MDP++ and DriveSuprim. Blue HAD trajectories better match expert green trajectories in interactions and long-distance driving.]]

![[x7 16.png|Figure 7: HUGSIM qualitative examples showing HAD avoidance and bypass behavior in dynamic closed-loop scenes.]]

## Method Details

### Hierarchical Diffusion

The first-stage decoder starts from $M_1=20$ predefined trajectory anchors and applies truncated diffusion-style perturbation:

$$
\tilde{\tau}_j^{(i)}=\sqrt{\bar{\alpha}^{(i)}}\tau_j+\sqrt{1-\bar{\alpha}^{(i)}}\epsilon,\quad \epsilon\sim N(0,I)
$$

After denoising, the model scores each candidate for whether it lies in the same driving sub-region as the expert trajectory and keeps the top $K=2$ candidates.

The expansion stage maps each selected trajectory from Cartesian to polar coordinates and applies:

$$
\rho_t^{(u,v)}=\lambda_u\rho_t,\quad \theta_t^{(u,v)}=\theta_t+\delta_v
$$

with $\lambda=\{0.92,0.96,1.0,1.04,1.08\}$ and $\delta=\{-6,-3,0,3,6\}$ degrees. This yields $5 \times 5=25$ trajectories per selected intention and $M_2=50$ local candidates total.

### Metric-Decoupled Policy Optimization

For each local candidate set, HAD predicts per-metric RL logits and computes a metric-wise softmax:

$$
p_j^{(m)}=\frac{\exp(\hat{s}_{lcl,j}^{(m-rl)})}{\sum_{i\in I_k}\exp(\hat{s}_{lcl,i}^{(m-rl)})}
$$

Rewards are normalized within each local expansion set:

$$
\bar{r}_j^{(m)}=\frac{r_j^{(m)}-\mathrm{mean}(r_i^{(m)})}{\mathrm{std}(r_i^{(m)})+\epsilon}
$$

Then per-metric rewards are weighted:

$$
J_k=\sum_m \alpha^{(m)}\sum_{j\in I_k}p_j^{(m)}\bar{r}_j^{(m)}
$$

This keeps credit assignment separated by driving objective instead of training one coupled scalar reward head.

### Offline Reward Retrieval

HAD precomputes metric scores for an 8192-trajectory vocabulary. During training, a predicted trajectory retrieves its nearest vocabulary trajectory:

$$
\tau_{ref,j^*}=\arg\min_{\tau_{ref,j}\in V}\mathrm{Dist}(\hat{\tau},\tau_{ref,j})
$$

The cached scores for $\tau_{ref,j^*}$ approximate simulator rewards. Reported reward acquisition latency drops from 0.2449s to 0.0042s per trajectory, and total training time drops from 64.4h to 13.6h.

## Main Results

### Table 1: NAVSIM v1

| Method | Input | NC | DAC | EP | TTC | C | PDMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Human | - | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |
| Transfuser | C+L | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| UniAD | C | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 |
| VADv2 | C | 97.9 | 91.7 | 77.6 | 92.9 | 100 | 83.0 |
| LAW | C | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| DRAMA | C+L | 98.0 | 93.1 | 80.1 | 94.8 | 100 | 85.5 |
| GoalFlow | C+L | 98.3 | 93.8 | 79.8 | 94.3 | 100 | 85.7 |
| Hydra-MDP | C+L | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 |
| DiffusionDrive | C+L | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| WoTE | C+L | 98.5 | 96.8 | 81.9 | 94.9 | 99.9 | 88.3 |
| DriveSuprim | C | 97.8 | 97.3 | 86.7 | 93.6 | 100 | 89.9 |
| **HAD** | **C+L** | **98.2** | **97.3** | **87.4** | **97.5** | **100** | **90.2** |
| HAD-L | C | 98.1 | 97.2 | 87.2 | 97.3 | 100 | 89.9 |

HAD improves over DiffusionDrive by +2.1 PDMS and matches or slightly exceeds DriveSuprim in this paper's comparison set, but it is below the broader wiki's current non-VLM leader DriveSuprim ViT-L at 93.5 PDMS.

### Table 2: NAVSIM v2

| Method | Input | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Human | - | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 90.3 |
| Ego Status MLP | - | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| Transfuser | C+L | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ | C | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | C | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| DiffusionDriveV2 | C+L | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 |
| DiffRefiner | C | 98.5 | 97.4 | 99.6 | 99.8 | 87.6 | 97.7 | 97.7 | 98.3 | 86.2 | 86.2 |
| EvaDrive | C | 98.8 | 98.5 | 98.9 | 99.8 | 96.6 | 98.4 | 94.3 | 97.8 | 55.9 | 86.3 |
| **HAD** | **C+L** | **98.2** | **97.3** | **99.2** | **99.8** | **87.4** | **97.5** | **95.2** | **98.3** | **86.2** | **88.6** |
| HAD-L | C | 98.1 | 97.2 | 99.3 | 99.8 | 87.2 | 97.3 | 95.3 | 98.3 | 86.4 | 88.5 |

The headline NAVSIM contribution is v2: HAD reports 88.6 EPDMS and camera-only HAD-L reports 88.5 EPDMS. In the wiki, this is below WAM-Diff's reported 89.7 and Vega BoN-6's 89.4, but it is a strong non-VLM, real-time result.

### Table 3: HUGSIM

| Method | Easy RC | Easy HDS | Medium RC | Medium HDS | Hard RC | Hard HDS | Extreme RC | Extreme HDS | Overall RC | Overall HDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UniAD* | 58.6 | 48.7 | 41.2 | 29.5 | 40.4 | 27.3 | 26.0 | 14.3 | 40.6 | 28.9 |
| VAD* | 38.7 | 24.3 | 27.0 | 9.9 | 25.5 | 10.4 | 23.0 | 8.2 | 27.9 | 12.3 |
| Latent TransFuser* | 68.4 | 52.8 | 40.7 | 24.6 | 36.9 | 19.8 | 25.5 | 8.1 | 41.4 | 24.8 |
| Latent TransFuser | 60.4 | 42.5 | 39.4 | 17.7 | 32.7 | 11.8 | 27.9 | 10.6 | 37.9 | 18.0 |
| GTRS-Dense | 64.2 | 55.5 | 50.0 | 39.0 | 20.7 | 11.7 | 22.3 | 14.3 | 38.0 | 28.6 |
| ZTRS | 74.4 | 60.8 | 50.9 | 34.2 | 32.7 | 20.5 | 21.9 | 11.0 | 42.6 | 28.9 |
| **HAD-L** | **76.9** | **65.9** | **49.3** | **31.4** | **36.4** | **18.0** | **39.1** | **22.5** | **47.5** | **30.8** |

HAD-L improves overall route completion over ZTRS by +4.9 and HD-Score by +1.9. Its largest relative gain is on Extreme route completion.

## Ablations

### Table 4: Metric Heads in Hierarchical Denoising

| Global dist. | Global safety | Global RL | Local dist. | Local safety | Local RL | EPDMS |
| --- | --- | --- | --- | --- | --- | ---: |
| Yes | No | No | Yes | No | No | 86.7 |
| Yes | No | No | Yes | Yes | No | 87.3 |
| Yes | No | No | Yes | Yes | Yes | 88.6 |
| Yes | Yes | Yes | Yes | Yes | Yes | 87.2 |

Complex safety/RL scoring helps in the local refinement stage but hurts when applied to the whole global action space.

### Table 5: Top-K and Expansion Density

| Train Top-K | Train expansion | Inference Top-K | Inference expansion | EPDMS |
| ---: | --- | ---: | --- | ---: |
| 1 | 5x5 | 1 | 5x5 | 88.1 |
| 2 | 5x5 | 2 | 5x5 | 88.5 |
| 2 | 5x5 | 2 | 7x7 | 88.6 |
| 4 | 5x5 | 4 | 5x5 | 86.4 |
| 20 | 5x5 | 20 | 5x5 | 79.8 |

Small top-K is critical. Expanding all 20 coarse trajectories collapses EPDMS to 79.8, validating the need to narrow the local search region before refinement.

### Table 6: Expansion Algorithm

| Expansion | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Noise | 98.1 | 93.3 | 99.3 | 99.8 | 87.6 | 96.9 | 95.6 | 98.3 | 85.4 | 84.9 |
| XY Expand | 98.2 | 93.8 | 99.4 | 99.8 | 87.6 | 97.1 | 95.4 | 98.3 | 87.5 | 85.9 |
| **Polar Expand** | **98.2** | **97.3** | **99.2** | **99.8** | **87.4** | **97.5** | **95.2** | **98.3** | **86.2** | **88.6** |

Polar expansion is the default because it preserves trajectory structure while providing balanced angular/radial exploration.

### Table 7: MDPO

| Reward | Decoupled | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dist. | No | 92.0 | 91.2 | 96.6 | 99.5 | 88.7 | 91.6 | 85.1 | 41.3 | 3.9 | 62.6 |
| PDMS | No | 97.9 | 96.7 | 99.3 | 99.7 | 88.1 | 97.1 | 94.7 | 98.2 | 84.5 | 87.8 |
| Dist. + PDMS | No | 97.9 | 96.2 | 99.1 | 99.8 | 88.0 | 96.9 | 94.3 | 98.3 | 80.8 | 86.9 |
| **Decoupled Metrics** | **Yes** | **98.2** | **97.3** | **99.2** | **99.8** | **87.4** | **97.5** | **95.2** | **98.3** | **86.2** | **88.6** |

MDPO beats a coupled PDMS reward by +0.8 EPDMS and avoids the poor EC of distance-only optimization.

### Table 8: NAVSIM Inference Coefficients

| Type | NC | DAC | DDC | TLC | EP | TTC | LK | HC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coefficient | 0.5 | 0.5 | 0.3 | 0.1 | 5.0 | 5.0 | 2.0 | 1.0 |

NC/DAC/DDC/TLC are treated as multiplicative penalties; EP/TTC/LK/HC are weighted average metrics with $\lambda_{avg}=6.0$.

### Table 9: Loss Coefficients

| Equation | Coefficient | Value |
| --- | --- | --- |
| 19 | box, label, BEV | 1.0, 10.0, 14.0 |
| 22 | global reg, global cls | 8.0, 10.0 |
| 27 | dist, safe, RL | 10.0, 1.0, 1.0 |
| 28 | percept, global, local | 1.0, 12.0, 12.0 |

Training: 8 NVIDIA A100 GPUs, 85 epochs, AdamW, initial LR $6e-4$, 3-epoch linear warmup, cosine decay to $1e-6$.

### Table 10: Parameters and Speed

| Model | Input | EPDMS | Params | FPS |
| --- | --- | ---: | ---: | ---: |
| Transfuser | C+L | 76.7 | 56M | 56.4 |
| HydraMDP++ | C | 81.4 | 53M | 32.0 |
| DriveSuprim | C | 83.1 | 61M | 27.2 |
| DiffusionDrive | C+L | 87.5* | 61M | 42.1 |
| **HAD** | **C+L** | **88.6** | **63M** | **30.4** |
| HAD-L | C | 88.5 | 63M | 29.4 |

HAD remains real-time at roughly 30 FPS on a single A100.

### Table 11: NavHard

| Method | Backbone | Stage 1 metrics | Stage 2 metrics | EPDMS |
| --- | --- | --- | --- | ---: |
| PDM-Closed | - | NC 94.4, DAC 98.8, DDC 100, TLC 99.5, EP 100, TTC 93.5, LK 99.3, HC 87.7, EC 36.0 | NC 88.1, DAC 90.6, DDC 96.3, TLC 98.5, EP 100, TTC 83.1, LK 73.7, HC 91.5, EC 25.4 | 51.3 |
| LTF | ResNet34* | NC 96.2, DAC 79.5, DDC 99.1, TLC 99.5, EP 84.1, TTC 95.1, LK 94.2, HC 97.5, EC 79.1 | NC 77.7, DAC 70.2, DDC 84.2, TLC 98.0, EP 85.1, TTC 75.6, LK 45.4, HC 95.7, EC 75.9 | 23.1 |
| GTRS-Dense | V2-99 | NC 98.7, DAC 95.8, DDC 99.4, TLC 99.3, EP 72.8, TTC 98.7, LK 95.1, HC 96.9, EC 40.4 | NC 91.4, DAC 89.2, DDC 94.4, TLC 98.8, EP 69.5, TTC 90.1, LK 54.6, HC 94.1, EC 49.7 | 41.7 |
| DriveSuprim | ResNet34 | NC 97.2, DAC 96.2, DDC 99.3, TLC 99.6, EP 71.4, TTC 96.7, LK 94.7, HC 97.3, EC 47.1 | NC 88.6, DAC 86.0, DDC 89.4, TLC 98.4, EP 74.7, TTC 86.5, LK 55.3, HC 97.7, EC 59.4 | 39.5 |
| DriveSuprim | V2-99 | NC 98.9, DAC 95.1, DDC 99.2, TLC 99.6, EP 76.1, TTC 99.1, LK 94.7, HC 97.6, EC 54.2 | NC 87.9, DAC 88.8, DDC 89.6, TLC 98.8, EP 80.3, TTC 86.0, LK 53.5, HC 97.1, EC 56.1 | 42.1 |
| DiffusionDrive | ResNet34* | NC 96.8, DAC 86.0, DDC 98.8, TLC 99.3, EP 84.0, TTC 95.8, LK 96.7, HC 97.6, EC 66.7 | NC 80.1, DAC 72.8, DDC 84.4, TLC 98.4, EP 85.9, TTC 76.6, LK 46.4, HC 96.3, EC 40.5 | 27.5 |
| **HAD-L** | **ResNet34*** | **NC 96.2, DAC 90.4, DDC 98.6, TLC 99.3, EP 83.6, TTC 95.8, LK 96.0, HC 97.8, EC 78.2** | **NC 80.2, DAC 75.2, DDC 83.7, TLC 98.4, EP 86.6, TTC 77.0, LK 47.2, HC 95.9, EC 70.5** | **32.3** |

HAD-L improves over DiffusionDrive and LTF under similar Transfuser-style backbones but remains below DriveSuprim and GTRS-Dense on NavHard.

## Relationships

- **vs. DiffusionDrive**: both use truncated/anchored diffusion over trajectories; HAD adds hierarchical coarse-to-fine filtering and local expansion.
- **vs. DiffusionDriveV2**: both use RL for diffusion planners, but HAD decouples rewards by metric and avoids online simulator reward calls through cache retrieval.
- **vs. DriveSuprim**: both exploit selection/coarse-to-fine ideas and 8192-trajectory vocabularies; HAD generates/refines local candidates, while DriveSuprim scores a fixed vocabulary with stronger ViT-L results.
- **vs. DreamerAD**: both use cached/learned alternatives to online simulator calls during training; DreamerAD learns a latent reward model from world-model features, while HAD retrieves precomputed metric scores from a trajectory vocabulary.

## Limitations

- **Not current absolute NAVSIM v1 SOTA**: HAD reports 90.2 PDMS, below wiki entries such as DriveSuprim 93.5, HybridDriveVLA 92.1, FLARE 91.4, and DiffusionDriveV2 91.2.
- **NAVSIM v2 comparison is strong but not final**: HAD's 88.6 EPDMS is below WAM-Diff's reported 89.7 and Vega BoN-6's 89.4 in the wiki.
- **Reward retrieval is approximate**: nearest-neighbor vocabulary rewards may miss fine-grained simulator effects for trajectories between anchors.
- **Transfuser-style BEV backbone is noise-sensitive**: NavHard analysis attributes the remaining gap to noisy 3DGS images degrading BEV features.
- **No VLM reasoning or language interface**: HAD is a strong non-VLM planner, not an interpretable VLA.
- **HUGSIM comparison scope is separate**: HUGSIM is closed-loop and interactive; numbers are not comparable to NAVSIM PDMS/EPDMS.
