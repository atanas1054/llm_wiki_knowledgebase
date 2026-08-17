---
title: "DriveLaW: Unifying Planning and Video Generation in a Latent Driving World"
type: source-summary
sources: [raw/papers/DriveLaW_ Unifying Planning and Video Generation in a Latent Driving World.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/diffusion-planner.md, concepts/nuscenes-waymo-evals.md, sources/simwam.md, sources/drivewam.md, sources/driveva.md, sources/epona.md, sources/policy-world-model.md, sources/drivevla-w0.md, sources/recogdrive.md, sources/sgdrive.md, sources/uniugp.md, sources/futuresightdrive.md, sources/dreameraD.md]
created: 2026-08-17
updated: 2026-08-17
confidence: high
---

**Paper**: DriveLaW: Unifying Planning and Video Generation in a Latent Driving World
**Authors**: Tianze Xia, Yongkang Li, Lijun Zhou, Jingfeng Yao, Kaixin Xiong, Haiyang Sun, Bing Wang, Kun Ma, Guang Chen, Hangjun Ye, Wenyu Liu, Xinggang Wang
**Orgs**: Huazhong University of Science and Technology + Xiaomi EV
**arXiv**: 2512.23421v3 (CVPR'26)
**Code**: https://github.com/xiaomi-research/drivelaw

---

## Summary

DriveLaW's argument is that existing "unified" world models are not actually unified: Epona, DriveVLA-W0, and VaVAM run video generation and planning as **parallel** output streams, so the trajectory is never grounded in the features that actually govern video synthesis. DriveLaW instead **chains** them — the Video DiT's internal mid-denoising latents are injected directly into the planner as its perception state.

Two components. **DriveLaW-Video** is a 2B LTX-Video-initialized DiT with an aggressively compressed spatiotemporal VAE (32×32×8, 1:192 latent compression, 1:8192 pixel-to-token), a hybrid decoding scheme that performs the last rectified-flow step in pixel space, and a **noise reinjection** mechanism that selectively re-perturbs high-frequency regions before each denoising step to force detail regeneration rather than smoothing. **DriveLaW-Act** is a 133M vanilla DiT trained with flow matching, cross-attending to cached Video DiT features. A three-stage curriculum learns long-horizon motion first, then spatial detail, then chains the latents into the planner.

Results: **FID 4.6 / FVD 81.3** on nuScenes video generation (best single-view in its table) and **89.1 PDMS on NAVSIM** with no RL and no learned scorer. But the two ablations are what make the paper valuable to this wiki — a controlled comparison of *representation types* under a fixed planner, and a denoising-step sweep whose result quietly undercuts the imagine-then-act framing the paper otherwise adopts.

---

## Core Idea: Chained, Not Parallel

![[CVPR-Fig2.png|DriveLaW architecture: chained video generation and planning through a shared latent space]]

**Figure 1**: Historical observations (images, actions) are encoded into a unified latent world representation by a video diffusion model. Noise reinjection explores and selects the optimal generation path early in denoising. The denoised video latents are passed as conditioning to the action planner, and the lightweight Action DiT predicts trajectories aligned with the visual scene evolution. Video and action models share the same latent space.

The paper's taxonomy of prior world-model roles is worth recording because it is a clean framing of the field:

1. **World-model simulators** — synthesize downstream data or serve as closed-loop environments (HUGSIM, RAD, ReSim, Vista). Indirect; the model's physical understanding never enters the planner's state.
2. **World-model supervision** — predict future visual/affordance signals as auxiliary loss (DriveVLA-W0, LAW, OccSora). Improves foresight but planning stays externally specified.
3. **Unified world-model** — co-generate video and trajectories (Epona, VaViM/VaVAM, FSDrive). Tighter, but still "two independent output streams," leaving a *representation disconnect*.

DriveLaW positions itself as a fourth option: reuse the generator's **mid-denoising features** as the planning state. Formally, with denoiser $\Psi_\theta$ producing a latent trajectory $z_{t-1}=\Psi_{\theta}(z_{t},t,c)$, it extracts $h_{t}=\phi_{\theta}(z_{t})$ and selects timesteps $t^\star$ to form the perception latent $h=h_{t^\star}$.

---

## Method

### DriveLaW-Video

**Spatiotemporal VAE.** 32×32×8 spatial-temporal downsampling, 128 channels, **1:192 compression** (1:8192 pixel-to-token) — roughly twice the compression of typical text-to-video pipelines (1:48 or 1:96). The stated purpose is longer prediction horizons under a fixed compute budget, needed for traffic-light changes and vehicle dynamics. A causal 3D-convolution encoder prevents temporal leakage.

**Hybrid decoding.** Rather than finishing all reverse-diffusion steps in latent space before one decode, the final rectified-flow step is executed by a time-conditioned decoder in pixel space:

$$x_{0}=D(z_{t_{1}},t_{1}),\qquad z_{t_{1}}=(1-t_{1})z_{0}+t_{1}\epsilon,\quad\epsilon\sim\mathcal{N}(0,\mathbf{I})$$

This recovers high-frequency detail (highlights, shadows, road texture) without a separate super-resolution stage.

**Backbone.** PixArt-α-adapted 3D Transformer: 28 self/cross-attention blocks, hidden 2048, FFN ×4, RMSNorm on queries and keys, normalized fractional RoPE with exponential frequency spacing. Tokens are serialized from VAE latents at 1×1×1 granularity — no patchification.

**Motion-conditioned prompting.** Instead of a dedicated motion encoder, ego kinematics are discretized into semantic bins ("low speed", "turning left") and slotted into a fixed natural-language template with numeric grounding, encoded by a frozen T5-XXL and cross-attended into every DiT layer. The rationale is that this reuses the pretrained text-to-video interface directly and avoids numeric encodings tied to dataset-specific scales.

### Noise Reinjection

![[simplerenoise_compressed.png|Noise reinjection restores structural and temporal consistency]]

**Figure 2**: Baseline generation degrades with (a) blurring, (b) structural inconsistency, and (c) artifacts. Noise reinjection preserves sharp details, maintains object structure, and produces artifact-free frames.

High-speed driving generation over-smooths boundaries and accumulates ghosting. Unlike methods that renoise globally, DriveLaW perturbs **only high-frequency regions**, computed in the pixel domain for fidelity. At step $t$: predict the clean latent $\hat{L}_{0}=\Psi_{\theta}(L_{t},t)$, decode it to a temporary image $\hat{I}_{0}=D(\hat{L}_{0})$, convert to grayscale $G_f$, apply a discrete Laplacian $K_L$ to get $H_{f}=|G_{f}*K_{L}|$, and threshold at $\tau=\beta\cdot\mathrm{std}(H_{f})$:

$$M_{f}(x,y)=\begin{cases}1,&H_{f}(x,y)>\tau\\0,&\text{otherwise}\end{cases}$$

The mask is nearest-neighbor downsampled to latent resolution, then noise is injected only there:

$$L^{\prime}_{t}=L_{t}+\sigma^{\prime}_{t}\cdot M\odot\varepsilon_{t},\quad\varepsilon_{t}\sim\mathcal{N}(0,\mathbf{I})$$

forcing the generative prior to "inpaint" plausible high-frequency detail while leaving smooth regions like sky untouched.

### DriveLaW-Act

A 133M vanilla DiT. Noised action $a_t=(1-t)a_0+t\epsilon$, ego status $s_t$, and command $g_t$ are encoded, and **latents from each Video DiT block are cached during the first denoising step** as $\{f_1,\dots,f_B\}$:

$$f_{\theta}(a_{t},t)=\mathrm{DiT}_{\mathrm{act}}\big([\,h_{\mathrm{act}};\,t\,]\,\big|\,h_{\mathrm{ctx}},\{f_{i}\}_{i=1}^{B}\big)$$

trained with flow matching:

$$\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{t,a_{0},\epsilon}\big[\lVert f_{\theta}(a_{t},t)-(a_{0}-\epsilon)\rVert_{2}^{2}\big]$$

Output is continuous $(x,y,\theta)$ at 2 Hz over 4 s. At inference **the planner runs purely in latent space with no video decoding**, and gradient isolation between the two modules is preserved during training.

### Three-stage progressive training

1. **Long-horizon motion**: 740×352×121 frames — low resolution, long clips, prioritizing temporal span (lane keeping, turning, speed variation).
2. **Spatial fidelity**: 1280×704×25 — high resolution, short clips, refining lane markings, vehicles, textures while preserving stage-1 coherence.
3. **Chaining**: condition DriveLaW-Act on DriveLaW-Video latents and train for planning.

---

## Results

### nuScenes video generation (Table 1)

| Metric | DriveGAN | DriveDreamer | DrivingGPT | DriveWorld | Vista | Epona | **DriveLaW** |
|---|---:|---:|---:|---:|---:|---:|---:|
| FID ↓ | 73.4 | 52.6 | 12.8 | 7.4 | 6.9 | 7.5 | **4.6** |
| FVD ↓ | 502.3 | 452.0 | 142.6 | 90.9 | 89.4 | 82.8 | **81.3** |

The abstract's "33.3% FID / 1.8% FVD" improvements are measured against different baselines — FID against Vista (6.9 → 4.6) and FVD against [[sources/epona.md]] (82.8 → 81.3). The FID gain is substantial; the FVD gain is marginal.

### NAVSIM navtest (Table 2)

† = trained with the same flow-matching objective.

| Method | Ref | Image | LiDAR | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---|:-:|:-:|---:|---:|---:|---:|---:|---:|
| Constant Velocity | – | | | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP | arXiv'23 | | | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| *Traditional End-to-End* | | | | | | | | | |
| VADv2-𝒱8192 | arXiv'24 | ✓ | | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD | CVPR'23 | ✓ | | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | TPAMI'23 | ✓ | ✓ | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | CVPR'24 | ✓ | | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| ReCogDrive-IL | arXiv'25 | ✓ | | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| DiffusionDrive | CVPR'25 | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | **82.2** | 88.1 |
| *World Model Methods* | | | | | | | | | |
| DrivingGPT | arXiv'24 | ✓ | | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| LAW | ICLR'25 | ✓ | | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| Epona | ICCV'25 | ✓ | | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| ReSim | NeurIPS'25 | ✓ | | – | – | – | – | – | 86.6 |
| DriveVLA-W0† | arXiv'25 | ✓ | | 98.4 | 95.3 | 95.2 | 100 | 80.9 | 87.2 |
| WoTE | ICCV'25 | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| PWM | NeurIPS'25 | ✓ | | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| **DriveLaW** | – | ✓ | | **99.0** | **97.1** | **96.7** | 100 | 81.3 | **89.1** |

**NC 99.0 is the highest No-at-fault-Collision score recorded anywhere in this wiki**, and TTC 96.7 is likewise exceptional. EP 81.3 is the trade-off — below DiffusionDrive's 82.2 and well below ReCogDrive's RFT 87.3. This is a conspicuously *safety-skewed* policy. Achieved with **no RL and no learned scorer**.

### nuScenes open-loop planning (Table 3)

| Method | L2 1s | L2 2s | L2 3s | **Avg ↓** | Col 1s | Col 2s | Col 3s | **Avg ↓** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Epona | 0.61 | 1.17 | 1.98 | 1.25 | **0.01** | 0.22 | 0.85 | 0.36 |
| **DriveLaW** | **0.44** | **1.10** | **1.91** | **1.15** | 0.15 | **0.10** | **0.48** | **0.24** |

Better on average, but note the 1s collision rate *regresses* (0.01 → 0.15) — the gain is at 2s and 3s.

![[contrast2_compressed.png|Qualitative video-generation comparison against Epona]]

**Figure 3**: Versus Epona on nuScenes validation. DriveLaW produces clearer vehicle detail and more stable structure; Epona degrades pedestrians to near-unrecognizable while DriveLaW preserves complete shapes; and Epona misclassifies an inconspicuous yellow van where DriveLaW maintains its appearance and position.

---

## Ablations

### Representation comparison under a fixed planner (Table 5) — the paper's most transferable result

| Representation | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| BEV Features | 97.6 | 93.0 | 92.9 | 100 | 79.1 | 84.1 |
| VLM Hidden State | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| **Video Latents** | **99.0** | **97.1** | **96.7** | 100 | **81.3** | **89.1** |

Same diffusion planner throughout: video latents beat BEV features by **+5.0 PDMS** and VLM hidden states by **+2.6**. This is the wiki's first *controlled* comparison of the three dominant conditioning representations. Note the VLM row lands at exactly 86.5 — ReCogDrive-IL's published score — so that row is effectively a ReCogDrive-representation reimplementation.

![[feature_compressed.png|PCA visualization of BEV, VLM, and video-generator latents]]

**Figure 4**: PCA to 3 principal components mapped to RGB, upsampled to 1280×704. Rows: input frame, BEV features (BEVFormer ResNet-101 backbone), VLM features (Qwen2.5-VL from ReCogDrive), and VGM features (DriveLaW-Video). BEV and VLM features appear diffuse, unstable, with irregular focus shifts; the video-generator features are sharper, less noisy, and show stronger spatial-structure awareness under severe driving motion.

### Which denoising step feeds the planner (Table 6) — the result that complicates the paper's framing

| Video denoise step | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **t = 1** | 99.0 | **97.1** | 96.7 | 100 | 81.3 | **89.1** |
| t = 5 | **99.2** | 93.7 | 95.6 | 100 | **81.8** | 86.9 |
| t = 10 | 81.7 | 63.4 | 67.6 | 0 | 15.4 | **23.2** |

Conditioning on **early** denoising latents is best; late latents **collapse the policy entirely** (23.2 PDMS, comfort 0). The paper's explanation is that "raw pixel-format videos frequently contain redundant, non-essential information, which can hinder the effectiveness of decision-making."

This deserves emphasis: DriveLaW's best configuration reads the latent after the **first** denoising step — before any recognizable future has actually been synthesized. The useful signal is the generator's *early-denoising internal state*, not its finished imagination. See the discussion in [[concepts/world-model-for-ad.md]].

### Video-pretraining scaling (Table 4)

| Video pretrain size | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 0 (scratch) | 98.2 | 93.8 | 94.1 | 99.9 | 80.8 | 85.9 |
| 76k | 98.7 | 94.7 | 95.3 | 99.9 | 80.8 | 87.0 |
| 3.8M | 98.6 | 95.8 | 94.8 | 100 | **82.2** | 87.8 |
| 7.6M | **99.0** | **97.1** | **96.7** | 100 | 81.3 | **89.1** |

Monotone and unsaturated: **+3.2 PDMS** from full driving-domain video pretraining versus none. This varies *pretraining data* at fixed model size — the complementary axis to [[sources/simwam.md]]'s finding that *model size* barely matters.

### Training strategy (Table 7) and noise reinjection (Table 11)

| Setting | FID ↓ | FVD ↓ |
|---|---:|---:|
| w/o first stage (long-horizon motion) | 5.0 | 109.3 |
| w/o second stage (high-res detail) | 5.0 | 93.2 |
| w/o noise reinjection | 6.1 | 102.1 |
| **Full** | **4.6** | **81.3** |

Dropping stage 1 costs the most FVD (+28.0) — temporal coherence comes from long-clip training, exactly as the curriculum intends. Noise reinjection is worth 1.5 FID and 20.8 FVD on its own.

### Cross-dataset and long-horizon generation (Table 8)

| Method | nuScenes FID ↓ | nuScenes FVD ↓ | OpenDV FID ↓ | OpenDV FVD ↓ | NuPlan FVD₂₄ | FVD₄₀ | FVD₈₀ | FVD₁₀₀ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Epona | 7.5 | 82.8 | 6.9 | 80.7 | 61.3 | 74.9 | 239.6 | **277.3** |
| **DriveLaW** | **4.6** | **81.3** | **4.6** | **72.9** | **55.6** | **71.2** | **230.2** | 296.1 |

Zero-shot OpenDV generalization is better on both metrics, but **Epona wins at the 100-frame horizon** (277.3 vs 296.1) — DriveLaW's advantage holds only to ~80 frames. The paper argues the trade is still favorable given the speed gap.

### Inference speed (Tables 9–10)

| GPU | Method | Resolution | Params | Traj. (s) | Per-frame (s) |
|---|---|---|---|---:|---:|
| 4090 | Epona | 1024×512 | ~1.9B | – | 0.88 |
| 4090 | DriveLaW | 768×512 | ~2.0B | – | **0.12** |
| 4090 | DriveLaW | 1024×512 | ~2.0B | – | **0.18** |
| 4090 | DriveLaW | 1280×704 | ~2.0B | – | 0.39 |
| H20 | Epona | 1024×512 | ~1.9B | **0.42** | 1.06 |
| H20 | DriveLaW | 1024×512 | ~2.0B | 0.71 | **0.21** |

Video generation is ~5× faster than Epona at matched resolution, but **trajectory planning is slower** (0.71 s vs 0.42 s on H20) — a detail the paper does not dwell on.

---

## Implementation Details

- **Video DiT**: 2B, initialized from LTX-Video. **Action DiT**: 133M.
- **Video pretraining**: 8 Hz frames from nuScenes + nuPlan; stage 1 at 740×352×121, stage 2 at 1280×704×25; 30k iterations each, batch 4, lr 1e-5, weight decay 5e-2; flow matching with token-wise uniform $\sigma\in[0,1]$.
- **Trajectory fine-tuning**: past four camera frames in, 2 Hz trajectory over next 4 s; batch 192, 44k steps, lr 3e-5, weight decay 1e-5; **both** Video DiT and Planning DiT updated.
- **Inference**: 30 sampling steps for video, 5 for trajectory.
- **Evaluation**: nuScenes for generation, NAVSIM navtest (12k) for planning.

![[supp1_compressed.png|Additional nuScenes video generation examples]]

**Figure 5**: (a) conventional urban driving with stable lane keeping, (b) complex urban scenes with dense multi-agent interaction, turning, and occlusion, (c) night driving under low light with preserved temporal consistency.

![[supp2.png|Qualitative planning results on Navtest]]

**Figure 6**: Representative Navtest planning cases showing safe, smooth predicted trajectories.

---

## Limitations

1. **"Sets a new record on NAVSIM" is scoped to its own table.** 89.1 PDMS is a genuine record among the methods DriveLaW compares against (best other: WoTE 88.3, DiffusionDrive/PWM 88.1), but it sits well below the wiki frontier: CLEAR 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, DynVLA 91.7, [[sources/simwam.md]] 91.5, FLARE 91.4, DiffusionDriveV2 91.2, [[sources/sgdrive.md]] 91.1, DriveVA 90.9. SimWAM explicitly beats DriveLaW by 2.4 and cites this exact 89.1.
2. **The comparison omits RL-trained methods by design** — fair as an SFT-only claim, but the paper's own framing ("without any post-training such as RL") reads as a strength when it is also the reason the number is lower than peers.
3. **Safety-skewed at the cost of progress.** NC 99.0 and TTC 96.7 lead the wiki, but EP 81.3 is mediocre and the model has no mechanism (RL, scorer) to recover progress. On nuScenes the 1s collision rate actually regresses versus Epona (0.01 → 0.15).
4. **Table 6's t=10 collapse (23.2 PDMS, comfort 0) is reported without diagnosis.** A total policy failure from a conditioning change that large suggests something more than "redundant information" — possibly a distribution mismatch between cached-latent statistics at different noise levels. The paper offers one sentence.
5. **The high-compression VAE is acknowledged to introduce motion artifacts** (Appendix D.1) in high-motion scenarios, propagating into generation. Noise reinjection mitigates but does not resolve this.
6. **Long-horizon generation degrades past ~80 frames**, where Epona overtakes it (FVD 296.1 vs 277.3).
7. **Planning latency is worse than Epona's** (0.71 s vs 0.42 s on H20) despite much faster video generation, and Appendix D.2 concedes DriveLaW remains slower than planners that skip video generation entirely.
8. **DriveVLA-W0 appears at 87.2**, a third distinct number for that method (the wiki records 90.2★ with anchors and 88.4 single-sample). It is marked † "trained with the same flow-matching objective," so it is a reimplementation, not the published configuration — worth remembering before treating it as a head-to-head.
9. **No NAVSIM-v2 / EPDMS, no navhard, no closed-loop reactive benchmark.** Generation is evaluated on nuScenes/OpenDV/nuPlan; planning on NAVSIM only.
10. **Both modules update during stage 3** ("updating both the Video DiT and the Planning DiT"), which sits awkwardly with the claim that the design "avoids gradient interference between the video generator and the planner" and that "gradient isolation ... is preserved" (Appendix A.2). The two statements are not obviously reconcilable from the text.

---

## Key Cross-References

- **World-model pattern**: [[concepts/world-model-for-ad.md]] — DriveLaW is the wiki's only method that uses a video generator's *mid-denoising internal features* as the planning state, and its Table 6 is directly relevant to whether test-time imagination helps.
- **Representation choice**: Table 5 (video latents 89.1 > VLM hidden state 86.5 > BEV 84.1 under a fixed planner) is the wiki's first controlled comparison of conditioning representations. See [[concepts/foundation-backbones-for-ad.md]].
- **Scaling axis**: Table 4 varies *pretraining data* at fixed model size (+3.2 PDMS from 0 → 7.6M), complementing [[sources/simwam.md]]'s finding that *model size* barely matters and [[sources/drivewam.md]]'s 4k → 100k clip study.
- **Direct predecessor**: [[sources/epona.md]] — DriveLaW's main generation baseline, beaten on FID/FVD and on speed, but not at the 100-frame horizon.
- **Generation quality**: FID 4.6 is the best in the wiki, displacing UniUGP's 7.4 — see the generation tables in [[concepts/world-model-for-ad.md]].
- **The competing answer**: [[sources/simwam.md]] reaches 91.5 by discarding future generation at inference entirely; DriveLaW reaches 89.1 by keeping the generator but reading it early. Both point away from conditioning on fully generated futures.
