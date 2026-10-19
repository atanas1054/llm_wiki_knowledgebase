---
title: "Driving on Registers (DrivoR)"
type: source-summary
sources: ["raw/papers/Driving on Registers.md"]
related: [sources/toad.md, sources/chainflow-vla.md, sources/clover.md, sources/dvgt-2.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/hugsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/perception-for-planning.md, concepts/visual-tokenization.md, concepts/inference-latency.md, concepts/best-of-n.md, concepts/evaluation-variance.md, concepts/data-driven-simulators.md, sources/world4scorer.md, sources/mm-future.md, sources/simscale.md, sources/drivefuture.md, sources/drivesuprim.md, sources/hydra-mdp-pp.md, sources/wa-jepa.md, sources/simvta.md, sources/idol.md, sources/maneuverflow.md, sources/hybriddriveVLA.md, sources/da-wam.md, research-directions.md]
created: 2026-10-14
updated: 2026-10-19
confidence: medium
---

# Driving on Registers (DrivoR)

**Paper**: Driving on Registers
**Authors**: Ellington Kirby, Alexandre Boulch, Yihong Xu, Yuan Yin, Gilles Puy, Éloi Zablocki, Andrei Bursuc, Spyros Gidaris, Renaud Marlet, Florent Bartoccioni, Anh-Quan Cao, Nermin Samet, Tuan-Hung Vu, Matthieu Cord
**Orgs**: valeo.ai (Paris); LIGM, ENPC (Marlet); Sorbonne Université, ISIR (Cord)
**arXiv**: 2601.05083v2 (v1 January 2026; listed as CVPR 2026 in later papers' tables)
**Code**: project page valeoai.github.io/driving-on-registers ("code and checkpoints will be made available"). [[sources/world4scorer.md]] later runs DrivoR's released model and software stack, so they were released.
**Source**: `raw/papers/Driving on Registers.md`. The clipping has all text and all 14 tables, but several figures are missing or partial: Figure 2(b), Figure 3(b), nine of Figure 5's ten tiles, Figure 7 (no image in the arXiv HTML either), the single-token panel of the Appendix F figure, and the second and third panels of Figures 8–11. The missing tiles were downloaded from the arXiv HTML (`raw/assets/drivor_*.png`; Figure 5 composed into one image, the extra panels of Figures 8 and 9 into strips), and Figure 7 was cropped from page 9 of the arXiv PDF (`drivor_fig7.png`).

**Confidence: medium.** A simple, well-ablated architecture with timing, released code, and an unusually careful appendix (a pre-fix / post-fix navhard table, HUGSIM code fixes, Best-of-N separated). But the ablations are single runs on navval; the v1 lead over DriveSuprim needs the extra navval data; the NAVSIM-v2 settings were chosen on a 7-scene validation set; the behavior-control demonstration compares two different checkpoints; and the figures and tables disagree on the training-set base of the data-scaling rows.

---

## What It Is

**A camera-only generate-and-score planner built from three plain transformers, with a ViT whose extra register tokens are the scene representation.**

1. **Perception encoder.** DINOv2 ViT-S/14, LoRA rank 32, applied to each of four cameras (front, front-left, front-right, back) at 672×1148. Each camera gets 16 learned **camera registers** appended to its tokens; after the ViT, only those registers are kept. 4 × 16 = **64 scene tokens** replace thousands of patch tokens.
2. **Trajectory decoder.** 64 learned queries plus encoded ego status (poses, velocities, accelerations, command) cross-attend to the scene tokens through 4 transformer layers (width 256); each query is decoded by an MLP into a whole 8-pose trajectory. Trained by winner-take-all L1 to the human trajectory.
3. **Scoring decoder.** Each decoded trajectory is **re-embedded by an MLP and detached from the generator's gradient**, becomes a query, cross-attends to the same scene tokens, and six heads predict the PDMS sub-scores (NC, DAC, DDC, TTC, EP, Comf) by BCE against the NAVSIM oracle scorer. Scorer gradients still reach the encoder.
4. **Selection.** The highest weighted combination of predicted sub-scores. The weights can be changed at inference ("behavior conditioning"): standard PDMS weights for NAVSIM-v1, tuned weights for NAVSIM-v2.

No BEV, no LiDAR, no perception labels, no trajectory vocabulary. About 40M parameters.

**Results**

| Benchmark | DrivoR | Notes |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest) | **93.1** navtrain / **93.7** trainval / 94.0 (+65k SimScale) / **94.6** (+134k SimScale) | DriveSuprim 93.5 and RAP-DINO 93.8 (10× data) in the same table |
| navhard two-stage EPDMS (corrected evaluator) | **48.3** / 52.3 (+65k) / **54.6** (+134k) | 45.3 under the pre-fix evaluator with identical sub-scores |
| HUGSIM, 345 "pre-challenge" scenarios, DrivoR's corrected code | 49.8 RC / **35.7** HD-Score | Zero-shot from the NAVSIM-v1 model |
| Speed (A100, batch 1, no acceleration) | **110 ms** per forward (107 ms ViT, 3 ms decoders), 41M parameters, 351 GFLOPs, 0.5 GB | GTRS-Dense ViT-L reproduced at 400 ms |
| Training | 4 × A100, about 1 h per epoch on navtrain | 25 epochs on trainval for v1; 10 on navtrain for v2 |

---

## Key Takeaways

- **64 register tokens carry the scene.** On navval, register compression (64 scene tokens) scores 90.0 against 90.2 for the full 16k patch tokens, 89.7 for pooling and 89.3 for a learned query decoder, with LoRA in each case. The differences are 0.2–0.9 PDMS, single runs. The registers specialize: front-camera registers are nearly decorrelated and attend to traffic lights, the lead vehicle and road edges, while the back camera's registers collapse to essentially one distinct token.
- **The scorer design is worth more than the encoder.** One shared decoder for generation and scoring: 84.7. Separate decoders: 86.8. Separate plus re-embedded, gradient-stopped trajectories: **90.0**. Six sub-score heads beat one aggregate head by +1.8 (88.2). Candidate count: 1 query 80.1, 8 87.6, 64 90.0, 128 90.0.
- **On navhard the decomposition is visible.** Putting DrivoR's encoder under GTRS-Dense's vocabulary and scorer ("GTRS-DrivoR") gives 45.8 against 45.0 for GTRS-Dense's V2-99 (+0.8, at a third of the latency); DrivoR's own generator and scorer on that encoder give 48.3 (+2.5). Adding SimScale's synthetic scenes gives 52.3 (+65k) and 54.6 (+134k). **The wiki has carried DrivoR's navhard score as 54.6, "second only to DriveFuture"; 6.3 points of that is SimScale data.**
- **The v1 lead needs the validation split.** Trained on navtrain, DrivoR (93.1) is below DriveSuprim (93.5) in its own Table 1. The 93.7 that "outperforms all other methods" is trained on navtrain + navval, as [[sources/mm-future.md]]'s 94.0 is. With SimScale data it reaches 94.6, the highest single-pass v1 score in the wiki, with extra real and synthetic data. *(lint 2026-10-19: passed by [[sources/chainflow-vla.md]], 94.8 on navtrain + navval with no synthetic data. [[sources/toad.md]] reaches 94.7 by test-time search on this 94.6 model.)*
- **A direct measurement of the NAVSIM-v2 evaluator fix on navhard.** Appendix Table 14 prints four methods before and after the official fix (Issue #151, human-error filtering) with identical stage sub-scores: DrivoR 45.3 → 48.3, GTRS-DrivoR 42.3 → 45.8, ZTRS 45.5 → 48.1, RAP-DINO 36.9 → 39.6. **The fix adds +2.6 to +3.5 combined EPDMS.** It also lists LTF at 23.1 among the pre-fix results, which identifies the wiki's "lineage B" navhard baselines as pre-fix (see [Table audit](#table-audit)).
- **DrivoR's HUGSIM numbers come from code it fixed.** The paper reports "inconsistent acceleration bounds in the comfort metric computation and an incorrect heading computation for the planned trajectory provided to the controller", fixed them, and re-ran every baseline. That is the "corrected evaluator and controller" [[sources/simvta.md]] later reuses, and it moves DrivoR's published row out of the wiki's "no heading fix" group on [[concepts/hugsim-benchmark.md]].
- **Behavior conditioning is claimed, not isolated.** The "safety-oriented agent" (Figure 7) and the passive agent (Figure 11) are the NAVSIM-v2 model, trained differently (10 epochs on navtrain) from the v1 model (25 epochs on trainval). Figure 11's two panels show different candidate sets, which re-weighting the scorer cannot produce. The v2 weights and training length were chosen on warmup-two-stage, a 7-scene validation set.
- **Efficient for its class.** 110 ms on an A100 for four cameras at 672×1148, of which 107 ms is the ViT; the decoders cost 3 ms. [[sources/world4scorer.md]] measures the same stack at 106.5 ms.

---

## Method

![[architecture_pdf.png|Figure 1: DrivoR architecture. The proposed architecture is composed of three transformer blocks: one encoder (perception) and two decoders (trajectory and scoring). The perception encoder compresses perceptual information in camera-aware registers for lightweight subsequent processing in the trajectory and scoring decoders. The decoded trajectories are re-embedded and detached from the gradient computation graph to disentangle scoring and generation. The final trajectory is chosen from the proposal set via the max predicted score.]]

**Figure 1**: Images and sensor registers enter the perception transformer encoder; the scene tokens (all sensor registers) are keys and values for both decoders. Trajectory queries plus ego status feed the trajectory decoder; its trajectories pass a "disentanglement" block (re-embedding, stop-gradient) to become score queries for the scoring decoder, which outputs scored trajectories.

![[architecture_encoder.png|Figure 2(a): Encoder: a ViT with LoRA finetuning takes frozen-embedded image tokens and learnable per-sensor registers; the registers from all sensors form the scene tokens.]]

![[drivor_architecture_decoder.png|Figure 2(b): Decoder: a vanilla transformer decoder (self-attention, cross-attention to scene tokens, FFN). Recovered from the arXiv HTML.]]

**Figure 2**: (a) Per-camera ViT with LoRA; image tokens plus learnable, sensor-specific registers; only the registers leave the encoder. (b) *(recovered from the arXiv HTML)* Each decoder is a standard stack of self-attention, cross-attention to the scene tokens, and FFN.

### Encoder (§3.2)

For each of $N$ cameras, $R$ new registers of size $D_\mathrm{ViT}$ are concatenated with DINOv2's own registers, the class token and the patch tokens. After the last ViT layer the $R$ camera registers are kept, giving $N\times R$ camera-aware scene tokens. LoRA (rank 32) fine-tunes the ViT; registers are initialized from $\mathcal N(0,10^{-6})$. Default $R=16$, $N=4$.

### Trajectories (§3.3)

$$\mathcal L_\mathrm{traj}=\min_i\lVert\tau_i-\hat\tau\rVert_1\qquad\text{(optional)}\qquad \mathcal L_\mathrm{traj}=\min_i\big(\lVert\tau_i-\hat\tau\rVert_1+\lVert\tau_i-\hat\tau'\rVert_1\big)$$

$\hat\tau'$ is a longer human trajectory ($T'>T$) resampled to $T$ with a cubic spline, an "accelerated" target that encourages progress. Each trajectory is decoded from **one token** (Appendix F: 90.0 against 83.9 for one token per pose).

### Scores (§3.4–3.5, Appendices B–C)

$$\mathcal L_\mathrm{score}=\sum_c\lambda_c\sum_i\mathrm{BCE}\big(\mathcal G_{\theta_c}(\tau_i),\mathcal G_c(\tau_i)\big),\qquad \mathcal L=\mathcal L_\mathrm{traj}+\lambda_s\mathcal L_\mathrm{score},\qquad \lambda_c=\lambda_s=1$$

$\mathcal G_c$ is the NAVSIM oracle for sub-score $c$, run on every predicted trajectory. At inference the predicted sub-scores are combined as in the PDMS / EPDMS formula with weights $\kappa_c$:

**Table 10: inference weights**

| Sub-score | Group | NAVSIM-v1 | NAVSIM-v2 (tuned) |
|---|---|---:|---:|
| NC | penalty | 1 | 10 |
| DAC | penalty | 1 | 13 |
| DDC | penalty | 0 | 6 |
| TTC | weighted | 5 | 14 |
| EP | weighted | 5 | 15 |
| Comfort | weighted | 2 | 2 |

The v2 weights were "validated with NAVSIM-v2 warmup-two-stage", which the paper says contains 7 scenes. The scorer predicts v1 comfort only; v2's traffic-light, lane-keeping and extended-comfort terms are never predicted.

### Tables 8–9 (Appendices A–B)

PDMS and EPDMS share the form $\prod_{c\in G_p}\mathcal G_c^{\kappa_c}\times\frac1Z\sum_{c\in G_b}\kappa_c\mathcal G_c$. PDMS: NC, DAC multiplicative; TTC 5, EP 5, Comfort 2. EPDMS: NC, DAC, DDC, TLC multiplicative; TTC 5, EP 5, LK 2, HC 2, EC 2. All training loss weights are 1.

---

## Results

### Table 13 (superset of Table 1): NAVSIM v1 navtest

Residual = closed form − reported, computed here; negative is normal for per-scene-averaged PDMS ([[concepts/navsim-benchmark.md#physwam]]).

| Method | Mod. | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PDM-Closed (privileged) | – | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 | −0.76 |
| Human driver | – | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 | −0.03 |
| *Test-time training* | | | | | | | | |
| Centaur | C | 99.5 | 98.9 | 98.0 | 100 | 85.9 | 92.6 | −0.80 |
| *Best-of-N* | | | | | | | | |
| AutoVLA (BoN) | C | 99.1 | 97.1 | 97.1 | 100.0 | 87.6 | 92.1 | −2.01 |
| DriveVLA-W0 (BoN) | C | 99.3 | 97.4 | 97.0 | 99.9 | 88.3 | 93.0 | −2.22 |
| TransDiffuser (BoN) | C+L | 99.4 | 96.5 | 97.8 | 99.4 | 94.1 | **94.9** | −2.31 |
| *Ensemble* | | | | | | | | |
| Hydra-MDP-C (V2-99) | C+L | 98.7 | 98.2 | 95.0 | 100 | 86.5 | 91.0 | −1.55 |
| *10× more training data* | | | | | | | | |
| RAP-DINO | C | 99.1 | 98.9 | 96.7 | 100 | 90.3 | 93.8 | −1.10 |
| *Camera + LiDAR* | | | | | | | | |
| TransFuser | C+L | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 | −3.91 |
| DistillDrive | C+L | 98.1 | 94.6 | 93.6 | 100 | 81.0 | 86.2 | −3.22 |
| TrajHF (EM) | C+L | 96.6 | 96.6 | 92.1 | 100 | 84.5 | 87.6 | −3.38 |
| DiffusionDrive | C+L | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 | −2.72 |
| WoTE | C+L | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | −2.18 |
| Hydra-MDP (V2-99) | C+L | 98.0 | 97.8 | 93.9 | 100 | 86.5 | 90.3 | −2.28 |
| GoalFlow (V2-99) | C+L | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 | −1.79 |
| ResAD (V2-99) | C+L | 98.9 | 97.8 | 94.9 | 100 | 87.0 | 90.6 | −1.17 |
| SeerDrive (V2-99) | C+L | 98.8 | 98.6 | 95.8 | 100 | 84.2 | 90.7 | −1.40 |
| *Camera only* | | | | | | | | |
| Ego-status MLP | C | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 | −9.77 |
| UniVLA | C | 96.9 | 91.1 | 91.7 | 96.7 | 76.8 | 81.7 | −5.50 |
| DrivingGPT | C | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 | −2.85 |
| UniAD | C | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 | −4.12 |
| LTF | C | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 | −4.18 |
| PARA-Drive | C | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | −4.01 |
| DriveX-S | C | 97.5 | 94.0 | 93.0 | 100 | 79.7 | 84.5 | −3.28 |
| World4Drive | C | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 | −3.70 |
| DRAMA | C | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 | −3.80 |
| VAD-v2 ⚠ | C | 98.1 | 94.8 | 94.3 | 100 | 80.6 | 86.2 | −2.93 |
| PRIX | C | 98.1 | 96.3 | 94.1 | 100 | 82.3 | 87.8 | −2.62 |
| DIVER | C | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 | −2.16 |
| AutoVLA | C | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 | −2.92 |
| DriveVLA-W0 | C | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 | −1.22 |
| ReCogDrive | C | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 | −2.61 |
| Hydra-MDP++ | C | 98.6 | 98.6 | 95.1 | 100 | 85.7 | 91.0 | −1.56 |
| iPad | C | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 | −1.68 |
| DriveSuprim | C | 98.6 | 98.6 | 95.5 | 100 | 91.3 | 93.5 | −1.63 |
| **DrivoR (navtrain)** | C | 98.9 | 98.3 | 96.2 | 100 | 89.1 | **93.1** | −1.84 |
| **DrivoR (trainval)** | C | 99.0 | 98.9 | 96.7 | 100 | 90.0 | **93.7** | −1.21 |
| **DrivoR (+65k SimScale)** | C | 99.1 | 99.0 | 96.9 | 100 | 90.3 | **94.0** | −1.12 |
| **DrivoR (+134k SimScale)** | C | 99.1 | 99.2 | 96.9 | 100 | 91.6 | **94.6** | −1.00 |

Main-text Table 1 is the camera-only block plus PDM-Closed, the human and RAP-DINO. Every row passes the check. ⚠ VAD-v2 here (86.2, cited as the 2024 arXiv paper) is the row [[sources/da-wam.md]] labels "ICLR'26"; the wiki also carries VADv2 at 80.9 (the original reproduction) and 89.3 ([[sources/maneuverflow.md]], also labelled ICLR 2026). TransDiffuser's best-of-N 94.9 is above the human score.

### Figure 3: data scaling

![[nav1_scaling_v2.png|Figure 3(a): NAVSIM-v1 PDMS against training data size: 93.1, 93.7, 94.0, 94.6 for DrivoR; RAP at 93.9 with about 565k samples; human driver 94.8.]]

![[drivor_nav2_scaling.png|Figure 3(b): NAVSIM-v2 navhard EPDMS against training data size: 48.3, 52.2, 54.6 for DrivoR; SimScale at 53.2 with about 270k samples. Recovered from the arXiv HTML.]]

**Figure 3**: (a) v1: DrivoR at about 85k samples (93.1), 103k (93.7), 168k (94.0) and 237k (94.6); RAP 93.9 at about 565k; human 94.8. (b) *(recovered from the arXiv HTML)* navhard: 48.3 at 85k, 52.2 at 150k, 54.6 at 219k; SimScale 53.2 at 270k. The x-positions put the v1 SimScale runs on the trainval base (103k + 65k, 103k + 134k) and the v2 runs on the navtrain base (85k + 65k, 85k + 134k); the text says both "jointly train with navtrain". The figure prints 52.2 where Table 3 has 52.3, and RAP at 93.9 where Table 1 has 93.8.

### Table 3 / Table 14: navhard two-stage

Stage closed forms and their product computed here (the combined score is approximately the product of stage scores, [[concepts/navhard-ood-evaluation.md]]). "Product − reported" is the navhard version of the residual fingerprint.

| Method | Evaluator | Stage-1 CF | Stage-2 CF | Product | Reported | Product − reported |
|---|---|---:|---:|---:|---:|---:|
| PDM-Closed (privileged) | pre-fix | 83.1 | 61.4 | 51.0 | 51.3 | −0.28 |
| Constant velocity | pre-fix | 21.6 | 27.1 | 5.8 | 10.9 | −5.05 |
| Ego-history MLP | pre-fix | 38.6 | 21.9 | 8.5 | 12.7 | −4.24 |
| LTF | pre-fix | 67.9 | 35.0 | 23.7 | **23.1** | +0.64 |
| RAP-DINO (ViT-H; 10× data) | pre-fix | 80.1 | 46.2 | 37.0 | 36.9 | +0.12 |
| RAP-DINO | **corrected** | 80.1 | 46.2 | 37.0 | **39.6** | −2.58 |
| GTRS-DrivoR (ViT-S) | pre-fix | 78.1 | 52.9 | 41.3 | 42.3 | −0.97 |
| GTRS-DrivoR (ViT-S) | **corrected** | 78.1 | 52.9 | 41.3 | **45.8** | −4.47 |
| GTRS-Dense (EVA-ViT-L) | pre-fix | 78.2 | 57.5 | 44.9 | 43.4 | +1.53 |
| GTRS-Aug (ViT-L) ⚠ | pre-fix | 81.1 | 56.7 | 46.0 | 44.7 | +1.28 |
| GTRS-Dense (ViT-L) | pre-fix | 80.2 | 59.0 | 47.3 | 45.3 | +2.02 |
| DrivoR (ViT-S) | pre-fix | 80.5 | 55.9 | 45.0 | 45.3 | −0.27 |
| **DrivoR (ViT-S)** | **corrected** | 80.5 | 56.0 | 45.1 | **48.3** | −3.23 |
| ZTRS (V2-99) | pre-fix | 78.5 | 59.3 | 46.6 | 45.5 | +1.09 |
| ZTRS (V2-99) | **corrected** | 78.5 | 59.3 | 46.6 | **48.1** | −1.51 |
| GTRS-Dense (V2-99) | corrected | 77.5 | 56.6 | 43.9 | 45.0 | −1.09 |
| GTRS-Aug (V2-99) ⚠ | corrected | 79.5 | 57.8 | 46.0 | 45.4 | +0.55 |
| **DrivoR + 65k SimScale** | corrected | 83.6 | 61.1 | 51.1 | **52.3** | −1.22 |
| SimScale + 185k (V2-99) | corrected | 79.3 | 65.7 | 52.1 | **53.2** | −1.15 |
| **DrivoR + 134k SimScale** | corrected | 84.1 | 62.7 | 52.7 | **54.6** | −1.87 |

The full stage sub-scores are in the clipping's Tables 3 and 14. Main-text Table 3 is the corrected block.

- **The fix at fixed sub-scores**: +3.0 (DrivoR), +3.5 (GTRS-DrivoR), +2.6 (ZTRS), +2.7 (RAP-DINO).
- **The fingerprint, calibrated.** Learned pre-fix rows sit at −1.0 to +2.0 (product minus reported); corrected rows at −1.1 to −4.5, except the suspect GTRS-Aug V2-99 row. The bands touch near −1, so the check is a tendency, not a classifier, on navhard.
- ⚠ The two pre-fix GTRS rows for V2-99 (not shown) have scrambled columns (Stage-1 EP 99.4, TLC 89.2); "GTRS-Aug (ViT-L)" and "DriveSuprim (EVA-ViT-L)" are identical in all 19 values; the corrected GTRS-Aug V2-99 row has Stage-2 EP 98.9 where every other row has 64–87.

### Table 2: HUGSIM closed loop (345 pre-challenge scenarios, zero-shot)

| Method | RC E / M / H / X | RC Avg | HDS E / M / H / X | HDS Avg |
|---|---|---:|---|---:|
| VAD | 51.3 / 31.1 / 25.3 / 26.5 | 31.4 | 36.3 / 9.5 / 8.0 / 11.5 | 13.4 |
| LTF | 67.8 / 35.1 / 26.2 / 40.5 | 38.9 | 58.9 / 18.0 / 9.8 / 25.9 | 23.7 |
| UniAD | 78.4 / 60.5 / 33.6 / 17.8 | 45.9 | 64.9 / 45.8 / 20.6 / 6.6 | 32.7 |
| **DrivoR** | 80.9 / 50.5 / 33.8 / 47.1 | **49.8** | 73.3 / 34.6 / 18.8 / 32.5 | **35.7** |

All four rows were re-run with DrivoR's corrected implementation (comfort acceleration bounds; heading of the planned trajectory passed to the controller). HD-Score combines RC with averaged NC, DAC, TTC and comfort. Weighting arithmetic on [[concepts/hugsim-benchmark.md#simvta]] shows these rows share one count weighting (≈ 49 / 126 / 86 / 84). On WA-JEPA's 436-scenario snapshot DrivoR scores 32.5 ([[sources/wa-jepa.md]]).

### Table 11: efficiency (A100, batch 1, no quantization; NAVSIM-v2 navhard EPDMS)

| Method | Image size | Cams | Encoder | Params (backbone ‖ rest) | GFLOPs | Memory | Latency (backbone ‖ rest) | EPDMS |
|---|---|---:|---|---|---:|---:|---|---:|
| RAP-DINO (10× data) | 448×768 | 4 | ViT-H | 888M (857M ‖ 31M) | 4760 | 4.2 GB | 690 ms (653 ‖ 37) | 39.6 |
| + GTRS-DP diffusion proposals | 512×2048 | 4 | + V2-99 | +116M | +1249 | +1.15 GB | +389 ms | – |
| GTRS-Dense | 512×2048 | 1 | V2-99 | 81M (69.5M ‖ 11.5M) | 404 | 1.0 GB | 96 ms (22 ‖ 74) | 45.0 |
| GTRS-Aug | 512×2048 | 1 | V2-99 | 171M (69.5M ‖ 101.5M) | 439 | 1.42 GB | 243 ms (22 ‖ 221) | 45.4 |
| GTRS-Dense (reproduced) | 512×2048 | 1 | ViT-L | 321M (313M ‖ 8M) | 1730 | 1.6 GB | 400 ms (352 ‖ 48) | 47.0 |
| GTRS-Dense (reproduced) | 512×2048 | 1 | ViT-S | 32M (24M ‖ 8M) | 234 | 0.38 GB | 303 ms (46 ‖ 257) | 45.9 |
| ZTRS | 512×2048 | 2 | V2-99 | 81M (69.5M ‖ 11.5M) | 840 | 1.0 GB | 193 ms (119 ‖ 74) | 48.1 |
| **DrivoR** | 672×1148 | 4 | ViT-S | **41M (24M ‖ 17M)** | 351 | 0.5 GB | **110 ms (107 ‖ 3)** | **48.3** |

FLOPs exclude scaled dot-product attention (an FVCore limitation the paper notes). GTRS's non-backbone cost is 48–257 ms across its variants; DrivoR's two decoders over 64 candidates cost 3 ms.

---

## Ablations (navval, single runs)

### Table 4: perception

| (a) Initialization | Random | ImageNet-21k | DINOv2 |
|---|---:|---:|---:|
| PDMS | 70.1 | 87.5 | **90.0** |

| (b) | Compression | Scene tokens | Encoder training | Trainable / total params (M) | PDMS |
|---|---|---:|---|---|---:|
| (a) | None | 16k | Frozen | 18.2 / 41.2 | 88.2 |
| (b) | None | 16k | LoRA | 18.8 / 41.8 | **90.2** |
| (c) | Pooling | 64 | LoRA | 18.2 / 41.2 | 89.7 |
| (d) | Query decoder | 64 | Frozen | 19.3 / 42.3 | 86.9 |
| (e) | Query decoder | 64 | LoRA | 19.9 / 42.9 | 89.3 |
| (f) | Registers | 64 | Full fine-tuning | 41.2 / 41.2 | 88.4 |
| (g) | Registers | 64 | Frozen (registers only trained) | 18.2 / 41.2 | 84.4 |
| (h) | **Registers** | 64 | **LoRA** | 18.8 / 41.8 | **90.0** |

Group labels are reconstructed from the text (the clipping's merged cells drop them); the text's "full finetuning (d, f)" conflicts with the table, which marks (d) frozen.

| (c) Tokens per camera | Per scene | Registers | PDMS |
|---|---:|---|---:|
| 5 | 20 | DINOv2's own | 88.1 |
| 5 + 16 | 84 | DINOv2's + random | 89.8 |
| 5 | 20 | random | 89.7 |
| 8 | 32 | random | 89.7 |
| **16** | **64** | random | **90.0** |
| 32 | 128 | random | 89.8 |

### Tables 5, 6, 12: trajectories and scoring

| Trajectory queries | 1 | 8 | 16 | 32 | **64** | 128 |
|---|---:|---:|---:|---:|---:|---:|
| PDMS | 80.1 | 87.6 | 88.1 | 89.5 | **90.0** | 90.0 |

| Scoring design | Separate decoders | Disentangled | Score heads | PDMS | Behavior control |
|---|:-:|:-:|---:|---:|:-:|
| (a) | ✗ | – | 6 | 84.7 | ✓ |
| (b) | ✓ | ✗ | 6 | 86.8 | ✓ |
| **(c)** | ✓ | ✓ | 6 | **90.0** | ✓ |
| (d) | ✓ | ✓ | 1 | 88.2 | ✗ |

| Trajectory representation | One token per pose | One token per trajectory |
|---|---:|---:|
| PDMS | 83.9 | **90.0** |

### Table 7: a longer, "accelerated" second target

| Targets | navval PDMS | warmup-two-stage EPDMS (7 scenes) |
|---|---:|---:|
| $(t+T)$ | 90.0 | **39.4** |
| $(t+T)$ & $(t+T')$ | **90.6** | 37.8 |

**Training length**: navval PDMS rises to a plateau at 25 epochs; warmup-two-stage EPDMS falls with more epochs and with the trainval split, so the v2 model is trained 10 epochs on navtrain. Both choices rest on 7 scenes.

---

## Visualizations

![[correlation_matrices.png|Figure 4: Cosine similarity between scene tokens. Darker indicates lower cosine similarity. Note the specialized tokens in the front cam, and collapsed tokens in the back cam, showing relative camera compression. Averaged on navval.]]

**Figure 4**: Register–register cosine similarity per camera (navval average). Front: a checkerboard of low similarities, so the 16 registers carry different content. Front-right and front-left: two or three blocks. Back: one distinct register and fifteen nearly identical ones.

![[drivor_fig5.png|Figure 5: Attention maps of scene tokens. Top row: front camera and four of its registers; bottom row: back camera and four of its registers. Front-camera tokens specialize to distinct regions (traffic light, lead vehicle, road edges), while back-camera tokens largely collapse to the same features, aside from a single distinct token. Composed from the arXiv HTML tiles.]]

**Figure 5** *(composed from the arXiv HTML; the clipping has only the first tile, `cam_f0_base.png`)*: Final-layer attention of individual registers. Front camera (top): registers 3, 5, 11 and 12 pick out, roughly, the lead vehicle, the traffic lights, the kerb on the right and the vehicles ahead. Back camera (bottom): register 2 has its own pattern; registers 1, 3 and 4 attend to almost the same scattered points along the road.

![[dominant_cameras.png|Figure 6: Scoring head disentanglement. Dominant cameras are identified via cross-attention between scene tokens and score or trajectory tokens. Trajectory prediction consistently relies on the front camera, while scoring shifts attention based on trajectory behavior—underscoring the need to separate the two pipelines.]]

**Figure 6**: A left turn. Every generated trajectory's dominant camera is the front one; when the same trajectories are scored, sharper turns draw on the left camera and some on the back.

![[drivor_fig7.png|Figure 7: Safety-oriented agent. Dark blue was tuned on warmup-two-stage, light blue is our NAVSIM-v1 model. The result of our behavior tuning is a more cautious, but safer agent. Cropped from page 9 of the arXiv PDF; the arXiv HTML has no image for it.]]

**Figure 7** *(cropped from the arXiv PDF; no image in the HTML or the clipping)*: Sub-scores of the default (v1) agent and the safety-oriented agent on warmup-two-stage. The safety agent improves DAC (≈ 0.875 → 0.97), NC (0.725 → 0.79), TTC (0.665 → 0.73), LK (0.41 → 0.445) and EC (0.62 → 0.725), and gives up EP (0.775 → 0.605); stage-2 EPDMS 0.53 → 0.56, EPDMS 0.445 → 0.48. Values read from the bars. Values like 0.875 are consistent with a set of 7–8 scenes.

![[multi_tok.png|Appendix F figure (a): trajectories decoded with one token per pose.]]

![[drivor_single_tok.png|Appendix F figure (b): trajectories decoded with one token per trajectory. Recovered from the arXiv HTML.]]

**Appendix F figure**: The same left turn decoded (a) one token per pose, noisy and spread across both lanes, and (b) *(recovered)* one token per trajectory, a tight bundle along the human path. The single-token candidates are smoother and also far less diverse: in this scene the 64 candidates are near-duplicates of one manoeuvre.

![[right_turn_bev_with_images.png|Figure 8: Visualization of generated trajectories (right turn) in a NAVSIM-v1 val set scene. Right turn scenario with right camera as dominant camera used in trajectory scoring and generation.]]

![[drivor_fig8_panels.png|Figure 8, right panels: dominant camera per trajectory for generation and for scoring. Recovered from the arXiv HTML.]]

**Figure 8**: A right turn; both generation and scoring rely mostly on the right camera, with a few scored trajectories drawing on front, left or back.

![[no_tlc_bev_with_images.png|Figure 9: Visualization of generated trajectories (intersection crossing) in a NAVSIM-v1 val set scene. The ego-vehicle crosses a signalized intersection, but the scoring attention focuses on the rear camera.]]

![[drivor_fig9_panels.png|Figure 9, right panels: generation relies on the front camera, scoring on the back camera. Recovered from the arXiv HTML.]]

**Figure 9**: Crossing a signalized intersection: generation uses the front camera, scoring mostly the back camera. The paper suggests a traffic-light sub-score would redirect it; its TLC on navhard is nonetheless high (98.6–100).

![[failure_mode_bev_with_images.png|Figure 10: Visualization of generated trajectories (failure case) in a NAVSIM-v1 val set scene. Failure case highlighting difficulty of navigation without historical (past) frames.]]

![[drivor_failure_mode_bev.png|Figure 10, BEV panel: the candidates undercut the turn. Recovered from the arXiv HTML.]]

**Figure 10**: A failure without history frames: an ambiguous front view, full attention on the right camera, and candidates that cut the turn toward the wrong side of the road.

![[aggressive_agent_trajs.png|Figure 11 (left): trajectories from the agent with default PDMS weights.]]

![[drivor_passive_agent_trajs.png|Figure 11 (right): trajectories from the agent with tuned weights. Recovered from the arXiv HTML.]]

**Figure 11**: The same scene for the "default" and the "tuned" agent. The candidate sets differ (left: long, tight bundle; right: shorter and more spread), so these are two checkpoints, not one checkpoint re-weighted.

---

## Table Audit {#table-audit}

1. **v1 is consistent.** All 40 rows pass the closed-form check (−0.03 for the human to −9.77 for the ego-status MLP; low-scoring rows have the largest gaps, as usual). iPad's 91.7 row matches DA-WAM's copy (Drive-JEPA's table has iPad ResNet-34 at 91.1 and ViT-L at 91.7 with different sub-scores).
2. **A third VADv2.** "VAD-v2" 86.2 (98.1 / 94.8 / 94.3 / 100 / 80.6), cited to the 2024 arXiv paper, is the row [[sources/da-wam.md]] labels ICLR'26. The wiki also carries 80.9 (the NAVSIM-paper reproduction of VADv2-𝒱₈₁₉₂) and 89.3 ([[sources/maneuverflow.md]], labelled ICLR 2026). Which ICLR 2026 number is the paper's, and which pairs with IDOL's v2 "VADv2 85.8", is open. *(2026-10-14: [[sources/dvgt-2.md]] cites the same 2024 arXiv paper at 80.9, so the citation does not identify the row.)*
3. **"Outperforms all other methods on NAVSIM-v1"** holds only for the trainval model (93.7): the navtrain model (93.1) is below DriveSuprim (93.5) and, setting RAP aside as orthogonal data, the 93.7 margin over DriveSuprim is 0.2 with navval added to training.
4. **Figure 3 and the text disagree on the scaling base** (trainval for v1, navtrain for v2 by the x-axis; navtrain for both in the text), and on two values (52.2 / 52.3; RAP 93.9 / 93.8). Figure 3 also places navtrain at about 85k samples, where [[sources/world4scorer.md]] and [[sources/maneuverflow.md]] count 103,288; the counting convention is not stated.
5. **navhard Table 14**: scrambled pre-fix GTRS V2-99 rows; a duplicated GTRS-Aug / DriveSuprim row; a GTRS-Aug Stage-2 EP of 98.9. PDM-Closed's Stage-1 HC is 97.7 here and 87.7 in [[sources/idol.md]]'s copy.
6. **Small text slips**: "full finetuning (d, f)" against a table that marks (d) frozen; Appendix F calls its figure "Fig. 7", which is the safety-agent figure.

---

## Limitations

The paper's own: no history frames (Figure 10), and compression of history, more sensors or maps left for future work.

The wiki adds:

1. **Single runs on navval** for every ablation; most compression and register effects are 0.2–0.5 PDMS.
2. **NAVSIM-v2 choices tuned on 7 scenes**: inference weights, training length and split, and the longer-target decision.
3. **Behavior control is not isolated**: the safety-oriented and passive agents are a different checkpoint. Re-weighting one trained scorer at inference is never evaluated on its own.
4. **Privileged supervision**: the scorer learns the NAVSIM oracle's sub-scores on every candidate; it is a scorer-cohort method ([[concepts/selection-based-planning.md]]). The v2 scorer predicts only v1 sub-scores.
5. **The headline numbers use extra data**: navval for 93.7, SimScale scenes for 94.0 / 94.6 and for the navhard 52.3 / 54.6.
6. **HUGSIM on its own corrected code** over 345 scenarios. The fixes are described in one sentence; whether they match the HUGSIM repository's later heading correction (PR #57, in WA-JEPA's pinned commit) is not stated.
7. **Four of eight cameras.** NAVSIM provides eight (Figure 10 shows them); DrivoR uses front, front-left, front-right and back.

---

## Relation to Other Wiki Pages

- [[sources/toad.md]] *(2026-10-19)*: DrivoR's group uses its scorer as the reward of a test-time CEM search around any planner's pick. On DrivoR it adds +0.1 PDMS (94.6 → 94.7) and +1.7 navhard EPDMS (54.6 → 56.3); on five other planners, +0.1 to +2.1 PDMS and +1.1 to +15.1 EPDMS. TOAD's "DrivoR" rows are this page's +134k-SimScale rows, unlabelled. Scorers trained on fixed vocabularies (GTRS, SparseDriveV2) fail as search rewards. The paper credits DrivoR's gradient-stopped scorer on freely decoded proposals for generalizing off the proposal set.
- [[sources/world4scorer.md]]: built on DrivoR's released encoder, generator and stack, and replaces its scorer. On DrivoR's own 64-candidate pools its scorer reaches 94.19 against DrivoR's 93.69 (oracle 99.33), and its inertial re-ranking lifts DrivoR from 90.66 to 92.27 EPDMS on navtest.
- [[sources/clover.md]]: keeps DrivoR's inference graph and retrains it against the evaluator (pseudo-experts, alternating scorer refits): 94.5 PDMS, 90.4 EPDMS, navhard 48.3 (equal to DrivoR's). Its baseline pools are DrivoR's released ones (oracle 99.33, selected 93.69), and its v1 and navhard tables copy DrivoR's camera-only and corrected blocks, including the suspect GTRS-Aug row. On its own decomposition +0.69 of its +0.74 is better ranking.
- [[sources/mm-future.md]]: takes DrivoR's register-token encoder, 93.4 / 94.0 against DrivoR's 93.1 / 93.7 on navtrain / trainval.
- [[sources/chainflow-vla.md]] *(2026-10-19)*: DrivoR's trainval model is its ablation baseline. It replaces the one-token-per-trajectory decoder with an autoregressive decoder through a bicycle model (+0.3), adds a residual DiT conditioned on a frozen 2B VLM (+0.7; +0.1 on DrivoR's own scene tokens) and scores the refined set: 93.6 / 94.8 PDMS on navtrain / trainval. The AR decoder works where DrivoR's own per-pose decoding failed (83.9 against 90.0 in Appendix F), plausibly because of the kinematic model.
- [[sources/hybriddriveVLA.md]]: uses a DrivoR-style disentangled scorer.
- [[sources/simscale.md]]: supplies the synthetic scenes DrivoR trains on, and DrivoR's after-fix table is where the circulating "SimScale 53.2" row appears with full sub-scores.
- [[sources/drivefuture.md]]: its navhard leaderboard copies DrivoR's after-fix rows (DrivoR 54.6, SimScale 53.2, ZTRS 48.1) next to pre-fix baselines; see [[concepts/navhard-ood-evaluation.md#evaluator-fix]].
- [[sources/drivesuprim.md]], [[sources/hydra-mdp-pp.md]]: the vocabulary-scoring lineage DrivoR argues against (no vocabulary, 64 learned queries).
- [[sources/wa-jepa.md]], [[sources/simvta.md]]: WA-JEPA rescored DrivoR on HUGSIM's 436-scenario snapshot (32.5); SimVTA reuses DrivoR's 345-scenario protocol (38.8 against 35.7).
- [[sources/idol.md]]: its navhard table showed the same residual split between its own row and its baselines; DrivoR's paired table calibrates that check.
- [[sources/maneuverflow.md]]: argues that learned queries without mode structure concentrate in one region (its Figure 1a); DrivoR's own single-token figure shows exactly that bundle in one scene, while scoring 93.1–94.6.
- Not ingested: **GTRS / GTRS-Dense / GTRS-Aug** and **ZTRS** (the vocabulary scorers DrivoR compares and swaps into), **RAP** (3D-rasterization augmentation, 10× data), **Centaur** (test-time training), **TransDiffuser** (best-of-N 94.9), **CtRL-Sim** (the offline-RL idea behind re-weighting).

---

## Open Questions

- **Does inference-time re-weighting change behavior on one checkpoint?** Score the v1 model with the v2 weights and the v2 model with the v1 weights on navtest and navhard.
- **How many tokens does a planner need once history is added?** The paper's own failure case is a missing-history case; register compression of past frames is untested.
- **Are DrivoR's HUGSIM fixes the same as the repository's PR #57?** If so, the 345 and 436 groups differ only in scenarios; if not, there are three controller states.
- **Is the +2.5 for DrivoR's heads over GTRS's on the same encoder robust?** One navhard run each.
