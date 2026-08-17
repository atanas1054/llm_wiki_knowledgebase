---
title: "DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving"
type: source-summary
sources: [raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/physicalai-av-benchmark.md, concepts/dual-system-vla.md, concepts/nuscenes-waymo-evals.md, sources/driveva.md, sources/epona.md, sources/drivevla-w0.md, sources/drivedreamer-policy.md, sources/alpamayo-r1.md, sources/futuresightdrive.md, sources/automot.md, sources/spanvla.md]
created: 2026-08-17
updated: 2026-08-17
confidence: high
---

**Paper**: DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving
**Authors**: Chen Shi, Jinrui Xu, Shaoshuai Shi, Kehua Sheng, Bo Zhang, Li Jiang
**Org**: The Chinese University of Hong Kong (Shenzhen) + Voyager Research, Didi Chuxing
**arXiv**: 2605.28544v1
**Project page**: https://chenshi3.github.io/drivewam.github.io/

---

## Summary

DriveWAM adapts a **pretrained video diffusion transformer (Wan2.2-TI2V-5B)** into an **autoregressive video-action policy** for end-to-end driving. Video and action streams are organized into a unified temporal token sequence and trained under a **joint flow-matching objective**: the model first generates the next 4-second video chunk, then decodes the ego action chunk conditioned on that generated future latent — an **inverse-dynamics readout** of imagined world evolution. Two additions address the video backbone's blind spots: (1) **scene-evolving driving guidance** — a *frozen* Qwen3-VL-8B produces fresh, chunk-specific two-sentence semantic intent from causally available context, injected via temporally localized cross-attention; and (2) **selective KV memory** — a training-free, inference-time cache policy (FlowCache-style relevance-redundancy scoring) that keeps bounded, modality-separated video/action memory pools for long-horizon rollout, cutting 300-second-rollout KV memory and attention FLOPs by over 12× vs. full caching.

Results: **90.1 PDMS on NAVSIM v1** with a single front camera, and **0.47/1.35 ADE/FDE@3s, 0.83/2.47 ADE/FDE@4s** on a curated 1,000-clip test subset of the [[concepts/physicalai-av-benchmark.md]] — substantially ahead of VaVAM and Alpamayo-1.5 under the paper's protocol. A data-scaling study (4k → 20k → 100k clips, fixed 50k iterations) shows monotone improvement with no saturation.

---

## Core Idea: Video Generative Model as the Policy Core

![[pipeline2.png|DriveWAM overview: pretrained video generation backbone adapted into a unified video-action policy]]

**Figure 1**: Overview. A pretrained video generation backbone becomes a unified video-action policy; a frozen VLM provides chunk-specific scene-evolving guidance; selective KV memory preserves compact prediction-relevant history for long-horizon rollout.

The positioning argument: VLA policies inherit semantic reasoning from VLMs pretrained on static image-text pairs, but must learn temporal dynamics (spatial layout, motion continuity, near-future scene evolution) from downstream driving data. Video generative models are pretrained on exactly those dynamics. Prior driving work uses video generation as an auxiliary signal on top of a VLM-centric policy ([[sources/futuresightdrive.md]], [[sources/drivevla-w0.md]], [[sources/drivedreamer-policy.md]]); prior driving world-action models rely on separate planners, discrete tokenizers, or custom architectures (WorldDrive, VaViM/VaVAM, [[sources/epona.md]]). DriveWAM instead makes a modern video foundation model the policy backbone itself, and delegates high-level semantics to a frozen VLM guide.

---

## Method

### 3.1 Autoregressive Video-Action Generation

A driving clip is divided into $K$ consecutive chunks. At decision step $k$, the model predicts the next video-action chunk $(x_{k+1}, a_{k+1})$ given history $H_k$, ego state $e_k$ (velocity, acceleration, curvature at the chunk's end frame), and text guidance $g_k$.

**Tokenization**: video chunks are encoded by the pretrained Wan VAE; action chunks (normalized ego-frame translation + yaw increments) by an MLP encoder $E_a$:

$$z_k = \mathrm{VAE}(x_k), \qquad u_k = E_a(a_k), \qquad H_k = \{(z_i, u_i)\}_{i \le k}$$

**World-action flow** (rectified flow, $\tau=1$ noise → $\tau=0$ clean):
- Video branch predicts the velocity of the next video latent: $\hat{v}^{z}_{k+1,\tau} = T_\omega(z_{k+1,\tau}; H_k, e_k, g_k, \tau)$
- Action branch is an **inverse-dynamics flow on the same shared transformer**, conditioned on the future world latent: $\hat{v}^{a}_{k+1,\tau} = D_a(T_\omega(u_{k+1,\tau}; \tilde{z}_{k+1}, H_k, e_k, g_k, \tau))$

where $\tilde{z}_{k+1}$ is the clean future latent during teacher-forced training and the *generated* latent $\hat{z}_{k+1}$ at inference (noisy-history augmentation reduces this train-test mismatch). The ego state is injected through a separate cross-attention branch.

**Joint objective**:

$$\mathcal{L} = \mathbb{E}_{k,\tau}\left[\|\hat{v}^z_{k+1,\tau} - v^z_{k+1,\tau}\|_2^2 + \beta_a \|\hat{v}^a_{k+1,\tau} - v^a_{k+1,\tau}\|_2^2\right], \qquad \beta_a = 1.0$$

The video term preserves the pretrained spatio-temporal prior during policy adaptation; the action term teaches the backbone to decode that prior into ego motion.

**Full-clip training / chunked rollout**: all chunks of a clip are denoised in parallel under a causal teacher-forcing mask (one forward pass); at inference the model rolls out one chunk at a time — sample $\hat{z}_{k+1}$, then sample $\hat{a}_{k+1}$ conditioned on it, then append the next *real* observation to history.

### 3.2 Scene-Evolving Driving Guidance

A video model captures near-future dynamics but not decision-level semantics (route intent, right-of-way). Existing WA methods use a single clip-level text condition; DriveWAM queries a **frozen Qwen3-VL-8B once per decision step** with only causally available inputs — latest observation $x_k$, recent ego trajectory $a_k$, route command $c_k$ — producing a concise two-sentence guidance $g_k = \Phi_{\mathrm{VLM}}(x_k, a_k, c_k)$ for the upcoming 4-second horizon (proceed / yield / stop / merge, etc.). No observation from the target chunk is used, so no future leakage from the VLM path.

**Temporally localized injection**: a block-diagonal text mask restricts chunk $k{+}1$'s video-action tokens to attend *only* to $g_k$'s tokens — preventing cross-chunk (including future-guidance) leakage.

![[kv_cache_vis.png|DriveWAM training attention mask]]

**Figure 2**: Training attention mask — colored entries are allowed attention; the block-diagonal text mask keeps guidance temporally localized.

**Guidance pipeline** (Appendix B): each chunk gets a route command from {straight, left, right}, derived from the ego yaw change over that chunk (>15° left, <−15° right) — a labeling construction, since explicit route annotations are unavailable. The VLM prompt receives the route command, the last front-camera frame, and a BEV visualization of the previous chunk's ego trajectory, and must output exactly two present-tense sentences (<50 words): road context, then qualitative ego guidance (no numbers/distances/coordinates). Guidance texts are precomputed for training; at inference the VLM is queried once per chunk and reused across denoising steps.

![[appendix_guidance.png|Examples of scene-evolving VLM guidance]]

**Figure 7**: Guidance adapts to changing scene context and route intent (pedestrians, traffic lights, construction barriers).

### 3.3 Selective KV Memory

Full KV caching grows linearly with rollout length; sliding-window/FIFO eviction discards old-but-critical evidence (occluded pedestrian, motion trend of a nearby vehicle) while keeping repeated static background. DriveWAM adapts FlowCache's relevance-redundancy criterion, **training-free and inference-only**:

- **Modality-aware pools**: separate bounded pools $H^v_k$ (video, budget $B^v$) and $H^a_k$ (action, budget $B^a$) — a single global cache would be dominated by numerous video tokens and under-preserve ego-motion history.
- **Retention score**: relevance $\rho^m_j$ = average attention mass from current queries to cached token $j$; redundancy $\eta^m_j$ = mean cosine similarity of its key to other cached keys; retain by $s^m_j = \lambda\rho^m_j - (1-\lambda)\eta^m_j$ (λ = 0.07). Repeated road surface/sky/buildings get filtered; moving vehicles and lane geometry are retained (Figure 3 — *image not present in the raw clipping*).
- **Update rule**: after chunk $k{+}1$, keep the top-scored history tokens to make room for the new KVs: $H^m_{k+1} \leftarrow \mathrm{Top}_{B^m - |\Delta H^m_{k+1}|}(H^m_k) \cup \Delta H^m_{k+1}$.

Cache capacities: 448 video tokens, 160 action tokens.

---

## Results

### NAVSIM v1 (Table 1)

∗ = with imitation learning; † = trained with multiple trajectory anchors (Hydra-MDP); MV/SV = multi/single-view cameras; L = LiDAR.

| Method | Ref | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Human | – | – | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| UniAD | CVPR'23 | MV | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| TransFuser | TPAMI'23 | MV & L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| PARA-Drive | CVPR'24 | MV | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LAW | ICLR'25 | SV | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| DiffusionDrive | CVPR'25 | MV & L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE | ICCV'25 | MV & L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| *VLA-based:* | | | | | | | | |
| ReCogDrive∗ | ICLR'26 | MV | 98.1 | 94.7 | 94.2 | 100.0 | 80.9 | 86.5 |
| DriveVLA-W0 | ICLR'26 | SV | 98.7 | 96.2 | 95.5 | 100.0 | 82.2 | 88.4 |
| AutoVLA | NeurIPS'25 | MV | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveDreamer-Policy | arXiv'26 | MV | 98.4 | 97.1 | 95.1 | 100.0 | 83.5 | 89.2 |
| DriveVLA-W0† | ICLR'26 | SV | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| *WA-based:* | | | | | | | | |
| Epona | ICCV'25 | SV | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| WorldDrive | arXiv'26 | SV | 98.4 | 95.8 | 95.2 | 99.8 | 83.3 | 89.0 |
| **DriveWAM** | – | SV | 98.3 | **98.1** | 95.2 | **100.0** | **84.3** | **90.1** |

Single front camera. Highest DAC (98.1) and EP (84.3) among single-sample entries in its table; the only entry above it, DriveVLA-W0† (90.2), uses trajectory anchors (multi-candidate selection, not single-sample).

### PhysicalAI-Autonomous-Vehicles (Table 2, curated 1,000-clip test subset)

∗ = evaluated with released checkpoint, which only supports up to 3s prediction.

| Method | Source | Sensors | # Params | ADE@3s ↓ | FDE@3s ↓ | ADE@4s ↓ | FDE@4s ↓ |
|---|---|---|---|---:|---:|---:|---:|
| VaVAM∗ | Valeo | SV | 1.3B | 2.31 | 4.32 | – | – |
| Alpamayo-1.5 | NVIDIA | SV | 10B | 0.80 | 2.31 | 1.44 | 4.18 |
| **DriveWAM** | – | SV | 5B + 8B | **0.47** | **1.35** | **0.83** | **2.47** |

Alpamayo-1.5 was trained on ~80,000 hours *including* the PhysicalAI-AV training set; VaVAM on ~1,700 hours of OpenDV. DriveWAM roughly halves both ADE and FDE vs. Alpamayo-1.5 at both horizons.

![[result_vis.png|Qualitative results on NAVSIM and PhysicalAI-AV]]

**Figure 4**: Jointly generated future scenes and ego trajectories on NAVSIM (left) and PhysicalAI-AV (right).

![[more_results.png|Additional qualitative results]]

**Figure 8**: NAVSIM (top rows: BEV map with predicted red vs. GT blue trajectory + generated frames at T=1–4s) and PhysicalAI-AV (bottom rows: GT/predicted trajectories overlaid on the front view).

---

## Ablations

All on PhysicalAI-AV, 100k clips / 50k iterations unless noted.

### Scene-evolving guidance × data scale (Table 3)

✗ = fixed global prompt as text conditioning.

| # Clips | # Iters | SE Guidance | ADE@4s ↓ | FDE@4s ↓ |
|---|---|---|---:|---:|
| 4k | 50k | ✗ | 1.21 | 3.65 |
| 4k | 50k | ✓ | 1.01 | 2.95 |
| 20k | 50k | ✗ | 0.95 | 2.94 |
| 20k | 50k | ✓ | 0.94 | 2.65 |
| 100k | 50k | ✗ | 0.92 | 2.75 |
| 100k | 50k | ✓ | **0.83** | **2.47** |

Chunk-specific guidance helps at every scale and the benefit does **not** vanish with more data (ADE −0.20 at 4k, −0.09 at 100k; FDE −0.70 → −0.28). This doubles as the **data-scaling study** (Figure 5, *image not in raw clipping*): monotone gains 4k → 100k regardless of guidance — not yet saturated.

### Video backbone initialization & joint video supervision (Table 4)

| Pretrained init. | Video sup. | ADE@4s ↓ | FDE@4s ↓ |
|---|---|---:|---:|
| ✗ | ✓ | 1.10 | 3.26 |
| ✓ | ✗ | 1.23 | 3.79 |
| ✓ | ✓ | **0.83** | **2.47** |

Both matter, and **action-only adaptation of a pretrained backbone is the worst configuration** (1.23/3.79 — worse than training from scratch with video supervision): dropping the video flow-matching term destroys the generative priors WA policy learning depends on. Mirrors [[sources/driveva.md]]'s +19.5 PDMS video-supervision ablation on the same backbone family.

### KV memory strategies (Table 5)

ADE/FDE on 20s clips; memory and GFLOPs profiled on a 300s rollout (KV summed over all DiT layers; attention GFLOPs of one causal self-attention layer per step).

| KV memory | ADE@4s ↓ | FDE@4s ↓ | Mem. (GB) ↓ | GFLOPs ↓ |
|---|---:|---:|---:|---:|
| Full | **0.83** | **2.47** | 3.07 | 17.37 |
| FIFO | 1.40 | 3.47 | **0.25** | **1.05** |
| Selective | 0.89 | 2.52 | **0.25** | 1.44 |

Selective memory nearly matches full caching (0.89 vs. 0.83 ADE) at the same budget where FIFO collapses (1.40), with >12× memory/FLOPs reduction vs. full caching.

### Per-chunk inference cost (Table 6, single H20 GPU)

∗ = action denoising steps reduced 10 → 5.

| Method | VLM (ms) | Video Gen (ms) | Action (ms) | ADE@4s ↓ | FDE@4s ↓ |
|---|---:|---:|---:|---:|---:|
| Alpamayo-1.5 | 570 | — | 330 | 1.44 | 4.18 |
| DriveWAM | 125 | 372 | 765 | 0.83 | 2.47 |
| DriveWAM∗ | 125 | 372 | 374 | 0.84 | 2.45 |

The 5-step-action variant totals ~871 ms per 4-second chunk — comparable to Alpamayo-1.5's ~900 ms — while additionally producing the generated future video. VLM guidance is cheap (125 ms, amortized per chunk, vs. Alpamayo's 570 ms per query).

---

## Dataset Curation (Appendix A)

PhysicalAI-AV: ~1,700 h, 306,152 clips of 20 s (153,625 train / 90,928 val / 61,599 test). DriveWAM tags every clip with frozen Qwen3-VL-8B over 20 uniformly sampled front-view frames using four structured prompts (scene attributes; vulnerable-road-user events; vehicle-interaction events; intersection/long-tail events), then computes a rule-weighted scalar **interest score** (accident scene 5.0, occluded pedestrian popout 4.0, animal on road 3.5, traffic-police gesture 3.0; common attributes 0.5–1.5).

- **Training subset (100k)**: keep all clips with score ≥ 2.0; uniformly sample 50% of the rest. 20k/4k subsets sampled from it for scaling.
- **Test subset (1,000)**: rare-event clips (tags in <1% of test clips, up to 30 top-scoring each) + high-interest clips (>75th percentile, quota-balanced by weather/lighting/road type) + 200 uniformly sampled common-scene controls.

![[appendix_tag.png|Scene tagging examples for dataset curation]]

**Figure 6**: Tagging examples — high-score clips capture rare/interaction-rich scenarios; low-score clips are ordinary driving.

---

## Implementation Details

- **Backbone**: Wan2.2-TI2V-5B video DiT, initialized from the *Causal World Modeling for Robot Control* base checkpoint (the paper's code framework); full DiT fine-tuned together with new action/ego-state modules. Action encoder/decoder: MLPs, hidden 3072.
- **Guidance VLM**: frozen Qwen3-VL-8B, one query per 4s chunk (vLLM).
- **Training**: 256×448 resolution, 48× NVIDIA H20; AdamW β=(0.9, 0.95), wd 0.1, lr 1e-5, per-device batch 1; β_a = 1.0. NAVSIM: 100k iters (lr ×0.5 at 50k/70k/90k), single-chunk setting (current frame → 4s horizon at 1 Hz). PhysicalAI-AV: 50k iters, 12s segments cropped from 20s clips; video at 1 Hz, actions at 10 Hz.
- **Inference**: Euler ODE — 3 steps for video (τ 1 → 0.6), 10 (or 5) steps for action (τ 1 → 0). Selective KV: λ = 0.07, budgets 448 video / 160 action tokens.

---

## Limitations

1. **NAVSIM comparison scope.** Table 1's 90.1 PDMS is strong for a single-camera WA model, but the comparison set omits the wiki's frontier: CLEAR (93.7), DriveSuprim (93.5), HybridDriveVLA (92.1), DynVLA/Reasoning-VLA (91.7), FLARE (91.4), DiffusionDriveV2 (91.2), WAM-Diff/ELF-VLA (91.0), [[sources/driveva.md]] (90.9), DriveFine (90.7). "Outperforming all competing methods under comparable training settings" is scoped to its own table — where anchor-based DriveVLA-W0† (90.2) still edges it. See [[concepts/navsim-benchmark.md]].
2. **DriveVA is conspicuously absent.** [[sources/driveva.md]] uses the *same* Wan2.2-TI2V-5B backbone with a joint video-action DiT target and reports 90.9 PDMS — higher than DriveWAM's 90.1 — yet is not cited or compared. The two papers are near-simultaneous tests of the same thesis with different coupling designs (joint denoising + sliding window vs. chunked AR + inverse dynamics + VLM guidance + selective memory); neither cross-compares.
3. **Self-curated test subset.** The PhysicalAI-AV 1,000-clip test set is constructed by the authors' own VLM tagging + interest-score pipeline; baselines were not involved in its design. VaVAM is evaluated via a released checkpoint capped at 3s; Alpamayo-1.5 is evaluated under a single-trajectory front-camera protocol that may not match its native multi-hypothesis setting. Favorable-selection risk is real even if the curation is principled.
4. **Open-loop metrics only on the large benchmark.** PhysicalAI-AV results are ADE/FDE — open-loop imitation metrics ([[concepts/nuscenes-waymo-evals.md]] caveats apply); NAVSIM is non-reactive. No closed-loop reactive evaluation (Bench2Drive, HUGSIM) anywhere.
5. **Selective KV memory accuracy is unverified at long horizon.** Accuracy is measured on 20s clips while memory/FLOPs are profiled at 300s; there is no accuracy evaluation in the 300s regime the mechanism exists for. Training uses full-history attention; the bounded memory is a train-test mismatch by construction (0.83 → 0.89 ADE degradation already at 20s).
6. **Route command derives from future ego yaw.** The {straight, left, right} command for the upcoming chunk is computed from the GT yaw change *over that chunk* — a navigation-intent proxy that leaks coarse directional future at training/eval time. Deployment would need a real router; the paper argues (reasonably) it mirrors standard navigation-command conditioning.
7. **System size and latency.** 5B DiT + 8B frozen VLM; ~871 ms–1.26 s per 4 s chunk on an H20. Amortized this is fine, but chunk-boundary latency spikes and the 13B total footprint are non-trivial for onboard deployment.
8. **No RL/RFT stage.** Pure flow-matching imitation; most NAVSIM-era peers add GRPO-style refinement.
9. **Guidance ablation baseline is coarse.** SE guidance is compared only against a *fixed global prompt* — it does not isolate per-chunk freshness vs. VLM quality (e.g., per-clip VLM guidance is untested).
10. **Extraction gaps.** Figure 3 (KV retention visualization) and Figure 5 (data-scaling plot) images are absent from the raw clipping; the References section is empty (footnotes only); the body text cites "Table 5" for both the backbone ablation (labeled Table 4) and the KV ablation — an off-by-one labeling quirk in the source.

---

## Key Cross-References

- **World-action pattern**: [[concepts/world-model-for-ad.md]] — DriveWAM vs. DriveVA is the wiki's cleanest controlled contrast of two ways to turn the same video backbone into a policy; also contrast with Epona (custom AR+diffusion, no pretrained video prior) and DriveVLA-W0 (world model training-time only).
- **NAVSIM standing**: [[concepts/navsim-benchmark.md]] — 90.1 PDMS, single camera, below wiki frontier; comparison-scope caveat.
- **PhysicalAI-AV benchmark**: [[concepts/physicalai-av-benchmark.md]] — first wiki source to report on the public release; supersedes the "internal evals only" framing in [[sources/alpamayo-r1.md]].
- **Backbone roles**: [[concepts/foundation-backbones-for-ad.md]] — Wan2.2-TI2V-5B as *the* policy core (second wiki use after DriveVA); frozen Qwen3-VL-8B in a new "inference-time semantic guide" role (also dataset tagger).
- **Frozen-VLM guidance vs. dual-system**: [[concepts/dual-system-vla.md]] — chunk-level free-text guidance via localized cross-attention is a third bridge type next to Senna-2's meta-actions and AutoMoT's layer-wise KV cache.
- **KV memory**: [[sources/automot.md]] (layer-wise shared KV cache for async execution) and [[sources/spanvla.md]] (sparse-KV action bridge) are the wiki's nearest KV-efficiency mechanisms; DriveWAM's is the only *content-based eviction* policy.
