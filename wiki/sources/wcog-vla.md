---
title: "WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/WCog-VLA_ A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/chain-of-thought-for-ad.md, concepts/diffusion-planner.md, concepts/perception-for-planning.md, concepts/rl-for-ad.md, concepts/foundation-backbones-for-ad.md, concepts/best-of-n.md, sources/sgdrive.md, sources/recogdrive.md, sources/autovla.md, sources/geoworldad.md, sources/adaptive-wam.md, sources/brainwam.md, sources/drivelaw.md, sources/da-wam.md, sources/orion.md, sources/drivevla-w0.md, sources/diffusiondrive.md, sources/wa-jepa.md, sources/driveva.md, sources/deepsight.md, sources/dynvla.md, sources/adathinkdrive.md]
created: 2026-09-04
updated: 2026-09-04
confidence: high
---

**Paper**: WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving
**Authors**: Xuerun Yan†, Zhexi Lian, Nuoheng Zhang, Shiyu Fang, Haoran Wang, Chen Lv†✉, Jia Hu✉, Binyang Song†✉ (†equal contribution, ✉corresponding)
**Orgs**: Tongji University + Nanyang Technological University
**arXiv**: 2607.08375v1

---

## Summary

WCog-VLA's world model predicts something no other entry in this wiki predicts: **the future trajectories of the surrounding agents, generated jointly with the ego's own**. Every world model tracked here forecasts pixels, video latents, semantic features, occupancy voxels, metric point maps, symbolic state, or an ego-action latent. This one forecasts *other cars' behaviour*, as a joint diffusion over $N_m$ agents.

Around that sits a two-level design: at the **semantic** level, agent tokens from a BEV/TrackFormer stack are injected into an InternVL3-2B VLM and decoded by a "world head" into current 3D boxes plus future agent trajectories; at the **generative** level, the **Aligned Decoupled Diffusion Transformer (ADDT)** synthesizes the joint multi-agent rollout. A third contribution is **Game-CoT**, an 85k-sample chain-of-thought dataset built around Stackelberg leader–follower reasoning.

**92.9 PDMS on NAVSIM v1** from a **2B** model — which would rank fourth in this wiki, above HybridDriveVLA's 92.1 and below Drive-JEPA's 93.3. That is a genuinely strong result for the parameter count.

Two things temper it. **The largest single ablation is not world cognition but reinforcement fine-tuning** (+3.6 PDMS, against +2.8 for the whole dual-level cognition stack). And the paper's own Table 5 contains the most brutal efficiency number for chain-of-thought in this wiki: **Game-CoT text reasoning costs 9.896 s and buys +0.5 PDMS** over answering directly.

---

## Positioning

![[ECCV_intro.png|Four paradigms for using a VLM in end-to-end driving: autoregressive action tokens, VLM-as-encoder with action decoder, fragmented world foresight, and WCog-VLA's dual-level world cognition]]

**Figure 1**: (a) VLA as autoregressive text/action-token generation; (b) VLM as cognitive encoder with a dedicated action decoder; (c) world cognition via VLM hidden states, treating world modeling as an auxiliary semantic task; (d) WCog-VLA, adding generative-level world evolution.

The paper's three stated complaints, and where each is tested:

1. **No 3D spatial awareness** — VLAs run on 2D image features. → Table 7 (+3.3 PDMS from the 3D perception module).
2. **Insufficient world cognition** — methods that use VLM hidden states for world modeling (it names SGDrive and UniDrive-WM) treat it as auxiliary semantic supervision and never synthesize *joint interactive* futures. → Table 4.
3. **No strategic social reasoning** — existing CoT is static scene description, lacking "if-what" game-theoretic imagination. → the Game-CoT dataset and Table 6.

The critique of "fragmented world foresight" is the interesting one and it is aimed squarely at the wiki's existing world-model roster: whether the future is predicted as images ([[sources/drivevla-w0.md]]) or as semantics ([[sources/sgdrive.md]]), it is a *perceptual* task about the scene, not a *behavioural* one about the other agents.

---

## Method

![[ECCV_Frame.png|WCog-VLA: VLM backbone with vision, text, and agent tokens feeding Game-CoT reasoning and a world head, coupled to the ADDT generative world model]]

**Figure 2**: The VLM integrates vision, text, and agent tokens to perform Game-CoT reasoning and semantic world forecasting; ADDT translates these representations into joint multi-agent trajectories.

### Semantic level

**Inputs**: six surround-view images $\mathcal{I}$, a navigation instruction $l_{\text{ins}}$, and ego state $\mathcal{S}=\{v, a, \mathcal{T}_{\text{hist}}\}$ with a 2 s history at 2 Hz. Backbone is **InternVL3-2B** (300M InternViT + Qwen2.5 LLM).

**3D spatial perception**: multi-view features are lifted to BEV by an off-the-shelf **BEVFormer** encoder, then a **UniAD-style TrackFormer** cross-attends learnable agent queries against $\mathcal{F}_{\text{BEV}}$ to produce $N_a$ sparse agent tokens.

**Role-decoupled hidden states.** Vision, text, and agent tokens are concatenated into the LLM, and the outputs are split by function:

$$O_{\text{vision}},O_{\text{text}},O_{\text{agent}}=\text{LLM}([\mathcal{T}_{\text{vision}},\mathcal{T}_{\text{text}},\mathcal{T}_{\text{agent}}])$$

$O_{\text{agent}}$ routes to a **world head** producing current 3D boxes *and* future trajectories for each surrounding agent; $O_{\text{vision}}$ and $O_{\text{text}}$ route to the language head for Game-CoT text. This is a cleaner separation than the wiki's other hidden-state world models, which read one pooled state for everything.

### ADDT: the generative world model

![[ADDT.png|ADDT architecture: a condition encoder with representation alignment to a VAE trajectory latent, and a generation decoder]]

**Figure 3**: A decoupled condition encoder and generation decoder, with representation alignment applied at an intermediate encoder block.

The motivating claim is an **optimization dilemma** in single-network DiTs: encoding low-frequency abstract semantics conflicts with decoding high-frequency continuous detail. In driving this is "modelling complex multi-agent interactions" vs. "generating precise trajectories."

**Condition encoder** ($N_1$ = 8 DiT blocks). Joint multi-agent action noise $x_t \in \mathbb{R}^{N_m \times H \times 3}$ is embedded and concatenated with history and pooled VLM features:

$$z_{t}=\text{Encoder}(F_{at},t,S,F_{\text{VLM}}),\qquad F_{at}=\text{concat}\bigl(E_{act}(x_{t}),\,E_{his}(\tau_{\text{his}}),\,\bar{F}_{\text{VLM}}\bigr)$$

Diffusion timestep $t$ and ego state $S$ enter via AdaLN; the full VLM token sequence enters via cross-attention.

**Representation alignment.** The 6th encoder block's feature $h_i$ is pulled toward a latent scene representation $r_*$ from a **GenAD-style VAE pretrained to reconstruct multi-agent trajectories** (MLP encoder, GRU decoder):

$$\mathcal{L}_{\text{align}}=1-\cos\bigl(r_{*},\,h_{\phi}(h_{i})\bigr)$$

The stated purpose is not fidelity but **stability**: it "maintains the local consistency of $z_t$ across adjacent denoising timesteps," which is what lets the model run few denoising steps. This is a REPA-style alignment applied to a *trajectory* latent space rather than a visual one.

**Generation decoder** ($N_2$ = 8 DiT blocks) receives $t$ and $z_t$ via AdaLN and $F_{\text{VLM}}$ via cross-attention, and denoises. Training uses an **agent-specific weight mask** $\mathbf{W}$ with separate penalties $\alpha_{\text{ego}}$ and $\alpha_{\text{surr}}$ — ego accuracy is prioritized over surrounding agents.

### Game-CoT

An automated pipeline (Qwen3-VL-Plus) produces four sequential steps: **scene description → critical object analysis → game-theoretic reasoning → payoff evaluation**. The third step frames traffic as a **Stackelberg game** with the ego as leader and surrounding agents as followers, enumerating candidate ego actions and inferring follower reactions ("if-what" imagination). The fourth scores each hypothetical for safety and efficiency.

**Ground-truth actions are supplied as hints** to reduce hallucination — the annotator is asked to "reconstruct explicit causal chains linking observed scene contexts to final GT actions." 85k annotations on NAVSIM.

### Four-stage training

![[train_stage.png|Four-stage training: 3D perception pretraining, VLM SFT, ADDT SFT with the VLM frozen, and DiffGRPO reinforcement fine-tuning]]

**Figure 4**: Three SFT stages plus one RFT stage.

| Stage | What trains | Data | Schedule |
|---|---|---|---|
| 1 | BEV encoder + TrackFormer | NAVSIM | 1 epoch; focal + L1 |
| 2 | VLM + world heads | 158k open VQA, then 170K NAVSIM-tailored | 1 + 3 epochs; $\mathcal{L}_{\text{LM}} + \lambda\mathcal{L}_{\text{world}}$ |
| 3 | **ADDT only** (VLM frozen) | NAVSIM | 200 epochs, DDPM; $\mathcal{L}_{\text{diff}} + \lambda\mathcal{L}_{\text{align}}$ |
| 4 | ADDT via **DiffGRPO** | NAVSIM | 10 epochs, group size 6 |

The RL reward decouples the two agent classes: $r_i = r_{\text{PDMS}} - \lambda_{\text{surr}}\mathcal{L}_{\text{L1}}(\tau_{\text{surr}})$ — NAVSIM's PDMS for the ego, an L1 displacement penalty for the surrounding agents' forecasts. All training on **4× A100 40GB**.

VQA corpus: 158k from DriveLM, CODA-LM, LingoQA, nuScenes-QA, NuInstruct, and DriveGPT4, plus 85k trajectory-specific VQA and 85k Game-CoT.

---

## Results

### Table 1 — NAVSIM v1 navtest (PDMS), after all four stages

| Method | Img | LiDAR | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|:-:|:-:|---:|---:|---:|---:|---:|---:|
| Constant Velocity | | | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP | | | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| VADv2-𝒱₈₁₉₂ | ✓ | | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| DrivingGPT | ✓ | | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| UniAD | ✓ | | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| BevDrive | ✓ | ✓ | 97.7 | 92.5 | 92.9 | 100 | 78.7 | 83.8 |
| TransFuser | ✓ | ✓ | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | ✓ | | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA | ✓ | ✓ | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP-𝒱₈₁₉₂-W-EP | ✓ | ✓ | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| iPad | ✓ | | 98.6 | 98.3 | 94.9 | 100 | **88.0** | 91.7 |
| *VLM-based* | | | | | | | | |
| QwenVL2.5-8B† | ✓ | | 97.8 | 92.1 | 92.8 | 100 | 78.3 | 83.3 |
| InternVL3-8B† | ✓ | | 97.0 | 92.4 | 91.8 | 100 | 78.9 | 83.3 |
| ReCogDrive-2B | ✓ | | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| **AutoVLA-3B** | ✓ | | 99.1 | 97.1 | 97.1 | 100 | 87.6 | **92.1** ⚠ |
| LatentVLA-3B | ✓ | | 98.9 | 98.2 | 95.2 | 100 | **88.2** | 92.4 |
| **WCog-VLA-2B (ours)** | ✓ | | **99.4** | **98.8** | **98.5** | 100 | 87.1 | **92.9** |

**NC 99.4 and TTC 98.5 are the second-highest in this wiki** — behind WA-JEPA's NC 99.5 and DriveVA's TTC 98.7 respectively. DAC 98.8 trails only DriveVLA-W0's 99.1. This is a conspicuously safety-strong policy, which the paper attributes to anticipating surrounding-agent intent.

**⚠ The AutoVLA row is the oracle Best-of-6 number, unlabelled.** AutoVLA's published single-sample post-RFT score is **89.11**; **92.12 is its Best-of-N oracle selection** ([[sources/autovla.md]], Table 1). Presented in a table of single-sample results without a marker, it makes AutoVLA look 3 points stronger than it is — and the paper's claim of surpassing it "by at least 0.8 PDMS" is measured against an oracle. See [[concepts/best-of-n.md]].

**Two methods new to the wiki**: LatentVLA-3B (92.4) and BevDrive (83.8). iPad 91.7 matches [[sources/geoworldad.md]]'s value.

### Table 2 — NAVSIM v2 navtest (EPDMS), after **three-stage SFT only**

| Method | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VADv2 | 97.3 | 91.7 | 98.2 | **99.9** | 77.6 | 92.7 | 66.0 | **100** | 97.4 | 76.6 |
| TransFuser | 97.7 | 92.8 | 98.3 | **99.9** | 79.2 | 92.8 | 67.6 | **100** | 95.3 | 77.8 |
| HydraMDP++ | 97.9 | 96.5 | 98.9 | **100** | 79.2 | 93.4 | 67.2 | **100** | 97.7 | 80.6 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | **100** | **98.3** | 83.1 |
| ReCogDrive-8B | 98.3 | 95.2 | **99.5** | 99.8 | **87.1** | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| WoTE | 98.5 | **96.8** | 98.8 | 99.8 | 86.1 | 97.9 | 95.5 | 98.3 | 82.9 | 84.2 |
| DiffusionDrive | 98.0 | 96.0 | **99.5** | 99.8 | 87.7 | 97.1 | **97.2** | 98.3 | 87.6 | 84.3 |
| **WCog-VLA-2B (ours)** | **98.8** | 96.6 | 99.3 | 99.8 | 85.8 | **98.2** | 96.4 | 98.3 | 86.3 | **85.9** |

**Note that v1 and v2 report different models.** The 92.9 is after four stages including RFT; the 85.9 is after three-stage SFT with **no RFT** — the stage worth +3.6 PDMS on v1. The paper states this in the table caption but does not flag it as a caveat.

**The v2 baselines are a fourth distinct set and 85.9 cannot be placed** against this wiki's v2 table — see Limitations.

---

## Ablations

### Table 3 — The four stages

| ID | S1 (3D perc.) | S2 (VLM SFT) | S3 (ADDT) | S4 (RFT) | PDMS ↑ |
|---:|:-:|:-:|:-:|:-:|---:|
| 1 | | ✓ | | | 84.4 |
| 2 | ✓ | ✓ | | | 85.5 |
| 3 | ✓ | ✓ | ✓ | | 89.3 |
| 4 | ✓ | ✓ | ✓ | ✓ | **92.9** |

**RFT is +3.6 and ADDT is +3.8 — together 7.4 of the 8.5 total.** 3D perception pretraining contributes +1.1 here. The headline "world cognition" story is real but it is not what most of the number is made of: switching from textual trajectory output to a continuous diffusion head, and then applying RL, account for nearly all of it.

Rows 1–2 output trajectories as *text tokens* from the VLM; rows 3–4 output continuous actions through ADDT.

### Table 4 — Dual-level world cognition (all with three-stage SFT)

| ID | Semantic: Cur | Semantic: Fut | Generative | PDMS ↑ |
|---:|:-:|:-:|:-:|---:|
| 1 | | | | 86.5 |
| 2 | ✓ | | | 87.0 |
| 3 | | ✓ | | 87.2 |
| 4 | ✓ | ✓ | | 88.1 |
| 5 | | | ✓ | 87.4 |
| 6 | ✓ | ✓ | ✓ | **89.3** |

The whole dual-level stack is worth **+2.8** over a planner with neither level. Individually: current 3D perception supervision +0.5, future agent-trajectory supervision +0.7, joint multi-agent generation +0.9. Combined semantic +1.6; combined everything +2.8, which the paper reads as synergy (1.6 + 0.9 = 2.5 < 2.8).

**Row 5 is the one this wiki should note**: generative joint multi-agent synthesis *alone*, with no semantic world supervision, is worth **+0.9 PDMS**. That is a shared future — one joint rollout conditioning one ego plan — measured positive, over a *behavioural* rather than photometric target. See [[concepts/world-model-for-ad.md]].

### Table 5 — ADDT design, PDMS and inference time

| Method | Denoise steps | PDMS ↑ | Infer time (s) ↓ |
|---|---:|---:|---:|
| VLM text, no reasoning | – | 85.0 | 1.131 |
| **VLM text, with Game-CoT reasoning** | – | **85.5** | **9.896** |
| VLM + SDT | 5 | 87.4 | 0.105 |
| VLM + SDT | 20 | 88.5 | 0.388 |
| VLM + DDT (no alignment) | 5 | 87.9 | 0.108 |
| VLM + DDT | 20 | 88.7 | 0.381 |
| VLM + ADT (no decoupling) | 5 | 88.6 | 0.103 |
| VLM + ADT | 20 | 89.1 | 0.392 |
| **VLM + ADDT** | **5** | **89.3** | **0.106** |
| VLM + ADDT | 20 | 89.6 | 0.383 |

Three results worth extracting:

1. **Game-CoT text reasoning costs 9.896 s and buys +0.5 PDMS** over answering directly (85.0 → 85.5). That is the most damning latency-per-point figure for inference-time textual CoT in this wiki, and the paper reports it without comment. The deployed system does not use the text path at all.
2. **Decoupling and alignment each help, and both help more at fewer steps.** At 5 steps: SDT 87.4 → +alignment 87.9 → +decoupling 88.6 → both 89.3. At 20 steps the spread narrows to 88.5 → 89.6. The mechanisms are substitutes for denoising budget.
3. **5 → 20 denoising steps is worth +0.3 PDMS** for a ~3.6× latency increase. Another entry in the wiki's converging finding that iterating the denoiser buys almost nothing — see [[sources/brainwam.md]], [[sources/drivelaw.md]], [[sources/adaptive-wam.md]].

### Tables 6 and 7 — Data and perception

| ID | Traj VQA | Drive VQA | Game-CoT | PDMS ↑ |
|---:|:-:|:-:|:-:|---:|
| 1 | ✓ | | | 86.7 |
| 2 | ✓ | ✓ | | 88.2 |
| 3 | ✓ | | ✓ | 87.5 |
| 4 | ✓ | ✓ | ✓ | **89.3** |

| 3D perception | PDMS ↑ |
|---|---:|
| ✗ | 86.0 |
| ✓ | **89.3** |

**Game-CoT data is worth +0.8 alone and +1.1 on top of open driving VQA.** Since the deployed model never generates the reasoning at inference (Table 5), this is a **training-time-only** contribution — CoT as a data-shaping signal rather than an inference-time computation. The wiki's CoT page has several adaptive-CoT entries but no other case of CoT supervision being retained while the reasoning path is discarded entirely.

**The 3D perception module is worth +3.3** here, against +1.1 in Table 3. The two measure it at different points — Table 3 before ADDT exists, Table 7 with the full three-stage stack — and the paper does not reconcile them.

---

## Qualitative

![[compare_with_previous.png|Comparison against ReCogDrive on navtest: WCog-VLA changes lane past a slow bus where the baseline stays trapped]]

**Figure 5**: Against [[sources/recogdrive.md]] in a complex urban scene. ReCogDrive remains in the slow lane behind a leading bus; WCog-VLA identifies it and changes lane, matching human ground truth.

The text describes two further qualitative results — a left-turn-at-intersection case where the joint multi-agent forecast lets the ego commit instead of decelerating, and a visualization of the world head's 3D perception and agent-trajectory outputs — but **their figures are absent from the source clipping** (the text refers to "Fig. 7" three times with no Figures 6 or 7 present).

---

## Limitations

1. **The AutoVLA baseline is an oracle Best-of-6 result presented as a single-sample row.** 92.1 ≈ AutoVLA's 92.12 BoN; its single-sample post-RFT score is 89.11. Nothing marks the difference, and the paper's "surpasses ReCogDrive and AutoVLA by at least 0.8 PDMS" is measured against it.

2. **The v1 table omits the frontier.** CLEAR/DA-WAM 93.7, DriveSuprim 93.5, and Drive-JEPA 93.3 are all above 92.9; also absent are WA-JEPA 91.8, SimWAM 91.5, FLARE 91.4, and DiffusionDriveV2 91.2. 92.9 is a strong fourth-place result in this wiki, not SOTA. That said, **at 2B parameters it is the best PDMS-per-parameter entry tracked here**, and its comparison against ReCogDrive-2B (90.8) is genuinely like-for-like.

3. **The v2 baselines are a fourth distinct set and 85.9 cannot be placed.** TransFuser appears at **77.8** with submetrics matching neither the wiki's 76.7 row nor GeoWAM's 84.0 — and its NC/DAC/EP/TTC (97.7 / 92.8 / 79.2 / 92.8) are *identical to its own NAVSIM-v1 row in Table 1*, with LK 67.6 against 92.7 elsewhere. HydraMDP++ is 80.6 (wiki 84.1, BrainWAM 81.4 — a third value) and DiffusionDrive 84.3 (wiki 84.5). ARTEMIS carries HC 100 where every other source says 98.3.

4. **v1 and v2 are different models.** The 92.9 includes RFT; the 85.9 does not, and RFT is worth +3.6 on v1. Stated in a caption, never flagged.

5. **RFT is the largest single contribution (+3.6) and its reward is the benchmark's own metric.** DiffGRPO optimizes $r_{\text{PDMS}}$ directly, so the headline is partly a measure of how well the policy was fitted to the scorer — the same caveat this wiki applies to Hydra-MDP-style distillation and to [[sources/adaptive-wam.md]]'s simulator-labelled verifier.

6. **This is a heavily supervised system.** It needs 3D bounding boxes, per-agent future trajectories, and BEV supervision, plus 328k VQA/CoT samples. Not comparable to annotation-free lines like FLARE or SimWAM; closest peer is [[sources/sgdrive.md]], which is also InternVL3-2B and also requires 3D structure.

7. **Game-CoT annotations are post-hoc rationalizations of the ground truth.** GT actions are supplied as hints so the annotator "reconstructs explicit causal chains linking observed scene contexts to final GT actions." The paper is candid about this, but it means the game-theoretic traces are constructed to justify a known answer rather than derived independently — a concern for anyone treating Game-CoT as a reasoning benchmark.

8. **The latency column's scope is undefined.** ADDT at 0.106 s cannot include a forward pass through InternVL3-2B over six surround views, and it certainly excludes generating any Game-CoT text (9.896 s in the same table). So the deployed end-to-end latency is unreported, and the 10.7× speedup claim compares an action head against a full text-generation pipeline.

9. **Table 3 and Table 7 disagree on the value of 3D perception** (+1.1 vs +3.3) with no reconciliation.

10. **No navhard, Bench2Drive, HUGSIM, or nuScenes.** No seed variance, single runs. No ADDT parameter count. No code release mentioned. Figures 6–7 are referenced but absent from the source.

11. **The paper's own stated limitation**: semantic cognition covers agents only and omits the future evolution of road geometry and map topology.

---

## Key Cross-References

- **A new world-model state space**: [[concepts/world-model-for-ad.md]] — joint multi-agent *trajectories* join pixels, video latents, features, occupancy, metric point maps, symbolic state, and ego-action latents as prediction targets. It is the only entry whose forecast is about *other agents' behaviour* rather than the scene, and its Table 4 row 5 measures a shared behavioural future at **+0.9 PDMS** on its own.
- **Chain-of-thought's price, measured**: [[concepts/chain-of-thought-for-ad.md]] — 9.896 s for +0.5 PDMS is the sharpest efficiency datum this wiki has on inference-time textual CoT, and WCog-VLA's resolution (keep the CoT *data*, discard the CoT *path*) is a pattern the adaptive-CoT entries do not cover.
- **Denoising steps, fifth data point**: 5 → 20 steps is worth +0.3 PDMS, converging with [[sources/brainwam.md]] (1 step ≈ 3), [[sources/drivelaw.md]] (t=1 best), and [[sources/adaptive-wam.md]] (one conditional forward).
- **Same backbone, direct comparison**: [[sources/sgdrive.md]] is also InternVL3-2B with structured world supervision and DiffGRPO-family RFT, scoring 87.4 SFT / 91.1 RFT against WCog-VLA's 89.3 / 92.9. Both need 3D labels; the difference is agent-trajectory forecasting versus scene-agent-goal symbolic queries.
- **Best-of-N contamination**: [[concepts/best-of-n.md]] — the AutoVLA 92.1 row is the first case in this wiki of an oracle BoN score appearing unmarked in another paper's single-sample comparison table.
- **Protocol drift**: [[concepts/navsim-benchmark.md]] — a fourth distinct TransFuser v2 value (77.8), with submetrics apparently carried over from its v1 row.
- **REPA-style alignment for planning**: [[concepts/diffusion-planner.md]] — aligning a diffusion encoder's intermediate feature to a pretrained VAE *trajectory* latent, used to stabilize few-step denoising, is a mechanism no other planner here uses.
