---
title: World Models for Autonomous Driving
type: concept
sources: [raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md, raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md, raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md, raw/papers/Epona_ Autoregressive Diffusion World Model for Autonomous Driving.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md]
related: [sources/uniugp.md, sources/futuresightdrive.md, sources/drivedreamer-policy.md, sources/drivevla-w0.md, sources/flare.md, sources/dreameraD.md, sources/vega.md, sources/epona.md, sources/driveva.md, sources/explorevla.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, concepts/rl-for-ad.md]
created: 2026-04-05
updated: 2026-04-23
confidence: high
---

## What Is a World Model in AD?

A world model learns to predict the future state of the environment — most commonly by predicting future video frames — from historical observations and optionally an action or trajectory. In autonomous driving, this means predicting what the scene will look like in the next N seconds given the current camera stream and the ego vehicle's intended motion.

**Core hypothesis**: learning to predict future visual states forces a model to internalize causal relationships in the scene — who will turn where, which objects will move, how the scene evolves. This visual causal reasoning then transfers to better planning.

## Why World Models Are Useful for Planning

Standard imitation learning (behavior cloning) trains a planner to mimic recorded trajectories, but the model sees no explicit signal about **why** specific objects matter for the future. A world model provides exactly this:

- Forces attention to **causally relevant** distant objects (e.g., a car about to run a red light far ahead)
- Enables **counterfactual reasoning**: "if I turn left, the future video should look like X; if I go straight, like Y"
- Provides a training signal from **unlabeled video** (no trajectory annotation needed for generative pre-training)

**Empirical evidence from UniUGP**: removing the generation expert degrades planning L2 from 1.45→1.72. Qualitatively, with the world model the VLA focuses more on distant, causally relevant objects; without it, attention is more near-field and reactive.

## Architecture Patterns

### 1. Sequential / Cascaded World Model
The world model is a separate module that follows the planner. It receives the planned trajectory and generates future video conditioned on it, as a form of visual verification.

**Example: UniUGP Generation Expert**
- Understanding + Planning experts (MoT-coupled) run first
- Generation expert (Wan2.1 DiT) is cascaded: conditioned on understanding hidden states AND planned action embeddings
- At inference, the generation expert is optional (can be disabled on mobile)
- During training, future video prediction loss back-propagates into the shared understanding representation

### 2. Autoregressive World Model + Diffusion Planner
The world model predicts future states autoregressively; trajectory planning is coupled via a shared latent conditioned on the same history.

**Example: Epona** ([[sources/epona.md]])

Epona (ICCV 2025, 2.5B params) combines a GPT-style causal transformer with twin diffusion transformers to solve two simultaneous problems: long-horizon video generation and real-time trajectory planning.

**Core insight**: instead of modeling all future frames jointly (video diffusion) or tokenizing frames (GPT-AR), Epona decomposes the problem: a causal MST extracts a compact latent F from T history frames, and two specialized DiTs consume F in parallel — TrajDiT for trajectory, VisDiT for the next frame. Both optimized via rectified flow loss. The shared F is the key: it must be predictive of future visual states *and* useful for trajectory planning simultaneously.

**Three-module architecture**:

| Module | Params | Role |
|--------|--------|------|
| **MST** (Multimodal Spatiotemporal Transformer) | 1.3B | Interleaved causal temporal + spatial attention → compact latent F |
| **VisDiT** (Next-frame DiT) | 1.2B | Rectified flow over next frame, conditioned on F + action |
| **TrajDiT** (Trajectory DiT) | 50M | Rectified flow over 3s trajectory, conditioned on F |

**MST design**: interleaves `CausalTemporalLayer` (across T frames, causal mask) and `MultimodalSpatialLayer` (within each frame), with action tokens ($\Delta\theta$, $\Delta x$, $\Delta y$) concatenated to visual latent patches along the spatial dimension.

**Chain-of-forward training** (key innovation for long-horizon generation): standard teacher-forcing creates a training/inference distribution mismatch that compounds into autoregressive drift past ~10–20 seconds. Fix: every 10 steps, run 3 forward passes using self-predicted frames as history — where the self-prediction uses a cheap 1-step velocity estimate $\hat{x}_{(0)} = x_{(t)} + t \cdot v_\Theta(x_{(t)}, t)$ rather than full denoising. This exposes the model to its own errors during training. Result: stable generation at 120s / 600 frames (Vista: 15s, DrivingWorld: 40s).

**Why joint training matters for planning** (ablation): disabling VisDiT while keeping TrajDiT drops NAVSIM PDMS 86.2 → 78.1 (−8.1). The world model supervision forces F to encode richer scene dynamics than trajectory prediction alone can achieve.

**Inference modes**: VisDiT can be deactivated → MST + TrajDiT runs at 20 Hz (0.05s, real-time planning). Full generation (+ VisDiT, 100 steps) takes ~2.3s/frame.

**Results**: FVD 82.8 on NuScenes (SOTA at time of publication, −7.4% vs Vista 89.4); generation length 120s (8× Vista). NAVSIM v1 86.2 PDMS (camera-only, no auxiliary supervision); NuScenes avg L2 1.25m / avg collision 0.36% (front camera only, no annotations). Best 1s collision rate (0.01%) — traffic rules learned purely from next-frame prediction.

**Successor**: DreamerAD ([[sources/dreameraD.md]]) adds latent RL on top of Epona (SF-WM + AD-RM + vocabulary sampling) to reach 88.7 PDMS and 87.7 EPDMS — making Epona the strongest pure-SFT world model planning baseline in the wiki.

### 3. Tokenized World Model
Future states represented as discrete tokens; model predicts token sequences for both future video and trajectory.

**Example: GAIA-1**
- Next-token predictor + auxiliary diffusion image decoder
- World knowledge encoded in discrete token space

### 4. Occupancy World Model
Predicts 3D occupancy grids rather than video frames.

**Example: OccWorld**
- Codebook-based discrete occupancy prediction
- Less computationally expensive than video generation; loses appearance detail

### 5. Visual CoT as Planning Intermediate (FSDrive)

**FSDrive** ([[sources/futuresightdrive.md]]) introduces a fundamentally different role for the world model: the generated future frame is not for video verification or auxiliary training signal — it is the **reasoning intermediate** (Chain-of-Thought) that planning conditions on.

**Dual-role VLA**:
1. **World model**: autoregressively generates unified future frame (red lane dividers + 3D detection boxes overlaid) via VQ-VAE token prediction
2. **Inverse dynamics model**: plans trajectory from current observations + generated visual CoT

$$P(W_t \mid I_t, Q_{CoT}, opt(T_{com}, T_{ego}))$$

**Vocabulary expansion** (key mechanism): MoVQGAN VQ-VAE tokens appended to the MLLM text vocabulary — no architectural change. Activates generation with ~0.3% of data used by prior methods (Janus, VILA-U).

**Progressive generation** (pre-training enforces physical laws):

$$P(Q_f \mid Q_l, Q_d) = \prod_{t=1}^{h \cdot w} P_\theta(q_i \mid q_{<i}, Q_l, Q_d)$$

Lane dividers $Q_l$ → 3D detection $Q_d$ → full frame $Q_f$: static road structure first, then dynamic agent layout, then appearance.

**Key empirical finding**: the visual CoT primarily reduces *collision rate* (31% improvement) rather than L2 accuracy. Text CoT and image-text CoT show diminishing intermediate gains — the spatial and temporal structure of the unified image is what drives collision avoidance.

**Contrast with UniUGP**: UniUGP uses its generation expert as a training-time causal learning signal (optional at inference). FSDrive uses the generated frame as a mandatory inference-time reasoning step. Both improve planning by grounding it in future visual prediction, but through different mechanisms.

### 6. Geometry-Grounded Causal WAM (DriveDreamer-Policy)

**DriveDreamer-Policy** ([[sources/drivedreamer-policy.md]]) extends the WAM paradigm by adding **explicit depth generation** as a 3D geometric scaffold before video and action prediction. The motivation: 2D appearance-only world models lack geometric grounding for occlusion reasoning, free-space estimation, and distance-to-collision cues.

**Causal depth → video → action ordering** (single LLM forward pass):
- Depth queries process scene + LLM context first
- Video queries additionally attend to depth context → geometry-aware video generation
- Action queries attend to both depth and video context → geometry+dynamics-informed planning

All three outputs produced by separate **flow-matching generators** (depth = pixel-space DiT, video = Wan-2.1-T2V-1.3B adapted, action = standalone DiT), each conditioned on LLM embeddings via cross-attention.

**Modular design**: can run in planning-only mode (action generator only), or full generation mode (depth + video + action). Planning-only mode implicitly benefits from world context because the LLM processes world queries even when generators are off.

**Key empirical findings** (Table 4 ablation):

| Depth | Video | PDMS |
|---|---|---|
| ✗ | ✗ | 88.0 |
| ✓ | ✗ | 88.5 |
| ✗ | ✓ | 88.9 |
| ✓ | ✓ | **89.2** |

Depth and video provide complementary planning cues: geometry (free space, distance) vs. temporal dynamics (agent motion). Neither alone matches the combined benefit.

**Depth improves video coherence** (Table 5): FVD 65.82 → 53.59 (−18.6%) when depth is jointly learned. Depth acts as a 3D scaffold that constrains the video generator's spatial consistency.

**Contrast with FSDrive (Pattern 5)**: Both add geometric priors to visual CoT. FSDrive overlays lane dividers + 3D boxes on a single generated future frame; DDP generates a dedicated metric depth map as a separate modality. FSDrive's CoT is mandatory at inference; DDP's depth/video are modular. DDP does not use the generated output as reasoning text — the LLM embeddings carry the world context directly to the action generator.

### 7. Training-Time-Only World Modeling for Data Scaling (DriveVLA-W0)

**DriveVLA-W0** ([[sources/drivevla-w0.md]]) frames world modeling as a solution to the **"supervision deficit"**: standard VLA fine-tuning maps high-dimensional visual inputs to sparse low-dimensional waypoints, leaving most representational capacity idle and preventing scaling. Future image prediction provides dense per-pixel self-supervision at every timestep, forcing the model to learn environment dynamics.

**Key distinction from Patterns 1–6**: the world model is used **exclusively during training** and bypassed at inference. There is no inference-time visual reasoning benefit — the improvement comes entirely from richer representations learned during training.

**Two variants** for the two VLA paradigms:

| Paradigm | Backbone | World Model Type | Predicts | Loss |
|---|---|---|---|---|
| VQ (discrete tokens) | Emu3-8B | AR next-token prediction | **Current** frame tokens | Cross-entropy |
| ViT (continuous features) | Qwen2.5-VL-7B | Latent diffusion | **Future** frame $I_{t+1}$ | MSE denoising |

The ViT variant predicts the *future* (not current) to avoid pure reconstruction — conditioned on action features $F_t^A$ to learn causal consequences of actions.

**Cross-dataset generalization finding** (Table 7): action-only VLAs overfit the pretraining action distribution and **degrade** on NAVSIM after NuPlan pretraining (VLA-VQ: −9.5% PDMS). World model VLAs learn transferable visual representations and consistently benefit (+6.1% for VQ, +1.7% for ViT). This is the clearest evidence in the wiki that world model training provides a representation quality benefit beyond what action-only supervision achieves.

**Data scaling finding** (Table 3, proprietary 70M-frame in-house dataset): at 70M frames, action-only VLAs saturate while world model VLAs continue improving. At 70M: +28.8% ADE for VQ, +15.9% collision reduction for ViT vs. action-only baselines. At 70k frames, the world model VQ variant *hurts* slightly — the benefit requires sufficient data to manifest.

**FID ↔ PDMS correlation**: 6VA (FID 4.6 → PDMS 85.6) outperforms 2VA (FID 9.8 → PDMS 84.1) — better generation fidelity links to better planning. (Only 2 data points; treat as directional evidence, not strong proof.)

**Comparison with UniUGP (Pattern 1)**: both use world modeling as training-time signal that improves planning representations. UniUGP's generation expert is optionally available at inference; DriveVLA-W0's world model is strictly training-time. UniUGP provides the world model signal via video consistency loss on annotated data; DriveVLA-W0 uses raw future frame prediction on unlabeled driving video — more scalable but less structured.

### 8. Semantic Feature Prediction as Self-Supervised Objective (FLARE)

**FLARE** ([[sources/flare.md]]) introduces a distinctly different approach: instead of generating future video frames (patterns 1–7), it predicts the DINOv2 **semantic patch features** of the next frame as an auxiliary loss. This bypasses pixel-level reconstruction entirely while still forcing the model to internalize scene dynamics.

**Core motivation**: predicting semantic features forces the model to learn object permanence and motion logic while remaining invariant to nuisance factors (lighting, appearance noise). Unlike pixel prediction, semantic feature prediction focuses supervision on the scene structure relevant to planning.

**Action-conditional future prediction** (key design): the Future Feature Predictor (FFP) is conditioned on the action decision vector **z** (produced by the MAP fusion module). This means the predictor must simulate *how a specific planned action changes the scene* — not just predict the general future. The FFP predicts:

$$\hat{\mathbf{F}} \in \mathbb{R}^{N_p \times d_f}$$

using spatial queries modulated by **z** via cross-attention over visual latents.

**Training objective**:
$$\mathcal{L}_\text{future} = \|\hat{\mathbf{F}} - \mathbf{F}_\text{gt}\|_1 + \alpha\left(1 - \frac{1}{N_p}\sum_j \text{CosSim}(\hat{\mathbf{F}}_j, \mathbf{F}_{\text{gt},j})\right)$$

Combined L1 for magnitude + cosine for directional (semantic) alignment.

**Prediction target ablation** (Table 3, NAVSIM SFT PDMS):

| Target | PDMS | Δ vs. none |
|--------|------|-----------|
| None (pure trajectory) | 83.4 | — |
| Image pixels | 84.7 | +1.3 |
| Global DINO feature | 85.9 | +2.5 |
| **Spatial DINO patches** | **86.9** | **+3.5** |

Spatial granularity matters: global DINO captures overall scene semantics but loses spatial structure that informs lane and obstacle positions. Spatial DINO preserves both.

**Result**: 86.9 PDMS SFT (best VLM-based SFT on NAVSIM-v1, 1 camera, no external pretraining); 91.4 PDMS after GRPO RFT (best single-sample VLM-based).

**Contrast with DriveVLA-W0 (Pattern 7)**:
| Aspect | DriveVLA-W0 | FLARE |
|--------|------------|-------|
| Prediction target | Pixel-level VAE latents | DINOv2 semantic patches |
| World model at inference | ✗ (training-time only) | ✗ (auxiliary loss only) |
| Language annotations needed | ✗ | ✗ |
| Single-sample PDMS | 88.4 (query-based expert) | 91.4 (RFT) |
| Dataset | In-house 70M frames (claimed) | NAVSIM navtrain (103K) |

Both avoid pixel-level generation overhead by predicting intermediate representations. DriveVLA-W0 predicts the full future frame via VAE latents; FLARE predicts the semantic token layout only.

**Contrast with FSDrive (Pattern 5)**: FSDrive generates a full visual CoT frame at inference time (mandatory), conditioning the planner on it. FLARE uses future prediction purely as an auxiliary training signal — no generation at inference.

### 9. Latent World Model as RL Reward Source (DreamerAD)

**DreamerAD** ([[sources/dreameraD.md]]) answers the open question "can a world model provide a reward signal for RL?" with a definitive yes — and does so without pixel-level generation at RL training time.

**Key insight**: denoised latent features from a Video DiT (Epona's flow-matching model) exhibit strong spatial and semantic coherence (confirmed via PCA), meaning these latent representations are rich enough to learn a reward model *without ever decoding to pixels*.

**The latent RL cycle**:
1. Candidate trajectories sampled from Gaussian-filtered vocabulary (physically constrained)
2. Shortcut-forced world model (1-step inference, 0.03s/frame) predicts future latent states $\hat{z}_{1:T}$ conditioned on each candidate trajectory
3. Autoregressive Dense Reward Model (AD-RM) scores each latent sequence → 8 reward dimensions × 8 time horizons
4. GRPO policy optimization over reward advantage estimates

**Contrast with all previous patterns**:
| Pattern | World model used at... | RL reward from... |
|---------|----------------------|------------------|
| 1–6 (UniUGP, FSDrive, DDP) | Training + inference (some optional) | Not used for RL |
| 7 (DriveVLA-W0) | Training only | Not used for RL |
| 8 (FLARE) | Training auxiliary loss only | Not used for RL |
| **9 (DreamerAD)** | **RL rollout (latent only)** | **Latent AD-RM (no simulator at RL time)** |

**Shortcut Forcing** is the enabling mechanism: compress Epona's 100-step diffusion to 1-step via recursive teacher-student distillation over power-of-2 step sizes. Performance is unchanged (87.7 EPDMS at 1-step = 16-step).

**AD-RM data efficiency**: 20% of labeled trajectories achieves 97% of full-data reward model performance — latent features are highly structured and reward learning converges rapidly.

**Limitations relative to simulator-based RL**: AD-RM rewards are learned approximations; edge-case behaviors outside the training distribution may produce unreliable rewards. Also, vocabulary constraint (256 trajectories) limits exploration breadth compared to free-form GRPO.

**Results**: 87.7 EPDMS (NAVSIM-v2), strongest safety gains among world-model methods (DAC +1.5, NC +0.9, TTC +1.1 over Epona). 80× faster RL rollouts than pixel-level diffusion world model baselines.

### 10. Instruction-Conditioned World Model for Open-Ended NL Driving (Vega)

**Vega** ([[sources/vega.md]]) extends world modeling to a new role: **dense supervision signal that bridges the instruction-to-action gap** — enabling open-ended natural language instruction following for the first time in the wiki.

**Core motivation**: a baseline VLA (Qwen2.5-VL + planning head) trained on 100K instruction-annotated scenes achieves only ~60 PDMS. Sparse trajectory supervision cannot ground high-dimensional instruction+visual inputs to low-dimensional actions. World modeling (future frame prediction) provides the missing dense signal:

| Training setting | PDMS | EPDMS |
|-----------------|------|-------|
| Action only | 51.8 | 48.9 |
| Random future frame | 77.3 | 75.2 |
| **Next frame (default)** | **77.9** | **76.0** |

Note: exact choice of future frame matters little — the task structure is what helps. This generalizes DriveVLA-W0's insight (Pattern 7) to the instruction-following domain.

**Unique contribution vs. Patterns 1–9**: all prior patterns use world modeling to improve *imitation* planning — they condition on expert trajectories. Vega conditions the world model on **user-specified instructions** → the generated future image must be *instruction-consistent*, not just expert-consistent. This enables multi-trajectory generation: same scene + different instructions → different valid trajectories + different future images.

**Architecture**: Integrated AR+Diffusion transformer with Mixture-of-Transformers (MoT) — all parameters (attention + FFN) duplicated for understanding vs. generation, initialized from Bagel-7B. AR pipeline (Qwen2.5 backbone) handles visual+text; diffusion pipeline generates future image and trajectory. A lightweight action expert (hidden=256 vs. 3584) handles action planning separately.

**InstructScene**: 100K automated annotation — Qwen2.5-VL-72B generates scene descriptions + driving instructions from future frames; rule-based ego-motion labels provide precision for ego-vehicle dynamics.

**CFG (classifier-free guidance)**: drops text/ViT/action tokens randomly during training → enables instruction guidance strength at inference.

**Results**: 86.9 EPDMS / 89.4 BoN-6 (NAVSIM-v2), 87.9 PDMS / 89.8 BoN-6 (NAVSIM-v1). No RL stage.

**Contrast with FSDrive (Pattern 5)**: FSDrive uses visual CoT as inference-time reasoning intermediate (mandatory at inference). Vega uses future frame prediction as training-time dense supervision (bypassed at inference, similar to DriveVLA-W0). Both generate future images during training; only FSDrive uses them at inference.

**Contrast with UniUGP (Pattern 1)**: UniUGP uses future video generation to improve expert imitation; Vega uses it to ground instruction-conditioned policy learning. UniUGP's generation expert is optionally available at inference; Vega's is training-time only.

### 11. Joint Video-Action DiT from Video Generation Backbone (DriveVA)

**DriveVA** ([[sources/driveva.md]]) answers a different framing of the world model question: instead of building a driving world model from scratch or adding video prediction to a VLM backbone, can we directly fine-tune a **large-scale pretrained video generation model** for AD planning?

**Core motivation**: VLMs pretrained on image-text pairs learn semantic knowledge ("what is what") but not spatiotemporal dynamics ("how the world moves"). Video generation models trained on web-scale video implicitly encode physically plausible motion patterns — richer priors for generalizable driving.

**Backbone**: Wan2.2-TI2V-5B (5B parameters) — the text-to-image-to-video variant of the Wan model family (same family as DriveDreamer-Policy's Wan-2.1-1.3B, but larger and with image-conditioning support). The 3D-causal VAE and frozen text encoder are inherited.

**Key architectural innovation — joint generative target**: instead of separate modules for video prediction and trajectory generation, DriveVA places both in the same noisy target:

$$\mathbf{Y}_0^{(l)} = [\underbrace{\mathbf{V}'_{l+1}, \ldots, \mathbf{V}'_{l+n_\text{pred}}}_\text{future video latents},\ \underbrace{\mathbf{A}_{l+1:l+K}}_\text{action tokens}]$$

A **single DiT** denoises both halves simultaneously at the same flow time $s$. This is the deepest video-action coupling in the wiki:

| Method | Video-action coupling mechanism |
|---|---|
| UniUGP | Cascaded: generation expert conditioned on planning expert output |
| Epona | Parallel branches (TrajDiT ‖ VisDiT) on shared MST latent |
| DriveDreamer-Policy | Causal stages: depth → video → action, separate FM generators |
| DriveVLA-W0 | Training-time auxiliary loss only, no coupling at inference |
| FLARE | Auxiliary semantic prediction only, no video generation at inference |
| **DriveVA** | **Single DiT over joint [video_latents ‖ action_tokens] target** |

**Video continuation module**: history observation buffer (m frames) encoded as condition latents; after each action chunk is executed, the window slides and a new short clip is predicted. Inference requires only **2 flow-matching steps** for near-optimal NAVSIM performance.

**Critical ablation** (Table 5.5): video supervision 71.4 → 90.9 PDMS (+19.5) over action-only optimization. This is the strongest single-component gain in the wiki for any technique. The authors argue the gain requires actions to be forced *consistent* with the imagined future — loose coupling (auxiliary loss) does not replicate it.

**Zero-shot generalization results** (key differentiator from all other wiki world-model methods):
- **nuScenes (zero-shot, trained on NAVSIM only)**: −78.9% avg L2, −83.3% collision vs. PWM
- **Bench2Drive (zero-shot, real→sim)**: −52.5% avg L2, −52.4% collision vs. PWM

No other wiki world-model paper demonstrates quantitative cross-dataset zero-shot transfer at this scale.

**Limitations**: Table 1 (NAVSIM sub-scores) truncated in source file — per-metric breakdown unavailable; comparison table methods unknown. No NAVSIM-v2/EPDMS. No RL stage. 5B backbone with no latency numbers. Video required at every inference step (unlike Epona's optional VisDiT). Zero-shot comparison baseline is PWM only, not full leaderboard.

**NAVSIM-v1**: 90.9 PDMS — between WAM-Diff (91.0) and DriveFine (90.7) in the wiki.

### 12. Dual-Role World Model: Dense Supervisor + Intrinsic Exploration Reward (ExploreVLA)

**ExploreVLA** ([[sources/explorevla.md]]) assigns the world model **two simultaneous roles** — a pattern not seen in any previous entry:
1. **Dense supervisory signal** (Stage 1 SFT): future RGB + depth masked token prediction provides rich visual and geometric supervision alongside trajectory prediction.
2. **Intrinsic exploration reward** (Stage 2 GRPO): the world model's token-level entropy measures trajectory novelty — high entropy indicates OOD trajectories that, if safe, are valuable learning opportunities.

**Key distinction from all prior patterns**:
- Patterns 1–8: world model provides supervision signal (pixels, latent features, semantic patches, instructions)
- Pattern 9 (DreamerAD): world model provides a *task-aligned learned reward* from latent features
- **Pattern 12 (ExploreVLA)**: world model provides an *uncertainty-based novelty reward* from prediction entropy — no separate reward model training; entropy is model-native

**RGB + depth dual supervision** (Table 3 ablation):

| RGB | Depth | PDMS |
|-----|-------|------|
| ✗ | ✗ | 86.2 |
| ✓ | ✗ | 87.9 |
| ✗ | ✓ | 87.8 |
| ✓ | ✓ | **88.5** |

Depth (Metric3D pseudo-labels) provides complementary geometric structure; joint supervision is additive (+2.3 PDMS over no image generation).

**Safety-gated entropy reward** (Stage 2 GRPO):

$$R_i = \begin{cases} \text{PDMS}_i + \lambda \cdot f(\mathcal{H}(\boldsymbol{\tau}_i)) & \text{PDMS}_i > \delta \\ \text{PDMS}_i & \text{otherwise} \end{cases}$$

where $\mathcal{H}$ = average entropy of MAGVIT-v2 token predictions across all future RGB + depth frames; δ = 0.9; λ = 0.5. The entropy bonus flows only to trajectories that are simultaneously safe and novel.

**Critical finding** (Table 4): image entropy reward alone = +0.03 PDMS; PDMS reward alone = +1.69; both = +1.86. The exploration signal is useless without the safety gate — discovery is only valuable when grounded by task performance.

**NAVSIM-v1**: 90.4 single / 93.7 BoN-6 (2nd in wiki after Curious-VLA 94.8). **NAVSIM-v2**: 88.8 EPDMS, EC = 86.8 (2nd in wiki after WAM-Diff 89.7; comparison table omits WAM-Diff, DDP, DreamerAD). **nuScenes** (Stage 1 only, no RL): avg L2 0.44m / collision rate 0.10% (ties OpenDriveVLA for best average collision).

**Contrast with DreamerAD (Pattern 9)**:
| Aspect | DreamerAD | ExploreVLA |
|--------|-----------|------------|
| World model reward type | Learned latent AD-RM (8 dims × 8 horizons) | Raw token entropy (no separate training) |
| Task alignment | High (explicitly trained on 8 EPDMS sub-metrics) | Indirect (entropy is novelty, not task reward) |
| Simulator needed for RL | No (latent inference only) | Yes (PDMS gate requires PDM simulator) |
| World model inference mode at RL | Latent (1-step shortcut, 0.03s) | Image generation (MAGVIT-v2 token decoding) |
| Primary PDMS result | 88.7 NAVSIM-v1 | 90.4 / 93.7 BoN-6 NAVSIM-v1 |

## Key Challenges

### 1. Coupling world model and trajectory planner
The world model must receive the planned trajectory as a condition, but the trajectory is what we're trying to optimize. Solutions:
- **Teacher forcing**: use ground-truth trajectories 50% of training time (UniUGP)
- **Feedback conditioning**: the world model is conditioned on the planning expert's output, training the planner to generate trajectories consistent with realistic future video

### 2. Computational cost
Video generation models (DiT-based, e.g., Wan2.1) are expensive. Solutions:
- Make generation expert optional at inference (UniUGP)
- Use lower-resolution occupancy instead of video (OccWorld)

### 3. Evaluation
World model quality (FID, FVD) and planning quality (L2, collision rate) can improve independently or diverge. UniUGP is notable for improving both simultaneously.

## Metrics for World Model Quality

| Metric | Meaning |
|--------|---------|
| FID (Fréchet Inception Distance) | Distribution-level image quality; lower is better |
| FVD (Fréchet Video Distance) | Distribution-level video quality; lower is better |

Note: FID/FVD measure distributional realism, not planning-relevant accuracy. A model with excellent FID could still predict unrealistic trajectories for edge-case scenarios.

## World Model vs. VLA: Complementary Strengths

| Capability | World Model | VLA (autoregressive) |
|-----------|-------------|----------------------|
| Visual causal learning | ✓ (from unlabeled video) | ✗ (needs annotations) |
| World knowledge / reasoning | ✗ | ✓ (pre-trained LLM) |
| NL interaction | ✗ (typically) | ✓ |
| Open-ended NL instruction following | ✗ | ✗ (typically) |
| Long-tail generalization | Partial | Partial |
| **UniUGP** | **Both** | **Both** |
| **Vega** | **World model as instruction bridge** | **Instruction-conditioned planning** |
| **DriveVA** | **✓ (joint DiT, video backbone)** | **✗ (no LLM reasoning)** |
| **ExploreVLA** | **✓ (RGB+depth masked prediction + entropy reward)** | **Partial (Show-o Phi-1.5 LLM)** |

## State of the Art (as of April 2026)

### nuScenes Future Frame Generation (FID ↓)

| Method       | Type                      | Resolution | FID ↓   | FVD ↓    |
| ------------ | ------------------------- | ---------- | ------- | -------- |
| DriveDreamer | Diffusion                 | 128×192    | 52.6    | 452.0    |
| Drive-WM     | Diffusion                 | 192×384    | 15.8    | 122.7    |
| Doe-1        | Autoregressive            | 384×672    | 15.9    | —        |
| FSDrive      | Autoregressive            | 128×192    | 10.1    | —        |
| [Epona](../sources/epona.md) | AR+Diffusion | — | 7.5 | 82.8 |
| **UniUGP**   | **AR+Diffusion (Wan2.1)** | **—**      | **7.4** | **75.9** |

Note: FID is resolution-dependent — methods at higher resolution (Doe-1 384×672) would achieve lower FID at lower resolution. FSDrive's 10.1 at 128×192 is competitive for its resolution tier and model size (2B).

### NAVSIM Future Video Generation (FVD ↓, front-view)

| Method | FVD ↓ | LPIPS ↓ | PSNR ↑ |
|--------|-------|---------|--------|
| PWM | 85.95 | 0.23 | 21.57 |
| **DriveDreamer-Policy** | **53.59** | **0.20** | **21.05** |

DDP substantially improves video coherence (−38% FVD) vs. PWM. The improvement is attributed to depth joint learning (−18.6% FVD alone) and LLM-conditioned generation. Note: front-view only for comparability with PWM (single-view model).

### nuScenes Planning (front/multi-camera, no heavy supervision, UniAD metrics)

| Method | Avg L2 (m) ↓ | Avg Collision (%) ↓ | Notes |
|--------|------------|-------------------|-------|
| Doe-1 | 1.26 | 0.53 | No ego status; Lumina-mGPT-7B |
| **FSDrive** | **0.96** | **0.40** | **No ego status; Qwen2-VL-2B** |
| [Epona](../sources/epona.md) | 1.25 | 0.36 | Front cam, no aux supervision |
| **UniUGP** | **1.23** | **0.33** | **multi-camera** |

## Open Questions

- Does trajectory-conditioned video generation improve **closed-loop** performance (NAVSIM PDMS), or only open-loop metrics? (FSDrive shows 85.1 PDMS, well below the NAVSIM SOTA of 90.7 — generation quality may not translate to closed-loop driving)
- **[Partially answered by DreamerAD]** Can the world model provide a reward signal for RL, replacing or augmenting the simulator? — DreamerAD shows a latent reward model can replace simulator calls *during* RL rollout (87.7 EPDMS), but still requires simulator for initial vocabulary annotation. True simulator-free RL from world model rewards remains open.
- Does higher-resolution visual CoT (e.g., 512×768) substantially improve collision avoidance over FSDrive's 128×192?
- FSDrive only generates a front-view CoT. Does generating surround-view visual CoT improve performance in lane-change and merge scenarios?
- Can world model pre-training on massive unlabeled video (e.g., internet dashcam footage) bootstrap planning performance without any trajectory labels? (FLARE's FFP is designed for this but has not been tested at scale)
- **FSDrive vs. UniUGP tradeoff**: UniUGP's generation expert is optional at inference (speed-critical deployment), while FSDrive's visual CoT is mandatory. Does the always-on generation cost hurt real-time deployment?
- **DDP depth grounding**: DDP uses Depth Anything 3 pseudo-labels for both training and evaluation — does real LiDAR depth provide further improvement? Is geometric grounding from pseudo-labels sufficient for embodied planning?
- **Comfort under extended metrics**: both DDP (EC=79.4) and WAM-Flow (EC=73.9) score poorly on NAVSIM-v2 extended comfort. Does world model training inherently produce more aggressive trajectories? (FLARE achieves EC=87.5 without video generation — suggests comfort is driven by RL reward design, not world model type)
- **FLARE multi-step**: does extending FFP to predict features at t+2, t+3 provide further planning gains over single next-frame prediction?
- **Video backbone scale**: DriveVA uses Wan2.2-TI2V-5B (5B params) and achieves 90.9 PDMS without RL. Would a smaller video backbone (e.g., 1.3B as in DDP) achieve comparable generalization? Is the zero-shot transfer a function of backbone size, joint coupling, or both?
- **Zero-shot transfer ceiling**: DriveVA's zero-shot nuScenes/Bench2Drive gains are measured relative to PWM only. How does DriveVA compare zero-shot against VLA methods (FLARE, DriveFine) that are fine-tuned on the target domain? Does joint video-action training provide a sustainable generalization advantage at matched data scale?
