---
title: World Models for Autonomous Driving
type: concept
sources: [raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md, raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md, raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md, raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md, raw/papers/Epona_ Autoregressive Diffusion World Model for Autonomous Driving.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/DynVLA_ Learning World Dynamics for Action Reasoning in Autonomous Driving.md, raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md, raw/papers/DeepSight_ Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving.md, raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md, raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md, raw/papers/DriveLaW_ Unifying Planning and Video Generation in a Latent Driving World.md, raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md]
related: [sources/da-wam.md, sources/geowam.md, sources/wa-jepa.md, sources/auto-jepa.md, sources/simwam.md, sources/sgdrive.md, sources/drivelaw.md, sources/uniugp.md, sources/futuresightdrive.md, sources/drivedreamer-policy.md, sources/drivevla-w0.md, sources/flare.md, sources/dreameraD.md, sources/vega.md, sources/epona.md, sources/driveva.md, sources/explorevla.md, sources/dynvla.md, sources/onevl.md, sources/latent-wam.md, sources/drive-jepa.md, sources/policy-world-model.md, sources/deepsight.md, sources/drivewam.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, concepts/rl-for-ad.md, concepts/physicalai-av-benchmark.md, concepts/counterfactual-prediction.md, sources/driving-wm-counterfactuals.md]
created: 2026-04-05
updated: 2026-09-02
confidence: high
---

## What Is a World Model in AD?

A world model learns to predict the future state of the environment — most commonly by predicting future video frames — from historical observations and optionally an action or trajectory. In autonomous driving, this means predicting what the scene will look like in the next N seconds given the current camera stream and the ego vehicle's intended motion.

**Core hypothesis**: learning to predict future visual states forces a model to internalize causal relationships in the scene — who will turn where, which objects will move, how the scene evolves. This visual causal reasoning then transfers to better planning.

## Why World Models Are Useful for Planning

Standard imitation learning (behavior cloning) trains a planner to mimic recorded trajectories, but the model sees no explicit signal about **why** specific objects matter for the future. A world model provides exactly this:

- Forces attention to **causally relevant** distant objects (e.g., a car about to run a red light far ahead)
- Enables **action-conditioned foresight**: "if I turn left, the future video should look like X; if I go straight, like Y" — this is *interventional* (Pearl rung 2), and despite widespread usage it is not counterfactual prediction; see [[concepts/counterfactual-prediction.md]]
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

**Modern instance**: [[sources/sgdrive.md]] (Pattern 20) revives occupancy forecasting inside a VLM, but predicts *geometry only* — deliberately dropping semantic class distributions to "remove redundant semantic dependencies" — and supervises it through a VAE decoder over the VLM's own query hidden states rather than a separate occupancy codebook.

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

**Result**: 86.9 PDMS SFT (strong VLM SFT on NAVSIM-v1, 1 camera, no external pretraining); 91.4 PDMS after GRPO RFT. Later wiki entries such as DynVLA report higher absolute VLM-style scores, so FLARE's "best" framing should be treated as comparison-scope limited.

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
| **DriveWAM** | **Shared DiT, sequential: generated future latent conditions the action flow (inverse dynamics)** |
| **SimWAM** | **Shared attention only, with an isolated mask: no coupling at inference by construction** |
| **DriveLaW** | **Chained: Video DiT's cached first-step block latents are cross-attended by a separate Action DiT** |

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

### Dynamics Tokens as Compact CoT (DynVLA)

**DynVLA** ([[sources/dynvla.md]]) introduces a world-model pattern that is neither full future image generation nor training-only auxiliary prediction: it learns a **Dynamics Tokenizer** whose discrete tokens are generated at inference time as the model's Chain-of-Thought before action tokens.

The tokenizer decouples dynamics into ego-centric and environment-centric branches, with two regularizers:

| Regularizer | Purpose |
|-------------|---------|
| Ego action regularization | Forces ego dynamics tokens to explain relative ego motion instead of collapsing into generic reconstruction codes |
| Image+BEV cross-view reconstruction | Forces the same dynamics tokens to predict future camera and BEV states, aligning appearance and spatial semantics |

Default representation: 8 dynamics tokens per transition (4 ego + 4 environment), codebook size 64 per branch, VQ dim 32. DynVLA reasons over K=2 transitions, producing a 16-token dynamics trace before action tokens.

**Why this matters for world models**: DynVLA uses future-state prediction to learn the token space, but at inference it does not decode pixels. The world model appears as a compact latent reasoning language:

| Method | World-model signal | Used at inference? | Output generated at inference |
|--------|--------------------|--------------------|-------------------------------|
| FSDrive | Future visual frame | Yes | Image tokens / visual CoT |
| DriveVLA-W0 | Future/current image prediction | No | None; training-time representation only |
| FLARE | Future DINOv2 feature prediction | No | None; auxiliary loss only |
| ExploreVLA | RGB+depth entropy | During RL | Entropy reward, not action reasoning |
| **DynVLA** | **Future image+BEV dynamics tokenization** | **Yes** | **Compact dynamics tokens before action tokens** |

Controlled CoT comparison on NAVSIM SFT stage: Dynamics CoT reaches 87.2 PDMS at 0.37s, compared with future-image CoT 86.3 PDMS at 2.29s and scene-description CoT 85.3 PDMS at 3.04s. This supports DynVLA's central claim that dynamics tokens preserve planning-relevant foresight while removing pixel/text redundancy.

### 13. Latent CoT with Training-Time Visual Decoder (OneVL)

**OneVL** ([[sources/onevl.md]]) adds another world-model role: the world model is a **training-time decoder that supervises latent reasoning tokens**, not a deployed generator or an RL rollout model. Visual latent tokens inside Qwen3-VL-4B are trained so an auxiliary decoder can predict future-frame visual tokens at +0.5s and +1.0s. A parallel language auxiliary decoder reconstructs text CoT from language latent tokens.

At inference, both decoders are discarded. The visual and language latent tokens are prefilled into the prompt, so the planner keeps the representation shaped by future-scene prediction without paying the cost of image generation. This places OneVL between FLARE/DriveVLA-W0 and DynVLA:

| Method | World-model signal | Used at inference? | Output generated at inference |
|--------|--------------------|--------------------|-------------------------------|
| FLARE | Future DINOv2 feature prediction | No | None |
| DriveVLA-W0 | Future/current image token prediction | No | None |
| DynVLA | Dynamics tokenization from future image+BEV | Yes | Dynamics tokens |
| **OneVL** | **Future-frame visual token decoder over latent CoT** | **No decoders; yes latent prefill** | **Trajectory tokens, optional post-hoc explanations** |

The ablation supports the world-model interpretation: removing the visual decoder drops NAVSIM PDMS from 88.84 to 87.97, while removing the language decoder drops it only to 88.53. The larger gain comes from the spatial-temporal future-frame target rather than linguistic reconstruction.

### 14. Compact Latent World Status Prediction (Latent-WAM)

**Latent-WAM** ([[sources/latent-wam.md]]) is the wiki's cleanest example of a world model that never decodes pixels and does not use a VLM. It compresses three-camera image patches into 16 scene queries per view, appends ego-status tokens, and trains a causal Transformer to predict future latent world status blocks.

The distinguishing feature is spatial-aware compression. Compression alone slightly hurts planning (87.9 -> 87.7 EPDMS), but geometric distillation from WorldMirror turns the compressed representation into a stronger planning state (88.3 -> 89.3 EPDMS in the full model). This separates Latent-WAM from video-generation WAMs: it does not need image reconstruction fidelity; it needs compact latent tokens that preserve lane/drivable-area geometry and ego dynamics.

| Aspect | Latent-WAM |
| --- | --- |
| World-model target | Future latent world status tokens |
| Visual decoder | None |
| Inference world model | No; SCWE + trajectory decoder only |
| Spatial supervision | WorldMirror geometric feature distillation |
| Dynamics supervision | Causal latent prediction + command/velocity/acceleration ego loss |
| NAVSIM-v2 | 89.3 EPDMS |
| Runtime | 104M params, 107ms on A100 |

Latent-WAM is closest to FLARE and DriveVLA-W0 in using world modeling as a training-time representation shaper, but its target is neither future pixels nor DINO patches. It predicts the latent world state itself.

### 15. JEPA Video Pretraining for Planning (Drive-JEPA)

**Drive-JEPA** ([[sources/drive-jepa.md]]) adapts V-JEPA to driving videos: masked context representations predict target latent representations without pixel reconstruction. The paper initializes from V-JEPA 2, curates 208 hours of front-view driving video, and pretrains a ViT-L encoder on 8-frame clips sampled at 2 Hz.

This is a world-model-like signal but not a deployed world model. Drive-JEPA does not decode future pixels or run a future-state predictor at inference. Instead, JEPA pretraining shapes the visual encoder before a proposal-centric planner is trained. The evidence is strongest in the perception-free table: a simple decoder on top of the Drive-JEPA encoder reaches 89.0 PDMS on NAVSIM-v1, compared with 86.2 for Epona and 86.1 for the base V-JEPA 2 checkpoint.

Drive-JEPA differs from Latent-WAM in the target and deployment path. Latent-WAM predicts compact future world-status tokens and uses WorldMirror geometric distillation; Drive-JEPA predicts latent video representations during pretraining, then relies on multimodal trajectory distillation and momentum-aware proposal selection during planner training.

### 16. Policy World Model: Forecasted Future Frames as Planning Rationales

**Policy World Model** ([[sources/policy-world-model.md]]) makes the strongest version of inference-time future forecasting among the compact AR world-model papers in the wiki. PWM pretrains on action-free OpenDV front-camera video, compresses each frame to 28 tokens, then rolls out future frame tokens before predicting action tokens.

The important distinction is ordering: the world model runs **before** the planner output is known. This avoids the action-conditioned setup where future video only verifies a candidate action. Instead, PWM uses generated future states as multimodal rationales for the action itself.

| Method | Future-state signal | Inference role |
| --- | --- | --- |
| DriveVLA-W0 | Future/current image prediction | Training-only representation shaping |
| FSDrive | Future visual CoT frame | Mandatory planning intermediate |
| DriveVA | Joint video-action DiT target | Video/action generated together |
| Latent-WAM | Future latent world status | Training-time latent dynamics, no pixel decoder |
| OneVL | Future-frame auxiliary decoder | Decoder discarded; latent tokens prefilled |
| **PWM** | **Action-free future frame tokens** | **Forecasted at inference before action prediction** |

Empirically, PWM's signature is collision reduction rather than best L2: with ego status it reports 0.41 average L2 and 0.04 average collision on nuScenes. The NAVSIM result is 88.1 PDMS with one front camera, which is no longer leaderboard-level in this wiki but remains useful evidence for the future-frame rationale mechanism.

### 17. Parallel Multi-Frame DINOv3 Latent Prediction in BEV (DeepSight)

**DeepSight** ([[sources/deepsight.md]]) is the wiki's clearest example of predicting **semantic latent features for several future frames at once**, rather than one frame autoregressively. A set of learnable **World Queries** $\mathbf{Q}_\text{world}=[q_0,\dots,q_4]$ lets the VLM (Qwen2.5-VL-3B) regress the DINOv3 features of five consecutive future BEV frames ($\Delta t=0.5$s → 2s) in a **single forward pass**, supervised by MSE against $\phi_\text{dino}(I^\text{bev})$ ground truth.

This combines three design choices that other patterns take separately:

| Choice | DeepSight | Closest prior |
|--------|-----------|---------------|
| Target | DINOv3 semantic features (not pixels/VAE tokens) | FLARE (DINOv2), Latent-WAM (latent status) |
| Horizon | 5 frames predicted **in parallel** | most predict 1 frame (FLARE, DriveVLA-W0) or AR-sequential (Epona, PWM) |
| Space | BEV (surrounding agents) | FSDrive/PWM front-view only |

**Why parallel latent prediction matters** (Table 6): predicting features rather than pixels, all frames in one pass, costs only **+3.57%** latency over a native VLM — versus **+60.71%** for FSDrive's autoregressive VQ-VAE pixel CoT. The world model is effectively "free" foresight.

**Two ablations that sharpen the pattern's claim** (Table 3, Dev-10 DS):
- **Semantic latent ≫ pixel reconstruction**: DINOv3 vs. VAE at one frame is +47.04 DS (74.79 vs. 27.75). Texture-oriented VAE codebooks lose the planning-relevant semantics.
- **Long horizon helps *only* latent modeling**: five-frame VAE *drops* −13.09 DS vs. one-frame VAE, while five-frame DINOv3 *gains* +11.78 DS vs. one-frame DINOv3. Pixel world models degrade over horizon; latent-feature world models improve.
- **BEV vs. front-view** (Table 4): +8.8 DS for BEV — surrounding-agent modeling is what long-horizon safety needs.

**Contrast with FLARE (Pattern 8)**: both predict DINO features as the world-model signal, but FLARE uses a *single* next-frame feature as an **auxiliary training loss** (no inference-time world output, action-conditioned), whereas DeepSight predicts a *five-frame* trajectory of features as a first-class output of the forward pass (produced before CoT and action). FLARE's target is front-view patches; DeepSight's is BEV.

**Contrast with DreamerAD (Pattern 9) / DynVLA**: DeepSight uses latent features as a **supervision target that shapes representation**, not as an RL reward (DreamerAD) or as a decoded CoT the planner reads (DynVLA). At inference, DeepSight's latents are an internal state, not a separately consumed reasoning artifact.

**Deployment note**: the DINOv3 targets are built from BEV-rendered images or semantic segmentation maps — a training-time rendering/annotation dependency (removed at inference). Evaluated only on Bench2Drive (closed-loop) and nuScenes (open-loop); no NAVSIM.

### 18. Chunked Autoregressive Video-Action Policy with VLM Guidance (DriveWAM)

**DriveWAM** ([[sources/drivewam.md]]) is the wiki's second method to make a pretrained video diffusion transformer *the policy itself* rather than an auxiliary branch — and it uses the **same backbone as DriveVA (Wan2.2-TI2V-5B)**, which makes the pair the cleanest controlled contrast available for "how should a video foundation model be turned into a driving policy?"

**Three design choices that differentiate it from DriveVA (Pattern 11)**:

| Aspect | DriveVA (Pattern 11) | DriveWAM (Pattern 18) |
|---|---|---|
| Video-action coupling | Single DiT denoises a **joint** `[video_latents ‖ action_tokens]` target at the same flow time | **Sequential inverse dynamics**: sample $\hat{z}_{k+1}$ first, then sample $\hat{a}_{k+1}$ *conditioned on* the generated future latent |
| Temporal structure | Sliding window over short predicted clips | Explicit **chunked autoregression** (4s chunks) with causal teacher-forcing mask, full-clip single-pass training |
| High-level semantics | None (video prior only) | **Frozen Qwen3-VL-8B** emits fresh chunk-specific guidance, injected by temporally localized cross-attention |
| Long-horizon memory | Not addressed | **Selective KV memory** (content-based eviction, bounded modality pools) |
| NAVSIM-v1 | 90.9 PDMS | 90.1 PDMS |

**Inverse-dynamics action generation** is the conceptual core: the action decoder $D_a$ reads out ego motion from the model's *own imagined future* ($\tilde{z}_{k+1}$ = clean latent under teacher forcing, generated latent at inference), rather than predicting a trajectory in parallel with the video. This makes the action an explicit function of the predicted world evolution instead of a sibling output — a stronger form of grounding than Epona's parallel TrajDiT ‖ VisDiT branches, and a looser one than DriveVA's single joint denoising target.

**The backbone ablation is the sharpest evidence in the wiki that video supervision is load-bearing, not decorative** (Table 4, ADE@4s / FDE@4s):

| Pretrained init. | Video sup. | ADE@4s | FDE@4s |
|---|---|---:|---:|
| ✗ | ✓ | 1.10 | 3.26 |
| ✓ | ✗ | **1.23** | **3.79** |
| ✓ | ✓ | **0.83** | **2.47** |

Initializing from the pretrained video backbone and then *removing* the video flow-matching term is **worse than training from scratch with video supervision**. Action-only fine-tuning does not merely fail to exploit the video prior — it actively destroys it. This complements DriveVA's +19.5 PDMS video-supervision gain and DriveVLA-W0's "supervision deficit" framing (Pattern 7) from the opposite direction: W0 shows adding a world-model loss helps a VLA scale; DriveWAM shows removing it from a video-native policy is catastrophic.

**Semantic guidance as a separable role**: DriveWAM keeps the VLM entirely frozen and outside the policy. Prior patterns either have no semantic module (DriveVA, Epona, Latent-WAM) or make the VLM the policy backbone with generation attached (FSDrive, DriveVLA-W0, DriveDreamer-Policy, DeepSight). DriveWAM inverts the usual hierarchy: **video model plans, VLM advises**. The guidance is chunk-specific (regenerated every 4s from causally available context) rather than the single clip-level text condition used by prior WA methods, and a block-diagonal text mask prevents chunk $k{+}1$ from attending to guidance produced at later decision steps. The ablation holds at every data scale (ADE@4s 1.21→1.01 at 4k clips, 0.92→0.83 at 100k) — the benefit does not wash out with more data, though the baseline is a fixed global prompt rather than a per-clip VLM caption, so freshness and VLM quality are not separately isolated.

**Selective KV memory** is the wiki's first *content-based* cache eviction policy for driving rollout. Tokens are scored $s^m_j = \lambda\rho^m_j - (1-\lambda)\eta^m_j$ (relevance = attention mass from current queries; redundancy = mean cosine similarity to other cached keys), with **separate bounded pools for video and action** so numerous video tokens cannot crowd out compact ego-motion history. It is training-free and inference-only. At a fixed budget it nearly matches full caching (0.89 vs. 0.83 ADE@4s) where FIFO collapses (1.40), with >12× reduction in KV memory and attention FLOPs on a 300s rollout. Caveat: accuracy is only measured on 20s clips, so the long-horizon regime the mechanism exists for is unvalidated.

**Data scaling**: 4k → 20k → 100k clips at fixed 50k iterations improves monotonically with no saturation ([[concepts/physicalai-av-benchmark.md]]), supporting the paper's claim that world-action modeling is a scalable policy foundation. This is the wiki's first real-world (non-proprietary) data-scaling curve for a world-model policy; DriveVLA-W0's is on an in-house 70M-frame set.

### 19. Video Backbone as Training-Time-Only Prior (SimWAM)

**SimWAM** ([[sources/simwam.md]]) completes a natural progression. DriveVA and DriveWAM both fine-tune a Wan-family video DiT into a policy and both generate the future at inference. SimWAM keeps the video generative backbone as the representation source but **deletes the future-frame branch at inference entirely**, using an isolated attention mask so the action tokens never depend on future-frame tokens in the first place.

The mask is the whole mechanism. The shared attention stream holds the current-observation latents $z(o_t)$, the future-frame latents $z_{t+1:t+N}$, and the action tokens. Both future-frame and action tokens attend to $z(o_t)$; the two are mutually invisible. Future-video prediction therefore shapes $z(o_t)$ during training and is discarded afterwards, collapsing the imagine-then-act integral to a direct policy $p_\theta(a\mid z(o_t), s_t, l)$.

This places SimWAM at the intersection of two existing patterns: it shares its *deployment* profile with Pattern 7 (DriveVLA-W0) and Pattern 8 (FLARE) — world model as training-time signal only — but its *representation source* is a pretrained video generative model, as in Patterns 11 and 18.

| Method | World-model backbone | Future generated at inference? | NAVSIM-v1 |
|---|---|---|---|
| DriveVA (P11) | Wan2.2-TI2V-5B | Yes (joint denoising target) | 90.9 |
| DriveWAM (P18) | Wan2.2-TI2V-5B | Yes (action is inverse dynamics from it) | 90.1 |
| **SimWAM (P19)** | **Wan2.2-5B (swappable)** | **No (isolated mask; branch dropped)** | **91.5** |
| DriveVLA-W0 (P7) | Emu3 / Qwen2.5-VL | No | 90.2★ |
| FLARE (P8) | DINOv2 features | No | 91.4 |

**Video co-training is where the gain lives** (Table 2): an action-only DiT reaches 86.6 PDMS; adding the video expert lifts it to 90.3 (+3.7, improving every sub-metric); RL adds 1.2 more. That +3.7 is the same phenomenon DriveVA measured as +19.5 PDMS and DriveWAM measured as a catastrophic 1.23-vs-1.10 ADE reversal — three independent confirmations that future-video supervision, not future-video *generation*, carries the benefit.

**Two scaling axes are both nearly flat.** Swapping the video backbone (Table 4) gives LTX-Video 88.7, Wan2.1-1.3B 90.2, Wan2.2-5B 90.3, Cosmos-Predict2.5 90.4 — prior *quality* matters (the lightweight LTX-Video loses 1.6) but prior *scale* barely does, and a driving-pretrained backbone (Cosmos) edges out a 4× larger general one. Scaling the action expert 0.21B → 1.02B (Table 5) buys only 0.4 PDMS. This is the wiki's only controlled comparison of interchangeable video priors under a fixed planner, and it argues the field's video backbones are already past the point of diminishing returns for this task.

**Temporal coverage beats frame density** (Table 8): shortening the supervision horizon 4 s → 2 s costs 0.4 PDMS, while halving the frame rate at fixed 4 s costs 0.1. What the representation needs is a long enough view of how the scene evolves, not a finely sampled one.

### 20. Structured Symbolic State Forecasting (SGDrive)

**SGDrive** ([[sources/sgdrive.md]]) forecasts the future without generating anything perceptual. Where Patterns 11/18/19 transfer appearance dynamics from a pretrained video model and Patterns 8/14/17 regress latent or semantic features, SGDrive predicts **structured symbolic state**: occupancy voxels, 3D agent boxes, and a goal pose — each at both the current time $t$ and a future time $t{+}n$.

The mechanism is supervised query tokens rather than a generator. A set of learnable ⟨world⟩ queries is appended to the VLM's token stream and decoded by three heads into a **scene → agent → goal** hierarchy meant to mirror human driving cognition: perceive the layout, attend to the agents that matter, then form a short-term objective. The queries' hidden states are then fed directly to a DiT planner, so nothing is explicitly decoded at inference.

| World-model target | Methods | Needs annotation? | Decoded at inference? |
|---|---|---|---|
| Pixels / video latents | DriveVA, DriveWAM, SimWAM, Epona, FSDrive, PWM | No (raw video) | Varies by method |
| Semantic features | FLARE (DINOv2), DeepSight (DINOv3 in BEV) | No (frozen extractor) | No |
| Latent world status | Latent-WAM, Drive-JEPA, OneVL | No | No |
| Dynamics tokens | DynVLA | No | Yes (as CoT) |
| **Structured symbolic state** | **SGDrive (occupancy + boxes + goal)** | **Yes (occupancy labels / LiDAR, 3D boxes)** | **No (hidden states condition the DiT)** |

**Two properties make this pattern distinct.** It is the only world model in the wiki whose targets are *human-interpretable by construction* — Figure 5 of the paper shows predicted occupancy, boxes, and goal directly against ground truth, which no pixel or latent world model can offer. And it is the only one requiring **3D annotation at training time**; every other pattern here is either self-supervised on raw video or distills a frozen feature extractor. That is a real cost when comparing against "camera-only" methods.

**Where the gain actually comes from** (Table 3, Stage-1 text-trajectory setting, isolating the representation from the planner): base 82.2 → current-state hierarchy 84.7 → adding future forecasting 85.5. **Structured perception of the present is worth +2.5 PDMS; forecasting the future adds +0.8.** This is a useful corrective to the paper's world-model framing — most of the benefit is knowing what is there now, not what happens next. It also echoes DeepSight's finding from the opposite direction, where the horizon mattered a great deal for *latent* targets.

**The hierarchy's components do distinguishable jobs** (Table 4, with the diffusion planner): agents mainly lift NC/DAC, the goal query mainly lifts Ego Progress (80.4 → 81.2, the single largest jump), and future forecasting mainly lifts NC/TTC. That functional separation is the strongest evidence that the scene-agent-goal decomposition is more than a multi-task loss.

**Anti-leakage via masking**, not parameter separation. A block-wise mask forbids attention between the scene/agent/goal blocks while allowing temporal attention within a block and free cross-attention to visual/text tokens. This is a third answer to the representational-interference problem that [[concepts/mixture-of-experts.md]] tracks: UniDriveVLA decouples expert *parameters*, OneDrive isolates heterogeneity in task FFNs, and SGDrive simply masks *attention* between query blocks. It is by far the cheapest of the three — and also the weakest measured effect, worth only +0.3 PDMS, entirely in EP.

### 21. Mid-Denoising Latents as the Planning State (DriveLaW)

**DriveLaW** ([[sources/drivelaw.md]]) makes a distinction the other video-prior methods do not: it separates the video generator's *output* from its *internal state*, and plans from the latter. The Action DiT cross-attends to latents cached from each Video DiT block **during the first denoising step** — the generator's early internal activations, not its finished prediction.

The paper's framing is that Epona, VaVAM, and DriveVLA-W0 are only nominally unified, running generation and planning as "two independent output streams" so the trajectory is never grounded in the features that actually govern synthesis. Chaining fixes that representation disconnect.

**The controlled representation comparison is the wiki's most direct evidence for the video-prior thesis** (Table 5, same diffusion planner throughout):

| Conditioning representation | PDMS |
|---|---:|
| BEV features (BEVFormer ResNet-101) | 84.1 |
| VLM hidden states (Qwen2.5-VL, ReCogDrive-style) | 86.5 |
| **Video-generator latents** | **89.1** |

Video latents beat BEV by +5.0 and VLM hidden states by +2.6 with everything else held fixed. Every other comparison of these representation families in the wiki is confounded by architecture and training data; this one is not. The PCA visualization (Figure 4 of the paper) supports it qualitatively — BEV and VLM features appear diffuse with irregular focus shifts, while video-generator features are sharper and spatially structured under severe motion.

**Pretraining data scales planning** (Table 4): 0 → 76k → 3.8M → 7.6M video samples gives 85.9 → 87.0 → 87.8 → 89.1 PDMS, monotone and unsaturated. This is the axis SimWAM did *not* test — SimWAM varied backbone *size* at fixed data and found it flat, DriveLaW varies *data* at fixed size and finds +3.2. Read together: for video priors, what you pretrain on matters much more than how big the model is.

**Cost profile.** NC 99.0 and TTC 96.7 are the highest in the wiki — a conspicuously safety-skewed policy achieved with no RL and no scorer — but EP 81.3 is mediocre, and there is no mechanism to recover progress. Video generation is ~5× faster than Epona at matched resolution, though trajectory planning is *slower* (0.71 s vs 0.42 s on H20).

### 22. The Ego Trajectory as the Prediction Target (Auto-JEPA)

Every pattern above predicts something about **the scene**: pixels, video latents, DINO features, occupancy voxels, BEV state, dynamics tokens. **Auto-JEPA** ([[sources/auto-jepa.md]]) predicts an encoding of **what the ego will do**, and treats scene evolution as relevant only through its effect on that.

The mechanism is JEPA applied to a trajectory latent space rather than to video. A trajectory autoencoder is trained first, its decoder discarded, and its encoder frozen — this defines an 8×1024 target space in which the ground-truth 4 s future trajectory has a fixed embedding $\mathbf{Z}^{+}$. A predictor (frozen V-JEPA 2 encoder + 24-layer Transformer) then maps four front-camera frames, four ego positions, and a route command to $\hat{\mathbf{Z}}$, trained with feature alignment, token-wise cosine alignment, and batch-level InfoNCE against $\mathbf{Z}^{+}$. No waypoint coordinates are ever supervised.

| | Scene-state world models | Auto-JEPA |
|---|---|---|
| Prediction target | Future observation / latent / occupancy | Latent of the future *ego trajectory* |
| What must be preserved | Enough of the scene to reconstruct it | Only what changes ego action |
| Annotation needed | None to heavy (SGDrive) | None |
| Inference role | Varies (see below) | The predicted latent is the retrieval key |
| Scene forecasts available? | Yes | **No — by construction** |

**Why this is a distinct position on the imagination question.** The synthesis below splits methods into imagine-then-act and training-time-only. Auto-JEPA fits neither. Its predictive model runs at inference and is entirely load-bearing — replace the predicted intent with a scene-independent constant and PDMS falls 91.3 → 52.6 — but the thing predicted is an *action* latent, not a world state. The paper's framing is that this is what a planning-oriented world model should predict in the first place, since "planning need not reconstruct the complete future world."

**The interesting evidence is not the benchmark number.** 91.3 PDMS is mid-frontier. The load-bearing result is the semantic-occlusion study: masking dynamic-agent regions across all four input frames changes the predicted intent 2.97× as much as equal-area random masks (mean $1-\cos$ 0.080 vs. 0.027 over 15,364 scenes, larger in 71.1%), and per-vehicle occlusion moves the plan much more for an interacting lead vehicle than for a non-interacting adjacent one. **The model was given no boxes, no agent identities, no interaction labels, and no surrounding-agent motion.** Agent selectivity emerged from an ego-motion target alone.

That is the pattern's actual claim, and it is a claim about *sufficiency of the supervision signal*: you do not need to model agents to attend to agents, if your target depends on them. It sits directly against SGDrive's route ([[sources/sgdrive.md]]), which buys the same selectivity by supervising safety-critical boxes explicitly and paying in 3D annotation. Both work; they cost different things. See [[concepts/perception-for-planning.md]] for the occlusion protocol as an evaluation method and for its controls.

**The cost of the target choice** is stated plainly in the paper's own limitations: the learned intent "does not provide the scene-level forecasts required by applications such as interactive simulation or counterfactual environment generation." A world model that only knows what the ego will do cannot be rolled out, queried under intervention, or used as a simulator. Everything in the [[concepts/counterfactual-prediction.md]] discussion is out of scope for it. This is the sharpest statement in the wiki of the trade the whole latent-world-model family is making, because Auto-JEPA takes it to the limit.

### 23. Generative Future Latents Jointly Denoised With Actions (WA-JEPA)

**WA-JEPA** ([[sources/wa-jepa.md]]) sits at the intersection of Patterns 15 (JEPA pretraining), 17 (parallel multi-frame latent prediction), and 11 (joint video-action denoising), and its contribution is to fix what it argues each gets wrong.

Its claim is that V-JEPA is the right representation and the wrong architecture, on three counts: **random spatiotemporal masking is a completion objective**, with no future-directed component; **deterministic regression cannot generate genuinely unseen tokens**, only interpolate observed ones; and V-JEPA 2's action-conditioned variant needs a goal image plus MPC, which is not online planning. The fixes are one-for-one — hybrid future masking, conditional flow matching over latents, and a joint scene-action MMDiT predictor.

| Pattern | Prediction target | Objective | Action coupling |
|---|---|---|---|
| 15 Drive-JEPA | Masked video latents | L1 regression | None (separate planner) |
| 17 DeepSight | 5 future DINOv3 BEV frames | MSE regression | Via VLM hidden states |
| 14 Latent-WAM | Future latent world status | Deterministic causal prediction | Trajectory decoder |
| 22 Auto-JEPA | Future ego-trajectory latent | Alignment + cosine + InfoNCE | The prediction *is* the query |
| **23 WA-JEPA** | **Future multi-view scene latents** | **Conditional flow matching** | **Joint denoising, asymmetric stop-grad** |

**The asymmetric stop-gradient deserves separate attention** because it inverts the usual arrangement. The scene stream reads action tokens but gradients from the scene loss are blocked at that interface; the action stream reads *differentiable* scene tokens. So action supervision shapes the world representation, but world-modeling never perturbs the policy. Most coupled WAMs in this wiki let gradients flow both ways or separate the modules entirely — this is a third option, and the paper's stated goal is to keep the scene representation biased toward *planning-relevant* future dynamics rather than generically accurate ones. It is also the paper's least-supported design: there is no ablation of it anywhere.

#### The prediction objective is a real design axis, not a detail {#objective-form}

This is WA-JEPA's most transferable result, and it is new to the wiki. In Stage 2, holding the joint architecture fixed:

| Configuration | EPDMS |
|---|---:|
| Cascaded baseline (historical latents only, cross-attention) | 89.9 |
| Separate flow-based future predictor, latents cross-attended in | 90.8 |
| Joint modeling, **no future-latent supervision** | 91.1 |
| Joint modeling + **regression** future prediction | **90.7** |
| Joint modeling + **flow matching** future prediction | **91.7** |

**Deterministic regression on a multimodal future is worse than not predicting the future at all** (90.7 vs. 91.1), while flow matching on the identical target is worth +0.6 over no prediction and +1.0 over regression. The wiki has been treating "does future prediction help?" as the question; this says the objective's *form* carries a swing larger than the margin separating the top four methods on NAVSIM-v2.

The diagnosis is measured rather than asserted, on the most dynamic $K{=}64$ token locations per instance:

| Objective | Directional-similarity collapse gap ↓ | Change-magnitude ratio (→1) |
|---|---:|---:|
| Direct regression | 0.30 | 0.45 |
| **Flow matching** | **0.10** | **0.80** |

Regression produces less than half the target's temporal variation and makes consecutive predicted frames excessively parallel — the signature of a conditional mean over a multimodal distribution. Figure 3 of the paper shows the same thing qualitatively: regression predictions grow progressively smoother across the horizon while flow-matched ones keep spatial structure.

**Which raises a question about several patterns above.** DeepSight (Pattern 17) regresses DINOv3 features for five future BEV frames with MSE. FLARE (Pattern 8) regresses DINOv2 features. Latent-WAM (Pattern 14) predicts latent world status deterministically. All three use exactly the objective WA-JEPA measures as harmful. The targets differ — frozen DINO features and compressed status tokens may be far less multimodal than EMA-updated ViT-L scene latents, which would make them much less exposed — and the architectures differ, so this is a hypothesis rather than a refutation. But it is cheap to test and no paper has run it.

**The entropy-of-the-target framing** resolves the apparent conflict with Pattern 22. Auto-JEPA uses a deterministic alignment objective and it works, because its target is a *single ego trajectory* — low-dimensional and weakly multimodal, where a conditional mean is still a usable prediction. WA-JEPA's target is a four-camera scene, where the conditional mean is a blur. **The right objective depends on the entropy of what is being predicted**, and Drive-JEPA sits in the uncomfortable middle: a high-entropy target under a deterministic objective.

### 24. Metric Geometry as the World-Model State Space (GeoWAM)

**GeoWAM** ([[sources/geowam.md]], Uber AV Labs) adds the one state space this page did not have. Patterns 1-13 and 19-21 predict pixels or video latents; 8, 14, 15, 17, 23 predict learned features; 4 and 20 predict occupancy or symbolic state; 22 predicts an action latent. GeoWAM predicts **dense metric point maps** — one 3D point per image pixel, per future step, per camera, in the ego coordinate frame.

**The argument is about entanglement.** Images encode geometry and motion only *indirectly*, mixed with appearance, texture, and illumination, so a video world model's objective "does not require it to explicitly recover the underlying physical dynamics that generate those observations." A model can satisfy that objective with photometric regularities while the 3D transformations stay implicit. Geometry inverts this — and, crucially, **scene geometry and ego trajectories are defined in the same coordinate space**, so forecasting geometry supervises exactly the structure planning consumes.

| World-model target | Methods | Annotation needed | Same frame as the action? |
|---|---|---|---|
| Pixels / video latents | DriveVA, DriveWAM, SimWAM, Epona, FSDrive, PWM, DriveLaW | No (raw video) | No |
| Semantic features | FLARE (DINOv2), DeepSight (DINOv3 BEV), WA-JEPA (EMA ViT-L) | No (frozen or EMA extractor) | No |
| Latent world status | Latent-WAM, Drive-JEPA, OneVL | No | No |
| Structured symbolic state | SGDrive (occupancy + boxes + goal) | **Yes** (occupancy, 3D boxes) | Partly (BEV) |
| Ego-trajectory latent | Auto-JEPA | No | Yes (but only the ego) |
| **Dense metric point maps** | **GeoWAM** | **No — pseudo-labels from geometry foundation models** | **Yes** |

**Two properties make it distinct.** It is the only pattern whose prediction target shares a coordinate frame with the output trajectory — the paper's central claim. And it gets explicit 3D structure **without annotation**: point-map targets come from off-the-shelf geometry foundation models, so training needs only RGB. That is the direct answer to Pattern 20's cost problem, where SGDrive buys interpretable 3D structure by paying for occupancy and box labels.

**A hybrid objective worth noting.** Supervision combines a JEPA-style term — cosine alignment to features from pushing *future* images through the same encoder with stop-gradient — with dense point regression (Euclidean + confidence-aware + multi-scale surface normals), plus the same point objective on the current frame to anchor the encoder. So GeoWAM is simultaneously in the latent-prediction family and the explicit-geometry family, which is unusual and probably load-bearing: [Pattern 23](#objective-form) measures deterministic cosine alignment on scene features as *harmful* in isolation (90.7 vs. 91.1), and the dense point terms are the obvious candidate for what rescues it here. Neither paper tests this. **It is the most testable cross-paper question the two raise.**

**The stop-gradient points the opposite way from WA-JEPA's.** Trajectory loss cannot propagate into predicted future geometry, so planning never reshapes the world model — the paper's "inverse-dynamics-like" reading, in which ego motion is inferred *from* scene evolution. WA-JEPA blocks the scene loss from touching the action stream so that action supervision shapes the world representation. Same mechanism, opposite priority, and **neither paper ablates it**.

**What the evidence actually supports.** Future-geometry accuracy beats video-then-reconstruct at long horizons (mean Abs Rel 0.257 vs. Epona+DVGT's 0.274; mean δ<1.25 0.754 vs. 0.655), though Epona wins δ<1.25 at the 1 s horizon and GeoWAM only pulls ahead from 2 s. For planning, the attribution is narrower than the framing: **+0.6 EPDMS over DVGT-2 on navtest, but +4.9 on navhard** — where DVGT-2 is GeoWAM's own initialization and already a geometry model. The paper never trains its own architecture with a pixel objective, so geometry-vs-pixels is tested only across papers.

**That navtest/navhard asymmetry may be the paper's most important unremarked result.** Whatever future-geometry forecasting adds is worth eight times more under the reactive protocol than the open-loop one — exactly what a world-model thesis predicts, since anticipation should matter most where errors compound. See [[concepts/navhard-ood-evaluation.md]].

### 25. One Future Per Candidate (DA-WAM)

**DA-WAM** ([[sources/da-wam.md]], HKUST-GZ + Leapmotor) targets an axis none of Patterns 1-24 vary: **how many futures are predicted, and whether each candidate trajectory gets its own.**

Its taxonomy of what everyone else does is worth reproducing, because it is the page's missing organizing principle:

| Design | Future reaches the scorer? | Per-candidate? | Examples |
|---|---|---|---|
| (a) Trajectory-only prediction | No | – | Most VLA planners |
| (b) Loosely coupled latent fusion | Yes | No — one proposal, nothing to compare | LAW, DriveFuture |
| (c) One future shared across candidates | Yes | **No — prediction-action mismatch** | WoTE, most WAM scorers, and structurally SimWAM/WA-JEPA |
| (d) **DA-WAM** | Yes | **Yes — one latent per candidate** | – |

The mechanism is a shared predictor with the **action as the query**: $\widehat{Z}_{i}=P_{\phi}(Q=a_{i},K=Z_{t},V=Z_{t})$ for each of 32 candidates. Parameters are shared deliberately, so differences between the $\widehat Z_i$ come from the action queries rather than from per-candidate weights. The scorer then evaluates the triplet $(Z_t, a_i, \widehat Z_i)$ **without pooling** — "preserving fine-grained token-level interactions rather than pooling futures into a coarse proposal-invariant vector."

**Two secondary design choices, both measured.** JEPA supervision stays *live during planner optimization* through a LoRA-adapted V-JEPA 2.1 online encoder and an EMA target, instead of freezing after pretraining — worth +2.42 PDMS cumulatively, far more than the per-candidate mechanism itself. And dense predictive supervision is restricted to the expert-matched candidate, since offline logs record exactly one future; the other 31 latents are shaped only by scorer gradients, which is honest about the data but leaves most of the "world model" unsupervised.

**What the numbers actually support** is covered in the synthesis immediately below, because DA-WAM's ablation is the most directly relevant experiment the wiki has on the test-time-imagination question.

## Does Test-Time Future Imagination Help? {#test-time-imagination}

This is now the central open dispute among world-model planners in the wiki, and SimWAM supplies the first controlled evidence.

**The imagine-then-act camp** conditions planning on generated future states at inference: FSDrive (mandatory visual CoT), PWM (future frame tokens rolled out before action), DriveVA (joint video-action denoising), DriveWAM (action as inverse dynamics from the generated latent), DriveLaW, and now WA-JEPA (future scene latents and actions denoised together over 12 sampling steps). DA-WAM belongs here too, and is the only member that predicts a *separate* future for every candidate rather than one future per scene — the distinction its ablation shows is decisive. The premise is that grounding the action in an explicit imagined future improves it.

**The training-time-only camp** uses future prediction purely to shape representations: DriveVLA-W0, FLARE, Latent-WAM, OneVL, Drive-JEPA, and now SimWAM.

**A third position** was missing from this framing until Auto-JEPA ([[sources/auto-jepa.md]], Pattern 22). It predicts at inference, and the prediction is indispensable — but its target is the ego trajectory latent, not a future world state. This matters for how the question is posed. The dispute below is often stated as "does predicting the future help at decision time?", when the results actually separate along a different axis: *what* is predicted. Auto-JEPA predicts an action and the prediction carries the whole system; SimWAM and DriveLaW predict a world and find the prediction contributes nothing at inference. Reframed, the surviving generalization across all of these papers is **future-prediction objectives are valuable; instantiated future world states at decision time are not** — and Auto-JEPA is the case that shows the first half does not require the second.

Until SimWAM, no paper varied *only* the inference-time dependency. SimWAM's Table 3 does exactly that — same backbone, same co-training, same data, three attention masks:

| Mask | Action sees future tokens? | NC | TTC | PDMS |
|---|---|---:|---:|---:|
| Bidirectional | Yes | 98.4 | 95.1 | 90.2 |
| Action → video | Yes | 98.5 | 95.5 | 90.1 |
| **Isolated** | **No** | **98.7** | **95.9** | **90.3** |

Access to future-frame tokens produces **no measurable benefit**, while forcing future-frame instantiation at inference. The isolated variant also has the best NC and TTC.

**How much weight this deserves.** The spread is 0.2 PDMS with no reported seed variance, so the supportable conclusion is that test-time future conditioning is *unnecessary here*, not that it is harmful. Three further caveats: the comparison is within SimWAM's own architecture (a shared-attention two-expert design where the action expert already reads a video-model-shaped representation), it is single-benchmark, and it uses a 4 s horizon at 2 Hz. A method whose future generation is longer-horizon, geometry-grounded (DriveDreamer-Policy's depth stage), or semantically guided (DriveWAM's per-chunk VLM intent) might still extract value the mask ablation cannot see.

**Corroboration from inside the imagine-then-act camp.** [[sources/drivelaw.md]] is classified above as imagine-then-act, and SimWAM treats it that way. But its own Table 6 sweeps *which* denoising step feeds the planner, and the result is striking:

| Video denoise step | PDMS | What the latent contains |
|---|---:|---|
| **t = 1** | **89.1** | Early internal state; no recognizable future yet |
| t = 5 | 86.9 | Partially denoised |
| t = 10 | **23.2** | Nearly clean generated future — **policy collapses** |

The closer the conditioning signal gets to an actual synthesized future, the worse the planning — catastrophically so at t=10, where comfort drops to 0 and PDMS falls below the Ego-Status-MLP baseline. DriveLaW's stated explanation is that "raw pixel-format videos frequently contain redundant, non-essential information, which can hinder the effectiveness of decision-making."

This matters because it is an **independent, differently-motivated result pointing the same way as SimWAM's mask ablation**. SimWAM removed the future-token dependency and lost nothing; DriveLaW kept the generator but found that useful signal lives in its early internal activations rather than its output. Neither paper set out to test the other's hypothesis. DriveLaW is therefore better described not as imagine-then-act but as **"borrow the generator's representation, not its imagination"** — closer to Pattern 19 than its own framing suggests.

**WA-JEPA does not test this and does not contradict it.** Its Table 4(c) removes the future-prediction training objective *and* the inference-time generation in the same row, exactly the confound SimWAM's isolated mask was designed to break. So its +0.6 EPDMS is evidence for the objective — which every paper here already supports — and says nothing about the inference path. Given SimWAM's and DriveLaW's results, the live hypothesis is that WA-JEPA's 12-step scene denoising at inference is wasted compute and an isolated-mask variant would score the same. One run would settle it. What WA-JEPA *does* add is orthogonal and more interesting: **the objective's form matters as much as its presence** (see [Pattern 23](#objective-form)).

### DA-WAM Supplies the Missing Variable: Shared vs. Per-Candidate

Every experiment above varies *whether* a generated future reaches the planner. [[sources/da-wam.md]] varies **how many futures there are**, and the result reorganizes the debate. Same data, same initialization, same proposal generator, same schedule, same checkpoint rule:

| Configuration | PDMS | vs. no future |
|---|---:|---:|
| No future prediction | 93.31 | — |
| **One future shared across all candidates** | **92.81** | **−0.50** |
| Current latent as an extra pathway | 93.25 | −0.06 |
| **One future per candidate** | **93.46** | **+0.15** |
| + safety-critical hard negatives | 93.68 | +0.37 |

**The negative half of this is the robust part, and it is the more useful finding.** A future *shared* across candidates is worse than predicting no future at all, and the submetrics say why: NC and TTC improve (99.02, 96.54) while ego progress collapses from 91.36 to 88.68. An averaged future cannot tell the scorer *which* candidate causes a hazard, so it makes the policy uniformly cautious instead of discriminative. The current-latent control rules out "extra pathway" as an explanation for anything.

**This retro-explains SimWAM and DriveLaW rather than contradicting them.** SimWAM's isolated-mask ablation removed the action expert's access to a *single* future stream and lost nothing; DriveLaW conditions one planner on one generated future and finds earlier latents better than cleaner ones. Both are configuration (c). DA-WAM measures (c) at −0.50 PDMS. The three results are consistent under a sharper statement than the one this page previously made:

> **Shared future conditioning is useless to harmful. Only per-candidate futures help, and then by little.**

**How much weight the positive half deserves: not much.** +0.15 PDMS, single run, no seed variance, against a no-future baseline of 93.31 that would itself rank third in this wiki. WA-JEPA measured 0.053 seed std for a stochastic sampler and training-seed variance is typically larger. Within DA-WAM's own paper the representation choices are worth +2.42 and the hard negatives +0.22 — **the mechanism the paper is named for is the smallest effect in it.** And its predicted future reaches only **0.5 seconds** while candidates span 8 poses, so whatever it is doing, it is not evaluating the multi-second consequences the introduction promises.

**What survives across all six papers**: every one finds video *supervision* or a video *prior* essential; none demonstrates that a *shared* generated future helps at inference, and one measures it as harmful; the only positive inference-time result requires a distinct future per candidate and is worth 0.15 PDMS unreplicated. DriveVA's +19.5 PDMS and DriveWAM's backbone ablation isolate the training objective; SimWAM's mask and DriveLaW's denoising sweep both isolate the inference path and find no benefit there. The efficiency implication is immediate — SimWAM reaches 91.5 PDMS at 518 ms while DriveWAM's imagine-then-act loop costs 871–1262 ms per 4 s chunk.

**The strongest remaining case for generation** is DriveLaW's own Table 5: video-generator latents beat VLM hidden states by 2.6 PDMS and BEV features by 5.0 under a fixed planner. The *generator* is clearly valuable as a representation learner. What is unsupported is running it forward to a clean future at decision time.

**What is still unresolved**: whether imagined futures matter for capabilities NAVSIM does not measure — long-horizon rollout, counterfactual evaluation of candidate maneuvers, reactive interaction, or the instruction-conditioned generation Vega targets. NAVSIM's 4 s non-reactive horizon may simply be too short for anticipation to pay off. Also unexplained is *why* DriveLaW's t=10 conditioning collapses so completely; a 66-point PDMS drop suggests a distribution or scaling pathology rather than merely "redundant information," and no paper has diagnosed it.

### Action-Conditioned ≠ Counterfactual

There is a stronger version of the "counterfactual maneuver evaluation" escape hatch above, and [[sources/driving-wm-counterfactuals.md]] tests it directly. Its target is the claim — made by Vista ("counterfactual reasoning ability"), Drive-WM ("can generate counterfactual events"), Waymo's world model, and Genie 3 — that feeding a world model an alternative ego action yields the counterfactual for a recorded episode.

The argument is a conditioning argument, not a capability argument. A counterfactual query is posed *after* the episode is recorded, so the factual continuation $F^{+}$ is available evidence; direct action-conditioned prediction discards it:

$$
\underbrace{p\big(Y_{a^{\prime}}\mid H,\,F^{+}\big)}_{\text{counterfactual (rung 3)}}\quad\text{vs.}\quad\underbrace{p\big(Y\mid H,\,a^{\prime}\big)}_{\text{direct prediction (rung 2 at best)}}
$$

Both integrate the same mechanism $p(Y\mid w,a')$ and differ only in the posterior over the world — $p(w\mid H)$ versus $p(w\mid H,F^{+})$. So no amount of generator scale closes the gap, and the gap is widest exactly for the events that matter: anything first revealed after the shared history.

Measured on 186 controlled CARLA cases with matched counterfactual ground truth, direct predictions from Vista (diffusion) and DrivingWorld (autoregressive) score a recovered fraction of **0.38 and 0.31** — closer to a replay in which the event never happened than to the replay in which it did. Performance tracks how much of the event is inferable from the history alone (side street 0.29/0.25, where the event is revealed only afterwards; lead brake 0.50/0.37, a confounded control where an already-visible lead looms under acceleration), which is what the conditioning-gap analysis predicts.

**How this bears on the debate above.** It is not a verdict on imagine-then-act planning: a planner at decision time has no $F^{+}$, so rung 2 is the correct and only available target, and every action-conditioned generator in this wiki is doing an appropriate computation *for planning*. What the result removes is the retrospective claim — that the same machinery answers "what would have happened in that recorded incident." Sections above establish that generated futures do not help planning; this one establishes that they are not counterfactuals either. Full treatment in [[concepts/counterfactual-prediction.md]].

### 1. Coupling world model and trajectory planner
The world model must receive the planned trajectory as a condition, but the trajectory is what we're trying to optimize. Solutions:
- **Teacher forcing**: use ground-truth trajectories 50% of training time (UniUGP)
- **Feedback conditioning**: the world model is conditioned on the planning expert's output, training the planner to generate trajectories consistent with realistic future video

### 2. Computational cost
Video generation models (DiT-based, e.g., Wan2.1) are expensive. Solutions:
- Make generation expert optional at inference (UniUGP)
- Use lower-resolution occupancy instead of video (OccWorld)
- Predict latent features instead of pixels, in parallel (DeepSight: +3.57% latency vs. native VLM)
- Few-step ODE solvers over the flow path (DriveVA: 2 steps; DriveWAM: 3 video / 5–10 action steps)
- **Bound the KV cache during rollout** (DriveWAM's selective KV memory): for autoregressive world-action policies, the dominant long-horizon cost is not the denoiser but the growing history cache — 3.07 GB and 17.37 GFLOPs per step at 300s under full caching, reduced to 0.25 GB / 1.44 GFLOPs by content-based eviction. Age-based FIFO achieves the same budget but degrades accuracy badly (ADE@4s 0.89 → 1.40), because old tokens can remain decision-relevant while new tokens are often redundant background.

### 3. Evaluation
World model quality (FID, FVD) and planning quality (L2, collision rate) can improve independently or diverge. UniUGP is notable for improving both simultaneously.

## Metrics for World Model Quality

| Metric | Meaning |
|--------|---------|
| FID (Fréchet Inception Distance) | Distribution-level image quality; lower is better |
| FVD (Fréchet Video Distance) | Distribution-level video quality; lower is better |
| LPIPS vs. a matched reference | Spatial perceptual distance to a *specific* target video; penalizes locally wrong content, seams, and blur |
| Recovered fraction (Rec) | Semantic preference for a counterfactual replay over an event-free null replay, rescaled so 0 = event omitted and 1 = reference event reproduced ([[concepts/counterfactual-prediction.md]]) |

Note: FID/FVD measure distributional realism, not planning-relevant accuracy. A model with excellent FID could still predict unrealistic trajectories for edge-case scenarios.

The last two rows require something the first two do not: a **ground-truth video to compare against**, which real driving cannot supply for any alternative action. [[sources/driving-wm-counterfactuals.md]] obtains one by replaying the same CARLA world under the alternative ego action, and this is the wiki's only example of scoring a generated future against a matched reference rather than against a distribution. The pair matters — recovered fraction is category-sensitive (a plausible event of the right *kind* nearly satisfies it) while LPIPS is identity-sensitive, and the two disagree sharply when evidence is transported from the wrong episode.

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
| **DynVLA** | **✓ (image+BEV dynamics tokenizer)** | **✓ (dynamics tokens as CoT before actions)** |
| **Latent-WAM** | **Compact latent future-status prediction** | **No VLM; DINOv2 + trajectory decoder** |
| **Drive-JEPA** | **V-JEPA latent predictive video pretraining** | **No VLM; ViT + proposal planner** |
| **Policy World Model** | **Action-free future video forecasting used as planning rationale** | **Show-o-style unified AR policy; no separate VLM reasoning focus** |
| **DeepSight** | **Parallel 5-frame DINOv3 latent prediction in BEV (training target)** | **✓ (Qwen2.5-VL-3B + adaptive CoT + tokenized trajectory)** |
| **DriveWAM** | **✓ (Wan2.2-TI2V-5B is the policy core; chunked AR video generation at inference)** | **Advisory only (frozen Qwen3-VL-8B emits chunk-level text guidance; never decodes actions)** |
| **SimWAM** | **✓ at training (Wan2.2-5B co-trained); ✗ at inference (isolated mask drops the branch)** | **✗ (no VLM; lightweight action DiT only)** |
| **SGDrive** | **Structured symbolic forecast (occupancy + agent boxes at t and t+n); no generation** | **✓ (InternVL3-2B hosts the ⟨world⟩ queries and does VQA)** |
| **DriveLaW** | **✓ (LTX-Video 2B DiT; best FID in wiki, and its early latents are the planning state)** | **✗ (no VLM; 133M action DiT reads video latents directly)** |
| **Auto-JEPA** | **✓ in objective, ✗ in content — predicts the future *ego trajectory* latent, never a scene state** | **✗ (no VLM; frozen V-JEPA 2 + Transformer predictor + retrieval)** |
| **WA-JEPA** | **✓ (flow-matched future multi-view scene latents, generated at inference alongside the action)** | **✗ (no VLM; V-JEPA 2 ViT-L + joint MMDiT predictor)** |
| **GeoWAM** | **✓ (dense metric future point maps, forecast at inference and conditioning the action head)** | **✗ (no VLM; DVGT-2 geometry encoder + deterministic regression head)** |
| **DA-WAM** | **✓ (one 0.5 s scene latent per candidate trajectory, generated at inference and fed to the scorer)** | **✗ (no VLM; LoRA V-JEPA 2.1 + EMA target + factorized scorer)** |

## Generation-Quality Tables (updated August 2026)

These tables cover *visual generation* quality, not planning. Most world-model entries ingested after April 2026 (DriveVA, DriveWAM, SimWAM, DeepSight, Latent-WAM, Drive-JEPA, SGDrive, Auto-JEPA, WA-JEPA, GeoWAM) report **no FID/FVD at all**, because they either never decode pixels or treat generation as a training-time means rather than an output — and for Auto-JEPA the metrics are not merely unreported but undefined, since nothing about the scene is ever predicted — so the table below is sparse for recent work by nature, not by neglect. [[sources/drivelaw.md]] is the exception and now leads on FID. For planning standings see [[concepts/navsim-benchmark.md]].

### nuScenes Future Frame Generation (FID ↓)

| Method       | Type                      | Resolution | FID ↓   | FVD ↓    |
| ------------ | ------------------------- | ---------- | ------- | -------- |
| DriveDreamer | Diffusion                 | 128×192    | 52.6    | 452.0    |
| Drive-WM     | Diffusion                 | 192×384    | 15.8    | 122.7    |
| Doe-1        | Autoregressive            | 384×672    | 15.9    | —        |
| FSDrive      | Autoregressive            | 128×192    | 10.1    | —        |
| [Epona](../sources/epona.md) | AR+Diffusion | — | 7.5 | 82.8 |
| Vista        | Diffusion                 | —          | 6.9     | 89.4     |
| UniUGP       | AR+Diffusion (Wan2.1)     | —          | 7.4     | **75.9** |
| **[DriveLaW](../sources/drivelaw.md)** | **Latent diffusion (LTX-Video 2B)** | **1280×704** | **4.6** | 81.3 |

DriveLaW holds the best FID (4.6, a 33% improvement over Vista's 6.9) while UniUGP retains the best FVD (75.9 vs DriveLaW's 81.3) — the two metrics do not agree on a single leader. FID is also resolution-dependent, and DriveLaW generates at by far the highest resolution here, which cuts against it rather than for it. On nuPlan, DriveLaW beats Epona up to 80 frames but **loses at 100 frames** (FVD 296.1 vs 277.3), so its advantage is horizon-limited.

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

- Does trajectory-conditioned video generation improve **closed-loop** performance (NAVSIM PDMS), or only open-loop metrics? (FSDrive shows 85.1 PDMS, well below the current wiki non-BoN frontier of DriveSuprim 93.5 and below multiple VLM-style 90+ PDMS methods — generation quality may not translate to closed-loop driving)
- **[Partially answered by DreamerAD]** Can the world model provide a reward signal for RL, replacing or augmenting the simulator? — DreamerAD shows a latent reward model can replace simulator calls *during* RL rollout (87.7 EPDMS), but still requires simulator for initial vocabulary annotation. True simulator-free RL from world model rewards remains open.
- Does higher-resolution visual CoT (e.g., 512×768) substantially improve collision avoidance over FSDrive's 128×192?
- FSDrive only generates a front-view CoT. Does generating surround-view visual CoT improve performance in lane-change and merge scenarios?
- Can world model pre-training on massive unlabeled video (e.g., internet dashcam footage) bootstrap planning performance without any trajectory labels? (FLARE's FFP is designed for this but has not been tested at scale)
- **FSDrive vs. UniUGP tradeoff**: UniUGP's generation expert is optional at inference (speed-critical deployment), while FSDrive's visual CoT is mandatory. Does the always-on generation cost hurt real-time deployment?
- **DDP depth grounding**: DDP uses Depth Anything 3 pseudo-labels for both training and evaluation — does real LiDAR depth provide further improvement? Is geometric grounding from pseudo-labels sufficient for embodied planning?
- **Comfort under extended metrics**: both DDP (EC=79.4) and WAM-Flow (EC=73.9) score poorly on NAVSIM-v2 extended comfort. Does world model training inherently produce more aggressive trajectories? (FLARE achieves EC=87.5 without video generation — suggests comfort is driven by RL reward design, not world model type)
- **FLARE multi-step**: does extending FFP to predict features at t+2, t+3 provide further planning gains over single next-frame prediction?
- **[Largely answered by SimWAM] Video backbone scale**: DriveVA uses Wan2.2-TI2V-5B (5B params) and achieves 90.9 PDMS without RL. Would a smaller video backbone achieve comparable results? — SimWAM's Table 4 holds the planner fixed and swaps the prior: Wan2.1-1.3B reaches 90.2 versus Wan2.2-5B's 90.3, so **scale is nearly irrelevant in this regime**, while a weak prior (LTX-Video, 88.7) does cost, and a driving-pretrained prior (Cosmos-Predict2.5, 90.4) helps most. Whether the same holds for *zero-shot transfer* — DriveVA's distinguishing claim — is untested.
- **[Partially answered by SimWAM] Joint vs. sequential video-action coupling**: DriveVA (joint denoising, 90.9) and DriveWAM (inverse dynamics from the generated latent, 90.1) use the *same* backbone but neither cites the other, and their other components differ, so the 0.8 gap is unattributable. SimWAM adds a third option — no inference-time coupling at all — and scores highest (91.5), but likewise differs in RL stage, resolution, and action expert. SimWAM's Table 3 *is* controlled and finds bidirectional and action→video coupling give no benefit over isolation within its own architecture. A controlled comparison across the three papers is still missing.
- **Does test-time future imagination ever pay off?** SimWAM shows it does not on NAVSIM's 4 s non-reactive horizon (see [Does Test-Time Future Imagination Help?](#test-time-imagination)). The open part is whether it matters for what NAVSIM cannot measure: long-horizon rollout, reactive closed-loop interaction, or instruction-conditioned generation. **The counterfactual branch of that question now has a partial, negative answer**: [[sources/driving-wm-counterfactuals.md]] evaluates counterfactual prediction head-on and finds action-conditioned generation does not produce counterfactuals at all — recovered fraction 0.38 (Vista) and 0.31 (DrivingWorld), below the 0.5 no-preference point. That result concerns *retrospective* counterfactuals over recorded episodes; comparing candidate maneuvers *before* acting is a rung-2 question the benchmark does not test. If the answer is no everywhere, the imagine-then-act line (FSDrive, PWM, DriveVA, DriveWAM, DriveLaW) is paying inference cost for nothing.
- **Can a world model do abduction?** Inferring the realized state of a *specific* episode from its observed continuation is the one operation no ingested method implements — every world model here conditions on history and action only. [[sources/driving-wm-counterfactuals.md]] closes the gap with monocular depth plus splatting rather than with the model, so whether a model *trained* to condition on the factual continuation would beat geometry is untested. See [[concepts/counterfactual-prediction.md]].
- **Does a frozen advisory VLM beat a fine-tuned VLA backbone?** DriveWAM's frozen Qwen3-VL-8B only emits text guidance and never decodes actions, yet the guidance helps at every data scale. Is the advisory role sufficient, or does it leave value on the table versus VLM-centric policies (DriveVLA-W0, DynVLA) that fine-tune the VLM into the action path? No paper compares the two arrangements at matched backbone and data.
- **Long-horizon memory validity**: DriveWAM validates selective KV memory's accuracy only on 20s clips while profiling cost at 300s. Does content-based eviction hold up over minutes of rollout, and does the training/inference mismatch (full-history attention at training, bounded pools at inference) compound?
- **Zero-shot transfer ceiling**: DriveVA's zero-shot nuScenes/Bench2Drive gains are measured relative to PWM only. How does DriveVA compare zero-shot against VLA methods (FLARE, DriveFine) that are fine-tuned on the target domain? Does joint video-action training provide a sustainable generalization advantage at matched data scale?
- **Is the ego-motion target the right minimal one?** Auto-JEPA argues a planning world model should predict only the ego trajectory latent, and its 2.97× occlusion selectivity is evidence that agent-relevance emerges from that target alone. What is untested is whether the *JEPA objective* is doing the work. No ablation compares it against the obvious baseline — regress waypoints, encode the regression through the same frozen trajectory encoder, retrieve with that. If the two are equivalent, the contribution is the shared latent retrieval space, not joint-embedding prediction. See [[sources/auto-jepa.md]].
- **Does agent selectivity predict driving quality?** Auto-JEPA measures selectivity in latent space (mean $1-\cos$ 0.080 vs. 0.027) and shows behavioral consequences on three hand-picked scenes only. No paper has correlated an occlusion-sensitivity statistic with PDMS, collision rate, or interaction-scenario performance across a dataset. Until someone does, "the model attends to the right things" remains a property of the embedding rather than a demonstrated cause of good driving.
- **Which deterministic latent predictors are leaving performance on the table?** WA-JEPA measures that regression future-prediction is *worse than no future prediction* on multi-view scene latents (90.7 vs. 91.1 EPDMS), and diagnoses it as temporal-mean collapse. DeepSight (DINOv3 BEV frames), FLARE (DINOv2 features), and Latent-WAM (latent world status) all use deterministic objectives on scene-level targets. Whether their targets are low-entropy enough to be safe, or whether swapping in flow matching would buy each of them a point, is untested and cheap to test. See [Pattern 23](#objective-form).
- **Is WA-JEPA's inference-time scene denoising doing anything?** It generates future latents jointly with actions over 12 sampling steps, but its ablations never separate the training objective from the inference computation — the exact control SimWAM ran. If SimWAM's finding generalizes, the scene stream could be dropped at inference for free.
- **Does world modeling buy open-loop accuracy or closed-loop robustness?** [[sources/geowam.md]] adds future-geometry forecasting to DVGT-2 and gains **+0.6 EPDMS on navtest but +4.9 on navhard** — the same architectural change worth eight times more under the reactive protocol. If that asymmetry replicates, it reframes what world-model pretraining is for and implies navtest is close to the wrong benchmark for evaluating it. Every world-model paper in the wiki optimizes and reports navtest; only GeoWAM and DriveLaW report navhard at all. See [[concepts/navhard-ood-evaluation.md]].
- **Does dense geometric supervision rescue a deterministic latent objective?** GeoWAM pairs JEPA-style cosine alignment on future features — the objective [Pattern 23](#objective-form) measures as *worse than nothing* in isolation — with dense point-map regression, and does not collapse. The natural explanation is that explicit metric targets anchor what a pure feature-alignment loss lets drift toward the temporal mean. Neither paper runs the ablation, and it is one training run for either of them.
- **Geometry versus pixels has never been tested under a fixed planner.** GeoWAM argues geometry beats pixels but compares against other papers' methods; DriveLaW argues video latents beat BEV and VLM hidden states and *does* hold the planner fixed, but geometry is not in its comparison. The controlled experiment — one planner, three conditioning representations including metric point maps — would settle the field's central representation dispute and nobody has run it.
- **Are the 31 unsupervised futures actually futures?** [[sources/da-wam.md]] predicts one latent per candidate but can only supervise the expert-matched one, since offline logs record a single outcome. The other 31 are shaped purely by scorer gradients, and no diagnostic shows they encode anything future-like — a hard-braking candidate's latent is never checked against a full-throttle candidate's for the divergence physics requires. WA-JEPA's temporal-collapse metrics are exactly the right instrument and nobody has pointed them at this. If those latents are just conditioning features, "decision-aligned future prediction" is a scorer-capacity result wearing world-model clothes.
- **Does the shared-vs-per-candidate distinction survive at a realistic horizon?** DA-WAM's per-candidate futures reach only 0.5 s while its trajectories span 8 poses, so the action-specific consequences it claims to exploit — collisions, lane departures, rule violations — mostly fall outside the predicted window. Whether the +0.15 PDMS grows, vanishes, or reverses at 2-4 s is untested and is the single most informative follow-up the design admits.
