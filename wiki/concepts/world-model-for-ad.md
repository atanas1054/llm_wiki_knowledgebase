---
title: World Models for Autonomous Driving
type: concept
sources: ["raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md", "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md, raw/papers/Unified Driving Tokens_ Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning.md, raw/papers/ReWorld_ Representation Learning for World Action Models.md, raw/papers/LWDrive_ Layer-Wise World-Model-Guided Vision-Language ModelPlanning for Autonomous Driving.md, raw/papers/WCog-VLA_ A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving.md, raw/papers/GeoWorldAD_ Geometry World Action Model for Autonomous Driving.md, raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md, raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md, raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md, raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md, raw/papers/Epona_ Autoregressive Diffusion World Model for Autonomous Driving.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/DynVLA_ Learning World Dynamics for Action Reasoning in Autonomous Driving.md, raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md, raw/papers/DeepSight_ Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving.md, raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md, raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md, raw/papers/DriveLaW_ Unifying Planning and Video Generation in a Latent Driving World.md, raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md]
related: [sources/metis.md, sources/suv.md, concepts/alpasim-benchmark.md, sources/qwen-drive-1.0.md, sources/drive-hwm.md, sources/coworld-vla.md, sources/drivefuture.md, sources/unified-driving-tokens.md, concepts/visual-tokenization.md, sources/reworld.md, sources/lwdrive.md, sources/wcog-vla.md, sources/geoworldad.md, sources/adaptive-wam.md, sources/brainwam.md, sources/foresight.md, sources/da-wam.md, sources/geowam.md, sources/wa-jepa.md, sources/auto-jepa.md, sources/simwam.md, sources/sgdrive.md, sources/drivelaw.md, sources/uniugp.md, sources/futuresightdrive.md, sources/drivedreamer-policy.md, sources/drivevla-w0.md, sources/flare.md, sources/dreameraD.md, sources/vega.md, sources/epona.md, sources/driveva.md, sources/explorevla.md, sources/dynvla.md, sources/onevl.md, sources/latent-wam.md, sources/drive-jepa.md, sources/policy-world-model.md, sources/deepsight.md, sources/drivewam.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, concepts/rl-for-ad.md, concepts/physicalai-av-benchmark.md, concepts/counterfactual-prediction.md, sources/driving-wm-counterfactuals.md]
created: 2026-04-05
updated: 2026-09-27
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

**DriveFuture re-runs this comparison and the sign of the regression term flips.** [[sources/drivefuture.md]]'s Table 4, on navhard with everything else fixed:

| Configuration | EPDMS |
|---|---:|
| No future frames in training (foresight latent still conditions the planner) | 30.9 |
| + **MSE regression** on the future latent | 32.1 |
| + **attention grounding, no prediction loss** | **34.6** |

Regression is **+1.2 over the null here**, where WA-JEPA measured it at **-0.4**. The ordering is the same in both papers and the magnitude of the gap to the better option is similar (+2.5 here, +1.0 there), but "regression on future latents is actively harmful" does not generalise — it was a property of WA-JEPA's target, not of regression.

**The entropy-of-the-target framing explains the difference and survives both.** WA-JEPA regresses multi-view EMA ViT-L scene latents; DriveFuture regresses a 16-token compression of a 64-token BEV bank, an order of magnitude lower in dimension and far less multimodal. Higher target entropy, worse regression — which is the same axis that reconciles Pattern 22's deterministic ego-trajectory alignment working at all.

**The third arm is the one that is new.** DriveFuture's winner has *no prediction objective*. So the design space is not "regression vs. flow matching" but:

| Objective on the future latent | Best result in its own paper | Target entropy |
|---|---|---|
| Deterministic regression | Worst or near-worst in both papers that test it | any |
| Generative (flow matching) | +1.0 over regression, +0.6 over null (WA-JEPA) | high |
| **None — planning loss plus attention grounding to a real future** | **+2.5 over regression (DriveFuture)** | low |

The two winners are not in competition: flow matching wins when the future latent must *stand alone* as a prediction, and no-objective-plus-grounding wins when it only has to be a useful conditioning context. **Nobody has run both arms on one target**, and that is now the cleanest open experiment on this page.

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
| **Dense optical flow** | **Drive-HWM** (slow branch) | **No — pseudo-labels from a flow estimator** | **No — 2D image space** ([tension](#flow-vs-geometry)) |

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

### 26. The Frozen Generator as the Planner's Primary Encoder (ForeSight)

**ForeSight** ([[sources/foresight.md]], Fudan + Shanghai Innovation Institute + Imperial + Surrey) is the maximal version of Pattern 1. Where every other cascaded design treats the generator as one input among several, ForeSight declares it **the** visual encoder: a frozen 2.5B Epona is run forward at inference to an actual imagined future, and the trajectory decoder's job is to read it. The 52M TransFuser current-frame branch is explicitly labelled "an additional supplement."

Three design elements are downstream of that commitment and are reusable independently of it:

- **WM-QFormer** — a spatiotemporal Transformer with $N_{\rm wm}$ learnable queries per generated frame, compressing $F_{\rm wm}\in\mathbb{R}^{T_{\rm wm}\times C_{\rm wm}\times H\times W}$ to $T_{\rm wm}\times N_{\rm wm}\times C$. Its stated purpose is to strip "abundant fine-grained textures and noise" from generated frames before the planner sees them — the same pathology DriveLaW diagnosed at t=10, addressed by filtering the finished future instead of reading an earlier one.
- **Time state queries** bound one-to-one to future timesteps (from the authors' BridgeAD), which is what makes per-frame temporal alignment between generated frames and trajectory poses possible at all.
- **Factorized attention** — present and future are consumed by two separate cross-attentions, with sinusoidal position embeddings so $T_{\rm wm}\neq T_{\rm f}$ is admissible.

**The one architectural question this pattern answers that no other does**: can a planner run on generated futures *alone*? Table 7 removes the current encoder entirely and scores **88.2 PDMS** — 1.4 above the same pipeline's no-world-model baseline (86.8) and level with WoTE. The current encoder is worth +1.1, concentrated in DAC and EP, i.e. exactly the side-view and geometry information a front-view-only generator cannot supply. Whether the current encoder can eventually be dropped is a question about multi-view generation, not about planning architecture.

**And the cost is fully disclosed**, which is rare: 900 ms total, **870 ms of it the world model**, on an H100. That number is what makes ForeSight the wiki's cleanest cost-benefit datapoint for the synthesis below.

### 27. Two Branches Meeting Only in Action Space (BrainWAM)

Every pattern above couples a world model to a planner through *representations* — features, latents, tokens, point maps. **BrainWAM** ([[sources/brainwam.md]], CASIA + Li Auto) is the first in the wiki to argue that representation-level coupling is the wrong interface when a VLM is also in the room, and it has the negative result to motivate the claim.

**The diagnosis first.** Putting VLM tokens, video-generator tokens, and action tokens into one shared attention space — the paper's **Tri-MoT** baseline — scores **87.8 PDMS, below its own WAM-only branch at 88.1**, despite strictly more information and comparable parameters. The mechanism is **modality competition**: action tokens attend more to VLM tokens than VGM tokens across most layers, because a large-scale-pretrained VLM stream is clean and stable while a rectified-flow video stream is still emerging from noise. The optimizer takes the semantic shortcut and the predictive dynamics go underused.

**The fix is an interface constraint, not a new objective.** Each branch first compresses itself to **8 action tokens at 1024 dim**; the two branches then communicate only through those, via zero-init gated cross-attention at two layers (**CAB**, 16.8M) followed by a 2-layer Transformer fusion and element-wise mean (**CIF**, 49.3M). Raw modality tokens never share an attention pool. Result: **89.5 PDMS / 89.6 EPDMS**, +1.4 over WAM-only and +1.7 over Tri-MoT.

**Where the boundary of the claim sits.** Joint video-action attention is fine on its own — [[sources/simwam.md]]'s bidirectional mask scores 90.2 against 90.3 isolated, [[sources/driveva.md]] reaches 90.9 and [[sources/drivewam.md]] 90.1 with joint denoising or shared attention. None of those has a VLM in the pool. **The harm is specific to mixing a clean semantic stream with a denoising one**, which makes this a constraint on VLA+WAM hybrids rather than on world-model coupling generally.

**Two further transferable results.** Cross-stream communication **saturates after two CAB blocks** — inserting 28 (every layer) scores the same 89.3 as 2 — which suggests that whatever the branches need to exchange is low-dimensional. And **freezing both pretrained branches beats end-to-end fine-tuning by 0.7 PDMS** (89.5 vs 88.8), attributed to a measured convergence-rate mismatch: the VLA branch reaches its plateau at 54K steps, the WAM branch needs 81K.

### 28. Quality-Routed Early Exit From the Generator's Middle (Adaptive-WAM)

Every pattern above fixes *where* the planner reads the world model — final layer, mid-denoising latent, compressed action tokens — and then argues about how much to denoise. **Adaptive-WAM** ([[sources/adaptive-wam.md]], AIR Tsinghua + USTC + Beihang) makes the readout point a *per-scene decision variable* and shows the choice matters more than the denoising schedule does.

**The representation.** A single conditional forward through Wan2.2-TI2V-5B at a fixed noise index — no denoising loop, no classifier-free-guidance unconditional branch, no VAE video decode. Six independent ReCogDrive-style 5-step trajectory heads hang off blocks {5, 9, 15, 18, 22, 30}, sharing architecture and training budget so depth is the only variable.

**The controller.** At each attempted exit, decode one trajectory, score it with a fine-tuned DINOv2-Small verifier that predicts the six NAVSIM components from the current image and the candidate poses alone, and terminate when the best trajectory accumulated so far clears a threshold $\eta$. Rejected exits cost only the *unevaluated* blocks — hidden states and scores are cached.

**Why route rather than just pick block 15.** Because depth ordering is not scene-wise dominance. Post-RL Jaccard overlap between exits' high-quality scene sets runs **0.69–0.82**, and while block 15 beats block 30 by ≥50 points on 598.6 scenes, block 30 beats block 15 on **422.4**. No fixed depth dominates.

**What it buys**: 90.79 PDMS at **170 ms** against 90.62 at 190 ms for the best fixed exit. The routing gain (+0.17) is within plausible noise; the latency result is the real one, and the permissive-threshold control ($\eta=70$: 112 ms but −2.13 PDMS) shows the saving comes from *conditional* allocation rather than from simply being shallow.

**Two design lessons that generalize past this architecture.** First, **the video prior must be adapted jointly with the action objective** — a frozen Wan scores 84.20, separately-tuned-then-cached features score 84.95, joint LoRA scores 90.62, and full fine-tuning adds 0.02 on top. Second, **the verifier should not be a ranker**: more than 95% of scenes contain candidate groups that are jointly perfect, jointly zero, or tied at the top, so Adaptive-WAM predicts un-binarized metric components with soft-label BCE and uses no rank loss at all.

**Classification note.** Adaptive-WAM keeps video prediction as a *training* objective and never decodes a future at deployment, so it belongs with the training-time-only camp (SimWAM, DriveVLA-W0, FLARE) rather than with imagine-then-act — despite running a generative backbone at inference. What it consumes is the generator's internal state, not its output, which is DriveLaW's position taken to its logical end.

### 29. Ego-Aligned Multi-Scale Geometry With Latent Future Tokens (GeoWorldAD)

[[sources/geoworldad.md]] (NTU + Xiaomi EV + Zhejiang) is the wiki's second geometry world-action model after [Pattern 24](#24-metric-geometry-as-the-world-model-state-space-geowam), arriving independently, from a different group, with the same DVGT-2 ancestor and no mutual citation. Its architecture differs from GeoWAM's on three axes, and each is ablated.

**Ego-aligned geometry, not anchor-frame geometry.** StreamVGGT reconstructs everything in the first frame's coordinate system; trajectories live in the moving ego frame, so misalignment accumulates. **EgoStreamVGGT** expresses each point map in the ego-camera frame of its own timestep and camera poses as adjacent-frame relative transforms. This is a pure re-parameterization with no added capacity, and it is worth **+2.5 PDMS** (84.8 → 87.3). The row above it is the more striking one: an off-the-shelf StreamVGGT with 4D reconstruction supervision beats a from-scratch planner by **0.6 PDMS while lowering NC, DAC, and TTC**. A geometry foundation model in the wrong frame is close to worthless — the first measurement of an argument [Pattern 24](#24-metric-geometry-as-the-world-model-state-space-geowam) makes rhetorically.

**Latent future tokens, not dense future point maps.** 4 chunks × 64 tokens spanning 2 s, built by a Q-Former that cross-attends to present geometry then applies causal self-attention across chunks. Future *depth* supervises them through the shared DPT head with a stop-gradient so the present decoder is not distorted, and **the decoder is not needed at planning inference** — only the latent tokens reach the planner. Where GeoWAM forecasts dense metric point maps and infers action from them inverse-dynamics style, GeoWorldAD keeps the future compressed and uses it as one refinement stage among five.

**Multi-scale, iteratively consumed.** Geometry tokens from layers {4, 11, 17, 23} of a 24-block decoder, each feeding one trajectory-refinement stage with supervision at every stage. Its Table 6 decomposes the two ideas cleanly: iterating on the final layer alone buys **progress** (EP 81.5 → 82.9) and nothing else; adding multi-scale buys **safety** (DAC 95.5 → 97.2, NC 98.6 → 98.9); and consuming all 24 layers in one stage is the *worst* of the three despite the most information. See [Pattern 28](#28-quality-routed-early-exit-from-the-generators-middle-adaptive-wam) — two papers, two backbone families, both finding that reading the final layer is the wrong default.

**Result**: 91.0 PDMS v1 / 90.4 EPDMS v2, camera-only, no map/box/occupancy supervision, with the future module worth +1.7 / +2.8 — the first sizeable positive shared-future result here, discussed [below](#shared-future-reopened). The planner uses 64 proposals with a scorer distilled from NAVSIM's own PDMS composition, so it is not a single-trajectory result. No latency, FPS, or parameter count is reported anywhere.

### 30. Joint Multi-Agent Trajectories as the Prediction Target (WCog-VLA)

Every pattern above forecasts something about the *scene*: pixels, video latents, semantic features, occupancy voxels, metric point maps, symbolic queries, or the ego's own action latent. [[sources/wcog-vla.md]] (Tongji + NTU) forecasts something about the *other agents* — a joint diffusion over $N_m$ agents' future trajectories, generated together with the ego's.

The argument for it is a gap in the taxonomy rather than a gap in fidelity. Predicting future images or future geometry is a **perceptual** task about how the world will look; it says nothing directly about the *reciprocal interplay* between ego and neighbours, because a scene forecast conditioned on one ego plan cannot express "if I go, they yield." A joint multi-agent rollout can.

**Two levels, both ablated.** At the semantic level, agent tokens from a BEVFormer + TrackFormer stack enter an InternVL3-2B VLM, and the returned `O_agent` hidden states are decoded by a world head into current 3D boxes *and* future agent trajectories. At the generative level, the **ADDT** — a 16-block DiT split into an 8-block condition encoder and an 8-block generation decoder — synthesizes the joint rollout conditioned on the VLM tokens, with an agent-specific loss mask prioritizing ego accuracy.

| Configuration (three-stage SFT) | PDMS |
|---|---:|
| Neither level | 86.5 |
| + semantic current perception | 87.0 |
| + semantic future agent trajectories | 87.2 |
| + both semantic | 88.1 |
| **+ generative joint multi-agent only** | **87.4** |
| Both levels | **89.3** |

**The row that matters for the debate below is the generative-only one: +0.9 PDMS from a shared joint rollout with no semantic world supervision at all.**

**One mechanism worth stealing regardless of the target.** ADDT's condition encoder is pulled toward a latent from a GenAD-style VAE pretrained to reconstruct multi-agent trajectories, via cosine similarity at the 6th block. Its stated purpose is *stability across denoising timesteps* rather than fidelity — and its measured effect is exactly that: at 5 denoising steps the alignment and decoupling together are worth +1.9 PDMS (87.4 → 89.3), but at 20 steps only +1.1 (88.5 → 89.6). **The architectural mechanisms substitute for denoising budget**, which is the cleanest statement in the wiki of why few-step planners can match many-step ones.

### 31. Future-Frame Supervision on VLM Hidden States, Consumed at Several Depths (LWDrive)

[[sources/lwdrive.md]] (Chongqing University) is the training-time-only counterpart to [Pattern 29](#29-ego-aligned-multi-scale-geometry-with-latent-future-tokens-geoworldad): the same interleaving of planner refinement stages with backbone depth, but on a **VLM** and with the future never instantiated at inference.

**The world model.** A frozen VAE encodes a future frame $I_{t+\Delta}$; a denoising head conditioned on the concatenated final-layer vision and action-query hidden states, $c_t=[F_t^{\mathrm{V}};h_t^{\mathrm{A}}]$, predicts the clean latent. This runs only in Stage 1, weighted $\lambda_{\mathrm{wm}}=15$ against $\lambda_{\mathrm{traj}}=1$ — the most lopsided world-model weighting recorded on this page — and is discarded at deployment.

**The consumer.** Qwen2.5-VL-3B emits an intent-aware coarse trajectory. A **Foresight Cascade Planner** initialises a proposal pool from the *pooled action-query latent* (not the decoded trajectory), then runs six refinement stages, each reading hidden states from a different Qwen layer through **Bridge Attention** — one attention over concatenated proposal-self, action-query/ego-state, and VLM-foresight memories — followed by BEV-grounded residual updates. A score head distilled from NAVSIM's PDMS composition selects. **92.0 PDMS v1 / 89.6 EPDMS v2, no RL.**

**Two things this pattern contributes, and they are both boundary conditions rather than confirmations.**

1. **Multi-depth readout on a VLM is worth +0.2**, against +4.80 on a video DiT ([Pattern 28](#28-quality-routed-early-exit-from-the-generators-middle-adaptive-wam)) and +1.1 on a geometry decoder (Pattern 29). The *direction* reproduces GeoWorldAD's — multi-scale buys DAC and TTC, costs ego progress — but the magnitude does not. Full treatment in [[concepts/foundation-backbones-for-ad.md]].
2. **The world-model contribution is measured in the wrong configuration.** Its Table 4 varies future-frame supervision only with **no refinement stack present** (84.5 → 86.3, +1.8 on a directly decoded VLM trajectory), and no row runs the full FCP without it. So LWDrive's headline claim — that the cascade consumes *foresight* features rather than merely deep ones — has no experiment behind it. Of the +7.5 from ablated baseline to headline, the refinement stack is +5.5, the world model +1.8, and the layer-wise schedule +0.2.

**Classification.** Training-time-only, unambiguously: no future image or latent is produced at inference, and the qualitative future frames in Figure 4 are decoded for illustration only. It is the first entry in that camp whose future target is supervised **directly on VLM hidden states** rather than on a video backbone's activations ([[sources/simwam.md]], [[sources/adaptive-wam.md]]) or a separate feature predictor ([[sources/flare.md]], [[sources/onevl.md]]).

### 32. Explicitly Optimizing the Latent Pathway Itself (ReWorld)

Every pattern above chooses *what* the world model predicts and *where* the planner reads it. [[sources/reworld.md]] (HUST + Xiaomi EV — the same group as [Pattern 21](#21-mid-denoising-latents-as-the-planning-state-drivelaw)) is the first to ask what should be *true of the intermediate states themselves*, and names the gap the **representation bottleneck of WAMs**: under standard output-level objectives, intermediate video states are supervised only through the final denoising output and action states only through the final trajectory, so nothing requires the connecting latents to be future-predictive, cross-modally grounded, or behaviour-sensitive.

Three objectives, all derived from the model's own targets — **no external encoder, no teacher branch, 1.003× per-step training cost**:

| Objective | Where | Mechanism |
|---|---|---|
| $\mathcal{L}_{\mathrm{Mid}}$ | Video DiT block 8 of 28 | Auxiliary head predicts the **same flow-matching velocity target** as the final head, moving the future constraint into representation formation |
| $\mathcal{L}_{\mathrm{align}}$ | Action DiT cross-attn layer 12 | Cosine alignment of the post-attention state to its **stop-gradient attended readout** $r_i=\sum_j\alpha_{ij}v_j$ — self-distillation of an attention operation |
| $\mathcal{L}_{\mathrm{RDE}}$ | Action DiT output | Repulsion in delta space from the **nearest low-scoring trajectory** among 64 PDM-simulator-scored candidates |

**Two mechanisms here are reusable outside driving entirely.** $\mathcal{L}_{\mathrm{align}}$ is a general fix for any cross-attention interface where conditioning may be used transiently rather than retained — the stop-gradient is what makes it a grounding constraint rather than a mutual-collapse objective. And the intermediate-head design turns "supervise the middle of the network" into a free operation whenever the training target is already a per-token regression target, which is true of every flow-matching or diffusion backbone on this page.

**What it achieves**: FVD 81.3 → **61.9** and FID 4.6 → **4.4** on nuScenes (the first entry here to lead both), PDMS 89.1 → **90.4** with no RL, UCF-101 frozen linear probe 68.3% → **80.2%**.

**What it does not show is the thing it is named for.** Its planning ablation (DriveLaW 89.1 → +align 89.5 → +RDE 89.8 → both 90.4) contains **no row for the video-side objective**, and its implementation section says Stage 2 initializes from the *DriveLaW* checkpoint. So the entire +1.3 PDMS is attributable to two action-side objectives — one of which ($\mathcal{L}_{\mathrm{RDE}}$) compares two trajectories in delta space and touches no world information at all. Meanwhile the +19.4 FVD is 17.0 sampling trick (self-guidance at $\gamma=1.4$) and 2.4 representation objective. See [the decoupling analysis](#generation-planning-decoupling) below.

### 33. The Future Latent as a Planning *Condition*, With a Training-Time Oracle Annealed Away (DriveFuture)

[[sources/drivefuture.md]] (CASIA and others) varies an axis orthogonal to every pattern above: not *what* the world model predicts, not *where* the planner reads it, but **whether the future latent is an output or an input**.

Patterns 8, 14, 15, 17, 22, 23, 24, 29 and 30 all define a prediction target and a distance to it. DriveFuture defines **no future-prediction loss at all**. Its total objective is $\lambda_{\mathrm{plan}}\mathcal{L}_{\mathrm{plan}}+\lambda_{\mathrm{bev}}\mathcal{L}_{\mathrm{BEV}}$; the 16-token foresight latent $\hat{\mathbf{Z}}_{t+T}$ is shaped entirely by (a) gradients from trajectory denoising and (b) an attention-based grounding against a real future observation. Appendix B.2 states the reframing directly: instead of asking whether the future latent *can be predicted*, ask whether it *improves denoising of the planned trajectory*.

**The grounding mechanism is the part worth stealing.** A single cross-attention layer, training only:

$$\mathbf{Z}_{t+T}=\operatorname{sg}(\phi_{\mathrm{enc}}(\mathbf{I}_{t+T}))\in\mathbb{R}^{64	imes d},\qquad 	ilde{\mathbf{Z}}_{t+T}=\operatorname{MHA}(\operatorname{LN}(\hat{\mathbf{Z}}_{t+T}),\,\mathbf{Z}_{t+T},\,\mathbf{Z}_{t+T})$$

with **no residual from the query side**, so the grounded condition is a pure *selection* of ground-truth future evidence, indexed by what the model predicted. The prediction supplies the question; the real future supplies the answer. Then LatentAlign anneals it out:

$$	ilde{\mathbf{Z}}^c_{t+T}=lpha(e)\,\mathbf{Z}^c_{t+T}+(1-lpha(e))\,\hat{\mathbf{Z}}_{t+T},\qquad lpha(e)=1-\sigma(eta(e-e_0))$$

This is **the first privileged training-time signal in the wiki that is explicitly scheduled to zero rather than simply dropped at test time.** The other privileged-supervision cases on file — Hydra-MDP distillation, [[sources/auto-jepa.md]]'s CLOVER scorer, [[sources/adaptive-wam.md]]'s pseudo-expert targets, [[sources/da-wam.md]]'s factor heads — all distil a *simulator's labels* into a head that keeps using them. DriveFuture hands the planner a real future *observation* and then withdraws it, and the withdrawal schedule turns out to be the most sensitive hyper-parameter in the system.

| Where the future enters | Trained against | Examples |
|---|---|---|
| Prediction target, planner reads the prediction | A distance to a future state | Patterns 8, 14, 15, 17, 23, 24, 29 |
| Generated observation/latent, planner reads it | Generation loss | Patterns 1, 11, 19, 21, 26, 27, 28 |
| Output-level scoring over candidates | Simulator labels | WoTE, GTRS, Pattern 25 |
| **Conditioning context, no prediction loss** | **Only the planning loss, plus attention grounding to a real future** | **Pattern 33** |

**What it is worth.** On navhard, adding future-frame grounding to an architecture that already carries the foresight latent moves 30.9 to 34.6 (+3.7) — but see the caveat below, and note that a GTRS-Dense scorer on top is worth **+20.9** on the same model.

**The pattern's own price is not measured.** DriveFuture's ablation baseline removes *future frames*, not the world model: the foresight latent still conditions the DiT in every reported row. `use_wm` and `use_wm_to_dit` are named in its Appendix C.3 and varied nowhere. So Pattern 33's headline claim — that a predicted future latent conditioning a diffusion planner beats not having one — is unmeasured in the paper that proposes it.

#### An oracle held too long is worse than no oracle {#oracle-annealing}

DriveFuture's $e_0$ sweep (the annealing inflection, as a fraction of total epochs) is the most transferable number in the paper and it is reported without comment:

| $e_0$ | EPDMS (navhard, no scorer) | Reading |
|---:|---:|---|
| 0.75 | 29.2 | Oracle withdrawn too early |
| **0.83** | **34.6** | Default |
| 0.95 | **28.9** | Oracle held nearly to the end — **below the 30.9 no-future-frame baseline** |

Two swept values land *below* not using future observations at all, and so does $q_s{=}4$ (28.2). The span of the $e_0$ sweep, 5.7 EPDMS, exceeds the +3.7 the whole mechanism is worth.

**This is the training-side counterpart of [[sources/drivelaw.md]]'s t=10 collapse**, and the two are worth stating together because neither paper cites the other's phenomenon:

| Paper | What was fed to the planner | Result |
|---|---|---|
| DriveLaW | A **generated** future at increasing denoising completion | 89.1 (t=1) → 86.9 (t=5) → **23.2** (t=10, near-clean) |
| DriveFuture | A **real** future, held for an increasing fraction of training | 34.6 ($e_0{=}0.83$) → **28.9** ($e_0{=}0.95$) |

Different signals, different mechanisms, same shape: **the closer a planner's conditioning gets to a future it cannot produce at inference, the worse it does.** DriveLaW reaches the cliff by making the conditioning signal more finished; DriveFuture reaches it by leaving the privileged signal in place longer. Both are exposure bias in a world-model interface, and DriveFuture's LatentAlign is the only remedy for it anyone here has implemented — which makes its sensitivity the most important open number in this pattern.

### 34. Four Future Targets in Parallel, in One Latent Space (CoWorld-VLA)

Patterns 1-33 each pick **one** thing the world model predicts. That choice is the organizing axis of this entire page, and no ingested paper had questioned whether one is enough. [[sources/coworld-vla.md]] (Afari Intelligent Drive + Southeast University + others) does, and runs four future-prediction objectives against four different teachers, all landing as typed expert tokens inside a single Qwen3-VL-2B's hidden states:

| Expert token | Target | Objective form | Pattern it instantiates |
|---|---|---|---|
| $H_{\mathrm{sem}}$ | Pooled frozen **V-JEPA** features of the *future* frame | SmoothL1 + cosine | 15 / 23 (JEPA latent prediction) |
| $H_{\mathrm{geo}}$ | Pooled frozen **VGGT** features of the *future* frame | MSE | 24 / 29 (geometry) |
| $H_{\mathrm{dyn}}$ | Future video latents, via a **Wan2.2-5B** DiT the token conditions | Flow matching | 19 (video prior, training-time only) |
| $H_{\mathrm{traj}}$ | GT waypoints | MSE | 22 (ego trajectory) |

**The dynamic branch is the mechanism worth extracting.** $H_{\mathrm{dyn}}$ *replaces the text condition* of the Stage-1 video model, and the video model's flow-matching loss then back-propagates into that one VLM token. The paper states the point precisely: *"The VLM itself does not directly decode future images... This design allows the dynamic tokens to learn future motion trends and temporal consistency without requiring the VLM backbone to act as a pixel-level generator."* A video generator used as a **differentiable critic on a latent** is strictly cheaper than making the planner generate, and it generalizes to any conditioning interface a generator already exposes.

**Are the four complementary?** Its Table 4 says yes, weakly, and the marginal contributions are order-dependent (NAVSIM-v1 PDMS, Stage-2 readout):

| | EgoT. | +Geo. | +Sem. | +Sem.+Dyn. | +Geo.+Sem. | All four |
|---|---:|---:|---:|---:|---:|---:|
| PDMS | 83.7 | 85.1 | 85.2 | 87.3 | 87.7 | **88.7** |

No expert is redundant and the full set wins, but **the dynamic expert is never measured alone** — the one missing row, EgoT.+Dyn., is exactly the world-model branch the paper is named for. And geometry beats dynamics as the third addition (87.7 vs. 87.3), which the paper's own learned fusion weights invert (dyn 0.35 > traj 0.31 > sem 0.19 > geo 0.15).

**Where the gains land is the transferable part.** Across the sweep DAC moves +3.8 and EP +3.8 while NC moves +0.7. That is the fourth instance on this page of [geometric and structural supervision buying progress and compliance rather than collision avoidance](#shared-future-reopened), after GeoWAM, GeoWorldAD and UDT.

#### Objective form is assigned by target entropy, correctly and silently {#entropy-assignment}

Read against [Pattern 23](#objective-form), CoWorld-VLA is an unintentional confirmation. The wiki's rule — assembled from [[sources/wa-jepa.md]]'s regression-vs-flow-matching result and [[sources/drivefuture.md]]'s reversal of its sign — is that **deterministic regression fails in proportion to the entropy of the target**. CoWorld-VLA assigns:

| Target | Entropy | Objective chosen |
|---|---|---|
| Future multi-frame video latents | High | **Flow matching** |
| *Pooled* V-JEPA future features | Low (pooled to a few tokens) | Regression (SmoothL1 + cosine) |
| *Pooled* VGGT future features | Low | Regression (MSE) |
| One future trajectory | Lowest | Regression (MSE) |

This is the correct assignment on every row, and the paper never states the principle or cites anyone who does. **Pooling is the move that makes it work**: it converts a high-entropy dense feature field into a low-entropy summary, at which point a conditional mean is a usable prediction. Neither WA-JEPA nor DriveFuture tried pooling as the lever, and it is cheaper than changing the objective.

#### The generator is discarded; the token it supervised is not

The paper's limitations section is explicit: *"the Wan model is not used in Stage 3 or during planning inference."* So the video world model is training-time-only, alongside Patterns 8, 13, 14, 19 and LWDrive. But $H_{\mathrm{dyn}}$ — the token that video model shaped — is produced by the VLM at inference and conditions a denoising branch directly. **That is a third position between the two camps below**: not "generate a future at decision time" and not "use future prediction only to shape a shared representation," but *keep the specific latent the generator supervised, and route it into the planner as its own conditioning stream*. [[sources/drivefuture.md]] is the closest relative, with one such latent instead of four.

The deployment cost lands elsewhere, and unpriced: Stage 3 still runs **frozen V-JEPA and VGGT on the current frame** for its scene stream. Discarding a 5B video model while keeping two foundation encoders in the input path is a trade nothing in the paper measures.

### 35. Two World Models at Two Rates, With Two Different Targets (Drive-HWM) {#two-rates}

**Drive-HWM** ([[sources/drive-hwm.md]]) varies an axis every pattern above holds fixed: **the rate at which the future is predicted, relative to the rate at which the agent decides.** Patterns 1–34 all run their world model and their planner on one clock. Drive-HWM runs two:

| | **Slow world model** | **Fast world model** |
|---|---|---|
| Period | every $N=8$ steps | every step |
| Backbone | V-JEPA-family (see the [naming note](../sources/drive-hwm.md#vjepa-naming)) | Emu3-8B |
| Predicts | $K=8$ future **optical-flow** fields, all in parallel from one history | the **next RGB frame** + the immediate action |
| Reaches the planner as | a **Dynamic-Aware Latent** (the pre-decoder hidden state), via FiLM | action tokens |
| Cost | 25.6 ms, amortized to **3.2 ms** | 81.6 ms |

**Three things here are new to this page.**

**1. Dense optical flow is a new target family, and it is annotation-free.** The page's target table gains a row that behaves unlike its neighbours:

| World-model target | Methods | Annotation needed | Same frame as the action? |
|---|---|---|---|
| **Dense optical flow** | **Drive-HWM** | **No — pseudo-labels from an off-the-shelf flow estimator** | **No — 2D image space** |

The motivating argument is about information density rather than fidelity: RGB features are "dominated by appearance semantics and spatial content, while the information most relevant to driving — including ego-motion, object displacement, and their temporal evolution — may occupy only a small portion of the representation." Flow inverts that ratio by construction. It shares [[sources/geowam.md]]'s annotation-free property (a foundation estimator supplies the targets) without its coordinate-frame property, which sets up the [tension below](#flow-vs-geometry).

**2. Flow is the objective; the latent is the interface.** The decoded flow field never reaches the planner. What crosses is the hidden state that had to be sufficient to reconstruct it, on the explicit grounds that raw displacement is uninterpretable — "similar image displacement may correspond to different driving implications depending on whether it originates from a vehicle, a pedestrian, the road surface, or camera motion." This is the same shape as [[sources/geowam.md]]'s point-map supervision feeding a latent planner and [[sources/coworld-vla.md]]'s Wan-as-differentiable-critic on one token, and it is the generalizable form: **any dense pseudo-labelled target can supervise a planner's conditioning latent without the dense field ever entering the inference path.**

**3. A fourth position in the training-time/inference-time taxonomy.** Both of Drive-HWM's world-model *objectives* are training-time-only and confined to nuPlan pretraining — the NAVSIM stage uses the action loss alone. But the **slow predictor still runs at inference**; only its flow decoder is discarded. So: decoders discarded (like Pattern 19), predictor retained (unlike Pattern 19), and the retained predictor's supervision came from a different dataset than the planner's. CoWorld-VLA keeps a supervised *token*; Drive-HWM keeps a whole supervised *predictor*.

**Parallel multi-offset prediction is the structural answer to rollout drift.** All $K$ flow fields are predicted from the same clean history, $p(\mathcal{F}_{\tau+1:\tau+K}\mid\mathcal{H}_\tau)=\prod_k p(\mathcal{F}_{\tau+k}\mid\mathcal{H}_\tau,k)$, so no early error is warped forward — the problem [[sources/epona.md]] attacks with chain-of-forward training and [[sources/drivewam.md]] with chunked rollout. The cost is that conditional independence given $h_\tau$ means the $K$ offsets need not describe one *coherent* future, which is exactly the property the paper claims for them. Nothing measures cross-offset consistency.

#### Target Content by Horizon: the finding, and its limits {#target-by-horizon}

This is Drive-HWM's most transferable result and the reason the pattern is worth a page entry. The paper ablates the prediction target **separately for each branch, in one codebase with everything else fixed**, and the ordering fully inverts:

| Branch | Horizon | Best target | Worst target | Spread |
|---|---|---|---|---|
| **Slow** | 8 steps ahead | **optical flow** | **RGB** | unknown — *Fig. 4's image is missing from the clipping* |
| **Fast** | 1 step ahead | **RGB** (93.8) | **optical flow** (93.5) | 0.3 over depth 93.6, 0.7 over no auxiliary target (93.1) |

> "optical flow is more suitable for learning dynamics-oriented representations in the slow world model, whereas next-frame RGB supervision better supports the fast model in generating accurate and responsive driving actions."

**The principle this supports is that the target should match the horizon.** Far futures are dominated by motion and their appearance is unpredictable, so predict motion; near futures are dominated by what is currently visible, so predict appearance. **That is orthogonal to [the entropy rule](#objective-form)**, which governs the *form* of the loss rather than the content of the target:

| Axis | Rule | Assembled from |
|---|---|---|
| **Objective form** | match the loss to the target's entropy — regression fails in proportion to it | [[sources/wa-jepa.md]], [[sources/drivefuture.md]], [[sources/coworld-vla.md]] |
| **Target content** | match what is predicted to how far ahead it is predicted | **Drive-HWM** |

The two compose and **nobody has run them together.** The obvious experiment — flow at long horizon under a generative objective versus a regression one — is one sweep and would say whether motion targets escape the entropy problem because flow is genuinely lower-entropy than appearance, or merely dodge it because the flow decoder absorbs the multimodality.

**How much weight this deserves: less than the framing suggests.** The slow-side sweep's numbers are not in the clipping, the fast-side spread is 0.3 across the top three, both are single runs with no seed variance, and the two branches also differ in backbone, horizon, loss form, and dataset stage — so "role" is confounded with several other variables. It is the best-supported hypothesis on this axis, not a result. It is also the first time any paper here has ablated the prediction target *twice within one system*; [[sources/coworld-vla.md]] ran four targets in parallel but all at one horizon and never separated them by role.

#### Image-Space Flow Against Ego-Frame Geometry {#flow-vs-geometry}

[[sources/geowam.md]] and [[sources/geoworldad.md]] make the coordinate frame the decisive variable — GeoWorldAD prices it at **+2.5 PDMS on its own** — on the argument that a world-model target sharing the trajectory's frame supervises exactly the structure planning consumes. **Optical flow does not share that frame.** It is 2D image-space displacement, and ego motion is recoverable from it only up to scale and only with camera geometry.

Yet Drive-HWM's linear probe (Table VI) puts its flow latent **first on future-ego-motion prediction at 83.7, with the BEV latent last at 71.2** — and BEV *does* share the ego frame. Both cannot be the general rule.

Three readings, none tested by anyone:
- the probe is near-tautological (a motion-trained latent probed on motion — see the caveat on [[sources/drive-hwm.md]]), so it may not measure what the coordinate-frame argument is about;
- BEV here is a *predicted* future BEV from a low-resolution front camera, which is a much weaker geometry target than GeoWorldAD's StreamVGGT point maps;
- the coordinate-frame advantage may be real for the *planner's* consumption while irrelevant to what a frozen encoder *encodes*.

**No paper compares a flow target against a metric-geometry target under a fixed planner**, and it is a one-sweep experiment in either codebase. It is now the cleanest open question on this page after the compose-the-two-rules experiment above.

#### What the Rate Hierarchy Is Actually Worth, and Why It Cannot Be Read Here

**+0.3 or +0.8 PDMS, depending on which of the paper's two mutually inconsistent result sets is correct** — see [[sources/drive-hwm.md]]'s [two-result-set section](../sources/drive-hwm.md#two-result-sets). Under either reading it is smaller than two other effects measured in the same paper: the slow-to-fast **conditioning mechanism** spans 1.3 (concatenation 92.5 → FiLM 93.8), and the fast model's **next-frame RGB auxiliary loss** is worth 0.7. **The mechanism in the title is again the smaller term** — the fifth consecutive ingest where that holds, after DA-WAM (+0.15), DriveFuture (+3.7 of 55.5), CoWorld-VLA (dynamic expert never ablated alone) and ReWorld (video-side objective never measured for planning).

**And the paper's actual claim for the hierarchy is untestable on its benchmark.** The justification for the fast branch is that "one-step action generation allows decisions to be continuously updated as new observations arrive" — a closed-loop property. NAVSIM is non-reactive and single-shot, and with $N=K=8$ at NAVSIM's 8-pose convention **the slow model runs exactly once per scenario and no new observation ever arrives.** So what Table IV measures is not slow-versus-fast scheduling at all; it is whether a flow-latent conditioning stream helps a single-shot planner. The scheduling claim needs [[concepts/hugsim-benchmark.md]] or Bench2Drive and the paper reports neither.

**What the pattern does establish cleanly is cost.** 25.6 ms for a parallel $K=8$ latent prediction, amortized to 3.2 ms per step, against [[sources/adaptive-wam.md]]'s 170 ms total, [[sources/simwam.md]]'s 518 ms and [[sources/foresight.md]]'s 870 ms of world model. **A flow-latent predictor is roughly two orders of magnitude cheaper than running a video generator to a finished future**, which makes it the best cost-per-point entry in the [comparison below](#test-time-imagination) if the gain is real. Note the peak step is 107.2 ms, so a real-time budget must be sized for $T_{\mathrm{peak}}$, not the amortized figure.

### 36. Perception Tasks as Extra Generated Video Streams (SUV) {#perception-as-video}

[[sources/suv.md]] (XJTU + USTC + Yinwang + Fudan) asks the question [[sources/coworld-vla.md]] asked (Pattern 34), whether one future target is enough, but answers it inside the *generator* instead of inside a VLM.
- Frozen SAM 3 and Depth Anything 3 turn each recorded future clip into three more videos: palette-rendered segmentation, Turbo-colormapped relative depth, and color-coded instance tracks.
- These are encoded by the same frozen Wan VAE, and **one Wan2.2-5B video expert generates all four streams** with no stream-specific heads. A modality prompt selects which stream is being generated.
- A ~1B MoT action expert reads every stream's latents at every block and denoising step. Streams cannot see each other, and the future cannot see the action.

| Contrast | CoWorld-VLA (34) | SUV (36) |
|---|---|---|
| Where the targets live | Typed VLM tokens | Native video latents in the generator |
| Teachers | V-JEPA, VGGT feature regression; Wan as a critic | SAM 3, DA3 outputs rendered as RGB video |
| Generator at planning time | Discarded | **Kept**; 2–10 joint denoising steps |
| Result | 88.7 PDMS v1 | 90.8 PDMS v1 / **91.0 EPDMS v2 (corrected)** / 36.9 navhard |

**Two findings are transferable:**
1. **Structured future supervision helps planning even when the planner never reads it.** It gives +1.0 navtest and +2.3 navhard with access disabled. This is the wiki's cleanest single-codebase evidence that *what* the generator is trained to predict shapes the planner's representation beyond RGB alone. It agrees with the wiki's objective-form and target-content threads.
2. **Native generation vs. generate-then-perceive is a split decision.** Native generation is better on depth (δ₁ 77.0 vs. 71.0) and worse on segmentation (64.2 vs. 66.8 mIoU) and tracks (84.4 vs. 86.9 AssA). It is measured against the teachers, not ground truth.

**Not addressed.** The mask blocks cross-stream attention to preserve the pretrained attention pattern, so consistency across streams rests on the shared prefix and weights, and is never measured. Track IDs are encoded modulo 7 per class, and depth is clip-relative.

### 37. The Future Reads the Action; the Action Never Reads the Future (Metis) {#future-reads-action}

[[sources/metis.md]] (Fudan + SII + Li Auto + others) keeps the SimWAM/SUV backbone (Wan2.2-5B + ~1B MoT action expert) and inverts the usual question. Instead of asking whether the planner should see the future, it lets the **future video attend to the action tokens**, so the video expert learns to generate the future *implied by the planned trajectory*. The action attends only to the current observation, so video is dropped at inference.
- Against the isolated mask at 320×384: **+0.5 navtest EPDMS, +2.2 navhard**.
- Against the bidirectional mask: +1.4 / +3.6.
- Video co-training overall is worth +1.6 EPDMS against the action expert alone.

**Two things are unmeasured:**
- *Why* it helps. Gradients through the action's keys/values versus action-conditioned video learning; a stop-gradient ablation would separate them.
- *Whether the generated video is any good*: no FVD, FID or PSNR is reported.

The design is the training-time counterpart of **action-conditioned generation**, which [Action-Conditioned ≠ Counterfactual](#action-conditioned--counterfactual) treats as the open problem for evaluation. Here it is used only as a training signal, never as a rollout.

## Does Test-Time Future Imagination Help? {#test-time-imagination}

This is now the central open dispute among world-model planners in the wiki, and SimWAM supplies the first controlled evidence.

**The imagine-then-act camp** conditions planning on generated future states at inference: FSDrive (mandatory visual CoT), PWM (future frame tokens rolled out before action), DriveVA (joint video-action denoising), DriveWAM (action as inverse dynamics from the generated latent), DriveLaW, WA-JEPA (future scene latents and actions denoised together over 12 sampling steps), **ForeSight**, which states the premise more explicitly than anyone — the generator *is* the encoder, run to a finished future, with everything else supplementary — and **BrainWAM**, whose video expert runs 1-3 denoising steps at inference to supply predictive context to a coordinated action stream. **GeoWorldAD** belongs here as well, with a shared latent-geometry future consumed as a refinement stage. **WCog-VLA** belongs here too, generating a joint multi-agent rollout at inference through a 5-step diffusion head. DA-WAM also belongs, and is the only member that predicts a *separate* future for every candidate rather than one future per scene — the distinction its ablation shows is decisive. **DriveFuture** belongs here in the weakest possible form: it instantiates a **16-token** predicted future latent at inference and routes it into every denoising step, but decodes nothing and predicts no pixels, features, or geometry - the latent exists only as a conditioning context. It is also the only member whose *training* signal is a real future observation that is then deliberately withdrawn (see [Pattern 33](#oracle-annealing)). The premise is that grounding the action in an explicit imagined future improves it.

**The training-time-only camp** uses future prediction purely to shape representations: DriveVLA-W0, FLARE, Latent-WAM, OneVL, Drive-JEPA, SimWAM, **Adaptive-WAM**, which runs a 5B generative backbone at inference but reads its intermediate activations rather than decoding any future, and **LWDrive**, whose future-frame VAE-latent objective shapes Qwen hidden states in Stage 1 and is then discarded. **CoWorld-VLA** belongs here for its video model — the Wan DiT is discarded before Stage 3 and never runs at planning time — but with a qualification that puts it between the camps: the **specific VLM token that video model supervised survives into inference** and conditions its own denoising branch. See [Pattern 34](#34-four-future-targets-in-parallel-in-one-latent-space-coworld-vla). **Drive-HWM** sits one step further from the camp again: both its objectives are training-time-only *and* confined to a separate pretraining dataset, yet the **whole slow predictor runs at inference** — only its flow decoder is dropped. Of everything in this list it instantiates the least at decision time (a latent, never decoded) while still paying for a forward pass, and at 3.2 ms amortized it pays the least. See [Pattern 35](#two-rates).

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

### SUV: The Access Effect Appears on navhard, Not navtest {#navhard-access}

[[sources/suv.md]] runs a 2×2 of structured supervision × future access on a Wan2.2-5B MoT WAM from the same family as SimWAM, and it reports **navhard as well as navtest**:

| Future access | navtest EPDMS | navhard EPDMS |
|---|---:|---:|
| Off (RGB supervision only) | 89.7 | 30.5 |
| **On** (RGB supervision only) | 90.6 (+0.9) | **35.0 (+4.5)** |
| Off (+ seg/depth/track supervision) | 90.7 | 32.8 |
| **On** (+ seg/depth/track supervision) | 91.0 (+0.3) | **36.9 (+4.1)** |

**This reconciles SimWAM's null with the imagine-then-act camp.**
- On navtest, access is worth +0.3 once supervision is rich. That is inside SimWAM's ±0.2 band, so SUV *reproduces* the null.
- On navhard, where Stage 2 re-renders the scene from a displaced ego pose, access is worth +4.1 to +4.5, with both stages rising by about 2.5.
- So the live hypothesis becomes: **test-time future access matters under observation shift, and navtest is not built to detect it.** Neither SimWAM nor WA-JEPA reported navhard.

**How much weight this deserves.** Single runs, and only 450 Stage-1 scenes, so this is suggestive, not settled. The direct test is still missing: SimWAM's isolated-mask checkpoint evaluated on navhard would settle it with one run.

**The price.** Access requires denoising the future at inference. SUV at 2 steps runs in 288 ms, against an unreported cost for the no-access variant. On navtest the trade is poor (+0.3). On navhard it is the largest single effect in SUV's ablations.

### The Mask Family: Three Papers, One Backbone {#mask-family}

[[sources/simwam.md]], [[sources/metis.md]] and [[sources/suv.md]] share Wan2.2-5B, a ~1B hidden-1024 MoT action expert, joint flow matching (λ=1), and a near-identical recipe (60 epochs, 8×H200, AdamW 1e-4, wd 0.01, cosine). They differ mainly in the attention mask, so this is the closest thing the wiki has to a controlled cross-paper sweep:

| Mask | Action → future | Future → action | Video at inference | SimWAM v1 PDMS | Metis @320×384 v2 / navhard | SUV v2 / navhard |
|---|:-:|:-:|:-:|---:|---:|---:|
| Bidirectional | ✓ | ✓ | yes | 90.2 | 87.4 / 28.0 | – |
| Action reads future | ✓ | ✗ | yes | 90.1 | – | 91.0 / 36.9 |
| Isolated | ✗ | ✗ | no | 90.3 | 88.3 / 29.4 | 90.7 / 32.8 |
| Future reads action | ✗ | ✓ | no | – | 88.8 / 31.6 | – |

**What holds across all three.** On navtest, every non-bidirectional mask is within 0.5 of every other. On navhard, both departures from isolation that avoid feedback help: +2.2 for future-reads-action (Metis) and +4.1 for action-reads-future (SUV). SimWAM's null stands only because it never reported navhard.

**The conflict, and one way to resolve it.** Metis's bidirectional mask contains SUV's helpful pathway (the action reads the future), yet it is the *worst* variant on navhard. The two results reconcile if **the damage comes from the feedback loop**: the future reads a noisy action that then reads that same future. [[sources/brainwam.md]]'s symmetric unmasked Tri-MoT losing to its isolated variant points the same way. **The deciding experiment is the missing cell**: SUV's one-way access and Metis's one-way action-conditioning in the same model, plus both at once, evaluated on navhard.

**Updated position on test-time imagination.** On navtest it remains unnecessary: three papers, no effect. On navhard, one-way access to the imagined future gave the largest single mechanism effect recorded for this backbone (+4.1). That favours the imagine-then-act camp *under observation shift*. Single runs throughout.

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

> **Shared *photometric* future conditioning is useless to harmful. Only per-candidate futures help, and then by little.**
>
> *(Scoped to photometric and feature-space targets after the GeoWorldAD ingest — see [below](#shared-future-reopened). The unqualified form no longer holds.)*

**How much weight the positive half deserves: not much.** +0.15 PDMS, single run, no seed variance, against a no-future baseline of 93.31 that would itself rank third in this wiki. WA-JEPA measured 0.053 seed std for a stochastic sampler and training-seed variance is typically larger. Within DA-WAM's own paper the representation choices are worth +2.42 and the hard negatives +0.22 — **the mechanism the paper is named for is the smallest effect in it.** And its predicted future reaches only **0.5 seconds** while candidates span 8 poses, so whatever it is doing, it is not evaluating the multi-second consequences the introduction promises.

#### DriveFuture Puts a Per-Candidate Future Inside the *Generator* {#per-candidate-in-generator}

DA-WAM's taxonomy files [[sources/drivefuture.md]] under **(b) loosely coupled latent fusion - one proposal, nothing to compare**. Reading the paper itself, that is right for two of its three inference branches and wrong for the third.

DriveFuture's Progressive Foresight Guidance runs three classifier-free branches per denoising step. The null and kinematic branches compute their future latents **once per scene** and broadcast them across all 100 proposals - configuration (c). But the third branch conditions the world model on a Tweedie estimate of the clean trajectory recovered from *that proposal's own noisy sample*, recomputed at every step where its weight is non-zero. That is one future latent per candidate - configuration (d) - and it dominates the guidance mixture over the last 30% of denoising.

**The difference from DA-WAM is where the per-candidate future is spent.** DA-WAM builds one future per candidate and hands the triple $(Z_t, a_i, \widehat{Z}_i)$ to a **scorer**, so the futures are used to *rank* proposals that already exist. DriveFuture builds one future per candidate and feeds it back into the **denoiser**, so the futures are used to *steer* proposals as they form - and its selection is then done by an unrelated GTRS-Dense scorer that never sees a future latent at all.

| | Futures per scene | Consumed by | Effect measured |
|---|---|---|---|
| WoTE, SimWAM, WA-JEPA, GeoWorldAD | 1 | Scorer or planner | -0.50 PDMS where isolated (DA-WAM's (c) row) |
| DA-WAM | 32 (one per candidate) | Scorer | +0.15 PDMS |
| **DriveFuture (Tweedie branch)** | **100 x per step, late phase** | **Denoiser, as CFG** | **Not isolated - bundled into the +2.6 for dual-source guidance** |

DriveFuture's Table 4 shows removing both guidance sources costs 2.6 EPDMS, but that row removes the kinematic branch and the GT training mode as well, so the per-candidate component is not separable. **Neither paper knows what a per-candidate future is worth inside a generator**, and DriveFuture is the only architecture here that could answer it cheaply - the switch (`use_dspcfg`) already exists.

### ForeSight Prices the Paradigm — and Disputes DriveLaW's Sweep

[[sources/foresight.md]] is the strongest statement of the imagine-then-act thesis in the wiki, and it supplies two numbers that bear on this section from opposite directions.

**The cost accounting is the more decisive one, and it is against the thesis.** ForeSight's Table 3 row 1 → row 2 is the same experiment SimWAM's mask ablation ran, from the other side: take a working planner, add a frozen 2.5B foundation world model, cross-attend to its generated future with vanilla attention, change nothing else.

| Configuration | PDMS | Δ |
|---|---:|---:|
| Baseline: current encoder + simple action decoder | 86.8 | — |
| **+ foundation world model, vanilla attention** | **87.1** | **+0.3** |
| + WM-QFormer | 87.9 | +0.8 |
| + state queries | 88.5 | +0.6 |
| + factorized attention | 89.3 | +0.8 |

**+0.3 PDMS is what an imagined future is worth when nothing is built to consume it.** The remaining +2.2 comes from a compression-and-routing stack, one component of which (state queries, from the authors' BridgeAD) has no intrinsic world-model dependency and is never tested without one. And the +0.3 is bought at **870 ms of a 900 ms inference budget** — the wiki's slowest NAVSIM planner, against SimWAM's 91.5 PDMS at 518 ms with generation removed at inference entirely.

**This is also the wiki's second measurement of DA-WAM's configuration (c).** ForeSight generates one future per scene and conditions all 20 trajectory modes on it. DA-WAM's matched ablation put that configuration at −0.50 PDMS versus no future; ForeSight measures +0.3 in a different architecture. Both are single-run, both are inside their own pipelines, and they bracket zero. The reasonable reading is that **a shared generated future is worth approximately nothing at inference**, which is a weaker and better-supported claim than either paper's.

**The dissenting number.** ForeSight's Table 5 sweeps the denoising budget, and it runs *opposite* to DriveLaW's Table 6:

| ForeSight — total denoising steps | PDMS | | DriveLaW — extraction step | PDMS |
|---:|---:|---|---:|---:|
| 25 | 88.0 | | t = 1 (earliest) | **89.1** |
| 50 | 88.3 | | t = 5 | 86.9 |
| 75 | 89.2 | | t = 10 (near-clean) | **23.2** |
| 100 | **89.3** | | | |

ForeSight: the more fully formed the future, the better the plan. DriveLaW: the more fully formed the future, the worse the plan, catastrophically so at the end of the schedule.

**These are not the same variable, and the difference matters.** DriveLaW holds the schedule fixed and moves the *extraction point* along it. ForeSight changes the *total schedule length*; its extraction step $t_{\rm d}$ is an explicitly adjustable parameter whose value the paper never reports. So ForeSight's 25-step row could be a genuinely coarser latent or an equivalently-positioned latent on a shorter schedule, and the paper gives no way to tell.

**BrainWAM breaks the tie, and it sides with DriveLaW.** [[sources/brainwam.md]] decouples its video and action rectified-flow timesteps, which makes "how many video denoising steps run before the features are cached" a free parameter — the axis closest to DriveLaW's, since both ask how *formed* the latent should be when the planner reads it.

| Video denoise steps | Latency | PDMS | EPDMS |
|---:|---:|---:|---:|
| 0 (pure noise) | 382 ms | 79.3 | 75.8 |
| **1** | **475 ms** | **89.3** | **89.4** |
| 2 | 565 ms | 89.5 | 89.6 |
| 3 | 644 ms | 89.4 | 89.6 |

**One step delivers 89.3 of an achievable 89.5**; the next two steps are worth 0.2 and then nothing, for 169 ms. That is DriveLaW's t=1 result reproduced in a completely different architecture — a Wan2.2-5B video expert coupled to a Qwen3-VL-4B semantic branch, versus an LTX-Video DiT chained into a 133M action DiT.

The tally on denoising depth is now:

| Paper | Variable | Verdict |
|---|---|---|
| [[sources/drivelaw.md]] | Extraction point, fixed schedule | Earliest is best; near-clean collapses |
| [[sources/brainwam.md]] | Steps executed before caching | One step is enough |
| [[sources/foresight.md]] | Total schedule length | More is better (+1.3 over 25→100) |

**Two independent results say the planner needs an early, barely-formed latent, and one says it needs a finished future.** ForeSight is also the one whose experiment cannot be interpreted, because $t_{\rm d}$ is unreported.

### Adaptive-WAM Shows the Axis Was Wrong {#three-axes}

[[sources/adaptive-wam.md]] is the first paper in the wiki to point out that "how much denoising" has been three questions wearing one name, and to vary them separately:

| Axis | What it controls | Measured effect |
|---|---|---|
| **1. Noise index** | Which diffusion timestep the backbone is *conditioned on*, in a single forward pass | **≤ 0.15 PDMS** across 5 indices of a 40-step schedule |
| **2. Denoising iterations** | How many times the denoiser is actually run before features are read | 1 step ≈ 3 steps (BrainWAM); t=1 ≫ t=10 (DriveLaW) |
| **3. Readout depth** | Which DiT block the features are taken from | **5.85 PDMS (IL) / 4.80 (RL)** across 6 depths |

Axis 1 is the one the field has been ablating, and it is worth almost nothing. Axis 3 is the one nobody had varied, and it dominates by roughly forty times:

| Block | 5 | 9 | **15** | 18 | 22 | 30 (full) |
|---|---:|---:|---:|---:|---:|---:|
| Imitation | 81.94 | 83.60 | **86.56** | 84.14 | 83.62 | 80.71 |
| + planner RL | 86.02 | 87.56 | **90.62** | 88.92 | 87.42 | **85.82** |

**The mid-network exit beats the full-depth exit by 4.80 PDMS.** All six exits share architecture, optimizer, batch size, epochs, and head capacity, so the difference is attributable to depth alone. This is a genuinely new design axis for this page — every other pattern above reads the final layer without comment.

**How the three papers reconcile.** DriveLaW's t=1, BrainWAM's one-step, and Adaptive-WAM's single conditional forward are all *the same operation*: one pass through the video DiT. Three papers, three architectures, three coupling schemes, one conclusion — **the planning-relevant signal is present after a single forward pass, and iterating the denoiser adds nothing.** ForeSight's 100-step schedule remains the outlier, and it is also the most expensive configuration in the wiki (870 ms of a 900 ms budget) for a claimed +1.3 over 25 steps.

**What Adaptive-WAM does not settle.** Its noise-index result is measured on a *single forward pass*, whereas DriveLaW's t=10 latents have been through ten actual denoising iterations and carry different activation statistics. So "noise level is nearly irrelevant" and "reading late in an iterative rollout collapses the policy" are compatible claims about different operations, and the t=10 collapse still has no diagnosis. The paper scopes this correctly and does not overclaim; neither should this page.

**The cost framing this supplies.** Adaptive-WAM is also the first to decompose what the alternative actually costs on identical hardware: **170 ms to plan from an intermediate feature, 13.22 s to synthesize the future that feature encodes** — 80 DiT forwards under classifier-free guidance plus VAE decode, at 31.19 GiB peak. A factor of 78 between reading the representation and rendering the imagination.

**One caveat on BrainWAM's own evidence.** Its 0-step row (79.3 PDMS, presented as proof that "video dynamics are essential to planning") feeds the action expert *pure Gaussian noise* through a pathway trained on partially-denoised features. That is a distribution-shift ablation, not a test of whether futures help; a −10.2 collapse is what feeding noise into any trained pathway produces. The clean version is SimWAM's isolated mask — retrain without the dependency — which BrainWAM does not run. Only the 1-vs-2-vs-3 rows carry information, and they say the marginal value of denoising is ≈0.2 PDMS after the first step.

**One thing ForeSight settles that nobody else tested**: its Table 7 runs the planner on generated futures *alone*, with no current-frame encoder, no multi-view images, and no LiDAR, and reaches 88.2 PDMS — above its own no-world-model baseline by 1.4. Whatever the generated future is contributing, it is not nothing; it is roughly what a competent BEV world model contributes (WoTE 88.3), for two orders of magnitude more compute.

**What survives across all nine papers**: every one finds video *supervision* or a video *prior* essential; none demonstrates that a *shared* generated future reliably helps at inference, and the two matched measurements of that configuration (DA-WAM −0.50, ForeSight +0.3) straddle zero; the only positive inference-time result requires a distinct future per candidate and is worth 0.15 PDMS unreplicated. DriveVA's +19.5 PDMS and DriveWAM's backbone ablation isolate the training objective; SimWAM's mask and DriveLaW's denoising sweep both isolate the inference path and find no benefit there; ForeSight isolates the price and BrainWAM isolates how little denoising that price needs to buy. The efficiency implication is immediate — SimWAM reaches 91.5 PDMS at 518 ms, DriveWAM's imagine-then-act loop costs 871–1262 ms per 4 s chunk, ForeSight's costs 900 ms for 89.3, and BrainWAM gets 89.3 at 475 ms by stopping its video branch after one step. [[sources/adaptive-wam.md]] closes the argument on cost: on one A100 it plans in **170 ms** from an intermediate feature, while synthesizing the future that feature encodes takes **13.22 s** — a factor of 78 between reading the representation and rendering the imagination.

**The strongest remaining case for generation** is DriveLaW's own Table 5: video-generator latents beat VLM hidden states by 2.6 PDMS and BEV features by 5.0 under a fixed planner. The *generator* is clearly valuable as a representation learner. What is unsupported is running it forward to a clean future at decision time.

**What is still unresolved**: whether imagined futures matter for capabilities NAVSIM does not measure — long-horizon rollout, counterfactual evaluation of candidate maneuvers, reactive interaction, or the instruction-conditioned generation Vega targets. NAVSIM's 4 s non-reactive horizon may simply be too short for anticipation to pay off. Also unexplained is *why* DriveLaW's t=10 conditioning collapses so completely; a 66-point PDMS drop suggests a distribution or scaling pathology rather than merely "redundant information," and no paper has diagnosed it.

### GeoWorldAD Reopens the Shared-Future Question {#shared-future-reopened}

The claim above was assembled entirely from methods whose future target is **photometric or feature-space** — video pixels, video latents, JEPA scene latents. [[sources/geoworldad.md]] is the first entry whose shared future is **geometric**, and it measures the same structural configuration at a very different value.

| Paper | Future target | Shape | Measured effect |
|---|---|---|---:|
| [[sources/da-wam.md]] | JEPA scene latents (0.5 s) | shared across 32 candidates | **−0.50 PDMS** |
| [[sources/simwam.md]] | video tokens | shared, single stream | **~0.0** (90.2 vs 90.3) |
| [[sources/foresight.md]] | generated video frames | shared across 20 modes | **+0.3 PDMS** |
| **[[sources/geoworldad.md]]** | **latent tokens supervised by future depth (2 s)** | **shared across 64 proposals** | **+1.7 PDMS / +2.8 EPDMS** |
| **[[sources/wcog-vla.md]]** | **joint multi-agent trajectories** | **one shared rollout, one ego plan** | **+0.9 PDMS** |

GeoWorldAD's ablation replaces its five-stage present-geometry planner's final refinement stage with one that attends to 256 latent future tokens (4 chunks × 64), and gets NC +0.1, TTC +0.1, and **EP +3.3**. The profile is the mirror image of DA-WAM's shared-future row, where NC and TTC rose while ego progress *collapsed* from 91.36 to 88.68. Same structure, opposite sign, on the sub-metric that distinguishes them.

**Two candidate explanations, and the paper does not separate them.**

1. **The target matters.** An averaged photometric future cannot attribute a hazard to a candidate, so it makes the policy uniformly cautious (DA-WAM's diagnosis). A shared *geometric* future does not need to attribute anything — it says where free space will be, which is useful to every candidate equally, and lets the planner commit instead of hedging. Under this reading the wiki's rule was never about sharing; it was about photometric futures carrying no decision-relevant signal once averaged.
2. **It is the extra training.** GeoAD is the Stage-2 checkpoint at 32K planner steps; GeoWorldAD adds **64K more**. The future block is zero-initialized so the two are identical at the start of Stage 3, which makes the delta attributable to Stage 3 — but Stage 3 varies the mechanism and triples the planner budget together. A GeoAD trained for a further 64K steps is the missing row, and it is one run.

**A second non-photometric target, same sign.** [[sources/wcog-vla.md]]'s Table 4 row 5 turns on joint multi-agent trajectory generation with no semantic world supervision and gains **+0.9 PDMS** over a planner with neither. Its target is neither photometric nor geometric but **behavioural** - what the other cars will do - and like GeoWorldAD it is a single shared future conditioning a single ego plan. The two positive results now share a property the four negative ones lack: **the forecast is of something the planner cannot read off the current frame**, whereas an averaged future image or scene latent is largely redundant with the observation it was conditioned on.

**How the synthesis should read for now.** The negative results remain the better-supported half, and they remain specific: *a shared photometric future is useless to harmful*. That statement survives GeoWorldAD untouched. What no longer survives is the general form — "shared future conditioning is useless to harmful" — because the one geometric instance measures +1.7, and the mechanism it claims (free-space anticipation buys progress, not safety) is visible in the sub-metrics rather than only in the aggregate. [[sources/geowam.md]] points the same way from a different direction, with a +4.9 navhard gain from future point-map forecasting against +0.6 on navtest.

**What would settle it**: the compute-matched GeoAD row, and a geometry-vs-pixel future target under one fixed planner. Neither geometry paper runs either.

**A third instance, from outside the world-model architecture entirely.** [[sources/unified-driving-tokens.md]] supervises a discrete *tokenizer* with adjacent-frame depth and relative pose, then measures the effect through a frozen 20M planning readout: **+1.5 PDMS, of which EP is +4.2 while TTC goes down 0.6**. That is neither a world model nor an inference-time future — it is geometric structure baked into a representation — and it lands on the same sub-metric as GeoWorldAD's shared future (+3.3 EP, safety flat) and as the effect [[sources/geowam.md]] argues for without ablating.

| Paper | Mechanism | EP | Safety |
|---|---|---:|---|
| [[sources/geoworldad.md]] | shared latent future-depth tokens at inference | **+3.3** | NC +0.1, TTC +0.1 |
| [[sources/unified-driving-tokens.md]] | depth + pose supervision on a discrete tokenizer, training-time only | **+4.2** | TTC **−0.6** |
| [[sources/geowam.md]] | dense metric point maps as the world-model state | — | +4.9 EPDMS on navhard, +0.6 navtest |

**Three papers, three different mechanisms, one sub-metric.** The generalization that survives is narrower than any of their claims and more useful: **geometric supervision buys ego progress rather than safety**, plausibly because knowing where free space will be lets a planner commit instead of hedging. It also means a geometry result reported only as aggregate PDMS is hiding which half of the trade it bought — and UDT's is the one case where the safety side actually goes backwards. WCog-VLA's row 5 is the better-controlled of the two positives — same three-stage SFT budget across all six rows of its Table 4 — but it is +0.9 rather than +1.7, and its ADDT also adds a continuous action head that the row-1 baseline lacks.

### ReWorld Makes the Decoupling Intra-Paper {#generation-planning-decoupling}

Everything above argues about whether a *generated future* helps at decision time. A prior question has been answered only across papers: **does a better world model produce a better plan at all?** The evidence for "no" was suggestive but confounded — [[sources/drivelaw.md]] leads the wiki on FID while planning at 89.1; [[sources/simwam.md]] plans at 91.5 with generation removed at inference; [[sources/foresight.md]] runs a 2.5B generator to a finished future for 89.3; [[sources/uniugp.md]] held the best FVD and is nowhere near the planning frontier. Different architectures, different data, different everything.

[[sources/reworld.md]] runs both halves inside one architecture, one codebase, and one paper — and does not connect them.

| Half | Mechanism | Video-quality effect | Planning effect |
|---|---|---|---|
| Stage 1 | $\mathcal{L}_{\mathrm{Mid}}$ + self-guided sampling | FVD 81.3 → **61.9**; UCF-101 68.3 → 71.7 | **never measured** |
| Stages 2–3 | $\mathcal{L}_{\mathrm{align}}$ + $\mathcal{L}_{\mathrm{RDE}}$ | not measured in isolation | 89.1 → **90.4** |

Its Table VII accounts for the full +1.3 PDMS with two action-side objectives and contains no Stage-1 row; its §IV-A says Stage 2 initializes from the *DriveLaW* checkpoint, which if read literally means the planning result never sees the improved video model at all. Meanwhile Stage 1's headline improvement is 88% attributable to an inference-time guidance scale that the paper explicitly states "improves video sampling **without changing the planner-facing latent interface**."

**So the sharpest available statement is now**: in a chained WAM, the mechanisms that improve future prediction and the mechanisms that improve action selection are separable, and no paper — including the one whose title asserts the link — has demonstrated that improving the first improves the second.

**Two things cut the other way and belong in the record.** Stage 3 unfreezes the Video DiT and lets planning gradients reshape it, and the video representation *does* change materially: UCF-101 frozen probing jumps 71.7 → **80.2** with the Video DiT frozen throughout Stage 2, so that entire +8.5 comes from hard-negative trajectory gradients propagating into the world model. That is a real, large, and completely unanalysed influence — running **action → world**, the reverse of the thesis. And the missing experiment is a single run on released code: Stages 2–3 from the vanilla video checkpoint versus from the Stage-1 checkpoint, PDMS both ways.

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

[[sources/foresight.md]] gives the sharpest illustration of the problem this section exists for: a 2.5B frozen Epona generator, 52M current encoder, and 21M action decoder, at **900 ms per frame on an H100 with 870 ms (96.7%) in the world model**, for 89.3 PDMS. Its own denoising sweep shows the last quarter of that budget buys +0.1 PDMS. This is the slowest NAVSIM planner in the wiki, and the papers that beat it (SimWAM 91.5 at 518 ms, DriveSuprim 93.5, CLEAR 93.7) all spend less.

[[sources/brainwam.md]] shows the other half of the same picture: with **decoupled video and action rectified-flow timesteps**, the video branch can stop after one denoising step, cache its features, and let the action stream keep going. That costs 93 ms over a no-video baseline (382 -> 475 ms) and recovers 89.3 of an achievable 89.5 PDMS. **Asynchronous, truncated video denoising with feature caching is currently the cheapest way to keep a generative branch in the inference loop**, and it is a strict improvement on ForeSight's 100-step schedule.

[[sources/adaptive-wam.md]] measures the gap this section exists to close, on one A100 at batch 1: **170 ms** to plan from an intermediate DiT feature versus **13.22 s** for a full 40-step classifier-free video rollout of the same nine-frame clip (12.05 s denoising over 80 DiT forwards, 0.27 s VAE encode, 0.90 s VAE decode, 31.19 GiB peak). It also adds a technique none of the entries below use: **route the backbone depth per scene**, terminating once a decoded trajectory clears a learned quality threshold. That is worth another 10% over a fixed mid-network exit (190 -> 170 ms), with 94% of scenes exiting within the first three of six blocks.

Video generation models (DiT-based, e.g., Wan2.1) are expensive. Solutions:
- Make generation expert optional at inference (UniUGP)
- Use lower-resolution occupancy instead of video (OccWorld)
- Predict latent features instead of pixels, in parallel (DeepSight: +3.57% latency vs. native VLM)
- Few-step ODE solvers over the flow path (DriveVA: 2 steps; DriveWAM: 3 video / 5–10 action steps)
- **Bound the KV cache during rollout** (DriveWAM's selective KV memory): for autoregressive world-action policies, the dominant long-horizon cost is not the denoiser but the growing history cache — 3.07 GB and 17.37 GFLOPs per step at 300s under full caching, reduced to 0.25 GB / 1.44 GFLOPs by content-based eviction. Age-based FIFO achieves the same budget but degrades accuracy badly (ADE@4s 0.89 → 1.40), because old tokens can remain decision-relevant while new tokens are often redundant background.

### 3. Evaluation
World model quality (FID, FVD) and planning quality (L2, collision rate) can improve independently or diverge. UniUGP is notable for improving both simultaneously.

## The First Closed-Loop Reproduction of Two WAM Leaders {#closed-loop-check}

Every world-model result on this page is measured on a non-reactive benchmark. [[sources/qwen-drive-1.0.md]] — which contains no world model at all — ran [[sources/simwam.md]] and [[sources/drivewam.md]] in the **AlpaSim** closed-loop simulator over 916 reconstructed real-log scenarios, and the result is the most uncomfortable data point this page carries.

| Method | NAVSIM-v1 PDMS | AlpaSim at-fault score | Progress | All-event close encounters |
|---|---:|---:|---:|---:|
| [[sources/simwam.md]] | **91.5** (best WAM here) | **0.30** (last) | 62 % | 35 % |
| [[sources/drivewam.md]] | 90.1 | 0.53 | **35 %** | **56 %** |
| Alpamayo-1.5 | not reported | 0.45 | 59 % | 37 % |
| Alpamayo-R1 | not reported | **0.58** | **67 %** | **19 %** |
| Qwen-Drive-1.0-RL (no world model) | 90.7 | 0.37 | 48 % | 41 % |

**Neither behaviour is what the NAVSIM ranking predicts.** SimWAM — whose training-time-only video co-training is one of this page's cleanest positive results — is characterized as "considerably more aggressive, frequently accelerating forward while failing to decelerate sufficiently for preceding vehicles or obstacles," with a 22 % at-fault close-encounter rate. DriveWAM "often remains stationary or advances only briefly," earning a good at-fault score from a metric that divides distance by events while being struck from behind in 56 % of scenarios.

Three things to hold alongside it before treating it as a verdict:

1. **These are third-party reproductions** under an input protocol neither paper designed for, at one run each.
2. **The AlpaSim score is a ratio** and flatters near-stationary policies; see [[concepts/alpasim-benchmark.md]] for why all six columns must be read together.
3. **The two leading closed-loop methods are Alpamayo variants, neither of which reports NAVSIM**, so this is not a controlled comparison of world-model designs — it is evidence that the two benchmarks rank differently.

What it does support is a sharper version of this page's standing caution. The [test-time-imagination](#test-time-imagination) thread has repeatedly found that generated futures are worth little *on NAVSIM*, and has repeatedly noted that NAVSIM cannot exercise the property a world model exists for. **The first reactive measurement does not rescue the world-model case** — the two WAMs here are separated by degenerate behaviour in opposite directions rather than by anticipation — but it does confirm the premise: **a 1.4-point PDMS lead carried no closed-loop information at all.**

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
| [DriveLaW](../sources/drivelaw.md) | Latent diffusion (LTX-Video 2B) | 1280×704 | 4.6 | 81.3 |
| **[ReWorld](../sources/reworld.md)** | **DriveLaW + intermediate supervision + self-guidance** | **1280×704** | **4.4** | **61.9** |

**[[sources/reworld.md]] is the first entry here to lead both metrics at once.** Until it, DriveLaW held the best FID (4.6) and UniUGP the best FVD (75.9), and the two did not agree on a leader; 4.4 / 61.9 takes both. FID is resolution-dependent and both DriveLaW and ReWorld generate at by far the highest resolution here, which cuts against them rather than for them. On nuPlan, DriveLaW beats Epona up to 80 frames but **loses at 100 frames** (FVD 296.1 vs 277.3), so its advantage is horizon-limited; ReWorld does not report the nuPlan horizon sweep.

**Read the 61.9 with its decomposition attached.** ReWorld's own ablation gives 78.9 FVD for the representation objective alone and 61.9 only after inference-time self-guidance at $\gamma=1.4$ — so **17.0 of the 19.4-point gain is a sampling mechanism**, non-monotonic in $\gamma$ and tuned on the reported set. FID barely moves (4.6 → 4.4), which is consistent with a temporal-coherence gain and equally consistent with a guidance-strength effect; FID at $\gamma=1.0$ is not reported.

**A second table in the same paper is more useful for method comparison**, because it is controlled: from-scratch training, 120k steps, 224×224×25 clips, no text conditioning, FVD on nuScenes. ReWorld 270.4 < Self-Flow 283.3 < SRA2 295.2 < REPA-DINOv2 295.9 < SRA 296.9 < **Vanilla Flow 304.1** < REPA-DepthAnything3 319.4 < REPA-VideoMAEv2 328.3 < **REPA-V-JEPA-2 331.6** < ReDi 421.7. **Three of four external teachers are worse than no teacher**, and the worst is V-JEPA 2 — scoped to *generator feature alignment*, not planning; see [[concepts/foundation-backbones-for-ad.md]].

Note: FID is resolution-dependent — methods at higher resolution (Doe-1 384×672) would achieve lower FID at lower resolution. FSDrive's 10.1 at 128×192 is competitive for its resolution tier and model size (2B).

### NAVSIM Future Video Generation (FVD ↓, front-view)

| Method | FVD ↓ | LPIPS ↓ | PSNR ↑ |
|--------|-------|---------|--------|
| SVD *(via CoWorld-VLA)* | 227.5 | — | — |
| DrivingGPT *(via CoWorld-VLA)* | 142.6 | — | — |
| PWM | 85.95 | 0.23 | 21.57 |
| **DriveDreamer-Policy** | **53.59** | **0.20** | **21.05** |
| **[[sources/coworld-vla.md]]** | **32.7** | — | — |

DDP substantially improves video coherence (−38% FVD) vs. PWM. The improvement is attributed to depth joint learning (−18.6% FVD alone) and LLM-conditioned generation. Note: front-view only for comparability with PWM (single-view model).

**CoWorld-VLA's 32.7 leads this table by a wide margin and needs three caveats attached.** Its own comparison set is SVD (227.5) and DrivingGPT (142.6) on NAVSIM plus Epona (61.3) and DriveLaW (55.6) on **nuPlan** — and its caption says so outright: *"Dataset and evaluation settings may differ across methods; cross-setting results are provided for contextual reference."* That is more honest than most generation tables here. But it **omits PWM and DriveDreamer-Policy**, the two NAVSIM entries this wiki already had, so the 32.7-vs-53.59 comparison is one this page is making rather than one the paper made. And **the generation resolution is never stated**, which FVD is strongly sensitive to. Treat it as the NAVSIM leader on the paper's own protocol, not as a settled −39% over DDP.

Worth noting what produced it: CoWorld-VLA's Stage-2 improvement over Stage 1 comes from **replacing a text condition with a VLM hidden state** as the video model's conditioning input. Figures 3, 5 and 6 attribute the gain to preserved turning intent, lane-level layout, and local object fidelity. None of it is measured separately — there is no Stage-1 FVD number anywhere in the paper.

### nuScenes Planning (front/multi-camera, no heavy supervision, UniAD metrics)

| Method | Avg L2 (m) ↓ | Avg Collision (%) ↓ | Notes |
|--------|------------|-------------------|-------|
| Doe-1 | 1.26 | 0.53 | No ego status; Lumina-mGPT-7B |
| **FSDrive** | **0.96** | **0.40** | **No ego status; Qwen2-VL-2B** |
| [Epona](../sources/epona.md) | 1.25 | 0.36 | Front cam, no aux supervision |
| **UniUGP** | **1.23** | **0.33** | **multi-camera** |

## Open Questions

- Does trajectory-conditioned video generation improve **closed-loop** performance (NAVSIM PDMS), or only open-loop metrics? (FSDrive shows 85.1 PDMS, well below the current wiki non-BoN frontier of CLEAR/DA-WAM 93.7 (Drive-HWM's 93.8 is disputed by its own prose) and below multiple VLM-style 90+ PDMS methods — generation quality may not translate to closed-loop driving)
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
- **Where should a planner read a video prior from?** [[sources/adaptive-wam.md]] shows readout *depth* dominates readout *noise level* by roughly 40x (4.80 vs 0.15 PDMS) and that the mid-network exit beats the final block, but it measures this on one backbone family with one head type. Whether block ~50% is a property of Wan2.2, of video DiTs generally, or of the planning task is untested — and no other wiki paper reports which layer it reads. Re-running DriveLaW's representation comparison across depths, or SimWAM's backbone swap at matched relative depth, would settle it cheaply.
- **How denoised should the conditioning latent be? The question is now better posed than answered.** Three papers agree that a single forward pass suffices (DriveLaW t=1, BrainWAM one step, Adaptive-WAM one conditional forward) and one disagrees (ForeSight, 100 steps, extraction point unreported). Adaptive-WAM shows the *noise index* is worth <=0.15 PDMS in a single pass, which removes one candidate explanation but leaves DriveLaW's t=10 collapse (89.1 -> 23.2) undiagnosed — those latents went through ten actual denoising iterations and carry different activation statistics than a one-shot forward at any index. A distribution-statistics diagnostic on cached latents at increasing iteration counts would answer it. See [Adaptive-WAM Shows the Axis Was Wrong](#three-axes).
m d}$, or running DriveLaW's extraction sweep at fixed schedule length inside ForeSight, settles it in one run. See [Does Test-Time Future Imagination Help?](#test-time-imagination).
- **Does the modality-competition result generalize beyond VLA+WAM?** BrainWAM measures a clean VLM stream suppressing a denoising video stream in shared attention (Tri-MoT 87.8 < WAM-only 88.1), and SimWAM's two-modality mask ablation shows no such effect without a VLM. Untested: whether the same competition appears between a VLM and *any* iteratively-refined stream (occupancy diffusion, flow-matched latents, JEPA predictors), and whether asymmetric masking - UniDriveVLA's and AutoMoT's design, where information flows one way - avoids it without needing an 8-token bottleneck. No paper has run the obvious control of re-weighting attention toward the suppressed modality to see whether Tri-MoT recovers.
- **[Partially answered by SUV and Metis] Does test-time future imagination ever pay off?** On navhard, yes, at least for one-way access: [[sources/suv.md]] measures +4.1 EPDMS for letting the action read the generated future, against +0.3 on navtest. [[sources/metis.md]] finds the *bidirectional* version worst on navhard. See [the mask family](#mask-family). The earlier framing follows. SimWAM shows it does not on NAVSIM's 4 s non-reactive horizon, and ForeSight prices the attempt at 870 ms for +0.3 PDMS in the one row that isolates it (see [Does Test-Time Future Imagination Help?](#test-time-imagination)). The open part is whether it matters for what NAVSIM cannot measure: long-horizon rollout, reactive closed-loop interaction, or instruction-conditioned generation. **The counterfactual branch of that question now has a partial, negative answer**: [[sources/driving-wm-counterfactuals.md]] evaluates counterfactual prediction head-on and finds action-conditioned generation does not produce counterfactuals at all — recovered fraction 0.38 (Vista) and 0.31 (DrivingWorld), below the 0.5 no-preference point. That result concerns *retrospective* counterfactuals over recorded episodes; comparing candidate maneuvers *before* acting is a rung-2 question the benchmark does not test. If the answer is no everywhere, the imagine-then-act line (FSDrive, PWM, DriveVA, DriveWAM, DriveLaW) is paying inference cost for nothing.
- **Can a world model do abduction?** Inferring the realized state of a *specific* episode from its observed continuation is the one operation no ingested method implements — every world model here conditions on history and action only. [[sources/driving-wm-counterfactuals.md]] closes the gap with monocular depth plus splatting rather than with the model, so whether a model *trained* to condition on the factual continuation would beat geometry is untested. See [[concepts/counterfactual-prediction.md]].
- **Does a frozen advisory VLM beat a fine-tuned VLA backbone?** DriveWAM's frozen Qwen3-VL-8B only emits text guidance and never decodes actions, yet the guidance helps at every data scale. Is the advisory role sufficient, or does it leave value on the table versus VLM-centric policies (DriveVLA-W0, DynVLA) that fine-tune the VLM into the action path? No paper compares the two arrangements at matched backbone and data.
- **Long-horizon memory validity**: DriveWAM validates selective KV memory's accuracy only on 20s clips while profiling cost at 300s. Does content-based eviction hold up over minutes of rollout, and does the training/inference mismatch (full-history attention at training, bounded pools at inference) compound?
- **Zero-shot transfer ceiling**: DriveVA's zero-shot nuScenes/Bench2Drive gains are measured relative to PWM only. How does DriveVA compare zero-shot against VLA methods (FLARE, DriveFine) that are fine-tuned on the target domain? Does joint video-action training provide a sustainable generalization advantage at matched data scale?
- **Is the ego-motion target the right minimal one?** Auto-JEPA argues a planning world model should predict only the ego trajectory latent, and its 2.97× occlusion selectivity is evidence that agent-relevance emerges from that target alone. What is untested is whether the *JEPA objective* is doing the work. No ablation compares it against the obvious baseline — regress waypoints, encode the regression through the same frozen trajectory encoder, retrieve with that. If the two are equivalent, the contribution is the shared latent retrieval space, not joint-embedding prediction. See [[sources/auto-jepa.md]].
- **Does agent selectivity predict driving quality?** Auto-JEPA measures selectivity in latent space (mean $1-\cos$ 0.080 vs. 0.027) and shows behavioral consequences on three hand-picked scenes only. No paper has correlated an occlusion-sensitivity statistic with PDMS, collision rate, or interaction-scenario performance across a dataset. Until someone does, "the model attends to the right things" remains a property of the embedding rather than a demonstrated cause of good driving.
- **Which deterministic latent predictors are leaving performance on the table?** WA-JEPA measures that regression future-prediction is *worse than no future prediction* on multi-view scene latents (90.7 vs. 91.1 EPDMS), and diagnoses it as temporal-mean collapse. DeepSight (DINOv3 BEV frames), FLARE (DINOv2 features), and Latent-WAM (latent world status) all use deterministic objectives on scene-level targets. Whether their targets are low-entropy enough to be safe, or whether swapping in flow matching would buy each of them a point, is untested and cheap to test. See [Pattern 23](#objective-form).
- **Is WA-JEPA's inference-time scene denoising doing anything?** *(Lint note: SUV shows the answer can depend on the benchmark, with future access null on navtest and +4.1 on navhard. Any test of this should report navhard.)* It generates future latents jointly with actions over 12 sampling steps, but its ablations never separate the training objective from the inference computation — the exact control SimWAM ran. If SimWAM's finding generalizes, the scene stream could be dropped at inference for free.
- **Does world modeling buy open-loop accuracy or closed-loop robustness?** *(Lint 2026-09-27: three more navtest-small / navhard-large effects on one backbone. SUV future access +0.3 / +4.1; Metis asymmetric mask +0.5 / +2.2; Metis video-prior scale ≈0 / +2.4. The asymmetry now looks systematic rather than GeoWAM-specific.)* [[sources/geowam.md]] adds future-geometry forecasting to DVGT-2 and gains **+0.6 EPDMS on navtest but +4.9 on navhard** — the same architectural change worth eight times more under the reactive protocol. If that asymmetry replicates, it reframes what world-model pretraining is for and implies navtest is close to the wrong benchmark for evaluating it. Every world-model paper in the wiki optimizes and reports navtest; only GeoWAM and DriveLaW report navhard at all. See [[concepts/navhard-ood-evaluation.md]].
- **Does dense geometric supervision rescue a deterministic latent objective?** GeoWAM pairs JEPA-style cosine alignment on future features — the objective [Pattern 23](#objective-form) measures as *worse than nothing* in isolation — with dense point-map regression, and does not collapse. The natural explanation is that explicit metric targets anchor what a pure feature-alignment loss lets drift toward the temporal mean. Neither paper runs the ablation, and it is one training run for either of them.
- **Geometry versus pixels has never been tested under a fixed planner.** GeoWAM argues geometry beats pixels but compares against other papers' methods; DriveLaW argues video latents beat BEV and VLM hidden states and *does* hold the planner fixed, but geometry is not in its comparison. The controlled experiment — one planner, three conditioning representations including metric point maps — would settle the field's central representation dispute and nobody has run it.
- **Are the 31 unsupervised futures actually futures?** [[sources/da-wam.md]] predicts one latent per candidate but can only supervise the expert-matched one, since offline logs record a single outcome. The other 31 are shaped purely by scorer gradients, and no diagnostic shows they encode anything future-like — a hard-braking candidate's latent is never checked against a full-throttle candidate's for the divergence physics requires. WA-JEPA's temporal-collapse metrics are exactly the right instrument and nobody has pointed them at this. If those latents are just conditioning features, "decision-aligned future prediction" is a scorer-capacity result wearing world-model clothes.
- **Does the shared-vs-per-candidate distinction survive at a realistic horizon?** DA-WAM's per-candidate futures reach only 0.5 s while its trajectories span 8 poses, so the action-specific consequences it claims to exploit — collisions, lane departures, rule violations — mostly fall outside the predicted window. Whether the +0.15 PDMS grows, vanishes, or reverses at 2-4 s is untested and is the single most informative follow-up the design admits.
- **Do the two supervision rules compose?** [[sources/drive-hwm.md]] adds a *target-content* rule (motion far, appearance near) orthogonal to the *objective-form* rule assembled from WA-JEPA and DriveFuture (match the loss to the target's entropy). Nobody has crossed them. The decisive sweep is one run: **long-horizon optical flow under a generative objective versus a regression one.** If flow escapes the entropy problem, it is because motion fields are genuinely lower-entropy than appearance; if it does not, Drive-HWM's flow decoder was absorbing the multimodality and the latent interface is doing more work than the target choice. See [Target Content by Horizon](#target-by-horizon).
- **Image-space flow or ego-frame geometry?** GeoWorldAD prices the coordinate frame at +2.5 PDMS on its own; Drive-HWM's linear probe puts an image-space flow latent first for future ego motion (83.7) and an ego-frame BEV latent last (71.2). No paper has compared a flow target against a metric-geometry target under a fixed planner, and it is a single sweep in either codebase. See [Image-Space Flow Against Ego-Frame Geometry](#flow-vs-geometry).
- **Does rate separation do anything a closed loop would show?** Drive-HWM's slow–fast hierarchy is worth +0.3 to +0.8 PDMS on NAVSIM — but NAVSIM is single-shot, so its slow model runs once per scenario and the high-frequency re-decision the design exists for never happens. The claim needs [[concepts/hugsim-benchmark.md]] or Bench2Drive, where new observations actually arrive between fast steps. Until someone runs it, rate hierarchy is an untested idea with a cheap conditioning stream attached. See [Pattern 35](#two-rates).
