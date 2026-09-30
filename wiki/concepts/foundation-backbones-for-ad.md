---
title: Foundation Backbones for AD
type: concept
sources: ["raw/papers/MM-Future_ Multi-Mode Joint World–Action Modeling for Autonomous Driving.md", "raw/papers/DriveReferee_ Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models.md", "raw/papers/WALT_ Learning World-Model-Aligned Latent Trajectories for Autonomous Driving.md", "raw/papers/ReDrive_ Shaping Representations with World Modeling for End-to-End Driving.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md", raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, raw/papers/Unified Driving Tokens_ Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning.md, raw/papers/ReWorld_ Representation Learning for World Action Models.md, raw/papers/LWDrive_ Layer-Wise World-Model-Guided Vision-Language ModelPlanning for Autonomous Driving.md, raw/papers/GeoWorldAD_ Geometry World Action Model for Autonomous Driving.md, raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md, raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md, raw/papers/CLEAR_ Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving.md, raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md, raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md, raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md, raw/papers/DriveLaW_ Unifying Planning and Video Generation in a Latent Driving World.md]
related: [sources/mm-future.md, sources/drivereferee.md, sources/walt.md, sources/redrive.md, sources/physwam.md, sources/ad-e2e-jepa.md, sources/metis.md, sources/suv.md, concepts/wam-attention-masks.md, concepts/general-capability-retention.md, sources/qwen-drive-1.0.md, sources/drive-hwm.md, sources/coworld-vla.md, sources/unified-driving-tokens.md, concepts/visual-tokenization.md, sources/reworld.md, sources/geoworldad.md, sources/adaptive-wam.md, sources/brainwam.md, sources/foresight.md, sources/da-wam.md, sources/geowam.md, sources/wa-jepa.md, sources/auto-jepa.md, sources/simwam.md, sources/sgdrive.md, sources/drivelaw.md, concepts/vlm-domain-adaptation.md, concepts/world-model-for-ad.md, concepts/dual-system-vla.md, concepts/adaptive-routing.md, concepts/r1-zero-like-training.md, sources/autovla.md, sources/nord.md, sources/elf-vla.md, sources/spanvla.md, sources/driveva.md, sources/alpamayo-r1.md, sources/explorevla.md, sources/onedrive.md, sources/onevl.md, sources/latent-wam.md, sources/drive-jepa.md, sources/policy-world-model.md, sources/clear.md, sources/understanding-r1-zero-like-training.md, sources/drivewam.md]
created: 2026-05-01
updated: 2026-09-30
confidence: high
---

## What It Tracks

Driving VLA papers increasingly differ less by whether they use a foundation model and more by which backbone is frozen, fine-tuned, paired with an action expert, or used only as a teacher.

## Backbone Roles

| Role | Examples | Notes |
| --- | --- | --- |
| Reasoning VLM backbone | Qwen2.5-VL, Qwen3-VL, InternVL | Usually paired with action tokens or a separate action expert. |
| Teacher/annotator | Qwen3-VL-32B, Gemini-style annotators, LRM critics | Used for CoT, failure feedback, or reward shaping. |
| Video/world backbone | Wan, Cosmos, Show-o/MAGVIT | Supplies future visual prediction or joint video-action generation. |
| Unified understanding/generation backbone | Show-o / PWM | Uses one autoregressive transformer for video tokens, text tokens, and action tokens. |
| Self-supervised video encoder | V-JEPA / Drive-JEPA | Learns planning-aligned predictive video representations before trajectory decoding. |
| **Frozen** off-the-shelf video encoder | V-JEPA 2 in Auto-JEPA | Used as-is with no driving adaptation at all; the JEPA objective is moved to a trajectory latent space instead of the encoder. |
| Video encoder re-pretrained with a *changed* JEPA objective | V-JEPA 2 in WA-JEPA | Future-masked instead of random-masked, flow matching instead of L1 regression; the backbone is kept but its training task is replaced. |
| Fully fine-tuned video encoder with live **action-conditioned** predictive supervision | V-JEPA 2 ViT-L in ReDrive | Driving-domain JEPA pretraining, then joint training in which a trajectory-conditioned predictor regresses EMA-target future latents while a flow-matching planner reads the same encoder. The predictor is discarded at inference. Unfreezing is worth +3.5 PDMS and the predictive loss +0.7. See [below](#redrive-sweep). |
| LoRA-adapted video encoder with live predictive supervision | **V-JEPA 2.1** in DA-WAM | Base frozen, LoRA updated by *both* future-prediction and planning gradients throughout planner training; paired with an EMA target. Beats full fine-tuning by 0.36 PDMS. |
| **Frozen image encoder as a rollout state space**, behind a learned compressing projector | DINOv3 ViT-L in AD-E2E-JEPA | The encoder is never updated during world-model training. A two-layer conv projector (16× fewer tokens, 4× narrower, SIGReg-regularized) defines the space the predictor rolls out in. For imitation learning the same encoder and projector are then fully fine-tuned. See [below](#encoder-by-job). |
| Hidden-state semantic router | Qwen 3.5 0.8B in CLEAR | Uses LLM hidden states for scheduling and trajectory scoring rather than text/action generation. |
| Base-model prior confound | Qwen2.5-Math in R1-Zero-like training | No-template QA behavior shows that apparent RL gains can depend heavily on pretraining and template choice. |
| Geometric teacher | WorldMirror / VGGT | Supplies training-time spatial features for Latent-WAM; removed at inference. |
| Geometry backbone **as** the policy trunk | DVGT-2 in GeoWAM | A driving visual geometry transformer is the encoder, the point-head initialization, and the strongest baseline all at once; fine-tuned rather than distilled or frozen. |
| Frozen understanding expert | AutoMoT-style UE | Preserves general reasoning and avoids catastrophic forgetting. |
| Shared attention backbone | OneDrive | Reuses VLM causal attention for image, perception, planning, and text tokens while replacing task FFNs. |
| Latent reasoning backbone | OneVL | Fine-tunes Qwen3-VL-4B so visual/language latent tokens can be decoded into future frames and text explanations during training. |
| Video backbone **as** the policy core | Wan2.2-TI2V-5B in DriveVA and DriveWAM | The video DiT is fine-tuned into the action path itself, not attached as a generation branch. |
| **Omnimodal world model as the whole policy** | **Cosmos 3 Nano (15.2B) in PhysWAM** | A Qwen3-VL-based Mixture-of-Transformers with a frozen understanding pathway and a 7.0B generation pathway that is fully fine-tuned. Video, metric depth and SE(3) pose rows are denoised in one sequence; the pretrained pose projection is reused for actions. The largest model in the wiki, at 9.4 GPU-seconds per plan. See [below](#cosmos3). |
| Swappable video prior | LTX-Video / Wan2.1-1.3B / Wan2.2-5B / Cosmos-Predict2.5 in SimWAM | Co-trained through shared attention only, so the backbone can be replaced without touching the action expert. |
| Video generator as feature extractor | LTX-Video 2B in DriveLaW | Mid-denoising block latents are cached and cross-attended by a 133M action DiT; the generator is repurposed as the perception encoder. |
| Frozen advisory VLM | Qwen3-VL-8B in DriveWAM | Emits chunk-level text guidance consumed by cross-attention; never decodes actions and is never fine-tuned. |
| VLM as frozen world model | InternVL3-2B in SGDrive | Fine-tuned in stage 1 to host structured ⟨world⟩ queries, then frozen in stage 2 while only the DiT planner trains. |
| Pretrained VLM **left architecturally untouched** | Qwen3.5-4B in Qwen-Drive-1.0 | No added tokens or experts; two external modules read it (a BEV head from its image-token outputs, a Planning Expert from its cached attention KV). Fully fine-tuned in one stage, then frozen. See [below](#vlm-adaptation-ladder). |
| VLM tower as a **3D detection encoder** | SigLIP-Qwen in Qwen-Drive-1.0 | The same SigLIP-style tower, initialized from Qwen3.5-4B, swapped into BEVFormerV2/PETR/PETRv2 in place of ResNet-50: **+6.6 to +10.1 mAP** with everything else fixed. |
| VLM read at **several depths** | Qwen2.5-VL-3B in LWDrive | Fine-tuned in stage 1 under a 15:1 future-frame world-model loss, then frozen while six planner refinement stages tap layers {6,…,36}. The multi-depth readout is worth only **+0.2 PDMS** over final-layer-only — see [below](#readout-depth-vlm). |

## Takeaways

- Bigger or newer backbones do not make benchmark comparisons fair by themselves; input cameras, training data, RL stage, and action head matter.
- Frozen-backbone designs can outperform fine-tuned VLMs when the action expert is well-coupled.
- Teacher-only use should be distinguished from inference-time use because it changes deployment cost and risk.
- OneDrive shows that even inside one VLM decoder, not all pretrained modules transfer equally: attention transfers to structured driving queries, while language FFNs may need task-specific replacement.
- OneVL shows a Qwen3-VL backbone can host latent reasoning tokens, but stable adaptation requires staged auxiliary-decoder training; direct joint fine-tuning collapses.
- Latent-WAM shows that DINOv2-Base can be turned into a compact planning encoder through geometric distillation, but LoRA is not sufficient for that distillation target.
- CLEAR shows a compact language model can be useful even when it does not emit actions: hidden states can route generation budget and score candidates.
- Understanding R1-Zero-like Training shows that base-model pretraining and templates can dominate the apparent benefit of RL; this caveat should carry over to Qwen-family VLA backbones.
- DriveWAM shows that a pretrained video backbone's prior is only retained if the video objective is retained: initializing from Wan2.2-TI2V-5B and then dropping video supervision is worse than training from scratch with it. Backbone choice and training objective cannot be selected independently.
- SimWAM shows the converse for capacity: with the objective held fixed, video-prior scale barely matters (1.3B ≈ 5B), while prior quality and driving-domain pretraining do. Read together, the two papers say the training signal dominates the backbone.
- The wiki now has three distinct answers to "what should a driving backbone be pretrained on": language/semantics, video appearance dynamics, and metric geometry. All three have papers reporting 89-92 on NAVSIM-v2, and no experiment holds the planner fixed while swapping across families. The comparison the field most needs is the one nobody runs.
- Auto-JEPA is the wiki's minimal position on backbone investment: the visual encoder is a stock V-JEPA 2 checkpoint, never adapted to driving, and only the predictor and small task modules train. It reaches 91.3 PDMS. Drive-JEPA spends 208 h of curated video and a 3-day 8-GPU pretraining stage adapting the same family of encoder and reaches 93.3. The 2.0-PDMS gap is the clearest available price tag on driving-domain encoder adaptation — though the two differ in planner design as well, so it is an upper bound on what adaptation buys, not a clean measurement.
- The adaptation ladder now has a data point in every backbone family. A frozen video DiT loses 6.42 PDMS to a jointly LoRA-adapted one (Adaptive-WAM); LoRA is insufficient for geometric distillation into DINOv2 (Latent-WAM); and a converged 3D head on **frozen** vision-language features trails a dedicated detector on the same features by 6.34 mAP, recovering +10.46 once the encoder and VLM are unfrozen (Qwen-Drive-1.0). Off-the-shelf use is the expensive option everywhere it has been measured.
- Vision-language pretraining is a strong *visual* initialization even for tasks with no language in them: swapping ResNet-50 for a Qwen-initialized SigLIP tower is worth 6.6-10.1 mAP to three separate 3D detectors under a fixed schedule.
- SGDrive makes the same point on the VLM side: InternVL3-**2B** with a scene-agent-goal query hierarchy reaches 87.4 PDMS, beating plain InternVL3-8B and QwenVL2.5-8B (both 83.3) by 4.1 and ReCogDrive-8B (86.8) at a quarter the size. Driving-specific representational structure buys more than 4× the parameters.

## Qwen Prior Caveat

[[sources/understanding-r1-zero-like-training.md]] finds that Qwen2.5-Math base models perform best with no template, likely because pretraining already included concatenated question-answer text. This is not direct evidence about Qwen-VL driving models, but it is a warning for backbone interpretation: when a VLA paper reports large RL gains from a Qwen-family base, the baseline prompt, output template, and hidden pretraining priors need to be separated from genuine RL-created capability.

## CLEAR Qwen Hidden-State Router

CLEAR ([[sources/clear.md]]) pairs a frozen Drive-JEPA visual encoder with a fully fine-tuned Qwen 3.5 0.8B model. The Qwen model is not used as an autoregressive planner. Instead, its hidden states feed an Adaptive Scheduler that picks `(alpha, N)` and a Cross-Attention Scorer that ranks generated trajectories.

This is a distinct backbone role from VLA action decoding. The LLM supplies traffic semantics and risk priors, while the trajectory generator remains a compact MLP-Mixer operating in VAE/PCA trajectory space. The result is 93.7 PDMS on NAVSIM-v1, suggesting hidden-state use can be more deployment-friendly than text-format action generation when the action head is strong.

## OneDrive Diagnostic

OneDrive's Table 1 isolates attention vs. FFN transfer for InternVL3-1B and Qwen2.5-VL-3B. Reusing attention while randomizing FFNs gives the best NDS for both tested backbones (32.05 for InternVL3-1B and 31.37 for Qwen2.5-VL-3B). Reusing FFNs can be actively harmful, especially for Qwen2.5-VL-3B where attention+FFN initialization drops to 27.14 NDS.

## OneVL Backbone Use

OneVL uses Qwen3-VL-4B-Instruct as the main VLM and keeps the auxiliary language/visual decoders training-only. The backbone role is therefore not just "planner" or "reasoner"; it is the shared latent-state generator whose hidden states must satisfy trajectory, text-CoT, and future-visual-token objectives. This makes OneVL a useful counterpoint to frozen-backbone designs: full fine-tuning works, but only after a warmup and decoder-alignment curriculum.

## Latent-WAM Backbone Use

Latent-WAM ([[sources/latent-wam.md]]) uses DINOv2-Base as the deployed visual encoder and WorldMirror, built on VGGT, as a frozen training-time geometry teacher. This is not a VLM backbone: the foundation-model value is spatial and geometric rather than linguistic.

The backbone ablation is unusually strong. DINOv2-Base full fine-tuning reaches 89.3 EPDMS, DINO-Small reaches 86.3, Small-LoRA reaches 84.7, and Base-LoRA collapses to 68.5. For geometric feature distillation, low-rank adaptation appears too restrictive; the model needs full backbone updates to align high-dimensional spatial features with planning.

## Drive-JEPA V-JEPA Use

Drive-JEPA ([[sources/drive-jepa.md]]) adds a self-supervised video-encoder role that is not a language model and not a pixel-generating video backbone. It initializes from V-JEPA 2, then pretrains a ViT-L encoder on 208 hours of curated front-view driving videos with a JEPA latent-prediction objective.

The vision-pretraining ablation is the main evidence: ImageNet ResNet34 reaches 76.0 PDMS, DINOv2 ViT/L 76.1, SigLIP ViT/L 83.4, V-JEPA 2 ViT/L 86.1, and Drive-JEPA's driving-video-pretrained ViT/L 89.0. MAE and DepthAnything did not converge in the paper's setup. This suggests that temporal latent prediction transfers better to planning than static image-level pretraining when the downstream decoder is intentionally simple.

## LoRA vs. Full Fine-Tuning: Two Papers, Opposite Answers

[[sources/latent-wam.md]] and [[sources/da-wam.md]] both adapt a frozen self-supervised visual backbone toward a predictive target, both ablate the adaptation method, and **they disagree**:

| Paper | Backbone | Distillation / prediction target | Full fine-tune | LoRA |
|---|---|---|---:|---:|
| Latent-WAM | DINOv2-Base | WorldMirror/VGGT **geometric features** | **89.3 EPDMS** | **68.5** (collapse) |
| DA-WAM | V-JEPA 2.1 | **EMA video latents** (JEPA) | 92.62 PDMS | **92.98** |

Latent-WAM concluded low-rank adaptation is "too restrictive" — the model "needs full backbone updates to align high-dimensional spatial features with planning." DA-WAM finds LoRA *better* than full fine-tuning by 0.36 PDMS, and frames the base network staying frozen as what "retains the pretrained backbone's representational capabilities while adapting the latent space to driving-specific objectives."

**The plausible reconciliation is the distance between the pretrained representation and the target.** Geometric distillation asks a semantically-pretrained DINOv2 to emit metric spatial features — a large representational move that a low-rank update cannot express. JEPA adaptation asks a video-predictive encoder to keep doing video prediction, slightly re-aimed; there the risk is *destroying* a prior that is already close to correct, which is exactly what full fine-tuning does. **Rule of thumb: LoRA when the pretrained objective already matches the downstream one, full fine-tuning when you are asking for a different kind of feature.**

Neither paper tests the other's setting, so this is a hypothesis fitted to two points. But it is a cheap experiment for anyone adapting a foundation backbone, and it predicts that [[sources/geowam.md]]'s DVGT-2 fine-tuning (geometry model → geometry forecasting, a small move) should also favour low-rank adaptation — which GeoWAM does not test, using a plain reduced learning rate of $2	imes10^{-5}$ instead.

**A related result on the target side.** DA-WAM ablates the target-encoder policy with LoRA fixed: frozen 92.98, separate 93.10, shared 93.34, **EMA 93.68**. That is the cleanest isolation of the EMA mechanism in the wiki — worth +0.70 over a frozen target — and it is a larger effect than DA-WAM's headline per-candidate-future contribution (+0.15). The EMA momentum coefficient is never reported.

**A LoRA use with no ablation, and a target that cannot be far away** *(2026-09-30)*. [[sources/mm-future.md]] adapts DINOv2-S with rank-32 LoRA, keeps its register tokens (the DrivoR design), and compresses four cameras and two frames into 64 learned tokens. Future targets come from an EMA copy of the same adapted encoder.

- It fits the distance reading above in a trivial way: the prediction target *is* the encoder's own output, so the representational move asked of the backbone is small and low-rank adaptation is enough to reach 93.4 PDMS.
- Neither the LoRA choice nor the EMA target is ablated, so it adds no third measurement to the table. DA-WAM's EMA-against-frozen result (+0.70) remains the only isolation of that mechanism.
- It is the smallest visual backbone among the 93+ PDMS entries (ViT-S), which repeats the DrivoR observation that a scorer-based planner does not need a large encoder.

### A third data point, and a new backbone: Cosmos 3 Nano in PhysWAM {#cosmos3}

[[sources/physwam.md]] adapts the generation pathway of Cosmos 3 Nano two ways and reports both:

| Adaptation | Trained parameters | Updates | Other differences | navtest "PDMS" / EPDMS |
|---|---:|---:|---|---:|
| LoRA, rank 128, attention projections | 122.7M | 16,000 | LR 2e-4, weight decay 0.05, no EMA | 88.2 / 86.5 |
| Full fine-tune of the generation pathway | 7.0B | 30,000 | LR 2e-5, no weight decay, EMA | 91.4 / 90.3 |

**Full fine-tuning is ahead by about 3.2 / 3.8, and the comparison is confounded** by training length, learning rate and weight averaging. It is the LoRA setting the paper uses for cheap ablations, not a controlled adaptation study. Against the rule of thumb above it is weak evidence for the "different kind of feature" branch: the model is asked for metric depth and SE(3) trajectories tied by a geometric loss, a larger move from web-video generation than re-aiming a JEPA encoder. The gap is largest on right turns (84.3 → 90.4 PDMS).

Three backbone facts worth recording:
- **The pretrained action interface is kept.** Cosmos 3 already has a pose projection for a 9-D relative-pose row (translation plus two rotation columns). PhysWAM reuses it unchanged, with translation in metres. New parameters (the ray embedding and a depth-modality embedding) are zero-initialized.
- **Depth goes through the RGB VAE**, as a log-encoded grey video, so no depth-specific tokenizer or head is added to the generator. [[sources/suv.md]] does the same with relative depth on Wan2.2.
- **Off the shelf, the same model is the worst future-depth baseline in GeoWAM's nuScenes table** (Cosmos 3 + DVGT: AbsRel 0.376 at +2 s, against GeoWAM's 0.245). Fine-tuned with depth supervision on NAVSIM it reaches 0.175, on a different dataset and rig. That is suggestive for the adaptation ladder further down this page and is not a measurement of it.

There is no backbone swap, so nothing here says whether Cosmos 3 is a better starting point than Wan2.2-5B. SUV, on Wan2.2-5B with a ~1B action expert, scores 91.0 EPDMS at 288 ms.

**A second Cosmos 3 paper from the same group, and a backbone-size pair** *(2026-09-30)*. [[sources/drivereferee.md]] fine-tunes the generation tower of two Cosmos 3 sizes with one front camera and imitation only.

| Backbone | Total / trained | PDMS | EPDMS |
|---|---|---:|---:|
| Cosmos3-Edge | 4B / 2B | 90.47 | 90.03 |
| Cosmos3-Nano | 16B / 8B | 91.08 | 90.69 |

- **Four times the parameters is worth +0.6.** A 462-pair preference fine-tune is worth +0.9 to +1.0 on either size, so the small model after that step (91.50 PDMS) is ahead of the large one before it.
- **The one-camera, video-plus-action base is 0.4 EPDMS above PhysWAM's three-camera, depth-generating system** (90.69 against 90.3) on the same backbone. The papers do not compare with each other and the recipes differ, so this is an observation, not an ablation of depth or of camera count.
- The two papers count the model differently (15.2B with a 7.0B trained pathway in PhysWAM; 16B with an 8B trainable tower here).

## Geometry Foundation Models Are a Third Backbone Family

Most of this page tracks two families: **language/VLM backbones** (Qwen, InternVL, Emu3) and **video-generation backbones** (Wan, Cosmos, LTX-Video, Show-o). [[sources/geowam.md]] makes a third one explicit — **visual geometry models**, the DUSt3R → CUT3R → VGGT → MapAnything lineage, specialized for driving by DVGT and DVGT-2.

| Family | Pretraining signal | What it supplies a planner | Wiki entries |
|---|---|---|---|
| Language / VLM | Web text + image-text | Semantics, reasoning, instruction following | Most VLA entries |
| Video generation | Raw video | Appearance dynamics, temporal priors | DriveVA, DriveWAM, SimWAM, DriveLaW, Epona |
| Self-supervised video (JEPA) | Raw video, latent prediction | Spatiotemporal representations | Drive-JEPA, Auto-JEPA, WA-JEPA |
| **Visual geometry** | **Multi-view images → metric 3D** | **Explicit metric structure in the action's coordinate frame** | **GeoWAM (DVGT-2), Latent-WAM (WorldMirror/VGGT as teacher)** |

The two geometry entries use the family very differently. [[sources/latent-wam.md]] treats WorldMirror as a **frozen training-time teacher**, distilling its features into compact latents and discarding it at inference — and its ablation found the distillation target demanded full backbone updates, with Base-LoRA collapsing from 89.3 to 68.5 EPDMS. GeoWAM treats DVGT-2 as **the policy trunk itself**: encoder and point head are initialized from it and fine-tuned at $2	imes10^{-5}$ while new components train at $10^{-4}$, and geometry decoding stays live at inference.

**The awkward part of GeoWAM's evidence is that DVGT-2 is also its strongest baseline.** DVGT-2 alone reaches 89.6 EPDMS on navtest to GeoWAM's 90.2, and 31.7 on navhard to GeoWAM's 36.6. So the measured value of GeoWAM's addition — future forecasting on top of a geometry backbone — is +0.6 open-loop and +4.9 reactive. That is a much narrower claim than "geometry beats pixels," and the paper does not separate what the backbone contributes from what the forecasting objective contributes. It is the same structural problem as the encoder ablations below, in a different guise: **a strong initialization and a novel objective are being credited together.**

## V-JEPA 2 for Planning: Three Independent Encoder Ablations

[[sources/wa-jepa.md]] and [[sources/drive-jepa.md]] each run a controlled encoder-initialization sweep with a fixed downstream planner, and they agree on the ordering. This is now the wiki's best-supported backbone claim. A third sweep, from [[sources/redrive.md]], is [below](#redrive-sweep).

| Encoder | Drive-JEPA (PDMS, NAVSIM-v1) | WA-JEPA (EPDMS, NAVSIM-v2) |
| --- | ---: | ---: |
| ImageNet ResNet-34 | 76.0 | – |
| DINOv2 ViT-L | 76.1 | – |
| DINOv3 | – | 83.8 |
| MAE ViT-L | did not converge | 83.8 |
| SigLIP ViT-L / SigLIP2 | 83.4 | 83.1 |
| **V-JEPA 2 ViT-L** | **86.1** | **89.5** |
| + driving-domain adaptation | 89.0 (208 h video) | 91.7 (nuPlan, future-masked) |

Two papers, different benchmarks, different planners, different years, same conclusion: **V-JEPA 2 initialization is worth +2.7 to +5.7 over the best image-level self-supervised or vision-language alternative**, and the alternatives cluster tightly among themselves (DINOv3 83.8 ≈ MAE 83.8 ≈ SigLIP2 83.1 in WA-JEPA; DINOv2 76.1 vs. SigLIP 83.4 in Drive-JEPA, a wider spread). Both papers also show that driving-domain adaptation of that checkpoint adds a further +2.2 to +2.9.

**Both share the same confound, and it matters.** Every alternative in both tables is **image-level** pretrained; V-JEPA 2 is the only video-pretrained entry. Neither paper includes a video-pretrained non-JEPA control — VideoMAE, an inflated DINOv3, or a video-generation encoder like the Wan/Cosmos backbones tracked below. So "the JEPA objective transfers to planning" and "video pretraining transfers to planning" are not separated by either experiment. WA-JEPA states the stronger of the two conclusions ("the gap tracks the V-JEPA 2 pre-training objective"), which its own design does not support.

The distinction is not academic. [[sources/simwam.md]] and [[sources/drivelaw.md]] show video-generation priors are also highly effective for planning, and those are video-pretrained *without* a JEPA objective. If the operative variable is temporal pretraining rather than joint-embedding prediction, the two families are converging on the same explanation from opposite directions — and the cheap experiment that would tell them apart has not been run.

### A Third Sweep, With the Image Encoders Given Every Advantage (ReDrive) {#redrive-sweep}

[[sources/redrive.md]] repeats the experiment with a control the two tables above lack. DINOv2, MAE and V-JEPA 2 encoders of comparable size start from their released checkpoints, are **further pretrained on the same driving video for the same number of epochs**, and then go through the **same joint training, including a trajectory-conditioned future-prediction loss**, with the encoder trainable.

| Encoder | PDMS after Stage 2 |
|---|---:|
| MAE | 83.9 |
| DINOv2 | 84.4 |
| V-JEPA 2, 4-frame driving pretraining | 89.9 |
| V-JEPA 2, 8 / 12 / 16 frames | 90.4 / 90.7 / 90.8 |

**Three papers now agree on the ordering.** V-JEPA 2 leads the best image-level alternative by +2.7 (Drive-JEPA), +5.7 (WA-JEPA) and +5.5 (ReDrive). What ReDrive removes from the list of explanations:
- *Driving-domain data.* The image encoders got it too.
- *A predictive objective during fine-tuning.* They got that too.

What it leaves: every alternative is still image-level, so the video-versus-JEPA confound stands. How the image encoders were "further pretrained" and applied to four frames is not described.

**The clip-length sweep is the best within-objective evidence that temporal pretraining is the operative variable.** Same backbone, same objective, same data, only the pretraining clip grows from 0.4 s to 1.6 s: +0.9 PDMS, monotone and saturating.

**Driving-domain adaptation is cheap to skip here.** Stock V-JEPA 2 with a trainable encoder scores 88.9. ReDrive's Stage 1 adds +0.3 at 4 frames (about +1.2 at 16), against +2.9 in Drive-JEPA and +1.5 to +2.2 in WA-JEPA. The likely reason is that ReDrive's baseline already fine-tunes the encoder end to end, which absorbs most of what domain pretraining would supply.

**The adaptation ladder for this encoder** (4-frame, no future predictor): frozen 85.7 → trainable 89.2. **+3.5 from unfreezing**, the largest single training choice in the paper, and a video-JEPA entry for the adaptation ladder further down this page.

**It also bears on a hypothesis recorded [further down](#encoder-by-job).** That section reads AD-E2E-JEPA's +5.2 (a future-prediction-pretrained projector on DINOv3) as a hint that temporal-predictive training in a small adapter might account for the V-JEPA gap. ReDrive's DINOv2 arm has a fully trainable encoder under a future-prediction loss and still trails V-JEPA 2 by 5.5. **On this evidence, predictive fine-tuning of an image encoder does not substitute for video pretraining**, and AD-E2E-JEPA's gain is better read as recovery from its randomly initialized bottleneck.

### Drive-HWM Runs the Missing Control — Partially {#jepa-vs-generative}

**[[sources/drive-hwm.md]] is the first paper here whose swap holds video pretraining constant and varies only the objective family.** Its slow world model is the future-prediction module, everything downstream is fixed (Emu3-8B fast model, FiLM conditioning, AR expert, identical recipe), and the three candidates are all video-pretrained:

| Slow-model backbone | Pretraining objective | NC | DAC | PDMS |
| --- | --- | ---: | ---: | ---: |
| CogVideo | video **generation** | 98.8 | 98.1 | 93.0 |
| WAN | video **generation** | 99.0 | 98.4 | 93.2 |
| **V-JEPA** | **joint-embedding prediction** | **99.6** | **99.0** | **93.8** |

**This is the complement to the two sweeps above, not a repeat of them.** WA-JEPA and Drive-JEPA vary the objective *and* the modality together, because every alternative they test is image-level. Drive-HWM varies the objective *alone*: all three arms saw video. **+0.6 over WAN and +0.8 over CogVideo is therefore the first estimate on this page of what the joint-embedding objective is worth once temporal pretraining is held fixed** — and it is roughly a fifth of the +2.7 to +5.7 the encoder sweeps attribute to V-JEPA 2 against image-level alternatives. If both numbers hold, most of that larger gap is *video pretraining*, not *JEPA*, which is the reading this page has been flagging as unsupported in WA-JEPA's stronger claim.

The paper's own explanation is about what the objective *discards*:

> "its latent-space predictive objective focuses more directly on temporally meaningful scene dynamics while avoiding the unnecessary complexity of reconstructing low-level visual details."

**Four reasons this only partially closes the confound.**

1. **It is the wrong slot.** The sweep varies the *future-predictor* backbone, not the planner's visual encoder — Emu3 does the encoding in all three arms. So it bears on "what should predict the future" rather than "what should see the present," which is the question the sweeps above are answering.
2. **Not compute- or capacity-matched.** No parameter count is given for any of the three, and CogVideo and WAN are large generative models being asked to emit a conditioning latent, which is not what they were built to do.
3. **The spread is small and unreplicated.** 0.8 PDMS, single runs, no seed variance, one benchmark.
4. **Which V-JEPA was run is ambiguous** — see [below](#vjepa-naming-ambiguity).

**What it does establish firmly** is consistent with [[sources/adaptive-wam.md]] (readout depth worth 4.80, video noise index ≤0.15), [[sources/foresight.md]] (a frozen 2.5B generator as the primary encoder, 870 ms, +0.3 in the row that isolates it), and [[sources/drivelaw.md]] (early denoising latents beat clean generated futures, 89.1 vs. 23.2): **the pixel-generation capacity of a video backbone is mostly not what the planner is buying.** Drive-HWM adds the cheapest version — skip the generator, predict a motion latent, 25.6 ms — and wins its own comparison.

**The same paper also swaps the fast-model VLM with the slow model fixed**, which is rarer than it should be:

| Fast-model backbone | NC | DAC | PDMS |
| --- | ---: | ---: | ---: |
| LLaVA-OneVision | 99.0 | 98.4 | 93.5 |
| "Qwen2.5-VL" *(cited ref. is the Qwen3-VL report)* | 99.1 | 98.6 | 93.3 |
| **Emu3** | **99.6** | **99.0** | **93.8** |

**0.5 PDMS across three general-purpose multimodal backbones** — a much flatter axis than the world-model slot, and flatter than the 1.3-point spread Drive-HWM measures for *how* the slow latent is injected. That ordering is worth holding onto: on this evidence the **conditioning interface matters more than the VLM identity**, which is the opposite of where most of this literature spends its architecture budget. The paper's explanation for Emu3 is its discrete-token interface — the same property that makes FiLM rather than concatenation the natural injection point, since Emu3's sequence layout is load-bearing. Note the caveat that this sweep's Qwen row cannot be attributed to a specific model ([below](#vjepa-naming-ambiguity)).

### Which V-JEPA? {#vjepa-naming-ambiguity}

Drive-HWM's slow backbone is named inconsistently in four places: Fig. 2's caption says **VL-JEPA**, Table V says **V-JEPA [55]**, §IV-D's prose says **V-JEPA**, and reference [55] is **VL-JEPA** (arXiv 2512.10942), a vision-*language* JEPA from a different group than Meta's V-JEPA / V-JEPA 2.

This matters because V-JEPA-family encoders now appear in six ingested papers ([[sources/drive-jepa.md]], [[sources/wa-jepa.md]], [[sources/auto-jepa.md]], [[sources/da-wam.md]], [[sources/coworld-vla.md]], Drive-HWM) and every cross-paper statement on this page assumes a common backbone. This wiki records Drive-HWM's slow backbone as **V-JEPA-family, exact variant undetermined**, and the swap above as evidence about the *objective family* rather than about a specific checkpoint.

**A second citation error sits in the same table**: the fast-model row labelled "Qwen2.5-VL" cites reference [57], which is the **Qwen3-VL technical report** (arXiv 2511.21631). The two are a generation apart, so that row's 93.3 cannot be attributed to either model with confidence.

### DINOv3 Returns, for a Different Job {#encoder-by-job}

[[sources/ad-e2e-jepa.md]] is the first JEPA paper in the wiki **not** built on V-JEPA. It uses a frozen DINOv3 ViT-L, and the reason is inherited: JEPA-WM (Terver et al., not ingested) ablates encoders for action-conditioned latent planning and prefers DINOv3 over DINOv2 and over V-JEPA 2. AD-E2E-JEPA attributes that to DINOv3's "dense semantic representations", adopts the default, and does not re-run the swap on driving data.

Set beside the sweeps above, the two literatures rank the same pair of encoders in opposite order:

| Use of the encoder | Preferred | Evidence | Measured on driving? |
|---|---|---|---|
| Initialization for a fine-tuned imitation planner | **V-JEPA 2** over DINOv3 (89.5 vs 83.8 EPDMS) | [[sources/wa-jepa.md]], [[sources/drive-jepa.md]] | Yes |
| Frozen state space for an action-conditioned rollout model | **DINOv3** over V-JEPA 2 | JEPA-WM, as cited by AD-E2E-JEPA | **No** (robotics tasks) |

This need not be a contradiction. A rollout target has to be spatially dense and stable under a frozen encoder. A planner initialization has to carry motion cues into fine-tuning. But **neither cross-cell has been run on NAVSIM**: V-JEPA 2 as the rollout state space, or DINOv3 inside WA-JEPA's planner with a temporal objective added. Until one is, "DINOv3 is the right world-model encoder for driving" is a borrowed default.

**The projector result bears on the video-versus-JEPA confound above.** In the same paper's imitation-learning experiment, everything is fixed except how a two-layer conv projector on top of DINOv3 is initialized:

| Encoder stack | Temporal-predictive training anywhere? | EPDMS (corrected) |
|---|---|---:|
| DINOv3 + random projector | No | 80.2 |
| DINOv3 + projector pretrained by action-conditioned next-latent prediction | **Yes, in the projector only** | **85.4** |
| *(WA-JEPA, other planner, 4 cameras)* DINOv3 | No | 83.8 |
| *(WA-JEPA, other planner, 4 cameras)* V-JEPA 2 | Yes, in the encoder | 89.5 |

**+5.2 from a small adapter trained to predict the future, against +5.7 from swapping to a video-pretrained encoder.** The two numbers come from different planners and camera counts and should not be subtracted. What the pair suggests is a hypothesis worth a run: the operative variable in the sweeps above may be *temporal-predictive training somewhere in the visual stack*, not the identity of the encoder. It would fit [[sources/drive-hwm.md]]'s finding that the JEPA objective itself is worth only +0.6 once video pretraining is held fixed.

*(2026-09-30, later the same day: [[sources/redrive.md]] supplies counter-evidence. Its DINOv2 encoder is trained end to end under a future-prediction loss and still scores 84.4 against V-JEPA 2's 89.9. See [the third sweep](#redrive-sweep). The hypothesis below is now unlikely in its strong form.)*

Three caveats keep this a hypothesis. The control is a **randomly initialized 16× bottleneck**, not DINOv3 with its full token grid, so part of the +5.2 may be recovery from a handicap. The paper does not say whether the transferred projector came from the 10 h or the 70 h world model. And it is a single run. The cheap decisive experiment is the same projector pretraining placed on top of V-JEPA 2: if it adds little there, the two effects are the same effect.

## Auto-JEPA: Freezing the Encoder, Moving the Objective

[[sources/auto-jepa.md]] inverts Drive-JEPA's allocation. Drive-JEPA applies the JEPA objective *to the encoder* — masked video representation prediction over 208 hours of curated driving footage — and then trains a proposal planner on top. Auto-JEPA leaves V-JEPA 2 exactly as released and applies the JEPA objective *to a trajectory latent space*, predicting the frozen encoding of the future ego trajectory.

| | Drive-JEPA | Auto-JEPA |
| --- | --- | --- |
| V-JEPA 2 role | Initialization for further pretraining | Frozen feature extractor, unmodified |
| JEPA objective applied to | Driving video representations | Future ego-trajectory latents |
| Driving-domain adaptation | 208 h curated video, 8 H800 GPUs, 3 days | None |
| Trained at planner time | ViT-L encoder + full planner | History/command encoders, 24-layer predictor, scorer, gate |
| Input resolution | 512×256 | 256×256 |
| NAVSIM-v1 | 93.3 PDMS | 91.3 PDMS |

The relevant lesson for this page is that **a general-purpose self-supervised video encoder is usable off the shelf for driving** if the downstream objective carries enough structure. Drive-JEPA's own Table 7 makes the same point from the other side: the un-adapted V-JEPA 2 ViT/L already reaches 86.1 PDMS with a trivial decoder, ahead of DINOv2 (76.1) and SigLIP (83.4). Auto-JEPA takes that starting point and invests in the target space rather than the encoder.

What this does *not* settle is whether the JEPA objective is load-bearing on the trajectory side. Auto-JEPA reports no comparison against encoding a regressed trajectory through the same frozen encoder and retrieving with that — so the contribution could be the shared latent retrieval space rather than joint-embedding prediction per se.

## DriveLaW: Which Representation Should Condition a Planner?

[[sources/drivelaw.md]] runs the comparison this page most needed. Holding the diffusion planner fixed and varying only the conditioning representation (NAVSIM-v1 PDMS):

| Representation | Source | PDMS |
| --- | --- | ---: |
| BEV features | BEVFormer ResNet-101 backbone | 84.1 |
| VLM hidden states | Qwen2.5-VL, ReCogDrive-style | 86.5 |
| **Video-generator latents** | **DriveLaW-Video (LTX-Video 2B)** | **89.1** |

Video latents beat VLM hidden states by **+2.6** and BEV features by **+5.0**. Every other comparison of these three families in the wiki is confounded by architecture, data, and training recipe; this one holds all of them constant. The VLM row landing at exactly 86.5 — ReCogDrive-IL's published score — suggests it is a faithful reimplementation of that representation rather than a weakened strawman.

A qualitative check accompanies it: PCA projections of the three feature types show BEV and VLM features diffuse and unstable with irregular focus shifts, while video-generator features stay sharp and spatially structured under severe ego motion.

**The complementary scaling axis.** DriveLaW varies *pretraining data* at fixed model size — 0 / 76k / 3.8M / 7.6M samples give 85.9 / 87.0 / 87.8 / 89.1 PDMS, monotone and unsaturated. SimWAM varies *model size* at fixed data and finds it nearly flat (1.3B ≈ 5B). Taken together the two results are consistent and jointly informative: **for video priors, what you pretrain on matters far more than how large the model is.** Note DriveLaW uses LTX-Video, the backbone SimWAM ranked weakest of four (88.7), yet reaches 89.1 — heavy driving-domain pretraining and the chained design appear to recover more from a modest prior than backbone choice alone predicts.

## Representation x Planner: The First Complete Factorial {#factorial}

Every result above holds the planner fixed and varies the representation ([[sources/drivelaw.md]]), or holds the readout fixed and varies the representation's training objectives ([[sources/unified-driving-tokens.md]]). **Neither can see an interaction**, and this page has been treating "the representation is worth several PDMS" as a standalone fact. [[sources/coworld-vla.md]]'s Table 5 varies both and the interaction is large and negative.

Its four cells, NAVSIM-v1 PDMS, everything else held constant:

| | Plain VLM trajectory readout | + HMEF diffusion planner | Effect of the planner |
|---|---:|---:|---:|
| **Ego-trajectory token only** | 83.7 | 88.9 | **+5.2** |
| **Full four-expert representation** | 88.7 | 90.0 | **+1.3** |
| **Effect of the representation** | **+5.0** | **+1.1** | interaction **-3.9** |

**Each intervention is worth about five points alone and about one point on top of the other.** Additively one would predict 93.9; the observed joint value is 90.0.

**What this does to the page's running claim.** "The representation is worth several PDMS" is true *when measured against a weak readout*, which is how both DriveLaW and UDT measured it — DriveLaW's planner is a 133M DiT and UDT's is a 20M MLP-plus-registers head. CoWorld-VLA shows that a competent diffusion action expert recovers most of the same ground from a plain trajectory token: **88.9 PDMS with no world representation at all**, above every no-RL VLA baseline in its own Table 1 (ReCogDrive 86.5, DriveVLA-W0 87.2, LaST-VLA 87.3, SGDrive 87.4).

So the honest restatement is: **representation quality and planner capacity are substitutes over most of this range, and the wiki has been reading substitution as addition.** Neither DriveLaW's +5.0 spread nor UDT's +6.3 is wrong; both were measured in the cell where the other factor is weak, which is the cell where a representation looks most valuable.

**Three caveats before generalizing.** PDMS is bounded and compresses near the top, so some sub-additivity is mechanical — though 90.0 is well short of the 94.8 human reference, so a ceiling effect cannot account for -3.9 alone. It is one architecture, one benchmark, single runs. And the two factors here are not the same objects DriveLaW and UDT varied: CoWorld-VLA's "representation" is four auxiliary token objectives, not a change of feature family. **The cheap replication is obvious**: DriveLaW already has three representations and a planner; running its weakest and strongest against a stronger head would close this.

## SimWAM: The First Controlled Video-Prior Swap

*(Second since [[sources/drive-hwm.md]] — see [JEPA vs. Generative](#jepa-vs-generative). The two are complementary: SimWAM varies **scale and domain within video generation**; Drive-HWM varies the **objective family** across video generation and joint-embedding prediction.)*

SimWAM ([[sources/simwam.md]]) holds the planner fixed and swaps the video backbone, which its architecture permits because its two experts share no parameters and communicate only through a shared attention stream. Four priors under an identical action expert and training recipe (NAVSIM-v1 PDMS):

| Video prior | Params | PDMS | Note |
| --- | --- | ---: | --- |
| LTX-Video | lightweight | 88.7 | Weak prior costs 1.6 PDMS |
| Wan2.1-1.3B | 1.3B | 90.2 | Essentially matches the 5B model |
| Wan2.2-5B | 5B | 90.3 | The default |
| Cosmos-Predict2.5 | – | **90.4** | Pretrained on driving video; best EP and TTC |

Two conclusions the field should absorb. **Prior scale is nearly irrelevant in this regime**: 1.3B versus 5B is a 0.1 PDMS difference, so papers reporting gains from a larger video backbone should check whether the gain is really from scale. **Domain relevance beats capacity**: Cosmos-Predict2.5, pretrained on driving video, edges out a substantially larger general-purpose model. Prior *quality* still matters — LTX-Video's 88.7 shows the floor is real.

SimWAM's action expert scales just as shallowly: 0.21B → 1.02B moves PDMS 89.9 → 90.3. Both capacity axes are flat, which makes the cheap configuration (small action expert on Wan2.1-1.3B) attractive and suggests the bottleneck lies in the training signal rather than either model's size.

## DriveWAM: Video Backbone as Policy, VLM as Advisor

DriveWAM ([[sources/drivewam.md]]) is the wiki's clearest split of the two backbone roles into separate models with separate jobs. Wan2.2-TI2V-5B is fully fine-tuned and *is* the policy: it hosts both the video flow and the action flow in one shared transformer. Qwen3-VL-8B stays frozen, is queried once per 4-second chunk, and contributes only two sentences of natural-language guidance injected through cross-attention.

This is different from every frozen-backbone design already tracked here. AutoMoT freezes an understanding expert that still sits inside the action model's attention path; CLEAR uses Qwen hidden states as routing/scoring features. DriveWAM's VLM communicates in text, has no gradient path, and could be swapped for another VLM without retraining the policy — but it also costs 125 ms and 8B parameters at deployment for a purely advisory signal.

The backbone-initialization ablation is the transferable lesson (ADE@4s / FDE@4s at 100k clips): pretrained init + video supervision reaches 0.83 / 2.47; no pretrained init but with video supervision reaches 1.10 / 3.26; pretrained init *without* video supervision is worst at 1.23 / 3.79. A pretrained video prior is not a free initialization — action-only fine-tuning erases it.

## Policy World Model Show-o Use

Policy World Model ([[sources/policy-world-model.md]]) uses Show-o as the unified autoregressive backbone rather than using a VLM only for language reasoning. Its token stream contains observed image tokens, ego/navigation tokens, generated text, future frame tokens, and action tokens.

The backbone is paired with a specialized tokenizer: a frozen high-resolution first-frame branch provides context, while a trainable low-resolution branch encodes each 128x224 future frame as 28 tokens with an 8192-entry codebook. This is a backbone-design lesson rather than just a compression trick: PWM keeps future video generation cheap enough to run before action prediction, which is what makes inference-time visual anticipation feasible.

## ForeSight: A Diffusion World Model as the Whole Visual Encoder

[[sources/foresight.md]] pushes the frozen-backbone idea further than anything else tracked here. Epona (2.5B, AR + diffusion) is not an initialization, not an auxiliary supervisor, and not an advisor — it is **the planner's primary visual encoder**, run forward at inference and read at a selected denoising step. The trainable stack downstream of it is 73M (52M TransFuser current encoder + 21M action decoder + WM-QFormer), a **35:1 frozen-to-trained parameter ratio**.

Three things this configuration establishes:

**A generator can carry a planner alone.** Table 7 deletes the current encoder — no multi-view images, no LiDAR, no present-frame features at all — and the planner still scores 88.2 PDMS, above its own no-world-model baseline of 86.8. No other paper here has run that experiment.

**But a front-view generator is not a perception system.** The current encoder is worth +1.1 PDMS, concentrated in DAC (+0.9) and EP (+1.8), which is exactly the drivable-area and progress information that side views and LiDAR geometry supply. ForeSight's own stated justification is that foundation world models "primarily process front-view images," so a generated-future-only planner is laterally blind. This is a backbone-selection constraint, not a planner one: it disappears if and when multi-view generation matures.

**Swapping the generator is tolerated, not rewarded.** Table 8 substitutes Vista for Epona with the planner fixed — the same controlled-swap design as SimWAM's video-prior table — and nuScenes results get worse on 6 of 8 columns (L2 0.62 → 0.64, collision 0.18 → 0.27). Compare SimWAM's swap, where a *driving-pretrained* prior (Cosmos-Predict2.5) edged out a larger general one and the spread across four priors was 1.7 PDMS. Both Epona and Vista are driving-pretrained, so ForeSight's result is a within-family comparison and the gap is more likely about the 2 Hz finetune Epona received (and Vista did not) than about architecture.

**The finetuning caveat is worth recording separately.** Epona is finetuned from its native 5 Hz to NAVSIM's 2 Hz before being frozen, and Table 6 shows this **costs generation quality** — FVD 50.77 → 54.63 on nuPlan. A frozen backbone that must first be adapted to the target frame rate is not the plug-and-play component the framing suggests, and the adaptation is measurably lossy on the backbone's own objective.

## Wan2.2-TI2V-5B: One Backbone, Four Coupling Strategies

Four ingested papers now build on the same video backbone with the same benchmark, which is the closest thing this wiki has to a controlled comparison of *how to attach a video prior to a planner* — controlled on the prior, not on the rest of the recipe, so read the ordering as suggestive rather than causal.

| Paper | Coupling strategy | Video at inference? | Second backbone | NAVSIM-v1 PDMS | Latency |
| --- | --- | --- | --- | ---: | ---: |
| [[sources/simwam.md]] | Isolated attention mask — video is a **training-time signal only** | No | — | **91.5** | 518 ms |
| [[sources/driveva.md]] | Single DiT over joint `[video latents ‖ action tokens]` | Yes (2 ODE steps) | — | 90.9 | — |
| [[sources/drivewam.md]] | Chunked AR video → action inverse dynamics | Yes (3 video steps) | Frozen Qwen3-VL-8B advisor (text only) | 90.1 | 871–1262 ms / 4 s chunk |
| [[sources/brainwam.md]] | Dual-MoT branch compressed to 8 action tokens, bridged to a VLA branch | Yes (1–3 steps, truncated + cached) | Qwen3-VL-4B VLA branch | 89.5 | 475–644 ms (H20) |
| [[sources/adaptive-wam.md]] | Quality-routed early exit from an intermediate DiT block | **No** — one conditional forward, no rollout, no VAE decode | — | 90.8 | **170 ms (A100)** |
| [[sources/coworld-vla.md]] | Video DiT as a **differentiable critic on one VLM token**; discarded after training | **No** — the generator never runs at planning time | Qwen3-VL-2B (the policy) | 90.0 | not reported |
| [[sources/metis.md]] | MoT; **future reads action, action never reads future**; video dropped at inference | **No** | — | 89.1 (89.5 EPDMS v2) | **147 ms** (2 steps, RTX 4090) |
| [[sources/suv.md]] | MoT; **four generated streams** (RGB, seg, depth, tracks); the action reads all of them | Yes (2–10 joint steps) | — | 90.8 (91.0 EPDMS v2) | 288 ms (2 steps, RTX 4090) |

Two things fell out when this table had four rows: **the ordering was inverse to how much video computation happens at inference** — the method that generates nothing at decision time scores highest, and each additional degree of inference-time video coupling costs about a point. And **BrainWAM is the only one that pairs the video backbone with a VLM inside the model**, which is also where its Tri-MoT ablation found the fusion problem; DriveWAM keeps its VLM outside the attention path entirely and scores 0.6 higher.

**CoWorld-VLA adds a fifth coupling and it is the loosest one yet.** The Wan DiT is never a feature source for the planner at all: in Stage 2 the VLM's dynamic-evolution token *replaces the video model's text condition*, and the flow-matching loss back-propagates into that token. Then the video model is thrown away. What survives into inference is one VLM hidden state that a 5B generator's gradient shaped. On the "how many DiT forwards at decision time" ladder below it sits at **zero**, with SimWAM — and like SimWAM it lands near the top of the score column. It also inverts BrainWAM's pairing: the VLM is the policy and the video model is the teacher, rather than two branches competing inside one attention pool.

**Adaptive-WAM breaks that pattern and clarifies it.** It runs the backbone at inference but performs *one* conditional forward to an intermediate block — no denoising loop, no unconditional CFG branch, no VAE decode — and lands second on score at a third of the next-fastest latency. So the real variable is not "does the backbone run at inference" but **how many DiT forwards it costs**: SimWAM 0 (video path dropped), Adaptive-WAM ~0.5 (a prefix of one forward), BrainWAM 1–3, DriveVA 2, DriveWAM 3, and a full rollout 80. Score tracks that ordering far more weakly than latency does.

**BrainWAM's contribution to backbone practice is the asynchronous schedule.** Because its video and action rectified-flow timesteps are independent, the video expert can stop after one denoising step and cache its features for the action stream to attend to. That costs 93 ms over a no-video baseline and recovers 89.3 of an achievable 89.5 PDMS — the cheapest way in the wiki to keep a generative branch live at inference, and a strict improvement on [[sources/foresight.md]]'s 100-step schedule at 870 ms.

**Also worth noting for the VLM side**: BrainWAM's VLA branch (Qwen3-VL-4B) reaches only 86.1 PDMS alone, against 88.1 for the video branch. On NAVSIM the video prior is simply the stronger of the two backbone families, which is consistent with [[sources/drivelaw.md]]'s controlled representation comparison (video latents 89.1 > VLM hidden states 86.5 > BEV 84.1) and worth remembering before reading VLA-vs-WAM results as a fair fight between equally-tuned systems.

**Metis and SUV complete a same-recipe mask family with SimWAM.** On navtest the "less video at inference scores higher" ordering still roughly holds (SimWAM 91.5 > SUV 90.8 > Metis 89.1). **On navhard it breaks**: SUV's action-reads-future mask is +4.1 over its own isolated variant. Metis also runs the first same-recipe *prior-scale* check on navhard: Wan2.1-1.3B matches Wan2.2-5B on navtest and loses 2.4 on navhard, qualifying SimWAM's "scale barely matters". See [[concepts/wam-attention-masks.md]].

## Readout Depth: The Axis Nobody Reported

Every backbone entry on this page implicitly reads the **final** layer. [[sources/adaptive-wam.md]] is the first to ask whether that is the right choice, and the answer is no.

Six trajectory heads on Wan2.2-TI2V-5B, identical architecture / optimizer / batch size / epochs, differing only in which DiT block feeds them (NAVSIM-v1 PDMS):

| Block | 5 | 9 | **15** | 18 | 22 | 30 (final) |
|---|---:|---:|---:|---:|---:|---:|
| Imitation | 81.94 | 83.60 | **86.56** | 84.14 | 83.62 | 80.71 |
| + planner RL | 86.02 | 87.56 | **90.62** | 88.92 | 87.42 | 85.82 |

**The mid-network exit beats the final block by 4.80 PDMS after RL, and by 5.85 after imitation alone.** For comparison, the same paper measures the *video noise index* — the parameter the field has actually been ablating — at ≤0.15 PDMS across five indices of a 40-step schedule. Depth is worth roughly forty times more than noise level, and it has never been reported.

This has immediate consequences for how this page's other comparisons should be read. [[sources/drivelaw.md]]'s representation sweep (video latents 89.1 > VLM hidden states 86.5 > BEV 84.1) holds the planner fixed but reads one depth; [[sources/simwam.md]]'s four-way video-prior swap likewise. If depth is worth 4.8 within one backbone, a cross-backbone comparison at unmatched relative depth could be measuring the readout point as much as the prior.

**A second backbone agrees the default is wrong, and goes further.** [[sources/geoworldad.md]] runs the analogous study on StreamVGGT (24 decoder blocks), comparing three aggregation strategies rather than three single depths:

| Geometry layers used | Refinement iterations | NC | DAC | EP | PDMS |
|---|---:|---:|---:|---:|---:|
| 24 (all, one stage) | 1 | 98.5 | 95.7 | 81.5 | 87.6 |
| 1 (final layer) | 4 | 98.6 | 95.5 | **82.9** | 88.2 |
| **4 (layers 4 / 11 / 17 / 23)** | **4** | **98.9** | **97.2** | 82.6 | **89.3** |

The two axes buy different things. **Iterating buys progress**: EP 81.5 → 82.9 going from one interaction stage to four, with collision metrics flat. **Multi-scale buys safety**: DAC 95.5 → 97.2 and NC 98.6 → 98.9 going from one layer to four, with EP flat. And feeding all 24 layers into a *single* interaction stage is the worst of the three despite carrying the most information — attributed to insufficient optimization depth for absorbing low-level boundary detail and high-level layout at once.

So the sharper statement across both papers is not "pick the right layer" but **"consume several layers progressively"**, with Adaptive-WAM's single-best-exit result as the special case where only one readout is permitted. Two backbone families, two head types, same verdict on the field's default of reading the last layer.

**Two caveats.** It is one backbone family with one head type, so whether "≈50% depth" is a property of Wan2.2, of video DiTs generally, or of the planning task is untested. And depth ordering is not scene-wise dominance: post-RL Jaccard overlap between exits' high-quality scene sets runs 0.69–0.82, and block 30 beats block 15 by ≥50 points on 422.4 scenes even while losing on 598.6 — which is what motivates routing rather than just picking block 15.

### The Third Backbone Family Says Almost Nothing Happens {#readout-depth-vlm}

[[sources/lwdrive.md]] runs the same experiment on a **VLM** — Qwen2.5-VL-3B, 36 layers, six trajectory-refinement stages tapping layers {6, 12, 18, 24, 30, 36} against the same six stages all reading the final layer. This is structurally the closest analogue available to GeoWorldAD's comparison: multi-depth versus final-depth at matched iteration count.

| | NC | DAC | EP | TTC | PDMS |
|---|---:|---:|---:|---:|---:|
| Final layer only, 6 stages | 98.7 | 98.0 | **88.1** | 95.7 | 91.8 |
| Layers {6,…,36}, 6 stages | 98.8 | **98.4** | 87.3 | **96.2** | **92.0** |
| Δ | +0.1 | **+0.4** | **−0.8** | **+0.5** | **+0.2** |

**The sub-metric direction reproduces GeoWorldAD exactly** — multi-depth buys drivable-area compliance and time-to-collision, and gives progress back. That is now two independent confirmations that multi-scale readout is a **safety** mechanism. The magnitude does not reproduce at all:

| Paper | Backbone | Final layer's committed output | Δ PDMS |
|---|---|---|---:|
| [[sources/adaptive-wam.md]] | Wan2.2-TI2V-5B video DiT, 30 blocks | predicted noise / velocity field | **+4.80** |
| [[sources/geoworldad.md]] | StreamVGGT geometry decoder, 24 blocks | point maps, depth, camera parameters | **+1.1** |
| [[sources/lwdrive.md]] | **Qwen2.5-VL-3B, 36 layers** | **language tokens** | **+0.2** |

A hypothesis worth stating as a hypothesis: **intermediate readout is worth most where the backbone's final layer is most committed to a non-planning output.** A video DiT's last block predicts noise, which is maximally far from a trajectory; a geometry decoder's predicts point maps, which at least share a coordinate frame with one; a driving-adapted VLM's last layer predicts language, and this particular VLM was already fine-tuned to emit trajectory-style responses, so its final hidden state is close to the target by construction. None of the three papers tests this, and the architectures differ on many other axes — head capacity, adaptation regime, proposal count.

**What is safe to conclude for this page**: Adaptive-WAM's +4.80 is a video-DiT result and **should not be carried over to VLM backbones**. The many designs catalogued above that read a VLM's final hidden state are, on the one available measurement, leaving roughly 0.2 PDMS on the table rather than 5. The axis is real; its size is backbone-specific.

**The mirror-image question: where to *inject* supervision.** [[sources/reworld.md]] sweeps which Video DiT block receives an auxiliary future-prediction head, on the same LTX-Video 2B backbone [[sources/drivelaw.md]] uses:

| Supervised block (of 28) | 2 | **8** | 12 | 16 | 20 |
|---|---:|---:|---:|---:|---:|
| nuScenes FVD ↓ | 65.5 | **61.9** | 62.7 | 63.0 | 64.3 |

**Block 8 of 28 — about 29% depth — wins**, with the stated reason being a balance between "representation maturity and subsequent refinement": earlier blocks carry less developed future estimates, deeper blocks leave less hierarchical separation from the final head. The spread is only 3.6 FVD, far flatter than Adaptive-WAM's readout sweep, and it measures a different quantity — but the shape of the answer is the same one this section keeps producing: **a mid-network block beats both ends**, at ~29% for injecting supervision and ~50% for reading features. Two datapoints, one backbone family, no theory.

## A Foundation Model as an Alignment *Target*, Not an Encoder {#alignment-targets}

Every entry above uses a foundation model as an **encoder** — something that consumes pixels and emits features a downstream head reads. [[sources/reworld.md]] measures a different role: the foundation model as a **frozen alignment target**, whose features an internal layer of a *generator* is pulled toward. This is the REPA recipe, imported from image diffusion, and the results are the sharpest negative data this page has on foundation-model transfer.

Controlled protocol: all methods trained from scratch on LTX-Video, 120k steps, nuPlan + nuScenes, 224×224×25 clips, no text conditioning, batch 32. FVD on the nuScenes test set.

| Alignment target | FVD ↓ | vs. no teacher |
|---|---:|---:|
| **None — ReWorld** (self-supervision on the generator's own flow target) | **270.4** | **−33.7** |
| None — Self-Flow | 283.3 | −20.8 |
| None — SRA2 | 295.2 | −8.9 |
| DINOv2 | 295.9 | −8.2 |
| None — SRA | 296.9 | −7.2 |
| *Vanilla Flow (no alignment at all)* | *304.1* | *—* |
| DepthAnything3 | 319.4 | **+15.3** |
| VideoMAEv2 | 328.3 | **+24.2** |
| **V-JEPA 2** | **331.6** | **+27.5** |
| ReDi | 421.7 | +117.6 |

**Three of the four external teachers are worse than no teacher, and the worst is V-JEPA 2.** That is worth stating carefully, because this page and the JEPA sections below treat V-JEPA 2 as the strongest available video representation for driving — and across [[sources/drive-jepa.md]], [[sources/auto-jepa.md]], [[sources/wa-jepa.md]], and [[sources/da-wam.md]] it demonstrably is, *for planning*.

**The reconciliation is about what the representation is being asked to do.** A JEPA encoder is trained to be predictive in latent space, which means it is trained to **discard appearance detail** that is not needed for prediction. That is exactly the property that makes it a good planning encoder and a bad alignment target for a model whose job is to put pixels back. DepthAnything3 (geometry, appearance discarded) and VideoMAEv2 (recognition-oriented) fail for the same reason; DINOv2, the only one that beats vanilla flow, is the one that retains the most dense appearance structure.

**The transferable rule**: *encoder quality for a discriminative downstream task does not predict target quality for a generative alignment objective, and can anticorrelate with it.* ReWorld's own framing is the same — external encoders "may supply semantic priors that are poorly matched to multi-second ego and agent dynamics," while its own target is the generator's native future-flow velocity, which is temporally structured by construction and free to compute.

**Two caveats on scope.** The protocol is 224×224×25 without text conditioning, far from the deployed 1280×704 setting, and each teacher gets one configuration with no per-teacher tuning of the alignment layer or weight — REPA's own results are known to be layer-sensitive. Treat the ordering as indicative and the sign of the V-JEPA 2 result as the finding.

**A second paper picks the same winner from the same family.** [[sources/unified-driving-tokens.md]] uses frozen **DINOv3-B** as both an encoder input and a decoding target for a discrete driving tokenizer, and that supervision is the largest single component of its ablation — **+3.9 PDMS** on a fixed 20M planning readout (85.5 → 89.4), moving every sub-metric. Different task, different mechanism, same conclusion as ReWorld's Table IV: **of the frozen encoders tried as supervision targets in driving, the DINO family is the one that transfers.** Neither paper explains why, and the two obvious candidates — dense appearance retention versus patch-level object structure — are not separated by either.

**A cost nobody prices.** UDT's tokenizer needs a DINOv3-B forward pass **at inference** to tokenize a frame, and reports no latency, throughput, or parameter count for it. A frozen foundation model in the *input* path is a deployment cost in a way that one used only as a training-time target is not, and this page has no numbers for it from any source.

### V-JEPA as an alignment target *works* — when the aligned model is discriminative {#target-role}

ReWorld's table is the strongest negative result on this page, and [[sources/coworld-vla.md]] is the case that bounds it. It aligns a **VLM hidden state** to pooled frozen **V-JEPA** features of the *future* frame (SmoothL1 + cosine) and that expert is worth **+1.5 PDMS** added to a bare trajectory token, +2.5 when a geometric expert is already present. Same family of mechanism as REPA — frozen teacher, feature-space alignment loss on an internal representation — and the opposite sign.

| | ReWorld | CoWorld-VLA |
|---|---|---|
| What is aligned | An internal layer of a **video generator** | An expert token inside a **VLM policy** |
| What the aligned model must do next | Put pixels back | Condition a trajectory |
| Target | V-JEPA 2, current clip | V-JEPA, **future** frame, **pooled** |
| Result | **331.6 FVD vs. 304.1 with no teacher** | **+1.5 PDMS** |

**The reconciliation is the rule this page already stated, now with the confirming case attached**: a JEPA encoder is trained to discard appearance detail that prediction does not need, which makes it a poor target for a model whose output is pixels and a good one for a model whose output is a plan. ReWorld supplies the failure, CoWorld-VLA the success, and the distinguishing variable is **what the aligned model produces**, not which teacher is used.

Two secondary observations from the same paper:

- **VGGT is a second geometry-as-target data point** (+1.4 PDMS alone, MSE to pooled future features), alongside DepthAnything3 failing as a *generative* target in ReWorld's table. Same split.
- **Pooling is the lever nobody else pulled.** Both of CoWorld-VLA's regression targets are pooled to a handful of tokens before the loss, which is what makes deterministic regression viable on them at all — see [the entropy argument](../concepts/world-model-for-ad.md#entropy-assignment). ReWorld aligns to dense features.

**And the cost is doubled, not avoided.** CoWorld-VLA discards its 5B video model before inference but runs **frozen V-JEPA *and* frozen VGGT on every frame** to build the planner's scene stream. Two foundation encoders in the input path, no latency reported — the same unpriced deployment cost flagged for UDT above, twice over.

**A third aligned model: a trajectory tokenizer** *(2026-09-30)*. [[sources/walt.md]] aligns the latent of a trajectory autoencoder to a frozen **driving world model** (EponaV2) with a contrastive loss, then generates that latent with the world model's own planning head.

- It fits the rule above. The aligned model produces a plan, the target is pooled (the token-pair score reduces to a dot product of mean vectors), and the sign is positive.
- The size is small: **+0.35 PDMS** over the same tokenizer without alignment, against +1.5 for CoWorld-VLA's pooled V-JEPA target.
- It runs a control the other two lack. Aligning the planner to a **trajectory-only** encoder is worth +0.07, so the gain comes from scene information in the target and not from alignment as a regularizer.
- The target is the planner's own frozen backbone, so there is no second encoder in the input path and no added inference cost, unlike UDT and CoWorld-VLA.

## Frozen Is Not Good Enough: The Adaptation Ladder

Adaptive-WAM also runs the cleanest available test of *how* a video prior should be attached, with everything else held fixed (NAVSIM-v1 PDMS):

| Wan training | Single trajectory | Fixed B22, 64 prop. |
|---|---:|---:|
| **Frozen** | **84.20** | 89.91 |
| Separate LoRA, then cache features | 84.95 | 90.80 |
| **Joint LoRA** | **90.62** | **92.59** |
| Full fine-tuning | 90.64 | 92.54 |

**A frozen backbone loses 6.42 PDMS to a jointly LoRA-adapted one**, and adapting the backbone *separately* before caching recovers only 0.75 of that. The video prior has to be trained against the action objective; using it as an off-the-shelf encoder leaves a large amount on the table.

Two wiki designs sit on the losing side. [[sources/foresight.md]] freezes Epona completely and makes it the planner's primary encoder; [[sources/drivelaw.md]] caches Video-DiT features for its planner (though it also updates both modules in stage 3, an inconsistency its own page flags). Neither architecture is tested here, so this is a strong prior rather than a refutation — but it is the most direct measurement of the question the wiki has.

**Third data point on LoRA vs. full fine-tuning**: full FT adds **0.02**, so LoRA is used. That agrees with [[sources/da-wam.md]] (LoRA beats full FT by 0.36 for JEPA latent adaptation) against [[sources/latent-wam.md]] (LoRA collapsed geometric distillation, 89.3 → 68.5 EPDMS). The reconciliation this page already records — LoRA is safe when the pretrained representation is close to the target and fails when a large representational move is required — survives: keeping a video DiT predicting video-like features is a small move.

**And against static encoders**: Wan intermediate features beat ViT-Large by 1.74 and ViT-Small by 6.71 in the single-trajectory setting, but the gap shrinks to 0.28 with 64 proposals. **Multi-proposal scoring masks representation quality**, which is a caution for reading any selection-based leaderboard as evidence about encoders.

## The VLM Side of the Adaptation Ladder {#vlm-adaptation-ladder}

Everything in the section above is about a *video* prior. [[sources/qwen-drive-1.0.md]] runs the same experiment on a *vision-language* prior, with a 3D perception head as the instrument, and gets the same answer.

Its encoder, **SigLIP-Qwen**, is the SigLIP-style vision tower initialized from Qwen3.5-4B's own weights. Because the paper reproduces three dedicated 3D detectors twice — once on ResNet-50, once on SigLIP-Qwen, same schedule, same $896\times512$ input, same relabelled nuScenes — it supplies the cleanest **encoder swap** measurement in the wiki:

| Detector | ResNet-50 | SigLIP-Qwen | Δ mAP |
|---|---:|---:|---:|
| BEVFormerV2 | 33.04 | 40.78 | **+7.74** |
| PETR | 29.77 | 37.61 | **+7.84** |
| PETRv2 | 25.98 | 36.10 | **+10.12** |
| BEVFormerV2* (multi-task) | 35.34 | 41.94 | **+6.60** |

**Vision-language pretraining is worth 6.6–10.1 mAP as a 3D-detection initialization**, with the architecture, schedule and labels held fixed. No other paper here runs this comparison at all; the backbone-choice discussion has been conducted almost entirely on planning scores.

### But the features do not contain 3D structure by themselves {#vlm-3d-probe}

The same table then prices freezing:

| Configuration | Encoder / VLM | nuScenes mAP | RayIoU |
|---|---|---:|---:|
| BEVFormerV2* trained for detection | SigLIP-Qwen, updated | 41.94 | **43.89** |
| Converged BEV head on frozen features | **frozen** encoder + frozen VLM | 35.60 | 36.98 |
| Qwen-Drive Stage 2 | encoder **and** VLM updated by perception loss | **43.95** | 37.02 |

> "This contrast indicates that vision-language-pretrained features support visual-text alignment but do not directly expose the 3D structure required for driving perception."

**−6.34 mAP for freezing; +10.46 for unfreezing.** The paper is careful to exclude the obvious alternative explanation — "the gains from Stage 2 cannot be explained by continued head optimization alone, since the head-only model had already converged."

**Three backbone families now agree.** [[sources/adaptive-wam.md]] on a video DiT: frozen 84.20 → joint LoRA 90.62 PDMS. [[sources/latent-wam.md]] on geometric distillation into DINOv2: LoRA collapses (89.3 → 68.5 EPDMS), full updates required. [[sources/qwen-drive-1.0.md]] on a VLM with a perception probe: −6.34 mAP for freezing. **Using a foundation model off the shelf costs several points in every family measured, and the deficit is recovered by adapting it against the downstream objective, not by adapting it separately.**

Two details that make the Qwen-Drive instance unusual:

- **The gradient path is doubled.** The head reads pre-VLM encoder features *and* post-VLM image-token features, so perception loss reaches the vision encoder both directly and through the entire language model. The "unfrozen" condition therefore means the language model is being shaped by detection and occupancy losses.
- **Adaptation is asymmetric by design**: the newly initialized head trains at **20× the VLM's learning rate.** None of the other adaptation-ladder papers report a differential rate.

### Qwen3.5-4B as a backbone

The paper is also the wiki's first entry to treat the *base model itself* as a reported baseline rather than an unmeasured starting point. Qwen3.5-4B unadapted scores **63.52 on a six-benchmark driving-QA average — above every driving- and embodiment-specialized model Qwen-Drive evaluates**, including Cosmos-Reason2-32B (46.62), MiMo-Embodied-7B (60.32), UniDriveVLA-8B (46.29) and Alpamayo-1.5-10B (33.33).

That bears directly on the **Physical-AI-backbone** claim this page records from [[sources/alpamayo-r1.md]]: Cosmos-Reason was adopted because it beat Qwen2.5-VL-7B by 6.4 points on zero-shot LingoQA. Two model generations later, under a third party's common protocol, the plain Qwen model leads the Cosmos family on driving QA by 14–28 points and on general capability by 6–20. **Domain-specific pretraining bought a real advantage over a 2024 general backbone; it has not obviously kept it against a 2026 one.** Caveats: different protocol, different judge, and the judge shares a family with the winner — see [[concepts/general-capability-retention.md#judge-confound]].

## Coordinate Frame Beats the Foundation Model

Everything above this line treats a foundation backbone as a black box whose value is set by its pretraining. [[sources/geoworldad.md]] measures a variable none of them vary: **what coordinate system the backbone's output lives in**, holding the model, the data, and the planner fixed.

StreamVGGT reconstructs in the anchor frame of the first video frame. Trajectories live in the *moving* ego frame, so misalignment grows across the clip. **EgoStreamVGGT** changes only the parameterization — each point map expressed in the ego-camera frame of its own timestep, camera poses as adjacent-frame relative transforms. No added capacity, no architectural change.

| Pretrained model | Aux. sup. | NC | DAC | TTC | EP | PDMS |
|---|---|---:|---:|---:|---:|---:|
| Scratch | – | 98.1 | 94.6 | 93.9 | 76.0 | 84.2 |
| StreamVGGT | 4D recon | 97.9 | 93.4 | 92.8 | 80.2 | **84.8** |
| EgoStreamVGGT | – | 98.4 | 95.1 | 95.0 | 81.7 | **87.3** |
| EgoStreamVGGT | 4D recon | 98.9 | 97.2 | 95.7 | 82.6 | **89.3** |

**Row 2 is the result this section exists for.** A pretrained streaming 4D geometry foundation model, with its reconstruction objective retained, is worth **0.6 PDMS over training from scratch** — and it *lowers* NC (98.1 → 97.9), DAC (94.6 → 93.4), and TTC (93.9 → 92.8), buying only ego progress. In the wrong frame, a geometry foundation model is close to a wash.

The re-parameterization alone recovers **+2.5**, with gains on every metric. Adding joint 4D reconstruction supervision during planner training adds **+2.0** more.

**Two things this generalizes to.** It is the first measurement of the argument [[sources/geowam.md]] makes rhetorically — that geometry's advantage over pixels is living in the same coordinate frame as the action — and it says the advantage is *conditional on actually doing the alignment*, not automatic from choosing a geometric target. And it belongs beside [[sources/adaptive-wam.md]]'s adaptation ladder (frozen Wan 84.20 → joint LoRA 90.62, cached separately-tuned features 84.95): both papers find that using a foundation model off the shelf costs several PDMS, that the fix is cheap, and that *how* the prior is attached matters more than which prior it is.

**A caveat on the geometry-quality tables.** GeoWorldAD's depth and pose comparisons (StreamVGGT vs. EgoStreamVGGT) show large improvements — nuScenes AbsRel 0.265 → 0.117, KITTI δ<1.25 72.2 → 95.5 — but EgoStreamVGGT is both re-parameterized *and* fine-tuned on four driving datasets while StreamVGGT is off the shelf, so those tables conflate alignment with domain adaptation. Table 4 above is the clean instrument. Note also that nuScenes **rotational** RPE regresses 0.47 → 1.31 under the change most likely to affect it, and the paper's prose excludes rotation by careful wording.

