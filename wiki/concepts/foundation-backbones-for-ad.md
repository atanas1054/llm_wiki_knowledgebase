---
title: Foundation Backbones for AD
type: concept
sources: [raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md, raw/papers/CLEAR_ Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving.md, raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md, raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md, raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md]
related: [sources/simwam.md, sources/sgdrive.md, concepts/vlm-domain-adaptation.md, concepts/world-model-for-ad.md, concepts/dual-system-vla.md, concepts/adaptive-routing.md, concepts/r1-zero-like-training.md, sources/autovla.md, sources/nord.md, sources/elf-vla.md, sources/spanvla.md, sources/driveva.md, sources/alpamayo-r1.md, sources/explorevla.md, sources/onedrive.md, sources/onevl.md, sources/latent-wam.md, sources/drive-jepa.md, sources/policy-world-model.md, sources/clear.md, sources/understanding-r1-zero-like-training.md, sources/drivewam.md]
created: 2026-05-01
updated: 2026-08-17
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
| Hidden-state semantic router | Qwen 3.5 0.8B in CLEAR | Uses LLM hidden states for scheduling and trajectory scoring rather than text/action generation. |
| Base-model prior confound | Qwen2.5-Math in R1-Zero-like training | No-template QA behavior shows that apparent RL gains can depend heavily on pretraining and template choice. |
| Geometric teacher | WorldMirror / VGGT | Supplies training-time spatial features for Latent-WAM; removed at inference. |
| Frozen understanding expert | AutoMoT-style UE | Preserves general reasoning and avoids catastrophic forgetting. |
| Shared attention backbone | OneDrive | Reuses VLM causal attention for image, perception, planning, and text tokens while replacing task FFNs. |
| Latent reasoning backbone | OneVL | Fine-tunes Qwen3-VL-4B so visual/language latent tokens can be decoded into future frames and text explanations during training. |
| Video backbone **as** the policy core | Wan2.2-TI2V-5B in DriveVA and DriveWAM | The video DiT is fine-tuned into the action path itself, not attached as a generation branch. |
| Swappable video prior | LTX-Video / Wan2.1-1.3B / Wan2.2-5B / Cosmos-Predict2.5 in SimWAM | Co-trained through shared attention only, so the backbone can be replaced without touching the action expert. |
| Frozen advisory VLM | Qwen3-VL-8B in DriveWAM | Emits chunk-level text guidance consumed by cross-attention; never decodes actions and is never fine-tuned. |
| VLM as frozen world model | InternVL3-2B in SGDrive | Fine-tuned in stage 1 to host structured ⟨world⟩ queries, then frozen in stage 2 while only the DiT planner trains. |

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

## SimWAM: The Only Controlled Video-Prior Swap

SimWAM ([[sources/simwam.md]]) is the wiki's only paper that holds the planner fixed and swaps the video backbone, because its two experts share no parameters and communicate only through a shared attention stream. Four priors under an identical action expert and training recipe (NAVSIM-v1 PDMS):

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
