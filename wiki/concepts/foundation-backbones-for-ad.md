---
title: Foundation Backbones for AD
type: concept
sources: [raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md]
related: [concepts/vlm-domain-adaptation.md, concepts/world-model-for-ad.md, concepts/dual-system-vla.md, sources/autovla.md, sources/nord.md, sources/elf-vla.md, sources/spanvla.md, sources/driveva.md, sources/alpamayo-r1.md, sources/explorevla.md, sources/onedrive.md, sources/onevl.md, sources/latent-wam.md, sources/drive-jepa.md, sources/policy-world-model.md]
created: 2026-05-01
updated: 2026-05-01
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
| Geometric teacher | WorldMirror / VGGT | Supplies training-time spatial features for Latent-WAM; removed at inference. |
| Frozen understanding expert | AutoMoT-style UE | Preserves general reasoning and avoids catastrophic forgetting. |
| Shared attention backbone | OneDrive | Reuses VLM causal attention for image, perception, planning, and text tokens while replacing task FFNs. |
| Latent reasoning backbone | OneVL | Fine-tunes Qwen3-VL-4B so visual/language latent tokens can be decoded into future frames and text explanations during training. |

## Takeaways

- Bigger or newer backbones do not make benchmark comparisons fair by themselves; input cameras, training data, RL stage, and action head matter.
- Frozen-backbone designs can outperform fine-tuned VLMs when the action expert is well-coupled.
- Teacher-only use should be distinguished from inference-time use because it changes deployment cost and risk.
- OneDrive shows that even inside one VLM decoder, not all pretrained modules transfer equally: attention transfers to structured driving queries, while language FFNs may need task-specific replacement.
- OneVL shows a Qwen3-VL backbone can host latent reasoning tokens, but stable adaptation requires staged auxiliary-decoder training; direct joint fine-tuning collapses.
- Latent-WAM shows that DINOv2-Base can be turned into a compact planning encoder through geometric distillation, but LoRA is not sufficient for that distillation target.

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

## Policy World Model Show-o Use

Policy World Model ([[sources/policy-world-model.md]]) uses Show-o as the unified autoregressive backbone rather than using a VLM only for language reasoning. Its token stream contains observed image tokens, ego/navigation tokens, generated text, future frame tokens, and action tokens.

The backbone is paired with a specialized tokenizer: a frozen high-resolution first-frame branch provides context, while a trainable low-resolution branch encodes each 128x224 future frame as 28 tokens with an 8192-entry codebook. This is a backbone-design lesson rather than just a compression trick: PWM keeps future video generation cheap enough to run before action prediction, which is what makes inference-time visual anticipation feasible.
