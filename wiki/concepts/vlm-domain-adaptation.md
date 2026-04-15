---
title: VLM Domain Adaptation for Autonomous Driving
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md, raw/papers/AutoDrive-R²_ Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md, raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md]
related: [sources/recogdrive.md, sources/uniugp.md, sources/senna2.md, sources/reasoning-vla.md, sources/hermes.md, sources/autovla.md, sources/automot.md, sources/autodrive-r2.md, sources/alpamayo-r1.md, sources/adathinkdrive.md, sources/futuresightdrive.md, sources/unidrivevla.md, sources/flare.md, sources/vega.md, sources/nord.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/world-model-for-ad.md, concepts/dual-system-vla.md, concepts/perception-for-planning.md]
created: 2026-04-05
updated: 2026-04-15
confidence: high
---

## The Domain Gap Problem

Vision-Language Models pre-trained on internet image-text data (e.g., InternVL3, Qwen2.5-VL) lack:
- Coverage of driving-specific scenarios and terminology
- Understanding of ego-vehicle dynamics, road topology, traffic rules
- Spatial precision needed for trajectory prediction (language tokens are imprecise for floating-point coordinates)

Directly applying such VLMs to driving yields suboptimal planning performance (e.g., InternVL3 baseline on NAVSIM: 83.3 PDMS).

## Two VLM Architectural Patterns in AD

### Single-System
VLM directly predicts trajectories and/or reasons about driving. Examples: EMMA, GPT-Driver, OpenEMMA, ReCogDrive.
- Pro: end-to-end optimization, joint reasoning + planning
- Con: VLM must handle both high-level reasoning and low-level action precision

### Dual-System
VLM generates high-level commands/low-frequency trajectories; a separate end-to-end module handles refinement. Examples: DriveVLM, Senna.
- Pro: separation of concerns; VLM focuses on scene understanding
- Con: two-module pipeline, harder to jointly optimize

## ReCogDrive's Adaptation Approach

### Data Construction (3.1M QA Pairs)
12 open-source datasets aggregated:
- Perception: Talk2Car, CODA-LM, NuScenes-QA, MAPLM, NuInstruct
- Prediction: motion forecasting QA
- Planning: DriveLM, LingoQA, OmniDrive, Senna-style QA, DriveLM
- General driving: DriveGPT4, DRAMA, SUTD

Plus 775K auto-annotated NAVSIM QA pairs covering: scene description, key object ID, driving behavior narration, road marking recognition, traffic light classification, vulnerable road user detection, motion prediction, planning command prediction, counterfactual reasoning.

### Data Quality Pipeline
1. **Normalization**: unify all bounding-box formats to InternVL3 standard (`<car><FRONT_VIEW><box>[x1,y1,x2,y2]</box>`, coordinates scaled to [0,1000])
2. **Augmentation**: LLM-paraphrased question templates + Qwen2.5-VL answer rewriting for linguistic diversity
3. **Filtering**: Qwen2.5-VL quality scoring; remove pairs scoring < 60 → retains 2.3M of 3.2M

### Scaling Law Finding
| Data | PDMS |
|------|------|
| 85K LQ | 83.3 |
| 1.5M LQ | 84.6 |
| 3.2M LQ | 85.3 |
| 3.1M HQ | 86.2 |

**Conclusion**: scaling data helps, but quality filtering (+0.9 PDMS) outweighs a 100K data increase. Quality > quantity.

## Post-Adaptation Capabilities

After stage-1 fine-tuning, the VLM produces:
- $T_{traj}$: textual trajectory
- $T_{reason}$: explicit reasoning trace (interpretable driving decisions)
- $T_{desc}$: scene description

These are consumed by the diffusion planner — the VLM provides semantic grounding while the planner handles continuous action generation. See [[concepts/diffusion-planner.md]].

## UniUGP's Adaptation Approach

**UniUGP** ([[sources/uniugp.md]]) adapts Qwen2.5-VL-3B through a four-stage curriculum with a distinct emphasis on **CoT reasoning** and **long-tail scenarios**:

### Multi-stage training (by expert, not by data only)
Unlike ReCogDrive (data-focused SFT then RL), UniUGP interleaves expert training:
- Stage 1: Understanding Expert only — broad scenario understanding (10 datasets)
- Stage 2: Generation + Planning Experts only — visual dynamics from large-scale video datasets
- Stage 3: Understanding Expert only — causal CoT reasoning (custom annotated dataset)
- Stage 4: All three jointly — multi-capability fusion (mixed 0.1:0.4:0.5)

The staged approach prevents interference between modalities during initial learning; fusion in Stage 4 resolves misalignments.

### CoT Integration
UniUGP integrates **4-component chain-of-thought** into the understanding expert:
1. **Scene description** (time, weather, traffic condition)
2. **Object detection** (traffic signals, dynamic agents, rare objects)
3. **Intention inference** (how objects affect ego vehicle's future)
4. **Action decision** (reasoning grounded in predicted future trajectories)

CoT is constructed by prompting a frontier VLM with the known future trajectory and scene context, then manually calibrating. This "trajectory-grounded CoT" approach ensures reasoning is physically consistent with the planned path.

**Effect of CoT on performance** (ablation): removing CoT degrades accident prediction (95.8% → 93.2%) and planning (L2: 1.45 → 1.58).

### Instruction Following via Data
Instruction-following capability is acquired purely through data: each training trajectory is assigned a natural-language navigation command (Turn Left / Go Straight / Turn Right). No architectural change needed — the VLM learns to condition its trajectory generation on these instructions. This is the only current unified AD system with demonstrated instruction-following trajectory modification.

### Long-tail Data as an Adaptation Priority
UniUGP curates anomaly/accident datasets (DADA-2000, Lost and Found, StreetHazards) into structured QA. The key insight: long-tail capability cannot emerge from standard urban-driving datasets alone — rare events at <0.003% frequency require explicit curation.

## Senna-2: Consistency-Oriented Adaptation

**Senna-2** ([[sources/senna2.md]]) takes a different angle on VLM adaptation: instead of adapting the VLM to produce better features for a planner, it explicitly enforces **consistency** between VLM decisions and E2E planned trajectories.

**Kinematic mapping $f_K$**: converts a continuous trajectory into a discrete meta-action category (4 speed × 5 direction = 20 classes). This is the key tool for measuring and enforcing consistency across all three training stages.

**Selective open-loop alignment (Stage 2)**: training signal is conditional on consistency:
$$\mathcal{L}_{stage2} = (1 - \mathcal{C}(\tau, d))(\mathcal{L}_{E2E} + \gamma \mathcal{L}_{VLM})$$
- Consistent samples → zero loss (self-reinforcing, no external correction needed)
- Inconsistent samples → full supervised correction from GT trajectories + decision labels

This is a novel training philosophy: *dynamic, consistency-conditioned supervision* rather than uniform supervision across all samples. The model is never penalized for being internally consistent, even if the consistent behavior deviates from GT — promoting autonomous coherence over pure imitation.

**Decision adapter**: bridges VLM and E2E planner with two token types: VLM tokens (from hidden states, semantic) + decision tokens (learnable category embeddings, categorical). Combined contribution: +27.4% collision rate reduction vs. pure E2E baseline.

See [[concepts/dual-system-vla.md]] for the full architectural pattern.

## Reasoning-VLA: Multi-Dataset Generalization via Unified Corpus

**Reasoning-VLA** ([[sources/reasoning-vla.md]]) takes the most aggressive generalization approach of any paper in this wiki: training on **8 heterogeneous datasets simultaneously** rather than adapting to a single domain.

### Unified 8-Dataset Corpus (75,000+ clips)

| Dataset | Region | Proportion |
|---------|--------|-----------|
| NAVSIM | US (Las Vegas) | ~39% |
| nuScenes | Singapore + Boston | ~19% |
| Waymo | US (multiple cities) | ~15% |
| ONCE + Mapillary | China + Global | ~11% each |
| Argoverse-V2 | US (Pittsburgh, Miami) | ~3% |
| KITTI | Germany | ~2% |
| IDD | India | ~1% |

**Challenge**: heterogeneous sensor configs, annotation formats, coordinate conventions, and vehicle platforms across datasets. The standardization pipeline converts all into a unified format before CoT generation.

### CoT Dataset Construction Pipeline

1. **Format normalization**: all datasets converted to standardized image-text format with ego status history (position, velocity, acceleration at 0.5s intervals)
2. **CoT generation**: a strong reasoning VLM (Qwen2.5-VL) generates `<think>...</think>` reasoning blocks for each clip, covering: scene description → physics extrapolation → intent inference → trajectory decision
3. **Rule-based verification**: temporal alignment, format checking, coordinate alignment, time-stamp alignment, scene/object checking
4. **Human verification**: annotators check logits, projections, labels, and images — final quality gate

**Comparison to ReCogDrive's data pipeline**:
- ReCogDrive: 3.1M high-quality QA pairs from 12 datasets, explicit quality scoring (Qwen2.5-VL), normalized bounding-box format
- Reasoning-VLA: 75K rich multi-step CoT clips from 8 datasets with full ego-status history — fewer examples but richer per-clip reasoning content

### Generalization Demonstrated

Zero-shot evaluation (train on NAVSIM/Waymo/KITTI/ONCE, test on unseen nuScenes/Argoverse/Mapillary/IDD):
- nuScenes zero-shot avg L2: 0.28 (vs. 0.23 in-distribution) — only 22% degradation
- Unified training outperforms nuScenes-only fine-tuning on closed-loop NeuroNCAP

**Key finding**: generalist models trained on the unified dataset outperform specialist models fine-tuned only on the target dataset in closed-loop evaluation — generalization generalizes.

## HERMES: Offline VLM Annotation as Safety Distillation

**HERMES** ([[sources/hermes.md]]) introduces a qualitatively different pattern: the VLM is not adapted or fine-tuned at all — it is used as a **structured offline annotator** that generates safety-critical knowledge, which is then distilled into a lightweight student planner via text embeddings.

### Teacher-Student Distillation Pattern

| Stage | System | Role | Timing |
|-------|--------|------|--------|
| Annotation | Qwen3-VL-Flash (cloud) | Generates structured safety annotations per frame | **Offline, once** |
| Encoding | BGE-M3 | Encodes annotations to fixed vectors | Offline |
| Planning | Tri-Modal Driving Module (ViT + transformer) | Consumes embeddings at training and inference | **Online** |

The cloud VLM never runs at inference time. This makes deployment practical (48ms-class latency for a ViT backbone) while still benefiting from billion-parameter reasoning.

### Structured 5-Field Annotation Protocol

The VLM is prompted to produce exactly five fields, organized into two semantic groups:

**Long-Tail Scene Context** (one field):
- Per-viewpoint hazard descriptions (front/rear/side-L/side-R)
- Explicitly targets: occlusions, abnormal agent behaviors, rare objects, adverse weather, compound edge cases

**Long-Tail Planning Context** (four fields):
- **Risk level**: Low / Medium / High — based on VRU presence, collision likelihood, reaction time
- **Intention**: Go straight / Turn left / Turn right / Stop
- **High-level plan**: concise directive grounded in observable scene elements
- **Plan rationale**: causal explanation (no unobserved inference allowed)

Strict 5-field output format ensures consistent downstream integration.

### Risk-Controlled Injection (α-scaled residual)

Rather than hard-conditioning the planner on VLM instructions, HERMES uses a **scaled residual** in the Risk Planning Cross-Attention module:

$$\text{plan\_context\_ctrl} = \text{LN}(\text{plan\_context\_intent} + \alpha \cdot c), \quad \alpha = 0.3$$

This limits VLM influence in common driving conditions while allowing semantic modulation in long-tail cases. The α parameter is fixed — a limitation, as the explicitly annotated risk level (low/medium/high) is not used to dynamically scale α.

The offline-only pattern sacrifices adaptability to truly novel test scenarios (the VLM cannot respond to new hazards it never annotated) in exchange for deployment efficiency and decoupled scaling. For the full cross-wiki strategy comparison, see the table in the AdaThinkDrive section below.

### Long-Tail as an Adaptation Axis

HERMES establishes a distinct axis in the VLM-for-AD design space: rather than improving general driving intelligence, it targets **rare and safety-critical scenario types** as a first-class objective. The WOD-E2E dataset is curated specifically for long-tail mixed-traffic (VRUs, cut-ins, construction zones, FOD). Ablation confirms that removing VLM semantic embeddings causes −0.60 RFS (the safety-aligned metric), but removing historical motion state causes −1.30 RFS — geometry still dominates semantics in terms of raw trajectory accuracy.

## AutoVLA: Dual-Mode SFT + Adaptive Reasoning via RFT

**AutoVLA** ([[sources/autovla.md]]) introduces two distinct adaptation contributions: a **dual-mode SFT** strategy and an **adaptive reasoning via RFT** approach.

### Dual-Mode SFT

AutoVLA trains on a mixture of fast (action-only) and slow (CoT) samples in the same SFT pass:

- **Fast samples**: short fixed prefix ("reasoning not needed") + action tokens — teaches direct trajectory generation
- **Slow samples**: 4-stage CoT (scene description → critical object ID → intention reasoning → decision + meta-action) + action tokens — teaches deliberate reasoning

Per-sample weighting: λ_cot=40 for slow samples (upweights CoT to compensate for fewer examples), λ_a=1 for action loss. Both loss components ($\mathcal{L}_{LM}$ + $\mathcal{L}_{action}$) are applied.

**Data scaling finding**: CoT underperforms action-only at < 50k training samples; it surpasses action-only only at larger data regimes. nuScenes (simple scenarios): action-only dominates throughout — domain difficulty determines CoT value.

### CoT Data Collection Pipeline

Qwen2.5-VL-72B annotates 4-stage CoT with **GT meta-action as hint** in the prompt, ensuring causal explanations link decisions directly to observable scene elements. 88.8% human quality score (3,000 sampled). Total: 45.6k nuPlan + 7.2k Waymo E2E + reformatted DriveLM.

**Key difference from other CoT pipelines**:
| Approach | Annotation model | Scale | Quality mechanism |
|----------|-----------------|-------|------------------|
| ReCogDrive | Qwen2.5-VL (various) | 3.1M QA pairs | Explicit quality scoring + filtering |
| Reasoning-VLA | Qwen2.5-VL | 75K CoT clips | Rule-based + human verification |
| UniUGP | Frontier VLM | Custom curated | Manual calibration |
| **AutoVLA** | **Qwen2.5-VL-72B** | **52.8k CoT** | **GT meta-action hint + 88.8% human check** |

### Adaptive Reasoning (learned via RFT)

The adaptive switching between fast/slow thinking is **not hardcoded** at inference — it is *learned* via the CoT length penalty in GRPO. The SFT stage provides the dual-mode capability; the RFT stage learns when each mode is appropriate based on reward signals. This is qualitatively different from DriveVLM's hard fast/slow separation (two separate modules). See [[concepts/rl-for-ad.md]] for full RFT details.

## AutoDrive-R²: Self-Reflection CoT and Minimal-Data SFT

**AutoDrive-R²** ([[sources/autodrive-r2.md]]) introduces the **nuScenesR²-6K** dataset — the first AD CoT dataset explicitly modeling a self-reflection step for trajectory validation — and shows that 6K curated samples with structured reasoning outperforms methods trained on 103K+ unstructured data.

### nuScenesR²-6K Dataset

6,000 image-trajectory pairs from nuScenes, annotated by Qwen2.5-VL-72B with GT trajectory as hint. Produces structured 4-step CoT in `<think>...</think><answer>...</answer>` format.

**Four-step chain** (Visualization → Calculation → Logic → Reflection):

| Step | Content | Key outputs |
|------|---------|-------------|
| 1. Visual Analysis | Obstacle/lane detection, traffic signals | Scene state (locations, signal status) |
| 2. Physics Calculation | Kinematic extrapolation: $x(t+1)=x_n + v_x\Delta t + \frac{1}{2}a_{avg}\Delta t^2$ | Predicted positions at each future step |
| 3. Logic Synthesis | Traffic rules, intersection handling, safety checks | Recommended action (stop/continue/slow) |
| 4. **Self-Reflection** | **Backward-check**: can predicted trajectory be achieved given vehicle's acceleration history? Flag contradictions, correct if needed | Validated or corrected trajectory |

The self-reflection step is structurally analogous to backward-checking in mathematical proof verification — the model re-examines its own prediction for physical consistency rather than treating the chain as strictly forward.

**"Aha Moment" (emergent)**: the model spontaneously self-corrects during reasoning in the reflection step, catching physics violations (e.g., implied impossible acceleration) and revising the trajectory. This behavior is not explicitly trained but emerges from the structured prompt design.

**Ablation**: both structured components are necessary:
- w/o 4-step structure: 0.25m avg L2 (+31.5% vs. full 0.19m)
- w/o self-reflection: 0.23m (+21.1%)

### Minimal-Data SFT Scaling

AutoDrive-R² achieves 0.19m L2 on nuScenes with only 6K SFT samples vs. EMMA+'s ~103K internal dataset (which achieves 0.29m). This confirms that **CoT structure quality outweighs data volume** for trajectory planning:
- Qwen2.5-VL-7B baseline (no SFT): 1.45m — VLMs without AD SFT perform poorly
- After nuScenesR²-6K SFT: 0.27m — 81.4% improvement from 6K structured samples
- After SFT + GRPO RL: 0.19m — further 29.6% improvement

**Comparison with other CoT approaches** in the wiki:

| Approach | CoT steps | Self-reflection? | Annotation model | Scale |
|----------|-----------|-----------------|-----------------|-------|
| ReCogDrive | Unstructured QA | ✗ | Various VLMs | 3.1M pairs |
| Reasoning-VLA | 4 (scene→physics→intent→trajectory) | ✗ | Qwen2.5-VL | 75K clips |
| UniUGP | 4 (scene→objects→intent→action) | ✗ | Frontier VLM | Custom curated |
| AutoVLA | 4 (scene→objects→intent→action) | ✗ | Qwen2.5-VL-72B | 52.8K |
| **AutoDrive-R²** | **4 (visual→calc→logic→reflection)** | **✓ (backward-check)** | **Qwen2.5-VL-72B** | **6K** |
| Alpamayo-R1 | 3 (decision→components→causal trace) | ✗ | GPT-5 + 2-stage human | 700K |
| AdaThinkDrive | 4 (road→agents→action→scene) | ✗ | Qwen2.5-VL-72B + rule-based | NAVSIM-scale |

The **self-reflection validation step** is unique across all wiki papers. All others produce forward-only CoT chains; AutoDrive-R² adds an explicit backward verification phase.

## AutoMoT: The Frozen VLM Case — Empirical Evidence of Catastrophic Forgetting

**AutoMoT** ([[sources/automot.md]]) makes the strongest empirical argument in the wiki for **not fine-tuning** VLMs on AD data, providing systematic evidence of catastrophic forgetting (Table 4):

| Benchmark | Task | Frozen UE | AD Fine-tuned UE | Δ |
|-----------|------|-----------|-----------------|---|
| LingoQA | AD scene understanding | 67.00 | 67.20 | +0.2 |
| OmniDrive | Counterfactual planning | 18.20 | **67.80** | +49.6 |
| ScienceQA | General knowledge | 88.60 | 87.80 | −0.8 |
| FigureQA | General knowledge | 97.60 | 91.20 | −6.4 |
| TallyQA | Multi-step general reasoning | **81.40** | 52.40 | **−35.3%** |
| InfographicVQA | Compositional general reasoning | **89.30** | 50.20 | **−43.8%** |
| VizWiz | Multi-step general reasoning | **75.60** | 50.20 | **−33.6%** |

**Pattern**: fine-tuning degrades simple tasks marginally (ScienceQA −0.8, FigureQA −6.4) but destroys complex multi-step reasoning (TallyQA −35%, InfoVQA −44%). AD scene understanding (LingoQA) sees only +0.2 gain — essentially no benefit. Counterfactual planning (OmniDrive +49.6) is the only clear win, confirming that action-level tasks genuinely require fine-tuning.

**Functional boundary principle** (AutoMoT's conclusion): pre-trained VLMs already support competitive AD scene understanding through semantic prompting alone; fine-tuning should be restricted to action-level components.

**Counterpoint**: Despite frozen UE, AutoMoT still achieves competitive planning (L2_avg 0.32) and best collision rate (0.07) on nuScenes, beating fine-tuned methods like AutoVLA (0.40 L2). This demonstrates that frozen scene understanding can effectively inform action-level learning via joint attention — the VLM doesn't need to be adapted to transfer its knowledge.

**Comparison with OpenEMMA** (also no VLM fine-tuning): OpenEMMA gets 2.81 L2 vs. AutoMoT's 0.32. The gap confirms that the improvement comes from the AE policy learning, not just from prompt-based VLM use — domain-specific adaptation at the *action level* remains essential.

## Alpamayo-R1: Physical AI Pre-Training + CoC Structured Reasoning

**Alpamayo-R1** ([[sources/alpamayo-r1.md]]) introduces two distinct contributions to VLM domain adaptation: a **domain-specific Physical AI backbone** (Cosmos-Reason) and a **structured causal reasoning dataset** (Chain of Causation, CoC) with hybrid labeling.

### Physical AI Backbone: Cosmos-Reason

Rather than adapting a general-purpose VLM (Qwen2.5-VL, InternVL), AR1 uses Cosmos-Reason — a VLM post-trained specifically on Physical AI data: 3.7M general VQA + 24.7K driving VQA grounded in physical dynamics and spatial causality. This is the first wiki paper to leverage a driving-optimized *pre-trained backbone* rather than fine-tuning a generic backbone on AD data.

**LingoQA zero-shot comparison** (Table 10):

| Model | Lingo-Judge |
|---|---|
| DeepSeek-VL-7B | 46.4 |
| Qwen2-VL-7B | 52.6 |
| InternVL3.5-8B | 58.6 |
| GPT-4V | 59.6 |
| Qwen2.5-VL-7B | 62.2 |
| **Cosmos-Reason-7B** | **66.2** |

+6.4% vs. Qwen2.5-VL-7B of the same size — achieved purely from Physical AI pre-training, not AD fine-tuning at this stage.

**Complementarity with AutoMoT's finding**: AutoMoT showed that fine-tuning general VLMs yields only +0.2 LingoQA with catastrophic forgetting on multi-step general reasoning. Cosmos-Reason avoids this by starting from a domain-aligned backbone, suggesting that the correct level to inject AD knowledge is pre-training or Physical AI SFT — not task-specific fine-tuning of general models.

### Chain of Causation (CoC) Dataset

700K video segments annotated with structured causal reasoning traces. CoC is designed around three desiderata:
- **Decision-grounding**: reasoning anchored to a closed 14-type decision set (prevents vague or unverifiable statements)
- **Causal locality**: evidence drawn only from observable history (no forward-leaking inference)
- **Annotation economy**: hybrid pipeline scales with auto-labeling

**Hybrid labeling pipeline**:

| Stage | Actor | Window | Goal |
|---|---|---|---|
| Stage I | Human | 0–2s observation only | Identify meta-action from closed decision set; limited window prevents retrospective rationalization |
| Stage II | Human | 0–8s full context | Compose causal trace from critical scene components |
| Auto-labeling | GPT-5 | Full | Scale to 700K; 92% LLM-human alignment |

**Quality outcome**: +132.8% causal relationship score vs. free-form reasoning baseline.

### CoT Comparison with Other Wiki Papers

| Approach | CoT steps | Self-reflection? | Causal locality? | Decision grounding? | Annotation model | Scale |
|----------|-----------|-----------------|-----------------|---------------------|-----------------|-------|
| ReCogDrive | Unstructured QA | ✗ | ✗ | ✗ | Various VLMs | 3.1M pairs |
| Reasoning-VLA | 4 (scene→physics→intent→trajectory) | ✗ | ✗ | ✗ | Qwen2.5-VL | 75K clips |
| UniUGP | 4 (scene→objects→intent→action) | ✗ | ✗ | ✗ | Frontier VLM | Custom |
| AutoVLA | 4 (scene→objects→intent→action) | ✗ | ✗ | ✗ | Qwen2.5-VL-72B | 52.8K |
| AutoDrive-R² | 4 (visual→calc→logic→reflection) | ✓ (backward-check) | ✗ | ✗ | Qwen2.5-VL-72B | 6K |
| **Alpamayo-R1** | **3 (decision→components→causal trace)** | **✗** | **✓ (observable history only)** | **✓ (closed 14-type set)** | **GPT-5 + 2-stage human** | **700K** |

CoC is the only approach with explicit causal locality enforcement and closed-set decision grounding. AutoDrive-R² has backward-check self-reflection but no causal locality constraint.

### Updated Strategy Comparison Table

| Approach | VLM role | VLM fine-tuned? | Data | Backbone | Deployment |
|----------|----------|----------------|------|----------|-----------|
| ReCogDrive | Fine-tuned feature extractor | ✓ | 3.1M QA pairs | InternVL3 | At inference |
| Reasoning-VLA | Unified multi-dataset SFT + GRPO | ✓ | 75K CoT clips | Qwen2.5-VL | At inference |
| UniUGP | Expert co-training, 4-stage | ✓ | Mixed video + QA | Qwen2.5-VL-3B | At inference |
| Senna-2 | Consistency-aligned dual-system | ✓ | NAVSIM + proprietary | LLaVA-NeXT | At inference |
| HERMES | Offline annotator only | ✗ | WOD-E2E + VLM annotations | Qwen3-VL-Flash | Not at inference |
| AutoMoT | Frozen scene understanding expert | ✗ | NuSync + nuScenes + CARLA | Qwen3-VL-4B | At inference (async) |
| AutoDrive-R² | SFT + GRPO on structured CoT | ✓ | 6K curated samples | Qwen2.5-VL-7B | At inference |
| **Alpamayo-R1** | **3-stage: inject→SFT(CoC)→RL** | **✓ (from Physical AI base)** | **700K CoC + 80K hr driving** | **Cosmos-Reason** | **At inference** |

## AdaThinkDrive: Scene-Complexity-Aware Dual-Mode SFT

**AdaThinkDrive** ([[sources/adathinkdrive.md]]) introduces the most explicit scene-complexity-aware adaptation strategy in the wiki: training on the same queries in two reasoning modes and using RL to teach the model when each mode is appropriate.

### Empirical Finding: CoT Hurts in Simple Scenarios

Study on InternVL3-8B/2B across 3 scene complexity levels establishes that CoT reasoning is not universally beneficial — it degrades performance in simple scenes (Level 1) while helping in complex ones (Levels 2–3). This motivates all subsequent design choices.

### Scene Complexity Categorization

Scenes are labeled with 3 complexity levels based on two binary geometric conditions:

| Level | Boundary proximity? | Critical objects? |
|---|---|---|
| 1 (Simple, $\mathcal{D}^-$) | ✗ | ✗ |
| 2 (Moderate) | One of two | — |
| 3 (Challenging, $\mathcal{D}^+$) | ✓ | ✓ |

**Critical objects** are classified into three types:
- CIPO-1: Closest In-Path Object (in ego lane)
- CIPO-2: Likely to merge (lane geometry + relative position)
- Motion Interaction: Predicted trajectory intersects ego path

These labels serve as cold-start initialization for RL's Adaptive Think Reward.

### Dual-Mode SFT: Same Query, Two Outputs

Unlike AutoVLA (which mixes separate fast/slow samples), AdaThinkDrive generates **both** Think and Non-Think outputs for the **same query** $q$:

$$\mathcal{D}^{SFT} = \{\{q, o^{Think}\}, \{q, o^{Non\text{-}Think}\}\}_{q \in \mathcal{Q}}$$

The model is supervised on both via standard CE loss, with no bias toward either style. This ensures both modes are available for any input — a prerequisite for the RL stage to select between them.

**Key contrast with AutoVLA's dual-mode SFT**:

| Aspect | AutoVLA | AdaThinkDrive |
|---|---|---|
| SFT data structure | Separate fast/slow samples (different queries) | Same query → both modes |
| Mode weighting | λ_cot=40 (upweights slow CoT) | Uniform (no bias) |
| Complexity labeling | None | 3-level (boundary + critical objects) |
| RL mechanism | Length penalty | Mode-comparison per scene |
| Adaptive goal | How *short* to reason | Whether to reason at all |

### Updated Strategy Comparison Table

| Approach | VLM role | VLM fine-tuned? | Data | Backbone | Deployment |
|----------|----------|----------------|------|----------|-----------|
| ReCogDrive | Fine-tuned feature extractor | ✓ | 3.1M QA pairs | InternVL3 | At inference |
| Reasoning-VLA | Unified multi-dataset SFT + GRPO | ✓ | 75K CoT clips | Qwen2.5-VL | At inference |
| UniUGP | Expert co-training, 4-stage | ✓ | Mixed video + QA | Qwen2.5-VL-3B | At inference |
| Senna-2 | Consistency-aligned dual-system | ✓ | NAVSIM + proprietary | LLaVA-NeXT | At inference |
| HERMES | Offline annotator only | ✗ | WOD-E2E + VLM annotations | Qwen3-VL-Flash | Not at inference |
| AutoMoT | Frozen scene understanding expert | ✗ | NuSync + nuScenes + CARLA | Qwen3-VL-4B | At inference (async) |
| AutoVLA | Dual-mode SFT + CoT length penalty RFT | ✓ | 52.8K CoT (nuPlan+Waymo) | InternVL2.5 | At inference |
| AutoDrive-R² | SFT + GRPO on structured CoT | ✓ | 6K curated samples | Qwen2.5-VL-7B | At inference |
| Alpamayo-R1 | 3-stage: inject→SFT(CoC)→RL | ✓ (from Physical AI base) | 700K CoC + 80K hr driving | Cosmos-Reason | At inference |
| **AdaThinkDrive** | **Dual-mode SFT + complexity-aware RL** | **✓** | **NAVSIM + open-source QA** | **InternVL3-8B** | **At inference** |

## FutureSightDrive: Visual CoT via Vocabulary Expansion

**FutureSightDrive** ([[sources/futuresightdrive.md]]) takes the most radical departure from textual CoT in the wiki: it replaces the entire CoT modality with a **generated future image frame** (visual spatio-temporal CoT), using vocabulary expansion to activate image generation in an existing MLLM without architectural change.

### Modality Gap Argument

FSDrive claims that textual CoT compresses continuous visual spatio-temporal relationships into lossy symbolic representations, creating a "modality gap" between perception and planning. Ablation evidence:

| CoT Type | Avg Collision ↓ | Improvement |
|---|---|---|
| None | 0.58 | — |
| Text CoT | 0.53 | −8.6% |
| Image-text CoT | 0.50 | −13.8% |
| **Visual ST-CoT** | **0.40** | **−31.0%** |

Text CoT and image-text CoT show diminishing gains; the unified visual representation captures spatial and temporal structure that text cannot.

### Vocabulary Expansion (Core Mechanism)

MoVQGAN VQ-VAE tokens are appended to the MLLM's existing text vocabulary. No encoder, decoder, or attention architecture is modified. The model predicts image tokens autoregressively using the same next-token prediction objective as text. This activates generation capability using ~0.3% of the data used by prior unified understanding+generation approaches (e.g., Janus, VILA-U) which train from scratch.

**Two-stage training**:
1. **Unified pre-training**: VQA + future frame generation (with progressive curriculum: lane dividers → 3D boxes → full frame) on nuScenes
2. **SFT**: DriveLM GVQA + nuScenes trajectory planning using visual CoT

### Positioning in the CoT Design Space

| Approach | CoT modality | Self-reflection? | Physical constraint? | Annotation model | Scale |
|----------|-------------|-----------------|---------------------|-----------------|-------|
| ReCogDrive | Text (unstructured QA) | ✗ | ✗ | Various VLMs | 3.1M pairs |
| AutoDrive-R² | Text (4-step + backward-check) | ✓ | Kinematic extrapolation | Qwen2.5-VL-72B | 6K |
| Alpamayo-R1 | Text (3-step causal) | ✗ | ✓ (causal locality) | GPT-5 + human | 700K |
| AdaThinkDrive | Text (4-step adaptive) | ✗ | ✗ | Qwen2.5-VL-72B + rule | NAVSIM-scale |
| **FSDrive** | **Visual image (future frame)** | **✗** | **✓ (physical priors overlaid)** | **nuScenes annotations** | **200K frames** |

FSDrive is the only wiki paper using image-modality CoT. Its physical constraint enforcement is architectural (lane dividers + 3D boxes as visual priors in the generated frame) rather than textual reasoning about physics.

## UniDriveVLA: MoT as Structural Solution to Representation Interference

**UniDriveVLA** ([[sources/unidrivevla.md]]) makes the strongest architectural argument in the wiki that VLM fine-tuning failures are not just a data or training recipe problem — they are a structural problem caused by **representation interference** when spatial perception and semantic reasoning share parameters.

### The Interference Diagnosis

UniDriveVLA measures cosine similarity between LLM tokens and perception tokens across transformer layers in a shared-weight decoder. The similarity progressively increases toward 1, indicating **feature collapse**: spatial perception tokens and semantic reasoning tokens converge to identical representations, causing each to undermine the other. This is not fixed by careful data mixing or learning rate schedules — it is an inherent consequence of joint optimization in shared parameter space.

**Comparison with AutoMoT's catastrophic forgetting evidence**:
- AutoMoT: fine-tuning degrades TallyQA −35%, InfoVQA −44% (temporal multi-step reasoning) while LingoQA gains only +0.2 — forgetting is task-specific and affects complex reasoning most
- UniDriveVLA: feature collapse is spatial and occurs at representation level, not just behavior level — the cause is geometric (shared-weight parameter conflict) rather than just data imbalance

Both papers converge on the same conclusion from different angles: **spatial perception and semantic reasoning should not share parameters** if both are needed at high quality simultaneously.

### MoT as the Fix

Three expert streams — Understanding (und), Perception (per), Action (act) — with asymmetric Masked Joint Attention:

```
und: causal self-attention only (never sees per or act tokens)
per: attends to und + self
act: attends to und + per + self
```

The und expert is fully protected from perception gradient interference. Per is enriched by und semantics but does not pollute und. Act aggregates both.

**Quantitative impact** (Table 7):

| Architecture | General VQA(%)↑ | DriveBench(%)↑ | L2(m)↓ | CR(%)↓ |
|---|---|---|---|---|
| Shared-Weight | 31.1 | 50.8 | 0.641 | 0.175 |
| MoT | **45.5** | **54.9** | **0.533** | **0.140** |
| Δ | **+14.4pp** | **+4.1pp** | **−0.108m** | **−0.035** |

### Three-Stage Progressive Training

| Stage | What changes | Key mechanism |
|-------|-------------|---------------|
| Stage 1 | Full VLM fine-tune | 3:7 driving-to-general ratio; lr=4e-5; 3 epochs — anchors semantic capability |
| Stage 2 | VLM (LoRA) + Per/Act experts | LoRA limits VLM drift; 0.5× VLM LR multiplier; 30 epochs joint optimization |
| Stage 3 | Per + Act experts only | VLM frozen; motion objective added; 15 epochs — final task specialization |

LoRA in Stage 2 combined with the reduced VLM learning rate explicitly limits how much perception and action gradients can modify the VLM representations. This is the most fine-grained LR control in the wiki, complementing ReCogDrive's single-LR approach and AutoMoT's frozen-VLM approach.

**Comparison of VLM protection strategies across wiki**:

| Approach | VLM parameter update | Protection mechanism |
|---------|---------------------|---------------------|
| AutoMoT | None (frozen) | Complete isolation |
| UniDriveVLA Stage 2 | LoRA, 0.5× LR | Partial — limited capacity + reduced rate |
| UniDriveVLA Stage 3 | None (frozen) | Complete isolation |
| ReCogDrive, AutoVLA | Full fine-tune | None — data quality as only protection |
| Alpamayo-R1 | Full fine-tune from Physical AI base | Domain-aligned starting point reduces forgetting |

### General VQA Degradation: Residual Even With MoT

Despite MoT's +14.4pp General VQA improvement vs. shared-weight, the adaptation still significantly degrades general reasoning relative to the base Qwen3-VL:

| Benchmark | Qwen3-VL-8B (base) | UniDriveVLA (after adaptation) | Δ |
|-----------|--------------------|-------------------------------|---|
| MMStar | 63.0 | 43.3 | **−19.7pp** |
| MMMU | 52.8 | 47.3 | −5.5pp |
| RealWorldQA | 69.0 | 49.9 | −19.1pp |
| VLMsAreBlind | 61.9 | 26.6 | **−35.3pp** |
| MME | 2364 | 1876 | −488 |

MoT substantially reduces the degradation that would occur with a shared-weight decoder (estimated base 31.1% General VQA → 45.5%), but the gap to the base model (63.0%) remains large. This suggests MoT is **necessary but not sufficient**: it prevents active interference, but driving SFT still displaces some general-domain knowledge.

**VLMsAreBlind** degradation (61.9→26.6, −35.3pp) is particularly striking — this benchmark tests basic visual grounding that the base model handles well. Driving-specific visual tokenization or attention patterns may specifically harm low-level visual reasoning.

### Updated Strategy Comparison Table

| Approach | VLM role | VLM fine-tuned? | Data | Backbone | Deployment |
|----------|----------|----------------|------|----------|-----------|
| ReCogDrive | Fine-tuned feature extractor | ✓ | 3.1M QA pairs | InternVL3 | At inference |
| Reasoning-VLA | Unified multi-dataset SFT + GRPO | ✓ | 75K CoT clips | Qwen2.5-VL | At inference |
| UniUGP | Expert co-training, 4-stage | ✓ | Mixed video + QA | Qwen2.5-VL-3B | At inference |
| Senna-2 | Consistency-aligned dual-system | ✓ | NAVSIM + proprietary | LLaVA-NeXT | At inference |
| HERMES | Offline annotator only | ✗ | WOD-E2E + VLM annotations | Qwen3-VL-Flash | Not at inference |
| AutoMoT | Frozen scene understanding expert | ✗ | NuSync + nuScenes + CARLA | Qwen3-VL-4B | At inference (async) |
| AutoVLA | Dual-mode SFT + CoT length penalty RFT | ✓ | 52.8K CoT (nuPlan+Waymo) | InternVL2.5 | At inference |
| AutoDrive-R² | SFT + GRPO on structured CoT | ✓ | 6K curated samples | Qwen2.5-VL-7B | At inference |
| Alpamayo-R1 | 3-stage: inject→SFT(CoC)→RL | ✓ (from Physical AI base) | 700K CoC + 80K hr driving | Cosmos-Reason | At inference |
| AdaThinkDrive | Dual-mode SFT + complexity-aware RL | ✓ | NAVSIM + open-source QA | InternVL3-8B | At inference |
| FSDrive | Vocabulary expansion + visual CoT generation | ✓ | 200K nuScenes + DriveLM | Qwen2-VL-2B | At inference (sequential gen→plan) |
| **UniDriveVLA** | **MoT: LoRA Stage 2 → frozen Stage 3; decoupled per/act experts** | **✓ (LoRA then frozen)** | **3:7 driving-to-general SFT** | **Qwen3-VL-2B/8B** | **At inference** |

## FLARE: Annotation-Free VLM Adaptation via Future Feature Prediction

**FLARE** ([[sources/flare.md]]) represents a fundamentally different adaptation paradigm: instead of using any language supervision (VQA, CoT, scene descriptions), it adapts a VLM-based planner entirely through **self-supervised spatial feature prediction**. This is the only wiki paper that requires zero language annotations for VLM driving adaptation.

### The Annotation-Free Argument

FLARE's core claim: language annotation pipelines (VQA, CoT) have three structural problems:
1. **Cost with diminishing returns**: scaling QA pairs past ~3M provides marginal gains (ReCogDrive ablation evidence)
2. **Semantic bias**: language descriptions of scenes introduce biases that misalign with actual driving decisions
3. **Modality mismatch**: discrete tokens compress continuous visual dynamics, creating a gap between text reasoning and trajectory generation

**Alternative**: reconstruct DINOv2 semantic patch features of the future frame as an auxiliary loss — dense, continuous, annotation-free, scalable to unlabeled video. The supervision signal is 100× denser than trajectory waypoints alone (all spatial patches of the future frame vs. 6 waypoints).

### Training Recipe

No VQA/caption dataset required. Two-stage:
1. **SFT**: LoRA fine-tune on NAVSIM navtrain trajectory data + future DINOv2 prediction loss (λ=0.1). LR=3e-5 for LoRA, 1e-4 for fusion heads. 80 epochs.
2. **RFT**: VLM frozen; GRPO with PDM-Score reward; G=16 samples; BC regularization (not KL).

**Baseline comparison**: QwenVL3-4B trained on navtrain trajectories only (same data, no feature prediction) achieves 80.8 PDMS. FLARE achieves 86.9 SFT (+6.1 PDMS) from the same backbone and data — the entire delta comes from the future feature prediction auxiliary task.

### Positioning in Adaptation Design Space

| Approach | Language annotations? | External driving data? | What supervises VLM? |
|---------|----------------------|----------------------|---------------------|
| ReCogDrive | ✓ (3.1M VQA pairs) | ✓ | VQA + scene description |
| AutoVLA | ✓ (52.8K CoT) | ✓ | 4-stage CoT |
| HERMES | ✓ (VLM-generated) | — | Safety annotations (offline) |
| **FLARE** | **✗** | **✗** | **Future DINOv2 patch features** |
| DriveVLA-W0 | ✗ | ✓ (70M frames) | Future VAE pixel latents |

FLARE and DriveVLA-W0 are the two annotation-free approaches. DriveVLA-W0 relies on a massive proprietary dataset for data scaling; FLARE achieves competitive results on the standard NAVSIM navtrain (103K frames) alone.

### Updated Strategy Comparison Table

| Approach | VLM role | VLM fine-tuned? | Data | Backbone | Deployment |
|----------|----------|----------------|------|----------|-----------|
| ReCogDrive | Fine-tuned feature extractor | ✓ | 3.1M QA pairs | InternVL3 | At inference |
| Reasoning-VLA | Unified multi-dataset SFT + GRPO | ✓ | 75K CoT clips | Qwen2.5-VL | At inference |
| UniUGP | Expert co-training, 4-stage | ✓ | Mixed video + QA | Qwen2.5-VL-3B | At inference |
| Senna-2 | Consistency-aligned dual-system | ✓ | NAVSIM + proprietary | LLaVA-NeXT | At inference |
| HERMES | Offline annotator only | ✗ | WOD-E2E + VLM annotations | Qwen3-VL-Flash | Not at inference |
| AutoMoT | Frozen scene understanding expert | ✗ | NuSync + nuScenes + CARLA | Qwen3-VL-4B | At inference (async) |
| AutoVLA | Dual-mode SFT + CoT length penalty RFT | ✓ | 52.8K CoT (nuPlan+Waymo) | InternVL2.5 | At inference |
| AutoDrive-R² | SFT + GRPO on structured CoT | ✓ | 6K curated samples | Qwen2.5-VL-7B | At inference |
| Alpamayo-R1 | 3-stage: inject→SFT(CoC)→RL | ✓ (from Physical AI base) | 700K CoC + 80K hr driving | Cosmos-Reason | At inference |
| AdaThinkDrive | Dual-mode SFT + complexity-aware RL | ✓ | NAVSIM + open-source QA | InternVL3-8B | At inference |
| FSDrive | Vocabulary expansion + visual CoT generation | ✓ | 200K nuScenes + DriveLM | Qwen2-VL-2B | At inference (sequential gen→plan) |
| UniDriveVLA | MoT: LoRA Stage 2 → frozen Stage 3; decoupled per/act experts | ✓ (LoRA then frozen) | 3:7 driving-to-general SFT | Qwen3-VL-2B/8B | At inference |
| **FLARE** | **LoRA SFT on trajectories + future DINOv2 feature prediction (no language annotations)** | **✓ (LoRA)** | **NAVSIM navtrain (trajectories only)** | **Qwen3-VL-4B** | **At inference** |

## Vega: Instructional Driving via World Model as Dense Supervision Bridge

**Vega** ([[sources/vega.md]]) introduces a novel VLM adaptation goal: instead of adapting for expert imitation, adapt for **open-ended natural language instruction following** — producing different trajectories in the same scene when given different user commands.

### The Instruction-to-Action Gap

A direct baseline (Qwen2.5-VL + planning head, same instruction-annotated data) achieves only ~60 PDMS. The gap arises because:
- Trajectory supervision is sparse (6 waypoints) vs. visual input dimensionality
- Language instructions introduce additional ambiguity that sparse action supervision cannot resolve
- LLMs are unreliable at precise numerical trajectory prediction from natural language

**Solution**: future image generation as dense supervision signal. This compels the model to learn the full causal chain: instruction → action → visual outcome.

### InstructScene: Automated Instruction Annotation

100K NAVSIM scenes annotated via two-stage automated pipeline:
1. Qwen2.5-VL-72B converts future frames + actions into scene descriptions + driving behavior narrations
2. Qwen2.5-VL-72B formulates concise driving instructions from those descriptions

Rule-based supplement: speed/acceleration/turn-rate thresholds → NL ego-motion labels → combined with VLM instructions as auxiliary prompts for accuracy.

This is the first instruction-annotated driving dataset at scale in the wiki; all other methods use scene-description or CoT annotations (not personalized user instructions).

### Architecture: Integrated AR+Diffusion with MoT

Vega uses an **Integrated Transformer** (not external diffuser) where understanding (AR, Qwen2.5-LLM) and generation (diffusion) share joint attention with separate weight sets via Mixture-of-Transformers. Initialized from Bagel-7B.

Key adaptation mechanisms:
- **Classifier-free guidance** (CFG): drops text/ViT/action tokens randomly during training → enables inference-time instruction guidance strength
- **Duplicate latent trick**: noisy copy for denoising, clean copy for conditioning → enables joint multi-task training (action + image) in a single forward pass
- **Lightweight action expert** (256-dim vs. 3584): separate from both understanding and generation modules

### Positioning in the Adaptation Space

| Dimension | Traditional VLA adaptation | Vega |
|-----------|---------------------------|------|
| Language role | Scene description, CoT reasoning | **User instruction (personalized)** |
| Planning goal | Imitate expert trajectory | **Follow user command** |
| Multi-trajectory | No (single expert) | **Yes (different instruction → different trajectory)** |
| Supervision | Sparse (waypoints) | Dense (future image + waypoints) |
| Dataset | Scene QA / CoT | **Instruction-annotated (InstructScene)** |

### Updated Strategy Comparison Table (with Vega)

| Approach | VLM role | VLM fine-tuned? | Data | Backbone | Deployment |
|----------|----------|----------------|------|----------|-----------|
| ReCogDrive | Fine-tuned feature extractor | ✓ | 3.1M QA pairs | InternVL3 | At inference |
| Reasoning-VLA | Unified multi-dataset SFT + GRPO | ✓ | 75K CoT clips | Qwen2.5-VL | At inference |
| UniUGP | Expert co-training, 4-stage | ✓ | Mixed video + QA | Qwen2.5-VL-3B | At inference |
| Senna-2 | Consistency-aligned dual-system | ✓ | NAVSIM + proprietary | LLaVA-NeXT | At inference |
| HERMES | Offline annotator only | ✗ | WOD-E2E + VLM annotations | Qwen3-VL-Flash | Not at inference |
| AutoMoT | Frozen scene understanding expert | ✗ | NuSync + nuScenes + CARLA | Qwen3-VL-4B | At inference (async) |
| AutoVLA | Dual-mode SFT + CoT length penalty RFT | ✓ | 52.8K CoT (nuPlan+Waymo) | InternVL2.5 | At inference |
| AutoDrive-R² | SFT + GRPO on structured CoT | ✓ | 6K curated samples | Qwen2.5-VL-7B | At inference |
| Alpamayo-R1 | 3-stage: inject→SFT(CoC)→RL | ✓ (from Physical AI base) | 700K CoC + 80K hr driving | Cosmos-Reason | At inference |
| AdaThinkDrive | Dual-mode SFT + complexity-aware RL | ✓ | NAVSIM + open-source QA | InternVL3-8B | At inference |
| FSDrive | Vocabulary expansion + visual CoT generation | ✓ | 200K nuScenes + DriveLM | Qwen2-VL-2B | At inference (sequential gen→plan) |
| UniDriveVLA | MoT: LoRA Stage 2 → frozen Stage 3; decoupled per/act experts | ✓ (LoRA then frozen) | 3:7 driving-to-general SFT | Qwen3-VL-2B/8B | At inference |
| FLARE | LoRA SFT on trajectories + future DINOv2 feature prediction (no language annotations) | ✓ (LoRA) | NAVSIM navtrain (trajectories only) | Qwen3-VL-4B | At inference |
| **Vega** | **Integrated AR+Diffusion (Bagel-7B); NL instruction following via world model dense supervision** | **✓ (full fine-tune)** | **InstructScene 100K + NAVSIM** | **Qwen2.5-LLM + Bagel-7B** | **At inference (CFG)** |

## NoRD: Reasoning-Free SFT as an Explicit Design Choice

**NoRD** ([[sources/nord.md]]) is the only paper in the wiki that deliberately eliminates reasoning supervision as a principled design decision — not due to data constraints, but to test the "Reasoning-Planning Decoupling Hypothesis": that CoT reasoning may be a byproduct of planning, not a causal determinant.

### The Reasoning-Free Hypothesis

Prior work (EMMA, SimLingo, S4-Driver) showed that reasoning-free VLAs can achieve strong results on simple benchmarks (nuScenes). NoRD asks whether this extends to harder benchmarks (NAVSIM, WaymoE2E) that are currently dominated by reasoning-centric models.

**Hypothesis**: if the RL optimizer is fixed (to handle the weaker SFT policy), a reasoning-free model can match reasoning-based performance — meaning reasoning provides no additional planning signal, only annotation cost and inference latency.

### Adaptation Design

- **Base**: Qwen-2.5VL-3B-Instruct
- **SFT**: trajectory token prediction only — no `<think>` blocks, no CoT, no scene descriptions
- **Data**: 80K NAVSIM samples (vs. 212K+ for reasoning-based SOTA)
- **Output**: k-disc tokens (vocab=2048) representing trajectory clusters — no text waypoints, no reasoning tokens

The reduction in token count (3× vs. reasoning VLAs) directly reduces training compute and inference latency.

### Key Finding: Reasoning Annotations Are Not the Bottleneck

NoRD's SFT-only baseline (76.66 PDMS) is weaker than reasoning-based models. But with Dr. GRPO post-training (85.62 PDMS), NoRD is competitive with AutoVLA (89.1) — which uses 2.65× more data and full CoT supervision. The remaining gap is closed by BoN-6 oracle selection (92.4 vs. AutoVLA-BoN 92.1).

**Implication**: reasoning annotations inflate performance in the SFT stage, but RL post-training can compensate — provided the optimizer handles the weaker initialization correctly. The cost of reasoning annotations is not justified by the final post-RL performance.

### Contrast with Other Reasoning Strategies

| Approach | Reasoning at train? | Reasoning at inference? | Data needed |
|----------|--------------------|-----------------------|------------|
| AutoVLA | ✓ (4-stage CoT) | ✓ (adaptive) | 212K+ |
| AdaThinkDrive | ✓ (Think+NonThink) | ✓ (adaptive) | NAVSIM-scale |
| FLARE | ✗ (DINOv2 features) | ✗ | NAVSIM navtrain |
| **NoRD** | **✗** | **✗** | **80K NAVSIM** |

NoRD and FLARE both achieve no-reasoning at inference, but via different mechanisms: FLARE uses future visual features as auxiliary supervision; NoRD uses no auxiliary supervision at all — pure trajectory SFT + Dr. GRPO.

### Updated Strategy Comparison Table (with NoRD)

| Approach | VLM role | VLM fine-tuned? | Data | Backbone | Deployment |
|----------|----------|----------------|------|----------|-----------|
| ReCogDrive | Fine-tuned feature extractor | ✓ | 3.1M QA pairs | InternVL3 | At inference |
| Reasoning-VLA | Unified multi-dataset SFT + GRPO | ✓ | 75K CoT clips | Qwen2.5-VL | At inference |
| UniUGP | Expert co-training, 4-stage | ✓ | Mixed video + QA | Qwen2.5-VL-3B | At inference |
| Senna-2 | Consistency-aligned dual-system | ✓ | NAVSIM + proprietary | LLaVA-NeXT | At inference |
| HERMES | Offline annotator only | ✗ | WOD-E2E + VLM annotations | Qwen3-VL-Flash | Not at inference |
| AutoMoT | Frozen scene understanding expert | ✗ | NuSync + nuScenes + CARLA | Qwen3-VL-4B | At inference (async) |
| AutoVLA | Dual-mode SFT + CoT length penalty RFT | ✓ | 52.8K CoT (nuPlan+Waymo) | InternVL2.5 | At inference |
| AutoDrive-R² | SFT + GRPO on structured CoT | ✓ | 6K curated samples | Qwen2.5-VL-7B | At inference |
| Alpamayo-R1 | 3-stage: inject→SFT(CoC)→RL | ✓ (from Physical AI base) | 700K CoC + 80K hr driving | Cosmos-Reason | At inference |
| AdaThinkDrive | Dual-mode SFT + complexity-aware RL | ✓ | NAVSIM + open-source QA | InternVL3-8B | At inference |
| FSDrive | Vocabulary expansion + visual CoT generation | ✓ | 200K nuScenes + DriveLM | Qwen2-VL-2B | At inference (sequential gen→plan) |
| UniDriveVLA | MoT: LoRA Stage 2 → frozen Stage 3; decoupled per/act experts | ✓ (LoRA then frozen) | 3:7 driving-to-general SFT | Qwen3-VL-2B/8B | At inference |
| FLARE | LoRA SFT on trajectories + future DINOv2 feature prediction (no language annotations) | ✓ (LoRA) | NAVSIM navtrain (trajectories only) | Qwen3-VL-4B | At inference |
| Vega | Integrated AR+Diffusion (Bagel-7B); NL instruction following via world model dense supervision | ✓ (full fine-tune) | InstructScene 100K + NAVSIM | Qwen2.5-LLM + Bagel-7B | At inference (CFG) |
| **NoRD** | **Reasoning-free trajectory SFT + Dr. GRPO; no CoT at any stage** | **✓ (full fine-tune)** | **80K NAVSIM (no annotations)** | **Qwen-2.5VL-3B** | **At inference (fast, 3× fewer tokens)** |

## Related Systems

- **DriveVLM**: dual-system, VLM generates chain-of-thought + trajectory commands
- **EMMA**: single-system, reformulates planning as next-token prediction
- **OmniDrive / Atlas**: enhance 3D scene perception with 3D tokenizer
- **Agency-Driver**: adds tool library + cognitive memory to LLM for driving
- **WiseAD / LMDrive**: VLM systems evaluated in closed-loop frameworks
