---
title: "LVDrive: Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model"
type: source-summary
sources: ["raw/papers/LVDrive_ Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model.md"]
related: [sources/raf-vla.md, sources/orion.md, sources/deepsight.md, sources/law.md, sources/drivevla-w0.md, sources/futuresightdrive.md, sources/hyworldvla.md, sources/raw2drive.md, sources/think2drive.md, sources/recogdrive.md, sources/momworld.md, sources/world4scorer.md, sources/wa-jepa.md, sources/foredrive.md, sources/dynvla.md, sources/linkvla.md, sources/automot.md, sources/sparsedrivev2.md, sources/autovla.md, sources/unidrivevla.md, sources/driveva.md, sources/hermes.md, sources/epona.md, concepts/bench2drive.md, concepts/world-model-for-ad.md, concepts/evaluation-variance.md, concepts/inference-latency.md, concepts/diffusion-planner.md, concepts/foundation-backbones-for-ad.md, research-directions.md]
created: 2026-10-19
updated: 2026-10-19
confidence: low
---

# LVDrive: Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model

**Paper**: LVDrive: Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model
**Authors**: Xiaodong Mei, Hongwei Xie, Guang Chen, Hangjun Ye, Dan Xu
**Orgs**: Xiaomi EV (Xie, Chen, Ye); The Hong Kong University of Science and Technology (Xu; Mei presumably). Hongwei Xie is an author of [[sources/orion.md]], the model LVDrive is built on; Xie and Chen are on MindDrive, and Xie, Chen and Ye on [[sources/driveva.md]].
**arXiv**: 2605.22089v1 (May 2026; NeurIPS-style template)
**Code**: none stated
**Source**: `raw/papers/LVDrive_ Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model.md`. All text, seven tables and eight figures (`nips_intro_0.png`, `nips_model_0.png`, `nips_vis_2.png`, `nips_vis_a1.png` … `nips_vis_a5.png` in `raw/assets/`).

**Confidence: low.** The headline is an ordinary full Bench2Drive run (220 routes, one run). Every mechanism result is on Dev10, ten routes, with success rates printed as counts out of ten and no repeats stated. No matched baseline is run on the full benchmark. Its VAE planner is described as reading the ground-truth trajectory during training with the KL term removed, and the paper does not say what replaces it at inference. The text and the table disagree on whether the method is faster or slower than its base.

---

## What It Is

**[[sources/orion.md]] with its VQA co-training stage replaced by front-camera future-latent regression, and a refiner that reads the predicted future.**

1. **Inputs and VLM (ORION's).** Multi-view images go through EVA-02-L and QT-Former (perception, scene and history queries, with detection, traffic-state and motion-prediction supervision). Vicuna v1.5 with LoRA is initialized from ORION's Stage-1 weights (vision–language alignment on Chat-B2D VQA).
2. **Future latents in one forward pass.** The output sequence is pre-filled with $F=6$ blocks of `<img_start>`, 256 `<img_i>` placeholders and `<img_end>`, then the planning token `<waypoint_ego>`. The LLM's last-layer hidden states at the placeholders pass through one linear layer (4096 → 256) and are regressed (cosine + L1) onto the **quantized code embeddings of a frozen VQGAN-ImageNet** (16,384 codes, 256-d, layer-normalized) for the **front camera** at 256×256, six future steps. Nothing is decoded to pixels.
3. **Two-stage trajectory decoding.**
   - *Coarse*: ORION's VAE + GRU generative planner turns the planning token into six command-conditioned trajectories. **The KL loss is removed**, and "during training, the ground-truth ego trajectory is concatenated with the planning embedding as the input to generate the distribution".
   - *Refined*: the GRU's motion states become queries in a two-layer transformer that cross-attends to the **LLM hidden states at the future placeholders** (not to the 256-d predicted codes). Two MLPs produce a base trajectory and an offset, which are summed.
4. **Loss**: $\mathcal L=\mathcal L_{vis}+\mathcal L_{plan}+\mathcal L_{plan\_r}+\mathcal L_{qt}+\mathcal L_{ce}$, with L2, boundary and collision terms on both trajectories. One end-to-end stage of 6 epochs on 32 H20 GPUs (ORION uses three stages of 6 epochs). No VQA data in this stage; the cross-entropy covers only the fixed placeholder tokens.

**Results**

| Benchmark | LVDrive | Notes |
|---|---:|---|
| Bench2Drive, 220 routes, base set (Think2Drive data) | **80.71 DS / 58.26 SR** | ORION 77.74 / 54.62; UniDrive-WM 79.31 / 56.42 in its table |
| Efficiency / Comfort | 155.77 / **14.34** | Comfort is the lowest of the 17 rows in its table that report it |
| Multi-ability mean | 54.90 | ORION 54.72; UniDrive-WM 59.23; MindDrive 56.94 |
| Open-loop L2 (3 s, 2 Hz) | 0.63 m | ORION 0.68; DriveMoE 0.38 |
| Inference (H20, batch 1) | **2.03 s** per frame | its planning-only base 0.93 s; a hypothetical autoregressive variant 36.62 s |

---

## Key Takeaways

- **80.71 DS, the top of its own table, below seven entries on the wiki's Bench2Drive ladder.** Five earlier methods above it are not in its table: LinkVLA 91.01, SparseDriveV2 89.15, DynVLA 88.34, AutoMoT 87.34 and SimLingo 85.07 (cited in its related work). SimLingo and AutoMoT train on PDM-Lite data and LinkVLA's protocol is undisclosed, so the like-for-like set is narrower. LVDrive states that it uses the official base set. [[sources/deepsight.md]] (86.23, same protocol) appeared about two weeks earlier.
- **The future-latent loss alone does nothing on Dev10; the gain arrives with the refiner.** Planning-only base 65.25 DS (4/10 routes); + future-latent loss 66.31 (3/10); + refiner reading the predicted future 82.39 (7/10). A one-stage variant, with the planning token itself as the cross-attention query, scores 60.43. The refined trajectory scores 82.39 and the same model's coarse trajectory 73.22.
- **The best target is a pixel-reconstruction codebook, not a semantic encoder.** With the refiner in place, the Dev10 scores by target are: VQGAN-ImageNet codes (256-d) 82.39; DINOv3-L (1024-d) 71.72; the model's own vision encoder 65.42, level with no target (65.25); Emu3's MoVQGAN codes (4-d) 59.91, below no target. The paper frames itself as "semantic, not pixel", and its winning target comes from a tokenizer trained only to reconstruct pixels. It beats DINOv3 by 10.7 DS. The same architecture with different targets spans 22 DS, so the refiner and placeholders alone are not the gain.
- **Its concurrent twin reaches the opposite conclusion.** [[sources/deepsight.md]] also predicts several future latents in one VLA forward pass on Bench2Drive and also ablates on Dev10. It finds DINOv3 far better than a VAE target (86.57 against 14.66 DS over five frames), and BEV better than the front view (86.57 against 77.77). The two papers differ in VLM, target view (front RGB vs BEV renders) and what "VAE" means.
- **The ability profile does not show broadly better scene understanding.** Compared with ORION, LVDrive is better on Merging (+14.7) and Traffic Sign (+5.1) and worse on Overtaking (−2.2), Emergency Brake (−6.7) and Give Way (−10.0). The means are 54.90 against 54.72, below UniDrive-WM (59.23) and MindDrive (56.94). The DS and SR lead comes from the two abilities with the most scenarios (18 and 16 of 44).
- **Give Way is two scenarios, ten routes.** Its Table 7 gives the scenario counts per ability. So Give Way scores move in steps of 10, which explains why every method scores 10, 20, …, 50 there. LVDrive's 20.00 is 2 of 10 routes. In its own failure figure the ambulance is visible in the rear camera input.
- **Slower than its base, not faster.** Table 6: 2.03 s per frame against 0.93 s for the base on an H20. The text calls this "an inference speed approximately twice that of $\mathcal M_{base}$". Pre-filled placeholders do avoid autoregressive decoding (a matched autoregressive variant: 36.62 s).

---

## Method

![[nips_intro_0.png|Figure 1: The comparison of different VLA paradigms. Standard VLA approaches, as depicted in (a), rely on sparse action supervision. Our LVDrive, illustrated in (c), performs the future visual and action representation learning jointly. Unlike VLA with the world modeling paradigm in (b), LVDrive predicts future scenes entirely in latent space, capturing rich semantic features without pixel-level reconstruction. The trajectory is then generated and explicitly refined through direct interaction with the learned future scene representations.]]

**Figure 1**: (a) a VLA supervised only by actions; (b) a VLA that also predicts future images (DriveVLA-W0, FSDrive, UniDrive-WM); (c) LVDrive: action, a future "vision latent" and an arrow from the latent to the action ("VA interaction").

![[nips_model_0.png|Figure 2: Overview of LVDrive. LVDrive is a VLA framework that unifies latent future scene representation learning and motion planning, with dense auxiliary supervision provided by a pre-trained vision backbone. Given multi-view images, the model encodes current and historical scene features and performs future-aware reasoning to predict both latent visual representations and motion features in a single forward pass. The trajectory is then generated through a two-stage decoding process: a coarse proposal is first decoded from the planning embedding, and then refined by explicitly conditioning on the learned latent future scene features to produce the final trajectory.]]

**Figure 2**: Vision space (left top): six cameras, a trainable vision encoder, multi-view features with detection, motion and traffic heads. Reasoning space (bottom): an instruction, a frozen LLM with trainable LoRA, and a text-guidance output. The output column holds the planning embedding (green) and future visual embeddings (yellow). Action space (right top): the generative planner gives a coarse trajectory and motion features, and the trajectory refiner gives the final trajectory. Future feature generation (right bottom): future frames pass through a frozen vision backbone into scene features, which supervise the latent visual space.

### Future latents (§3.2, A.1)

$$V_{t+j}=\mathrm{ViS}_\theta(H_{t+j}),\quad H_{t+j}\in\mathbb R^{256\times4096},\ V_{t+j}\in\mathbb R^{256\times256},\quad \mathcal L_{vis}=w_c\mathcal L_{cosine}+w_r\mathcal L_{reg}$$

$\mathrm{ViS}_\theta$ is one linear layer. Targets: VQGAN-ImageNet's post-quantization embeddings of the front image at 256×256 ($16\times16\times256$), layer-normalized, for six future steps. The step interval is not stated; the trajectory also has six steps at 2 Hz. Only the front view is predicted, "motivated by the fact that human drivers attend primarily to the forward perspective". The weights $w_c$, $w_r$ are not given.

### Two-stage decoding (§3.3, A.2)

$$z\sim\mathcal N(\mu,\sigma^2),\ (\mu,\sigma^2)=\mathrm{DiS}_\theta(H_p);\qquad s_{ego}=\mathrm{State}_\theta(z,H_p)\in\mathbb R^{6\times8192}$$

$$Q_{ego}=\mathrm{Proj}(s_{ego}),\ K_{fut},V_{fut}=\mathrm{Proj}(H_{t+1:t+F});\qquad s^*_{ego}=\mathrm{TrajRefiner}(Q_{ego},K_{fut},V_{fut})$$

- $\mathrm{DiS}$: three 1×1 convolutions (4096 → 2048), average pooling, one more convolution to $\mu$ and $\log\sigma$ with $C_z=32$. The ground-truth trajectory is concatenated to its input during training, and there is no KL term. What it receives at inference is not stated.
- $\mathrm{State}$: four GRU layers (1024), three MLPs to 4096, concatenated with the repeated planning embedding.
- Refiner: two transformer layers with cross-attention ($C_r=1024$). The keys and values are written as $F\times C_r$, so each future frame's 256 tokens appear to be reduced to one; how is not said. The output goes to base and offset MLPs.
- $K=6$ trajectories, one per Bench2Drive navigation command.

---

## Results

### Table 1: Bench2Drive closed loop (220 routes) and open loop, base training set

\* expert feature distillation. C/L camera/LiDAR. Avg. L2 over 3 s at 2 Hz.

| Method | Scheme | Mod. | DS | SR (%) | Efficiency | Comfort | Avg. L2 |
|---|---|---|---:|---:|---:|---:|---:|
| *Traditional end-to-end* | | | | | | | |
| TCP* | IL | C | 40.70 | 15.00 | 54.26 | 47.80 | 1.70 |
| VAD | IL | C | 42.35 | 15.00 | 157.94 | 46.01 | 0.91 |
| MomAD ⚠ | IL | C | 44.54 | 16.71 | 170.21 | 48.63 | 0.87 |
| UniAD-Base | IL | C | 45.81 | 16.36 | 129.21 | 43.58 | 0.73 |
| TCP-traj* | IL | C | 59.90 | 30.00 | 76.54 | 18.08 | 1.70 |
| DriveDPO | RL | C | 62.02 | 30.62 | 166.80 | 26.79 | – |
| ThinkTwice* | IL | C | 62.44 | 31.23 | 69.33 | 16.22 | 0.95 |
| DriveTransformer-Large | IL | C | 63.46 | 35.01 | 100.64 | 20.78 | 0.62 |
| DriveAdapter* | IL | C&L | 64.22 | 33.08 | 70.22 | 16.01 | 1.01 |
| DiffAD | IL | C | 67.92 | 38.64 | – | – | 1.55 |
| Raw2Drive | RL | C | 71.36 | 50.24 | 214.17 | 22.42 | – |
| *Vision-language-action* | | | | | | | |
| Drive π0 | IL | C | 60.45 | 30.00 | 168.41 | 14.88 | 0.56 |
| ReCogDrive ⚠ | RL | C | 71.36 | 45.45 | 138.18 | 17.45 | – |
| DriveMoE | IL | C | 74.22 | 48.64 | 175.96 | 15.31 | 0.38 |
| ORION | IL | C | 77.74 | 54.62 | 151.48 | 17.38 | 0.68 |
| MindDrive | RL | C | 78.04 | 55.09 | – | – | – |
| UniDrive-WM-AR | IL | C | 79.22 | 56.36 | 158.44 | 28.01 | 0.64 |
| UniDrive-WM-AR+Diff | IL | C | 79.31 | 56.42 | 158.65 | 27.93 | 0.63 |
| **LVDrive** | IL | C | **80.71** | **58.26** | 155.77 | 14.34 | 0.63 |

⚠ MomAD's row is the one [[concepts/bench2drive.md#momworld]] attributes to SparseDrive (MomAD's own: 47.91 / 18.11). ReCogDrive's row: see the [audit](#table-audit).

### Table 2: multi-ability (%)

| Method | Scheme | Merging | Overtaking | Emerg. Brake | Give Way | Traffic Sign | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| TCP* | IL | 16.18 | 20.00 | 20.00 | 10.00 | 6.99 | 14.63 |
| UniAD-Base | IL | 14.10 | 17.78 | 21.67 | 10.00 | 14.21 | 15.55 |
| VAD | IL | 8.11 | 24.44 | 18.64 | 20.00 | 19.15 | 18.07 |
| TCP-traj* | IL | 8.89 | 24.29 | 51.67 | 40.00 | 46.28 | 34.22 |
| ThinkTwice* | IL | 27.38 | 18.42 | 35.82 | 50.00 | 54.23 | 37.17 |
| DriveTransformer-Large | IL | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| DiffAD | IL | 30.00 | 35.55 | 46.66 | 40.00 | 46.32 | 38.79 ⚠ |
| DriveAdapter* | IL | 28.82 | 26.38 | 48.76 | 50.00 | 56.43 | 42.08 |
| Raw2Drive | RL | 43.35 | 51.11 | 60.00 | 50.00 | 62.26 | 53.34 |
| Drive π0 | IL | 29.35 | 36.58 | 48.83 | 40.00 | 54.45 | 41.84 |
| DriveMoE | IL | 34.67 | 40.00 | 65.45 | 40.00 | 59.44 | 47.91 |
| ReCogDrive | RL | 29.73 | 20.00 | 69.09 | 20.00 | 71.34 | 42.03 |
| ORION | IL | 25.00 | 71.11 | 78.33 | 30.00 | 69.15 | 54.72 |
| MindDrive | RL | 32.89 | 75.56 | 68.33 | 50.00 | 57.89 | 56.94 |
| UniDrive-WM-AR | IL | 28.81 | 74.04 | 79.84 | 40.00 | 71.30 | 59.00 ⚠ |
| UniDrive-WM-AR+Diff | IL | 29.97 | 74.64 | 79.98 | 40.00 | 71.54 | 59.23 |
| **LVDrive** | IL | **39.74** | 68.89 | 71.67 | 20.00 | **74.21** | 54.90 |

⚠ The printed means for DiffAD (38.79; the five values average 39.71) and UniDrive-WM-AR (59.00; 58.80) do not match their abilities. Every other row does, within 0.01.

### Table 7: scenarios per ability (appendix)

| | Traffic Sign | Merging | Emergency Brake | Overtaking | Give Way |
|---|---:|---:|---:|---:|---:|
| Scenarios | 18 | 16 | 12 | 9 | 2 |

Bench2Drive runs five routes per scenario, so Give Way is ten routes.

### Tables 3 and 5: components on Dev10 (10 routes)

| Variant | Latent future | Decoding | DS | SR |
|---|:-:|---|---:|---:|
| $\mathcal M_{base}$ (planning-only, as ORION's Stage 2) | – | coarse only | 65.25 | 4/10 |
| $\mathcal M_{vis}$ | ✓ | coarse only | 66.31 | 3/10 |
| $\mathcal M_{one}$ (planning token cross-attends to the future, then VAE) | ✓ | one-stage | 60.43 | 3/10 |
| $\mathcal M_{coarse}$ (full model, coarse output) | ✓ | two-stage, coarse read | 73.22 | 5/10 |
| **LVDrive** | ✓ | two-stage | **82.39** | **7/10** |

The text cites "Table 4" for the component ablation and "Table 6" for the two-stage one; they are Tables 3 and 5.

### Table 4: which frozen backbone supplies the target (Dev10)

| Variant | Target | Feature dim. | DS | SR |
|---|---|---:|---:|---:|
| $\mathcal M_{base}$ | none | – | 65.25 | 4/10 |
| $\mathcal M_1$ | its own vision encoder | 1024 | 65.42 | 4/10 |
| $\mathcal M_2$ | MoVQGAN (Emu3's tokenizer) | 4 | 59.91 | 3/10 |
| $\mathcal M_3$ | DINOv3-Large | 1024 | 71.72 | 5/10 |
| **LVDrive** | VQGAN-ImageNet | 256 | **82.39** | **7/10** |

The paper's explanations are capacity at both ends: 4-d codes carry too little, and DINOv3 is "excessively rich, potentially introducing redundant information as noise".

### Table 6: inference time (Bench2Drive validation, H20, batch 1)

| Variant | Time |
|---|---:|
| $\mathcal M_{base}$ | 0.93 s |
| $\mathcal M_{AR}$ (hypothetical: same visual and action tokens generated autoregressively) | 36.62 s |
| **LVDrive** | **2.03 s** |

---

## Qualitative Results

![[nips_vis_2.png|Figure 3: Qualitative results of our LVDrive and the base model in an Overtaking scenario from Bench2Drive. The ego vehicle encounters an accident ahead that blocks its driving lane, while a steady stream of oncoming traffic occupies the adjacent lane. The blue line denotes the generated trajectory. The ego vehicle controlled by the base model becomes immobilized at the accident site. In contrast, our LVDrive successfully and smoothly overtakes the accident cars with the safe planning.]]

**Figure 3**: Accident ahead, oncoming traffic beside. Both models stop (T1). LVDrive then plans a path into the opposite lane (T2) and passes (T3), while the base model stays stopped.

![[nips_vis_a3.png|Figure 4: Qualitative results of LVDrive and the base model in an Emergency Brake scenario. The ego vehicle is required to first perform an unprotected left turn at an intersection without a traffic light, then yield to a bicycle crossing its path. Upon encountering the crossing bicycle, the ego vehicle should execute an emergency brake, wait for the bicycle to clear the road, and subsequently resume driving. The base model fails to turn, resulting in the collision with the car at the intersection. In contrast, LVDrive successfully navigates the entire sequence.]]

**Figure 4**: Rain and fog. The base model collides with a car during the left turn and is then stuck; LVDrive turns, brakes for a crossing cyclist and resumes.

![[nips_vis_a2.png|Figure 5: Qualitative results of LVDrive and the base model in a Merging scenario. Navigating highway exits and merging onto narrow roads requires precise perception of the road layout and fine-grained trajectory planning. The base model fails to capture the exact road boundary, leading to a lateral deviation that results in a collision with the guardrail. In contrast, LVDrive accurately perceives the off-ramp geometry and plans a safe trajectory to merge into the narrow road without incident.]]

**Figure 5**: A highway exit onto a narrow road. The base model drifts laterally into the guardrail.

![[nips_vis_a4.png|Figure 6: Qualitative results of LVDrive and the base model in a Traffic Sign scenario. The ego vehicle is required to make a right turn at a non-signalized intersection without traffic lights or signs, while avoiding collisions with surrounding vehicles. The base model plans an inaccurate trajectory that deviates from the intended route and ultimately collides with a traffic sign. In contrast, LVDrive navigates through the junction without incident.]]

**Figure 6**: A right turn at an unsignalized junction. The base model leaves the route and hits a sign.

![[nips_vis_a1.png|Figure 7: Qualitative results of LVDrive and the base model in an Overtaking scenario. The ego vehicle drives forward in its lane, and the front car in the adjacent lane stops and opens the door, which blocks the driving lane. The base model stops and gets stuck in place. In contrast, LVDrive successfully plans the safe trajectory to bypass the front car with the open door and drives forward continuously.]]

**Figure 7**: At night, a parked car's door opens into the lane. Both models stop; the base model stays stuck, and LVDrive edges around the door.

![[nips_vis_a5.png|Figure 8: Failure case of LVDrive in a Give Way scenario. The ego vehicle is required to yield to the emergency vehicle that approaches from behind. LVDrive maintains the straight route and fails to yield to the ambulance.]]

**Figure 8**: An ambulance approaches from behind and is clearly visible, boxed, in the rear camera (bottom centre of the mosaic) for four frames, while LVDrive keeps lane-following at about 4.9 m/s. The paper blames the front-view prediction target for drawing attention forward. ORION, which has no such target, also scores 30 on Give Way, and the Think2Drive expert scores 90 ([[concepts/bench2drive.md#expert]]).

---

## Table Audit {#table-audit}

1. **Comparison scope.** Absent although earlier: LinkVLA (91.01; March 2026), SparseDriveV2 (89.15; March), DynVLA (88.34; March), AutoMoT (87.34; March, PDM-Lite), SimLingo (85.07–86.02; PDM-Lite; cited in its related work), AutoVLA (78.84; cited) and UniDriveVLA (78.37; April). DeepSight (86.23) appeared about two weeks before. "Outperforms all existing approaches" holds within the table.
2. **The MomAD row** (44.54 / 16.71 / 170.21 / 48.63 / 0.87) is the third ingested table to carry it under MomAD's name, after ORION's and DeepSight's. MomAD's own first author prints 47.91 / 18.11 for MomAD and 44.54 / 16.71 for SparseDrive ([[concepts/bench2drive.md#momworld]]).
3. **A ReCogDrive Bench2Drive row with its own sub-scores**: 71.36 DS, 45.45 SR, efficiency 138.18, comfort 17.45, abilities 29.73 / 20.00 / 69.09 / 20.00 / 71.34. [[sources/world4scorer.md]]'s table also prints "ReCogDrive 71.36", which the wiki noted is identical to Raw2Drive's DS. Here the two rows differ in every other column (Raw2Drive 50.24 / 214.17 / 22.42), so ReCogDrive's 71.36 is a separate result and not a copied Raw2Drive row. The wiki's ReCogDrive page (arXiv v1) records no Bench2Drive result.
4. **Two printed ability means do not match their rows** (DiffAD −0.92, UniDrive-WM-AR +0.20).
5. **Text–table contradictions**: "inference speed approximately twice that of $\mathcal M_{base}$" against 2.03 s vs 0.93 s; ablation table numbers in the text off by one.
6. **SR is not a whole number of routes out of 220** (58.26% would be 128.2 routes), and the same holds for ORION (54.62%) and Raw2Drive (50.24%). Bench2Drive's SR is probably not a plain fraction of 220 runs; no paper says how it is computed.

---

## Limitations

The paper's own: it does not study how language supervision could help the visual and action representations, and a driving-specific vision backbone might be a better target.

The wiki adds:

1. **Every mechanism result is on ten routes.** Dev10 SR moves in steps of one route; the components move DS by −5 to +17 with no repeats stated. [[sources/raw2drive.md]] averages three repeats on the same set. CARLA DS can vary by 9–12 between runs for weaker policies ([[concepts/evaluation-variance.md]]). The paper calls Dev10 "a diverse yet low-variance evaluation signal".
2. **No matched baseline on the full benchmark.** $\mathcal M_{base}$ is run only on Dev10. ORION's own planning-only result on 220 routes is 74.10 / 44.66 ([[sources/orion.md]], Table 6), so the full-set effect of LVDrive's additions is roughly +6.6 DS. That comparison is across papers and training schedules, and LVDrive also drops ORION's VQA co-training and KL term.
3. **The coarse planner's inference input is undefined.** The distribution generator is trained with the ground-truth trajectory concatenated to its input and no KL term tying it to an inference-time prior. As described, the coarse trajectory can be decoded from a latent that encodes the answer during training. What is fed at test time is not stated.
4. **The "semantic, not pixel" thesis is contradicted by its own best target** (a VQGAN code embedding), and the target ablation is one Dev10 run per backbone. It cannot separate the target's content from its dimension or entropy: VQGAN codes come from a 16,384-entry codebook, while DINOv3 features are continuous and four times wider.
5. **Front view only** (256×256), with six multi-view inputs. The paper's give-way explanation is plausible but untested; there is no surround-view arm.
6. **One closed-loop run on 220 routes**, and a lead of +1.40 DS / +1.84 SR over UniDrive-WM, inside the run-to-run spread recorded for CARLA.
7. **2 s per frame**, twice its base.
8. **The ability gains are concentrated**: merging and traffic sign up, three abilities down against ORION, and comfort the lowest reported in the table.

---

## Relation to Other Wiki Pages

- [[sources/raf-vla.md]] *(2026-10-19)*: the NAVSIM counterpart, four months later. Future-frame targets from a frozen encoder (Cosmos rather than VQGAN) are regressed from extra query tokens in a VLA and read by the action side at inference. It is +1.21 PDMS in SFT and survives GRPO, whereas LVDrive's loss alone was null on Dev10 until a refiner read it.
- [[sources/orion.md]]: the base, sharing an author. Same QT-Former, Vicuna and VAE + GRU planner, initialized from ORION's Stage-1 weights. ORION's VQA co-training (worth +3.64 DS in ORION's ablation) is replaced by future-latent regression and a refiner, and the KL loss is dropped. ORION found its VAE planner better than a 20-anchor diffusion planner (77.74 vs 71.97).
- [[sources/deepsight.md]]: the concurrent near-twin. Both run parallel future-latent prediction in one VLA forward pass on Bench2Drive's base set, with Dev10 ablations. DeepSight uses DINOv3 on BEV renders, five frames and MSE (86.23 DS, world model +26.4 on the full set); LVDrive uses VQGAN codes on the front view, six frames and cosine + L1 (80.71). On target choice they disagree: DeepSight's DINOv3 beats its VAE by 47–72 DS on Dev10, while LVDrive's VQGAN beats its DINOv3 by 10.7.
- [[sources/law.md]]: cited as the latent-world-model idea for conventional planners. LVDrive differs in reading the prediction at inference.
- [[sources/drivevla-w0.md]], [[sources/futuresightdrive.md]]: the pixel-prediction VLAs it argues against (dense supervision, but autoregressive and texture-level). [[sources/hyworldvla.md]] (later, NAVSIM) also finds a pixel-token target better than a VAE latent under one VLA.
- [[sources/wa-jepa.md]], [[sources/foredrive.md]]: the regression-objective results on NAVSIM. LVDrive regresses a low-dimensional quantized target and finds the wider DINOv3 target worse, the same direction as the entropy reading of objective form ([[concepts/world-model-for-ad.md#objective-form]]).
- [[sources/raw2drive.md]], [[sources/think2drive.md]]: Dev10's other user, and the expert behind the base set LVDrive trains on (91.85 DS; LVDrive is 11.1 below it).
- [[sources/recogdrive.md]], [[sources/momworld.md]], [[sources/world4scorer.md]]: rows its table carries ([audit](#table-audit)).
- [[sources/dynvla.md]], [[sources/linkvla.md]], [[sources/automot.md]], [[sources/sparsedrivev2.md]], [[sources/autovla.md]], [[sources/unidrivevla.md]]: earlier Bench2Drive entries absent from its table.
- [[sources/driveva.md]], [[sources/hermes.md]], [[sources/epona.md]]: cited world models; DriveVA shares three Xiaomi authors.
- Not ingested: **UniDrive-WM** (autoregressive future images in a VLA, 79.31; its closest comparison), **MindDrive** (ORION's group, online RL, 78.04), **DriveMoE / Drive π0**, **DriveWorld-VLA**, **World4Drive**, **FutureX**, **Latent chain-of-thought world modeling** (NVIDIA), **DiffAD**, **DriveDPO**, **Doe-1**, **VQGAN** (Esser et al.), **GenAD** (the VAE planner's origin).

---

## Open Questions

- **Is it the target's content or its size?** Regress DINOv3 features projected to 256 dimensions, or VQGAN codes widened to 1,024, under the same refiner.
- **What does the coarse planner receive at inference?** And does restoring ORION's KL term change the coarse or refined scores?
- **Does the full-benchmark gain hold against a matched planning-only baseline**, run on all 220 routes with the same schedule?
- **Would a surround-view or BEV target fix give way?** DeepSight's BEV arm (+8.8 DS over front view on Dev10) suggests yes; give way is ten routes, so it needs repeats to show.
- **Is the refiner reading a future, or just extra capacity attached to the LLM?** The internal-encoder target (65.42, with the same refiner) suggests the content matters. A current-frame VQGAN target would separate prediction from representation.
