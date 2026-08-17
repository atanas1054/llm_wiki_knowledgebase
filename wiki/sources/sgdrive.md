---
title: "SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving"
type: source-summary
sources: [raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/perception-for-planning.md, concepts/intent-conditioned-planning.md, concepts/diffusion-planner.md, concepts/foundation-backbones-for-ad.md, concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, sources/recogdrive.md, sources/simwam.md, sources/percept-wam.md, sources/unidrivevla.md, sources/latent-wam.md, sources/deepsight.md, sources/orion.md, sources/futuresightdrive.md, sources/drivewam.md]
created: 2026-08-17
updated: 2026-08-17
confidence: high
---

**Paper**: SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving
**Authors**: Jingyu Li, Junjie Wu, Dongnan Hu, Xiangkai Huang, Bin Sun, Zhihui Hao, Xianpeng Lang, Xiatian Zhu, Li Zhang
**Orgs**: Shanghai Innovation Institute + Tongji University + Li Auto + University of Surrey + Fudan University
**arXiv**: 2601.05640v2 (CVPR'26)
**Code**: https://github.com/LogosRoboticsGroup/SGDrive

---

## Summary

SGDrive's thesis is that generalist VLMs lack *driving-specific* spatial-temporal structure, and that the fix is to impose an explicit **scene → agent → goal** knowledge hierarchy on the VLM's representation rather than to add a generative world model. A set of learnable **⟨world⟩ queries** is appended to the VLM's multimodal token stream and trained to decode three complementary kinds of knowledge: **scene geometry** (occupancy, supervised via a VAE decoder), **safety-critical agents** (DETR-style detection restricted to agents that can actually affect the ego), and a **short-term driving goal** (the ego pose ~4 s ahead). Each is predicted at both the current time $t$ and a future time $t{+}n$, so the hierarchy is anticipatory rather than purely perceptual.

Two design details carry most of the weight. A **block-wise structured attention mask** forbids attention *between* the scene/agent/goal subquery blocks while permitting temporal attention within each block and free cross-attention to the visual and text tokens — preventing representational contamination across cognitive levels. And the ⟨world⟩ query hidden states are fed **directly** to a DiT diffusion planner as its conditioning latent, so no explicit decoding of occupancy or boxes is needed at inference.

Results on NAVSIM: **87.4 PDMS with SFT only** (InternVL3-**2B**, front camera only), beating ReCogDrive-8B's 86.8 with a quarter of the parameters, and **91.1 PDMS after RFT** under ReCogDrive's own RL configuration. On NAVSIM-v2, **86.2 EPDMS**. The best sub-metrics throughout are the collision-related ones (NC, TTC), which the paper argues validates the anticipatory hierarchy.

---

## Core Idea: Structured World Knowledge, Not Generated Futures

![[intro_fig1.png|Three paradigms for VLM-based driving: text actions, action embeddings, and SGDrive's structured world knowledge]]

**Figure 1**: (a) VLMs that emit driving actions directly as text. (b) VLMs that emit action embeddings decoded into a trajectory. (c) SGDrive explicitly learns and forecasts scene, agent, and goal knowledge, providing structured driving-world understanding that strengthens action reasoning and generalization.

The paper's diagnosis of VLM-based planners lists three deficits: **no spatial perception** (VLMs are trained for semantics, not geometry), **no discrimination of critical information** (attending to the whole scene rather than the agents that matter), and **no future world-state forecast**. SGDrive addresses all three with one mechanism — supervised query tokens — rather than by attaching a video generator.

This is a meaningfully different position from the video-prior world-action line ([[sources/drivewam.md]], [[sources/simwam.md]], DriveVA): those transfer *appearance dynamics* from a pretrained video model, while SGDrive supervises *structured symbolic state* (occupancy voxels, 3D boxes, a goal pose) and never models pixels at all.

---

## Method

![[pipeline1.png|SGDrive pipeline: hierarchical world queries, VLM fusion, and DiT trajectory generation]]

**Figure 2**: Hierarchical ⟨world⟩ queries (scene, agent, goal) for world modeling and trajectory generation. The **world query encoder** initializes these queries by integrating multi-modal priors from the ego state, historical trajectory, and visual features; these prior-informed queries are then processed by the VLM alongside text and visual embeddings, fusing all signals into a compact hierarchical world representation.

### Formulation

At time $t$ the model receives a language instruction $L_{ins}$, ego state $S_{ego}$, and camera input $I_{cam}$:

$$O_{\text{world}}=\mathrm{VLM}\bigl(I_{cam},\,L_{ins},\,S_{ego}\,\vert\,\langle\text{world}\rangle\bigr)$$

Hierarchical world heads $\mathcal{D}$ then decode structured knowledge at both current and future steps:

$$w=\mathcal{D}\bigl(O_{world}\bigr)=\{w^{t,t+n}_{geo},\,w^{t,t+n}_{agt},\,w_{goal}\}$$

Note the asymmetry: scene geometry and agents are predicted at $t$ *and* $t{+}n$; the goal is a single future pose and has no current-time counterpart.

### 1. Geometric scene layout

Deliberately **geometry without semantics** — the model predicts occupancy structure, not semantic class distributions, "removing redundant semantic dependencies." Occupancy ground truth comes from dataset annotations where available, otherwise generated from point clouds. The VLM output $W_{\text{geo}}$ is treated as a latent embedding and reconstructed by a standard VAE decoder. Because driving occupancy is highly sparse, a resampling strategy pairs a dense cross-entropy term with a resampled binary term:

$$\mathcal{L}_{\text{geo}}^{t,t+n}=\frac{1}{M}\sum_{i=1}^{M}\mathrm{CE}(o_{i}^{t,t+n},\hat{o}_{i}^{t,t+n})+\frac{1}{N}\sum_{j=1}^{N}\mathrm{BCE}(p_{j}^{t,t+n},\hat{p}_{j}^{t,t+n})$$

### 2. Safety-critical agent detection

Rather than detecting everything visible, SGDrive selects target agents (vehicle, pedestrian, cyclist) by **ego-trajectory relevance and front-camera frustum visibility**, forcing finite representational capacity onto the agents that can actually influence the ego decision. Supervision is the DETR set-based loss with bipartite matching $\hat\sigma$, predicting 3D states at $t$ and $t{+}n$:

$$\mathcal{L}_{\text{agent}}^{t,t+n}=\sum_{i=1}^{N_{q}}\bigl[\lambda_{\text{cls}}\mathcal{L}_{\text{cls}}(\hat{c}_{i},c_{\hat{\sigma}(i)})+\mathbf{1}_{c_{\hat{\sigma}(i)}\neq\emptyset}\mathcal{L}_{\text{reg}}(\hat{b}_{i},b_{\hat{\sigma}(i)})\bigr]$$

with $\lambda_{\text{cls}}=10$ and $L_1$ regression.

### 3. Short-term driving goal

The ego pose ~4 s ahead, decoded by a lightweight MLP head under an $L_1$ loss:

$$\mathcal{L}_{\text{goal}}=\lVert\hat{p}_{\text{goal}}-p_{\text{goal}}\rVert_{1}$$

Importantly, goal reasoning is **not** conditioned on the scene or agent representations — it "emerges implicitly from a holistic understanding of the scene and task instructions," which is what the structured mask enforces. The paper frames this as disentangling high-level decision-making from low-level trajectory planning.

### Block-wise structured attention mask

![[attn-mask1.png|Causal attention versus structured block-wise attention mask]]

**Figure 3**: (a) Causal mask — input tokens attend to all preceding tokens. (b) Structure mask — prohibits all mutual attention between the different subquery sets (scene, agent, goal).

The ⟨world⟩ queries are split into **five subqueries**: three encoding current-world knowledge and two forecasting future states. The mask blocks attention across knowledge categories while allowing temporal attention within a category (so a future-state subquery can see its own current-state counterpart), and all subqueries retain free cross-attention to the visual and text embeddings. The stated failure mode being prevented is **representational contamination** — leakage across cognitive levels degrading each specialized representation.

### Diffusion planner

The ⟨world⟩ query hidden states serve **directly** as the DiT's conditioning latent, avoiding "intermediate, lossy representations." Notably, $\mathbf{A}_T$ is not pure Gaussian noise: it is initialized by adding noise to a **learned prior** projected from the ⟨world⟩ queries and the historical ego trajectory, grounding denoising in the VLM's world understanding. Standard $L_2$ objective:

$$\mathcal{L}_{\text{diff}}=\mathbb{E}_{t,\mathbf{A}_{0},\epsilon}\bigl[\lVert\epsilon-\epsilon_{\theta}(\mathbf{A}_{t},t,c)\rVert_{2}^{2}\bigr]$$

### Two-stage training

**Stage 1 (SFT)** trains the VLM for VQA plus all three world-knowledge heads:

$$\mathcal{L}_{\text{Stage1}}=\mathcal{L}_{\text{text}}+\mathcal{L}_{\text{occ}}^{t,t+n}+\lambda_{\text{agent}}\mathcal{L}_{\text{agent}}^{t,t+n}+\mathcal{L}_{\text{goal}},\qquad \lambda_{\text{agent}}=0.1$$

**Stage 2** freezes the VLM entirely — using it as a "high-fidelity world model" — and trains only the diffusion planner on $\mathcal{L}_{\text{diff}}$.

---

## Results

### NAVSIM v1 navtest (Table 1)

† = fine-tuned on the NAVSIM trajectory dataset. Bold is best within SFT and RFT settings respectively.

| Method | Image | LiDAR | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|:-:|:-:|---:|---:|---:|---:|---:|---:|
| Constant Velocity | | | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP | | | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| VADv2-𝒱8192 | ✓ | | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| Hydra-MDP-𝒱8192 | ✓ | ✓ | 97.9 | 91.7 | 92.9 | 100 | 77.6 | 83.0 |
| UniAD | ✓ | | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| LTF | ✓ | | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 |
| BevDrive | ✓ | ✓ | 97.7 | 92.5 | 92.9 | 100 | 78.7 | 83.8 |
| TransFuser | ✓ | ✓ | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | ✓ | | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA | ✓ | ✓ | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Epona | ✓ | | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| Hydra-MDP-𝒱8192-W-EP | ✓ | ✓ | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| ARTEMIS | ✓ | ✓ | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 |
| DiffusionDrive | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| SeerDrive | ✓ | ✓ | 98.4 | 97.0 | 94.9 | 99.9 | 83.2 | 88.9 |
| *VLM-based (SFT)* | | | | | | | | |
| AutoVLA-3B | ✓ | | 96.9 | 92.4 | 88.1 | 99.1 | 75.8 | 80.5 |
| QwenVL2.5-8B† | ✓ | | 97.8 | 92.1 | 92.8 | 100 | 78.3 | 83.3 |
| InternVL3-8B† | ✓ | | 97.0 | 92.4 | 91.8 | 100 | 78.9 | 83.3 |
| ReCogDrive-2B | ✓ | | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| ReCogDrive-8B | ✓ | | 98.3 | 95.1 | 94.3 | 100 | 81.1 | 86.8 |
| **SGDrive-2B (SFT)** | ✓ | | **98.6** | 95.1 | **95.4** | 100 | 81.2 | **87.4** |
| *VLM-based (RFT)* | | | | | | | | |
| AutoVLA-3B | ✓ | | 98.4 | 95.6 | **98.0** | 99.9 | 81.9 | 89.1 |
| ReCogDrive-2B | ✓ | | 97.9 | 97.3 | 94.9 | 100 | **87.3** | 90.8 |
| ReCogDrive-8B | ✓ | | 97.8 | 97.7 | 94.9 | 100 | 86.3 | 90.5 |
| **SGDrive-2B (RFT)** | ✓ | | **98.6** | **97.8** | 96.2 | 100 | 85.8 | **91.1** |

Three claims the table supports cleanly: a 2B model beats **InternVL3-8B and QwenVL2.5-8B by 4.1 PDMS** (so the gain is architectural, not scale); it beats **ReCogDrive-8B by 0.6** at a quarter the size; and camera-only SGDrive beats most camera+LiDAR end-to-end methods. Best NC and TTC in both the SFT and RFT blocks — the collision metrics the hierarchy targets.

### NAVSIM v2 navtest, extended metrics (Table 2)

SGDrive-2B evaluated with the two-stage SFT strategy.

| Method | NC ↑ | DAC ↑ | EP ↑ | TTC ↑ | HC ↑ | TL ↑ | DDC ↑ | LK ↑ | EC ↑ | EPDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfuser | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 99.9 | 98.3 | 67.6 | 95.3 | 77.8 |
| VADv2 | 97.3 | 91.7 | 77.6 | 92.7 | 100 | 99.9 | 98.2 | 66.0 | 97.4 | 76.6 |
| Hydra-MDP | 97.5 | 96.3 | 80.1 | 93.0 | 100 | 99.9 | 98.3 | 65.5 | 97.4 | 79.8 |
| Hydra-MDP++ | 97.9 | 96.5 | 79.2 | 93.4 | 100 | 100.0 | 98.9 | 67.2 | 97.7 | 80.6 |
| ARTEMIS | 98.3 | 95.1 | 81.5 | 97.4 | 100 | 99.8 | 98.6 | 96.5 | 98.3 | 83.1 |
| ReCogDrive-8B | 98.3 | 95.2 | 87.1 | 97.5 | 98.3 | 99.8 | 99.5 | 96.6 | 86.5 | 83.6 |
| DiffusionDrive | 98.0 | 96.0 | 87.7 | 97.1 | 98.3 | 99.8 | 99.5 | 97.2 | 87.6 | 84.3 |
| **SGDrive-2B** | **98.6** | 94.3 | 86.0 | **97.9** | 98.3 | 99.9 | 99.5 | 96.1 | 85.9 | **86.2** |

Best NC and TTC again, +2.6 EPDMS over ReCogDrive-8B — but **DAC 94.3 and EC 85.9 are the weakest in the table** among the modern entries, and the comparison set stops well short of the wiki's v2 field (see Limitations).

![[comparison_on_diff_model1.png|Qualitative comparison against ReCogDrive on navtest]]

**Figure 4**: Two navtest scenarios versus ReCogDrive. In a multi-agent scene, ReCogDrive's trajectory deviates toward a potential collision while SGDrive — with explicit safety-critical agent detection — stays collision-free; on a curved road, ReCogDrive drifts out of lane into a roadside barrier while SGDrive tracks the geometric layout.

---

## Ablations

### Driving-world knowledge forecast, Stage 1 only (Table 3)

Trajectories produced in **text form** here, isolating the VLM representation from the diffusion planner.

| Exp. | Base | Current | Future | NC ↑ | DAC ↑ | TTC ↑ | EP ↑ | PDMS ↑ |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|
| a | ✓ | ✗ | ✗ | 97.3 | 91.1 | 92.9 | 76.8 | 82.2 |
| b | ✓ | ✓ | ✗ | 98.3 | 93.0 | 94.9 | 78.2 | 84.7 |
| c | ✓ | ✓ | ✓ | 98.4 | 93.6 | 94.9 | 79.3 | 85.5 |

Current-state hierarchy is worth **+2.5 PDMS**; adding future forecasting a further **+0.8**. The bulk of the gain is structured *perception*, not anticipation — a useful counterweight to the paper's world-model framing.

### World subquery ablation, Stage 2 planner (Table 4)

| Exp. | Scene | Agent | Goal | Future | NC ↑ | DAC ↑ | TTC ↑ | EP ↑ | PDMS ↑ |
|---|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|
| a | ✓ | ✗ | ✗ | ✗ | 98.2 | 94.1 | 94.4 | 80.2 | 86.0 |
| b | ✓ | ✓ | ✗ | ✗ | 98.3 | 94.5 | 94.8 | 80.4 | 86.3 |
| c | ✓ | ✓ | ✓ | ✗ | 98.5 | 94.9 | 95.1 | 81.2 | 87.0 |
| d | ✓ | ✓ | ✓ | ✓ | 98.6 | 95.1 | 95.4 | 81.2 | 87.4 |

Each level contributes, and the *shape* of each contribution matches its intent: agents mainly improve NC/DAC (+0.3 PDMS), the **goal mainly improves EP** (80.4 → 81.2, +0.7 PDMS — the largest single jump), and future forecasting mainly improves NC/TTC (+0.4). The goal query is doing exactly the efficiency job the paper claims for it.

### Structured versus causal attention (Table 5)

| Method | NC ↑ | TTC ↑ | EP ↑ | PDMS ↑ |
|---|---:|---:|---:|---:|
| Causal | 98.4 | **95.6** | 80.1 | 87.1 |
| Structure | **98.6** | 95.4 | **81.2** | **87.4** |

Only +0.3 PDMS, and the gain is **entirely in EP** (+1.1) while TTC slightly regresses. The paper's explanation is behavioral: causal attention leaks cross-category noise, corrupting the representations and making the vehicle "overly conservative (e.g., slowing excessively to avoid potential collisions)." So the mask's real effect is removing excess caution, not improving safety.

### Hidden-state fusion in the diffusion planner (Table 6)

| Exp. | Strategy | NC ↑ | TTC ↑ | EP ↑ | PDMS ↑ |
|---|---|---:|---:|---:|---:|
| a | Inject subqueries incrementally across successive cross-attention layers | 98.2 | 95.0 | 80.6 | 87.1 |
| b | Distinct cross-attention layers per subquery | 98.1 | 95.1 | 79.7 | 86.9 |
| c | **Concatenate all subquery states, interact at every layer** | **98.6** | **95.4** | **81.2** | **87.4** |

A 0.5 PDMS spread — the paper's own reading is that all three work, confirming the subqueries carry rich information regardless of routing.

---

## Qualitative

![[comparison_prediction_gt1.png|Predicted hierarchical world states versus ground truth]]

**Figure 5**: Model predictions (top) versus ground truth (bottom) across the scene-agent-goal hierarchy, showing close alignment for both current state and short-horizon future evolution.

![[Adaptive_occ1.png|Ego-motion based adaptive geometric scene perception]]

**Figure 6**: SGDrive adapts its perceptual focus to ego motion and navigation command — expanding the perceptual horizon at high speed, and redirecting attention toward the turning direction during maneuvers. The paper offers this as evidence that the VLM's world-modeling ability is genuinely being elicited rather than the heads merely memorizing.

![[supp_gostraight1.png|Additional navtest qualitative results, straight driving]]

**Figure 7**: Additional navtest results for straight-driving scenarios.

![[supp_turn1.png|Additional navtest qualitative results, turning]]

**Figure 8**: Additional navtest results for turning scenarios.

![[supp_fail1.png|Representative failure cases]]

**Figure 9**: Failure cases. With only a single front-view image, the model deviates under **extreme turning**, where the absent viewpoints make long-horizon prediction difficult and can produce lane-change errors. Multi-view input is flagged as future work.

---

## Implementation Details

- **Backbone**: InternVL3-2B — 300M InternViT visual encoder + Qwen2.5 LLM. Front camera only, multi-frame.
- **Stage 1**: domain adaptation on **3.1M QA pairs** (perception/prediction/planning, following ReCogDrive) for 1 epoch, then 3 epochs on 85k trajectory-specific QA pairs while training the world heads. $\lambda_{\text{agent}}=0.1$.
- **Stage 2**: VLM frozen; diffusion planner trained alone for **220 epochs**.
- **Hardware**: 4 nodes × 8 NVIDIA H20 = 32 GPUs.
- **RFT**: same RL configuration as ReCogDrive (diffusion GRPO against NAVSIM PDM reward).
- **Data**: NAVSIM navtrain (1,192 scenarios) / navtest (136 scenarios).

---

## Limitations

1. **The SOTA claim is scoped to camera-only VLM methods in its own table.** SGDrive's 91.1 RFT PDMS is below the wiki's frontier: CLEAR (93.7), DriveSuprim (93.5), Drive-JEPA (93.3), HybridDriveVLA (92.1), DynVLA (91.7), [[sources/simwam.md]] (91.5), FLARE (91.4), DiffusionDriveV2 (91.2) — most of them camera-only. SimWAM's table lists SGDrive at 91.1 and beats it by 0.4, which is the one direct external check available.
2. **The NAVSIM-v2 claim is the weaker of the two.** 86.2 EPDMS is compared against only seven baselines, the newest being DiffusionDrive and ReCogDrive-8B. In the wiki it sits below WAM-Diff (89.7), Latent-WAM (89.3), ExploreVLA (88.8), DriveDreamer-Policy (88.7), HAD (88.6), CLEAR (88.6), Drive-JEPA (87.8), DreamerAD (87.7), DriveSuprim/ELF-VLA (87.1), and Vega (86.9) — roughly mid-table, not state of the art.
3. **DAC and EC are the cost.** On v2, DAC 94.3 and EC 85.9 are the lowest among modern entries in its own table (DiffusionDrive: 96.0 / 87.6). The hierarchy buys collision avoidance and progress at some expense of drivable-area compliance and extended comfort.
4. **Occupancy supervision is a real annotation dependency.** Scene geometry needs occupancy labels or LiDAR point clouds to derive them, and agent supervision needs 3D boxes. So although SGDrive is *camera-only at inference*, it is not annotation-free at training — unlike FLARE's DINOv2 features or SimWAM's raw-video objective. The comparison against camera-only methods should account for this.
5. **The structured mask contributes only +0.3 PDMS**, and its gain is entirely EP while TTC regresses slightly. It is presented as a core contribution but is the smallest effect measured, with no seed variance reported.
6. **Stage 2 freezes the VLM**, so the planner cannot correct representation errors and the world heads receive no gradient from trajectory quality. The paper does not test joint or unfrozen fine-tuning.
7. **A minor cross-paper discrepancy**: SGDrive reports DiffusionDrive at 84.3 EPDMS; the wiki records 84.5 from other sources. Small, likely a scorer-version difference, but it means v2 numbers are not perfectly commensurate across papers.
8. **Front-camera-only failure mode is acknowledged**: extreme turns produce lane-change errors because the relevant viewpoints are simply absent (Figure 9).
9. **Compute is substantial** for a "compact" model — 32 H20 GPUs, 3.1M QA pairs of domain adaptation, and 220 planner epochs. The 2B parameter count understates the training cost.
10. **No latency numbers**, despite the argument that feeding ⟨world⟩ hidden states directly to the planner "reduces inference overhead." The claim is structural, not measured.

---

## Key Cross-References

- **A non-generative world model**: [[concepts/world-model-for-ad.md]] — SGDrive forecasts *structured symbolic state* (occupancy, agent boxes, goal pose) rather than pixels, latents, or features, making it the clearest counterpoint to the video-prior WAM line ([[sources/simwam.md]], [[sources/drivewam.md]], DriveVA).
- **Perception as planning supervision**: [[concepts/perception-for-planning.md]] — the ego-relevance filter on detection targets is a sharper version of the sparse-query perception in [[sources/unidrivevla.md]] and [[sources/percept-wam.md]].
- **Goal as intent**: [[concepts/intent-conditioned-planning.md]] — a continuous 4 s goal pose is a different intent representation from DIAL's discrete labels or PaIR-Drive's intention tokens, and it is the component that most improves Ego Progress.
- **RL comparability**: [[concepts/rl-for-ad.md]] and [[sources/recogdrive.md]] — SGDrive reuses ReCogDrive's RL configuration exactly, making its +3.7 PDMS SFT→RFT gain unusually comparable across the two papers.
- **Backbone efficiency**: [[concepts/foundation-backbones-for-ad.md]] — InternVL3-2B beating InternVL3-8B by 4.1 PDMS is direct evidence that driving-specific structure beats backbone scale.
