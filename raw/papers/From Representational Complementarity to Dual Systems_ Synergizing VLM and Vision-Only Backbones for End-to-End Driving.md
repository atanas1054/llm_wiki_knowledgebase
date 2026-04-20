---
title: "From Representational Complementarity to Dual Systems: Synergizing VLM and Vision-Only Backbones for End-to-End Driving"
source: "https://arxiv.org/html/2602.10719v1"
author:
published:
created: 2026-04-19
description:
tags:
  - "clippings"
---
Sining Ang    Yuguang Yang    Chenxu Dang    Canyu Chen    Cheng Chi    Haiyan Liu    Xuanyao Mao    Jason Bao    Xuliang    Bingchuan Sun    Yan Wang

###### Abstract

Vision-Language-Action (VLA) driving augments end-to-end (E2E) planning with language-enabled backbones, yet it remains unclear what changes beyond the usual accuracy–cost trade-off. We revisit this question with 3–RQ analysis in RecogDrive by instantiating the system with a full VLM and vision-only backbones, all under an identical diffusion Transformer planner. RQ1: At the backbone level, the VLM can introduce additional subspaces upon the vision-only backbones. RQ2: This unique subspace leads to a different behavioral in some long-tail scenario: the VLM tends to be more aggressive whereas ViT is more conservative, and each decisively wins on about 2–3% of test scenarios; With an oracle that selects, per scenario, the better trajectory between the VLM and ViT branches, we obtain an upper bound of 93.58 PDMS. RQ3: To fully harness this observation, we propose HybridDriveVLA, which runs both ViT and VLM branches and selects between their endpoint trajectories using a learned scorer, improving PDMS to 92.10. Finally, DualDriveVLA implements a practical fast–slow policy: it runs ViT by default and invokes the VLM only when the scorer’s confidence falls below a threshold; calling the VLM on 15% of scenarios achieves 91.00 PDMS while improving throughput by 3.2 $\times$. Code will be released.

Machine Learning, ICML

![[aligned_feature_kde.png|Refer to caption]]

Figure 1: Shared vs. model-specific representation geometry after alignment. Each model’s features are aligned to the ResNet-101 feature space (used as the reference) via (orthogonal) Procrustes alignment 22, then visualized using a 2D PCA projection fitted on the concatenation of all aligned features. For each model, the curves are KDE probability-mass contours enclosing the smallest regions containing 60/70/80% of the KDE mass; the shaded region is a robust 95% covariance ellipse (MinCovDet), and the star denotes the robust mean. Vision-only backbones form a tightly overlapping cluster, while the VLM shares a substantial core with them yet also occupies additional regions, indicating a mixture of shared and model-specific subspaces rather than strict containment.

![[cka_backbone_vs_dit.png|Refer to caption]]

Figure 2: Backbone vs. DiT representation similarity measured by CKA. We compute pairwise linear CKA between feature representations from different model branches and visualize the resulting similarity matrices for (left) the visual backbone features and (right) the DiT features. While backbone representations show high agreement among vision-only encoders (e.g., ViT/ResNet/EVA-CLIP) but low similarity to the VLM branch, the DiT representations become markedly more aligned across all branches, with substantially increased VLM-to-vision similarity (e.g., ∼ \\sim 0.21–0.22 → \\rightarrow 0.50–0.54), indicating that downstream planning modules compress heterogeneous visual representations into a more shared decision-relevant space.

## 1 Introduction

End-to-end driving planning maps multi-view observations directly to future trajectories, reducing reliance on hand-crafted modules. In this setting, the visual backbone impacts not only representation quality but also latency, memory, and deployment cost. Large vision–language models (VLMs) promise stronger open-world semantics and long-tail reasoning, yet their performance–cost trade-off against standard vision backbones (ViT/ResNet/EVA) is still unclear on in-distribution planning benchmarks such as NAVSIM test (PDMS): do backbone advantages survive policy learning and yield measurable planning gains, and if benefits are long-tail, can we invoke VLMs only when needed?

We study RecogDrive with a full VLM (InternVL-2B) and vision-only variants that differ only in the visual backbone, and we connect representation analysis to system design via three research questions.

RQ1 (Representation). How similar are VLM and vision-only representations, and where do differences matter—at the backbone feature or after the diffusion Transformer policy? Using linear CKA, a scale-invariant similarity measure that compares the pairwise geometry of two representation spaces (i.e., how samples relate to each other), we quantify alignment without requiring a one-to-one correspondence between feature dimensions. We find substantially stronger alignment at the policy-level than at the backbone level (about $0.22\rightarrow 0.54$; Fig. 2), suggesting policy learning compresses heterogeneous visual signals into a more homogeneous decision space. To localize where this residual mismatch comes from beyond a single CKA scalar, we further align the two spaces with a Procrustes transform and analyze the shared versus model-specific variance under PCA. A complementary Procrustes+PCA view further shows a large shared core alongside model-specific regions for each backbone (Fig. 1). These results help explain why replacing an expensive VLM with a cheaper ViT yields only moderate PDMS drops on NAVSIM test (IL $86.27\rightarrow 85.56$; RL $90.80\rightarrow 88.88$).

These observations motivate *per-scenario gating*: at test time we choose between a strong but expensive VLM-based policy and a cheaper vision-only policy for each scenario, invoking the VLM only when the vision-only branch is likely to fail. However, across a range of gating strategies using representation/alignment statistics (e.g., CCA and Shared–Unique SAE features), gains are marginal (best PDMS $90.80\rightarrow 90.96$), indicating that global alignment signals poorly predict scenario-level winners and motivating a shift to behavioral/trajectory-level cues.

RQ2 (Behavior). Do these representational differences translate into *scenario-dependent strengths*, i.e., are there consistent subsets of scenarios where VLM or vision-only policies win? We find complementarity effect on a small long tail rather than a containment relation: under $\Delta\mathrm{PDMS}>20\%$, each side strongly outperforms the other on roughly $2$ – $3\%$ of test scenarios (about $1.5\%$ under $\Delta\mathrm{PDMS}>50\%$). These differences also reflect distinct driving styles along both longitudinal (e.g., speed/aggressiveness) and lateral (e.g., path choice and merging timing) axes, suggesting exploitable decision diversity beyond average score.

RQ3 (System). How can complementarity be converted into deployable gains in both performance and cost? We propose two frameworks. *HybridDriveVLA* pools VLM and vision-only trajectories (plus interpolations) and ranks them with a scoring module, improving NAVSIM test PDMS from $90.80$ to $92.10$ without changing policy training. *DualDriveVLA* is a fast–slow deployment that runs the low-cost ViT first and triggers the VLM only for low-score cases; calling the slow system in only $\sim$ 15% of scenarios improves PDMS to $91.00$ (vs. $90.80$) while increasing end-to-end throughput by $3.2\times$.

## 2 Related Work

VLMs for autonomous driving. Prior work leverages VLMs either in *dual-system* pipelines that generate high-level commands/waypoints for downstream planners [^25] [^11], or in *single-system* formulations that cast planning as language generation, often with prompting for interpretability [^27] [^20] [^1] [^23] [^34] [^19] [^35] [^10] [^28] [^7]. We follow the dual-system spirit in using VLMs as a complementary signal, but focus on *understanding and exploiting* the differences between a VLM branch and vision-only backbones (e.g., ViT/ResNet/EVA) within the same planning architecture, turning their complementarity into practical trajectory selection and efficient deployment.

Diagnosing VLM/VLA driving stacks. Most analyses emphasize interpretability or benchmarked reasoning (e.g., explanation-centric datasets and evaluation protocols) [^5] [^21] [^26] [^24] [^32] [^33]. In contrast, we study a more mechanistic question inside a unified planning stack: how a VLM branch and alternative vision backbones differ in *internal representations*, how these differences are transformed (often compressed) after policy learning, and when they translate into actionable scenario-level gains. To our knowledge, this work is among the first to systematically analyze, within a unified VLA-style planning architecture, the representational discrepancy between a VLM branch and alternative visual backbones and how such discrepancy is transformed (and often compressed) at the decision level.

Overall, our work complements prior VLM-for-driving systems by providing an analysis-to-mechanism pipeline: from representation isomorphism (RQ 1), to behavioral complementarity (RQ 2), to trajectory selection and efficient deployment (RQ 3).

![[x1 21.png|Refer to caption]]

Figure 3: Overview of our dual-branch RecogDrive system and analysis points. A VLM branch (as in the original RecogDrive) and a vision-only branch (ViT/ResNet/EVA-CLIP) provide alternative visual representations to a diffusion Transformer planner (DiT) and action decoder. The two branches use the same planner architecture but are instantiated as separate policies (no weight sharing), producing two candidate trajectories with distinct behaviors. We expand candidates by interpolating between the two trajectories and select the final output using a learned scorer. Our analyses (e.g., Fig. 1 and Fig. 2 ) probe representations at the backbone feature and the DiT (decision) feature indicated in the figure.

## 3 Preliminaries

### 3.1 Formulation and Analysis Points

We study end-to-end driving planning on NAVSIM. Given a front-view image $I_{\mathrm{cam}}$, navigation signal $L_{\mathrm{nav}}$, and ego state $S_{\mathrm{ego}}$, a policy predicts a future trajectory

$$
\hat{\tau}=\Phi(I_{\mathrm{cam}},L_{\mathrm{nav}},S_{\mathrm{ego}})\in\mathbb{R}^{T\times 3},
$$

where each waypoint contains $(x_{t},y_{t},\theta_{t})$.<sup>1</sup>

For RQ1–RQ2, representations are probed at two locations in the stack (Fig. 3): (i) a backbone feature $\mathbf{h}^{\mathrm{bb}}$ (after the backbone adapter), and (ii) a decision feature $\mathbf{h}^{\mathrm{dec}}$ (the planner output immediately before the action head).

### 3.2 Two-branch RecogDrive and Fair Comparisons

Figure 3 overviews our dual-branch setup. The VLM branch uses a frozen VLM encoder to provide token-level visual conditioning to a diffusion Transformer planner (DiT), while the vision-only branch replaces the VLM encoder with a standard visual backbone. In both branches, backbone outputs are linearly projected to the same planner width and fed through an identical planner/action stack, enabling controlled backbone comparisons.

To ensure fairness, all variants share the same NAVSIM split, planner/action architecture, and training schedule; differences are restricted to the backbone initialization and which backbone parameters are updated. Full architectural details (tensor shapes, pooling/tokenization, and training stages) are provided in Appendix A.1.

## 4 RQ1: Representation Analysis

### 4.1 Paired Features and Preprocessing

We analyze *paired* representations extracted from the same driving scenarios. Let $(x_{i},y_{i})_{i=1}^{n}$ denote aligned feature pairs, where $x_{i}$ comes from the VLM branch and $y_{i}$ from a vision-only branch. We stack them as

$$
X\in\mathbb{R}^{n\times d},\quad Y\in\mathbb{R}^{n\times d}.
$$

We consider two extraction levels:

- Backbone level ($d=384$). The VLM yields a token sequence after the RecogDrive adapter, $F_{\mathrm{vlm}}\in\mathbb{R}^{L\times 384}$ ($L$ is the number of VLM tokens(visual + text tokens), $L=2800$), and we use a global descriptor $x=\mathrm{MeanPool}(F_{\mathrm{vlm}})\in\mathbb{R}^{384}$ for cross-backbone comparability.<sup>2</sup> The vision-only backbone outputs a single global embedding, which is linearly mapped to the same width (384) by the backbone adapter; we use this adapter output directly as $y\in\mathbb{R}^{384}$ (no pooling).
- Decision (DiT) level ($d=512$). We extract the planner representation after the final denoising step and immediately before the action head, denoted as $x,y\in\mathbb{R}^{512}$ for the two branches.

Unless stated otherwise, we center each feature dimension across samples. When training SAE, we additionally z-score features using training-split statistics (Appendix A.2).

### 4.2 Linear Similarity: CKA and CCA

We use two complementary linear tools. Linear CKA measures global geometric similarity, while CCA characterizes the maximally correlated linear subspaces.

##### Linear CKA.

With centered features, linear CKA is

$$
\mathrm{CKA}(X,Y)=\frac{\|X^{\top}Y\|_{F}^{2}}{\|X^{\top}X\|_{F}\,\|Y^{\top}Y\|_{F}}.
$$

We report CKA at both backbone and DiT levels to test whether policy learning increases cross-backbone isomorphism at the decision level.

##### PCA-whitened CCA.

We compute canonical correlations after PCA truncation and whitening for numerical stability (full details in Appendix A.4). Let $\rho_{1}\geq\cdots\geq\rho_{k}$ be the canonical correlations; we summarize alignability using the spectrum and aggregates such as mean@k.

### 4.3 Shared–Unique Sparse Autoencoder (SAE)

CKA/CCA quantify similarity but do not explicitly separate *shared* vs. branch-specific components, nor test whether the shared component is functionally interchangeable. We therefore introduce a Shared–Unique Sparse Autoencoder (SAE) on standardized features.

##### Additive shared/unique decomposition.

For each pair $(x,y)$, SAE encodes shared and unique latents for each branch, $z_{s}^{x},z_{u}^{x}=f^{x}(x)$ and $z_{s}^{y},z_{u}^{y}=f^{y}(y)$, and reconstructs with an *additive linear decoder*:

$$
\hat{x}=W_{s}^{x}z_{s}^{x}+W_{u}^{x}z_{u}^{x}+b^{x},\qquad\hat{y}=W_{s}^{y}z_{s}^{y}+W_{u}^{y}z_{u}^{y}+b^{y},
$$

which makes shared/unique contributions interpretable in the original feature space. Unless stated otherwise we use $d_{s}=64$ and $d_{u}=16$.

##### Training objective.

We optimize (i) full reconstruction, (ii) shared-only reconstruction, and (iii) cross shared-only reconstruction to enforce *interchangeability*: decoding branch $x$ using $z_{s}^{y}$ (and vice versa). Regularizers (anti-collapse, shared–unique separability, sparsity) follow standard practice and are specified in Appendix A.6.

##### Metrics.

We report explained-variance ($R^{2}$) for full reconstruction, shared-only self reconstruction, and shared-only cross reconstruction, and use the self–cross gap $\Delta_{\mathrm{cross}}=R^{2}_{\mathrm{sh}}-R^{2}_{\mathrm{cross}}$ as our primary interchangeability statistic (Appendix A.7).

### 4.4 Results & Discussion

##### Policy-level representations become substantially more isomorphic.

We first quantify representational isomorphism between the VLM branch and the vision-only branch via *linear CKA* in the original feature space (Appendix A.3). As shown in Fig. 2, DiT-level features are markedly more aligned than backbone-level features (CKA increases from $\approx 0.22$ to $\approx 0.54$).

We corroborate this trend using PCA-whitened CCA (Appendix A.4). While the average canonical correlation is similar (backbone: $0.61$; DiT: $0.63$), the *high-correlation aligned subspace* expands substantially after policy learning: the number of canonical directions with $\rho>0.8$ increases from $5/28$ (backbone) to $28/78$ (DiT). Consistently, the fraction of feature energy captured by this aligned subspace increases from $28\%\!\rightarrow\!53\%$ for VLM and from $56\%\!\rightarrow\!77\%$ for ViT, suggesting that the planner compresses heterogeneous backbone evidence into a larger set of strongly shared decision factors.

##### Sanity check for SAE: shared-space saturation and shuffled-pair control

To avoid over-interpreting near-1 shared-space CKA, we treat it as a sanity check and verify it with a shuffled-pair control. We additionally report similarity measured *inside* the learned shared space (shared-space CKA) in Table 1 and Table 6. These quantities often saturate near $1$ by design, because the SAE objective explicitly enforces invariance/alignment between paired shared latents. With expressive encoders, much of the alignment can be absorbed by the encoders themselves, making shared-space similarity primarily a *training-validity check* rather than a discriminative metric for comparing backbone vs. DiT. To rule out trivial solutions and validate that high shared-space alignment relies on correct pairing, we perform a shuffled-pair control by randomly permuting pairings $(x_{i},y_{i})$ while keeping marginals fixed (Appendix A.9). Under shuffling, shared-space alignment drops substantially (e.g., shared-space CKA $\approx 0.81/0.79$ for backbone/DiT under the main setting) and original-space alignment collapses to near zero, confirming that the model does not trivially “align everything.” We therefore base our main comparisons on original-space CKA and interchangeability metrics ($R^{2}_{\mathrm{cross}}$, $\Delta_{\mathrm{cross}}$).

Table 1: SAE interchangeability summary (standardized space) with CKA/CCA alignability. We report original-space CKA (CKA <sub>orig</sub>), shared-space CKA (CKA <sub>shared</sub>), cross shared-only reconstruction $R^{2}_{\mathrm{cross}}$, and the self–cross gap $\Delta_{\mathrm{cross}}$ (smaller is better). We additionally summarize PCA-whitened CCA by (i) mean@10, i.e., the mean of the top-10 canonical correlations, and (ii) CCA AER (Aligned Energy Ratio): the fraction of *original-space* feature energy captured by the CCA-aligned subspace defined by $\rho>0.8$ (reported as VLM/ViT; Appendix A.5). DiT exhibits smaller gaps (more interchangeable shared factors) and stronger linear alignability. Setting: use\_raw\_mse=FALSE, cross\_weight=0.1.

| Feature | CKA <sub>orig</sub> | CKA <sub>shared</sub> | $R^{2}_{\mathrm{cross}}(x{\leftarrow}z_{s}^{y})$ | $R^{2}_{\mathrm{cross}}(y{\leftarrow}z_{s}^{x})$ | $\Delta_{\mathrm{cross}}(x)$ | $\Delta_{\mathrm{cross}}(y)$ | CCA mean@10 | CCA AER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Backbone | 0.213 | 0.981 | 0.537 | 0.623 | 0.098 | 0.160 | 0.800 | 0.286/0.556 |
| DiT | 0.537 | 0.986 | 0.546 | 0.763 | 0.071 | 0.063 | 0.972 | 0.534/0.771 |

### 4.5 Negative Result: Representation-only Gating is Insufficient for Reliable Sample-wise Selection

Before turning to complementarity (RQ2), we ask whether per-scenario selection between the VLM and ViT trajectories can be made using representation-level signals alone, without introducing an external trajectory scorer. Specifically, the gate takes as input (i) backbone and/or DiT features from the two branches and (ii) statistics derived from the SAE shared/unique decomposition, and outputs a binary decision indicating which branch to use.

##### Rule-based gates from shared/unique energies.

We construct handcrafted rules based on the SAE additive decoder contributions (Appendix A.11). For each branch, we compute the squared- $\ell_{2}$ energy in the shared and unique components, $E_{s}$ and $E_{u}$, and derive indicators such as unique ratio, shared ratio, uniqueness strength, and shared-dominance. We evaluate four deterministic strategies that map these indicators to a signed score (positive favors VLM; negative favors ViT) and then choose the branch by thresholding it.

##### Learned gates and evidence for limited separability.

We also formulate gating as supervised prediction. Inputs include the two branches’ representations (backbone or DiT), and the label indicates which branch yields better closed-loop performance for that scenario (Appendix A.13). We evaluate tree-based models (Random Forest / boosting / GBDT), an MLP gate, and an attention-based gate operating on the VLM token sequence with the vision-only global embedding as a cross-attention query. Across settings, representation-only gating does not reliably outperform the VLM-only baseline and remains far below the oracle best-of-two (Table 2). This gap suggests that the information needed to predict trajectory superiority is not cleanly encoded in the static representation alone, at least under our current feature definitions and labels. As a qualitative diagnostic, we visualize features using t-SNE and color points by the “VLM-better vs. ViT-better” label; both backbone and DiT spaces exhibit poor class separation (Appendix A.13), consistent with the observed difficulty of gating from representations alone. This motivates RQ2 to incorporate trajectory-/behavior-level evaluation signals rather than relying solely on representation-level cues.

### 4.6 Takeaways (RQ1)

Our key findings are: (i) in the original feature space, decision-level (DiT) representations are substantially more isomorphic than backbone-level representations (CKA $\approx 0.22\rightarrow 0.54$); (ii) with an explicit Shared–Unique SAE, we observe stronger cross-branch interchangeability at the decision level (higher $R^{2}_{\mathrm{cross}}$ and smaller $\Delta_{\mathrm{cross}}$); (iii) shared-space cosine/CKA saturates by design and should be treated as a sanity check, validated via shuffled-pair controls. Finally, representation-only gating yields only marginal improvements over the VLM baseline and remains far below the oracle, motivating trajectory-/behavior-level signals for complementarity analysis in RQ2.”

Table 2: Representation-only gating on NAVSIM. Higher is better. The oracle selects the better branch per scenario (upper bound).

| Method | Score |
| --- | --- |
| RecogDrive-VLM-internvl2B (baseline) | 90.80 |
| RecogDrive-ViT-large (baseline) | 88.88 |
| RecogDrive-ViT-base (baseline) | 85.62 |
| RecogDrive-resnet-101 | 87.69 |
| RecogDrive-resnet-50 | 86.02 |
| RecogDrive-Evaclip02-base | 87.89 |
| Oracle best-of-two(VLM+ViT-large) | 93.58 |
| Rule: More-unique wins ($\uparrow\,\mathrm{unique\_ratio}$) | 89.95 |
| Rule: Shared-dominant conditional | 90.22 |
| Rule: Smoothed shared-dominance | 90.29 |
| Rule: ViT-prior fallback | 89.92 |
| Random Forest | 90.87 |
| Gradient Boosting | 90.75 |
| GBDT | 90.65 |
| MLP classifier | 90.80 |
| Self-attention (binary) | 90.82 |
| Self-attention (score regression) | 90.96 |
| Self-attention (partial score terms) | 90.82 |

## 5 RQ2: Behavioral Complementarity

### 5.1 Scenario-level comparison: complementarity rather than containment

To characterize complementarity at the behavior level, we assign each scenario $i$ and policy $m$ an offline trajectory-quality score $s(m,i)$, where higher is better (NAVSIM v1 PDMS; Appendix B). For a pairwise comparison between the VLM policy and a vision-only policy $m$, we define the per-scenario advantage

$$
\displaystyle\Delta_{i}(m)\;=\;s(\mathrm{VLM},i)-s(m,i),
$$
$$
\displaystyle\text{significant win}\iff|\Delta_{i}(m)|>\tau,
$$

where $\tau$ is a significance threshold.

A recurring pattern is that complementarity is primarily a long-tail phenomenon: for most scenarios, score differences are small; yet both sides exhibit a non-trivial subset of scenarios where they win decisively. Importantly, these decisive-win subsets are not nested—neither policy strictly “contains” the other on the hard cases. This motivates treating VLM and vision-only planners as complementary candidate generators, rather than arguing dominance purely via average score.

##### Stability-aware win counting (strict and conservative).

Because offline scores can be sensitive to small trajectory perturbations, we adopt a stability-aware counting protocol: we only count a scenario as a decisive win if its advantage exceeds $\tau$ under a fixed evaluation protocol (Appendix C). Under this conservative view, decisive wins are rare but persistent. Concretely, with $\tau=0.2$, we observe 257 scenarios where VLM decisively outperforms ViT and 253 where ViT decisively outperforms VLM; with a stricter $\tau=0.5$, the counts are 159 (VLM) vs. 153 (ViT). These near-symmetric tails support “real but long-tailed” complementarity: each policy has its own failure modes and corresponding cases where the other is substantially better.

### 5.2 Driving style differences: beyond speed

Complementarity is not only about who achieves a higher score, but also about systematic style differences. We therefore analyze both longitudinal and lateral behaviors using the expert trajectory as a reference.

##### Longitudinal (speed profile).

For a predicted trajectory $\tau$ with horizon $T$, define mean speed $\bar{v}(\tau)=\frac{1}{T}\sum_{t=1}^{T}v_{t}$. Comparing $\bar{v}(\tau_{\mathrm{VLM}})$ and $\bar{v}(\tau_{\mathrm{ViT}})$ per scenario, we find that the VLM policy is faster in a majority of cases (about $66\%$ in our analysis). This more aggressive longitudinal profile is also consistent with the observed prevalence of rear-end collisions among VLM collisions. Notably, in many scenarios the expert’s longitudinal behavior lies between VLM and ViT, suggesting that the two models represent distinct but plausible driving styles rather than one being a uniformly noisier version of the other.

##### Lateral (path / lane-level preference).

Beyond speed, we observe systematic lateral differences (e.g., lane-centering preference, left/right bias, merge timing). In many cases the expert trajectory again lies between the two or follows an intermediate path. Due to space constraints, we place representative visualizations and additional cases in Appendix D.

### 5.3 Best-of-nn: key evidence from set complementarity

If complementarity is real, then selecting the better trajectory from a combined candidate set should yield a meaningful upper bound improvement. We evaluate this via Best-of- $n$. For each scenario $i$, given a candidate set $\mathcal{C}_{i}$, define $S_{\mathrm{BoN}}(i)\;=\;\max_{\tau\in\mathcal{C}_{i}}s(\tau,i).$ In the Best-of-2 case with $\mathcal{C}_{i}=\{\tau_{\mathrm{VLM}}(i),\tau_{\mathrm{ViT}}(i)\}$, the overall metric improves from $90.80$ to $93.58$ (Table 2). This gain directly supports *set-level* complementarity: each policy wins on different subsets of scenarios, so an oracle trajectory-level selector yields a clear benefit.

### 5.4 Implications: selection needs trajectory-level signals

RQ2 shows that complementarity is expressed primarily in *trajectory outcomes* and consistent driving styles, and is concentrated in a long tail that remains visible under conservative win-counting. Together with RQ1 (representation-only gating has limited predictive power), this suggests that effective gating should be driven by *trajectory-/behavior-level* signals. Rather than generating many samples and re-ranking them, a practical implication is to use a lightweight selector to score and choose among a small set of complementary candidates—in our case, the trajectories produced by VLM and ViT branches (and their interpolations)—with the goal of capturing Best-of- $n$ -like gains from long-tail selection.

## 6 RQ3: From Behavioral Complementarity to Trajectory Selection

RQ2 suggests that VLM and vision-only policies differ in systematic driving style and exhibit long-tail yet stable complementarity at the trajectory level. Moreover, an oracle selector over cross-model candidates yields a clear upper bound. RQ3 turns this observation into a practical mechanism: construct a small, structured candidate set that reflects cross-model style variation, and select trajectories using trajectory-level signals (rather than representation-only cues shown insufficient in RQ1).

A practical constraint in our setting is that *within-model* sampling diversity is limited: increasing the number of diffusion samples for a single ViT or a single VLM yields only marginal Best-of- $n$ gains (Table 2). This shifts the focus from “sampling more from one model” to *cross-model candidate construction* plus *trajectory-level selection*.

##### Evaluation context.

All methods in this section are evaluated in closed loop on NAVSIM navtest. We report results under both NAVSIM metric versions: PDMS for NAVSIM-v1 and EPDMS for NAVSIM-v2; their computation is summarized in Appendix B. Besides the ablation-style evidence in Table 3, we provide full comparisons against prior and concurrent approaches on NAVSIM-v1 in Table 4 and the corresponding NAVSIM-v2 (EPDMS) results in Table 5.

Table 3: RQ3: Trajectory selection/fusion on NAVSIM (navtest).

| Method | PDMS(%) $\uparrow$ |
| --- | --- |
| ReCogDrive-ViT (single) | 88.88 |
| ReCogDrive-VLM (single) | 90.80 |
| Best-of- $n$ (ViT, $n{=}1/3/6$) | 88.88 / 89.13 / 89.32 |
| Best-of- $n$ (VLM, $n{=}1/3/6$) | 90.80 / 91.57 / 91.95 |
| Cross-model oracle (Best-of-2) | 93.58 |
| Cross-model oracle (Best-of-6) | 94.00 |
| Trajectory mean (two endpoints) | 91.18 |
| Rule-based selection (grid search) | 91.21 |
| Adaptive weighting (predict $\alpha$) | 91.31 |
| Scorer selection (endpoints only) | 91.75 |
| HybridDriveVLA | 92.10 |

### 6.1 HybridDriveVLA: style-axis interpolation + scorer-based selection

##### Candidate construction via an interpretable style axis.

For each scenario, we start from two endpoint trajectories predicted by the two branches, $\tau_{\mathrm{vit}}$ and $\tau_{\mathrm{vlm}}$. Motivated by the RQ2 observation that expert behavior often lies between the two styles, we construct intermediate candidates along the linear segment connecting them:

$$
\displaystyle\tau_{\alpha}\;=\;\alpha\cdot\tau_{\mathrm{vit}}+(1-\alpha)\cdot\tau_{\mathrm{vlm}},\alpha\in\{0.1,\ldots,0.9\}.
$$

This yields an $11$ -trajectory candidate set

$$
\displaystyle\mathcal{C}\;=\;\{\tau_{\mathrm{vit}},\tau_{\mathrm{vlm}},\tau_{0.1},\ldots,\tau_{0.9}\}.
$$

Unlike unconstrained diversity sampling, this construction restricts candidates to a single, interpretable *style axis* induced by cross-model complementarity, while still allowing “in-between” solutions that neither endpoint provides.

##### Trajectory-level scorer (PDMS sub-score prediction).

We adopt a DrivoR-style scorer [^12] to evaluate candidate trajectories at the trajectory level by predicting PDMS-related sub-score components from (i) the *decoded* trajectory and (ii) perceptual scene tokens. Concretely, for each candidate trajectory $\tau$, we first embed its decoded waypoints into a $D_{\text{score}}$ -dimensional *score query* using a small MLP, rather than reusing intermediate planner/decoder tokens. This explicit re-embedding enforces a separation between information used to *generate* trajectories and information used to *score* them: the scorer only observes the finalized trajectory and scene tokens, not additional latent details in the generator.

The scoring network then applies a lightweight decoder with cross-attention from score queries to scene tokens, producing trajectory-conditioned score features. Finally, we predict each PDMS sub-score component with a dedicated prediction head (one MLP per component). To keep the roles separated, we block gradients from the scorer back into the trajectory generator. Let $\mathcal{G}_{\theta}$ denote the learned scorer and $\mathcal{G}$ the oracle evaluator used to compute sub-scores during training. We train with component-wise supervision:

$$
\mathcal{L}_{\mathrm{score}}\ =\ \sum_{c}\lambda_{c}\,\frac{1}{\lvert\mathcal{D}\rvert}\sum_{(\tau,i)\in\mathcal{D}}\ell_{c}\!\left(\mathcal{G}_{\theta_{c}}(\tau,i),\mathcal{G}_{c}(\tau,i)\right),
$$

where $c$ indexes PDMS sub-score components and $\ell_{c}$ is a suitable per-component loss (e.g., BCE for binary/indicator terms; regression losses for continuous terms when applicable). At inference, we combine predicted components into a meta-score $\hat{s}(\tau)$ following the PDMS composition and select

$$
\displaystyle\tau^{\star}\;=\;\arg\max_{\tau\in\mathcal{C}}\hat{s}(\tau).
$$

Table 4: Performance comparison on NAVSIM-v1 navtest using PDMS.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| DrivingGPT [^3] | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| UniAD [^9] | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser [^4] | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive [^29] | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA [^31] | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP [^17] | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| ImagiDrive [^13] | 98.1 | 96.2 | 94.5 | 100 | 80.5 | 86.9 |
| DiffusionDrive [^18] | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE [^15] | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| AutoVLA [^14] | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 [^36] | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| ReCogDrive [^16] | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| WAM-diff [^30] | 99.1 | 98.3 | 96.5 | 99.9 | 84.4 | 91.0 |
| DiffusionDriveV2 [^37] | 98.3 | 97.9 | 94.8 | 99.9 | 88.0 | 91.2 |
| iPad [^8] | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 |
| HybridDriveVLA(ours) | 98.6 | 98.6 | 96.2 | 100 | 87.3 | 92.1 |

Table 5: Performance comparison on NAVSIM-v2 navtest using EPDMS.

| Method | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | TL $\uparrow$ | DDC $\uparrow$ | LK $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transfuser [^4] | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 99.9 | 98.3 | 67.6 | 95.3 | 77.8 |
| VADv2 [^2] | 97.3 | 91.7 | 77.6 | 92.7 | 100 | 99.9 | 98.2 | 66.0 | 97.4 | 76.6 |
| Hydra-MDP [^17] | 97.5 | 96.3 | 80.1 | 93.0 | 100 | 99.9 | 98.3 | 65.5 | 97.4 | 79.8 |
| Hydra-MDP++ [^17] | 97.9 | 96.5 | 79.2 | 93.4 | 100 | 100.0 | 98.9 | 67.2 | 97.7 | 80.6 |
| ARTEMIS [^6] | 98.3 | 95.1 | 81.5 | 97.4 | 100 | 99.8 | 98.6 | 96.5 | 98.3 | 83.1 |
| ReCogDrive [^16] | 98.3 | 95.2 | 87.1 | 97.5 | 98.3 | 99.8 | 99.5 | 96.6 | 86.5 | 83.6 |
| DiffusionDriveV2 [^37] | 97.7 | 96.6 | 88.9 | 97.2 | 97.8 | 99.8 | 99.2 | 96.0 | 91.0 | 85.5 |
| HybridDriveVLA(ours) | 98.6 | 92.2 | 89.7 | 98.5 | 98.3 | 99.8 | 99.3 | 96.6 | 87.0 | 85.5 |

### 6.2 DualDriveVLA: fast–slow deployment with scorer confidence

To convert complementarity into a more efficient system, we adopt a fast–slow design. We run the vision-only policy (fast path) by default to generate $\tau_{\mathrm{vit}}$ and score it with the trajectory scorer. If the predicted meta-score satisfies $\hat{s}(\tau_{\mathrm{vit}})\geq\gamma$, we directly output $\tau_{\mathrm{vit}}$; otherwise we invoke the VLM (slow path) to obtain $\tau_{\mathrm{vlm}}$, form the hybrid candidate set (Eq. (3)), and select with Eq. (6). Sweeping $\gamma$ yields an accuracy–compute trade-off (Fig. 4), where a substantial fraction of scenarios can be handled by the fast path while preserving (or improving) overall score.

![[Dual.png|Refer to caption]]

Figure 4: DualDriveVLA accuracy–compute trade-off by varying the confidence threshold γ \\gamma. The x-axis shows the fraction of scenarios routed to the fast path (ViT-only). The left y-axis reports the overall PDMS score, and the right y-axis reports inference speed (throughput / latency, as measured in our setup). Higher ViT selection ratio increases speed, while the scorer-based fallback preserves performance by invoking the slow VLM+selection path on low-confidence cases.

## 7 Conclusion

This paper develops a framework to diagnose, explain, and exploit complementarity in the widely used VLA paradigm for autonomous driving. We analyze the relationship between VLM-based and vision-only backbones at three levels (representation, behavior, and system), and use representation diagnostics to separate shared from model-specific subspaces, establishing a testable evidence chain for whether complementarity exists and where it comes from. We show that gating based solely on representation similarity or alignment strength does not reliably predict trajectory quality or consistently surpass a strong VLM baseline, indicating that complementarity is not captured by simple confidence proxies and instead appears as behavioral and style differences. Under strict cross-backbone-family and multi-seed evaluation, each model consistently wins on a stable long-tail subset, and the expert behavior often lies between the two styles. Based on this finding, we convert complementarity into trajectory-level control using a scorer and an interpretable candidate set constructed along the VLM–vision-only style axis via endpoints and interpolations, yielding stable gains. Finally, we operationalize the approach with a fast/slow system that outperforms the baseline with 85% fast-path acceptance and a 3.2 $\times$ speedup.

## References

## Appendix A Detailed Formulas for RQ1

This appendix specifies the full definitions and implementation details omitted from the main text of RQ1, including preprocessing, linear CKA/CCA (with PCA truncation and whitening), and the Shared–Unique SAE objective and metrics.

### A.1 RecogDrive Details and Training Protocol

#### A.1.1 Architecture and Tensor Shapes

##### VLM branch (InternVL-2B).

Given $I_{\mathrm{cam}}$ and a fixed textual prompt, the VLM produces last-layer hidden states

$$
H_{\mathrm{vlm}}\in\mathbb{R}^{L\times d_{\mathrm{vlm}}}.
$$

A linear adapter maps tokens to the planner width $d$:

$$
F_{\mathrm{vlm}}=H_{\mathrm{vlm}}W_{\mathrm{vlm}},\qquad F_{\mathrm{vlm}}\in\mathbb{R}^{L\times d}.
$$

For feature analysis, the backbone feature is defined by mean pooling:

$$
\mathbf{h}^{\mathrm{bb}}_{\mathrm{vlm}}=\mathrm{MeanPool}(F_{\mathrm{vlm}})\in\mathbb{R}^{d}.
$$

##### Vision-only branch (ViT/ResNet/EVA-CLIP).

Each vision-only backbone outputs a global embedding

$$
\tilde{\mathbf{h}}^{\mathrm{bb}}_{\mathrm{vis}}\in\mathbb{R}^{d_{\mathrm{vis}}},
$$

which is mapped to the same planner width:

$$
\mathbf{h}^{\mathrm{bb}}_{\mathrm{vis}}=\tilde{\mathbf{h}}^{\mathrm{bb}}_{\mathrm{vis}}W_{\mathrm{vis}}\in\mathbb{R}^{d}.
$$

For interface consistency, $\mathbf{h}^{\mathrm{bb}}_{\mathrm{vis}}$ can be implemented as a length-1 token sequence when the planner expects token inputs.

##### Diffusion planner and decision feature.

A diffusion Transformer planner (DiT) iteratively denoises to produce a latent trajectory representation. The *decision feature* is the planner output immediately before the action head:

$$
\mathbf{h}^{\mathrm{dec}}\in\mathbb{R}^{d_{\mathrm{dec}}}.
$$

A lightweight MLP action head maps it to $\hat{\tau}\in\mathbb{R}^{T\times 3}$.

##### Default dimensions.

Unless stated otherwise, we use $d{=}384$, $d_{\mathrm{dec}}{=}512$, and $T{=}8$.

#### A.1.2 Training Protocol and Fairness Controls

All variants share the same NAVSIM configuration (split, optimizer, schedule, epochs) and identical planner/action-head architecture.

##### VLM branch.

We follow the RecogDrive recipe: (1) domain adaptation of the VLM on refined image–text driving data with trajectory supervision; (2) imitation learning on NAVSIM with the VLM backbone frozen and the planner/action head trained; (3) reinforcement learning on NAVSIM with the VLM still frozen and only the planner/action head updated.

##### Vision-only branch.

Vision-only backbones are initialized from public pretrained weights (without the image–text domain-adaptation stage). On NAVSIM, we train the vision backbone jointly with the planner/action head during imitation learning. During RL refinement, the vision backbone is frozen and only the planner/action head is updated, matching the VLM branch to isolate backbone effects.

### A.2 Paired Features, Centering, and Standardization

##### Paired features.

Let $(x_{i},y_{i})_{i=1}^{n}$ be aligned feature pairs extracted from the same driving scenarios, and stack them as

$$
X=[x_{1}^{\top};\ldots;x_{n}^{\top}]\in\mathbb{R}^{n\times d},\qquad Y=[y_{1}^{\top};\ldots;y_{n}^{\top}]\in\mathbb{R}^{n\times d}.
$$

At the backbone level, $x_{i}$ is obtained by mean-pooling the VLM token sequence after the RecogDrive adapter (to support dataset-level statistics and cross-backbone comparability), while $y_{i}$ is the vision-only global embedding after its adapter (no pooling).

##### Centering across samples.

We center each feature dimension across samples using

$$
H=I-\frac{1}{n}\mathbf{1}\mathbf{1}^{\top}\in\mathbb{R}^{n\times n},\qquad\tilde{X}=HX,\ \tilde{Y}=HY.
$$

Unless stated otherwise, CKA/CCA use centered features.

##### Per-dimension z-scoring (used for SAE and reported R2R^{2}).

For SAE training and all reported $R^{2}$ values, we standardize each dimension using *training-split, dataset-level* statistics:

$$
x^{\prime}=(x-\mu_{x})\oslash\sigma_{x},\qquad y^{\prime}=(y-\mu_{y})\oslash\sigma_{y},
$$

where $\mu_{x},\sigma_{x}\in\mathbb{R}^{d}$ are computed per-dimension over the training split (and then reused for validation/test). For notational simplicity we drop primes in the SAE sections. (For some visualizations only, we may apply plot-specific normalization; such choices do not affect the quantitative RQ1 results.)

### A.3 Linear Similarity: Linear CKA

Given centered matrices $\tilde{X},\tilde{Y}$, linear CKA is

$$
\mathrm{CKA}(X,Y)=\frac{\|\tilde{X}^{\top}\tilde{Y}\|_{F}^{2}}{\|\tilde{X}^{\top}\tilde{X}\|_{F}\;\|\tilde{Y}^{\top}\tilde{Y}\|_{F}}.
$$

### A.4 PCA-Truncation and Whitening for CCA

Given centered feature matrices $\tilde{X}\in\mathbb{R}^{n\times d_{x}}$ and $\tilde{Y}\in\mathbb{R}^{n\times d_{y}}$, define sample covariances

$$
\displaystyle\Sigma_{xx}
$$
 
$$
\displaystyle=\frac{1}{n-1}\tilde{X}^{\top}\tilde{X},\qquad\Sigma_{yy}=\frac{1}{n-1}\tilde{Y}^{\top}\tilde{Y},\qquad\Sigma_{xy}=\frac{1}{n-1}\tilde{X}^{\top}\tilde{Y}.
$$

##### PCA truncation (explained-variance threshold).

We eigendecompose

$$
\displaystyle\Sigma_{xx}=P_{x}\Lambda_{x}P_{x}^{\top},\qquad\Sigma_{yy}=P_{y}\Lambda_{y}P_{y}^{\top},
$$

where $\Lambda_{x}=\mathrm{diag}(\lambda^{x}_{1},\ldots,\lambda^{x}_{d_{x}})$ and $\lambda^{x}_{1}\geq\cdots\geq 0$ (similarly for $y$). We choose the smallest $k_{x}$ (resp. $k_{y}$) such that the cumulative explained variance reaches $\eta=0.99$:

$$
k_{x}=\min\left\{k:\frac{\sum_{j=1}^{k}\lambda^{x}_{j}}{\sum_{j=1}^{d_{x}}\lambda^{x}_{j}}\geq\eta\right\},\qquad k_{y}=\min\left\{k:\frac{\sum_{j=1}^{k}\lambda^{y}_{j}}{\sum_{j=1}^{d_{y}}\lambda^{y}_{j}}\geq\eta\right\}.
$$

We then keep

$$
P_{x}^{(k_{x})}=P_{x}[:,1{:}k_{x}],\ \Lambda_{x}^{(k_{x})}=\Lambda_{x}[1{:}k_{x},1{:}k_{x}],\qquad P_{y}^{(k_{y})}=P_{y}[:,1{:}k_{y}],\ \Lambda_{y}^{(k_{y})}=\Lambda_{y}[1{:}k_{y},1{:}k_{y}].
$$

##### Whitening (ridge-stabilized).

We use ridge $\epsilon=10^{-8}$:

$$
W_{x}=P_{x}^{(k_{x})}\left(\Lambda_{x}^{(k_{x})}+\epsilon I\right)^{-1/2},\qquad W_{y}=P_{y}^{(k_{y})}\left(\Lambda_{y}^{(k_{y})}+\epsilon I\right)^{-1/2},
$$

and compute whitened features

$$
\hat{X}=\tilde{X}W_{x},\qquad\hat{Y}=\tilde{Y}W_{y}.
$$

##### Canonical correlations.

CCA is obtained via SVD:

$$
\hat{X}^{\top}\hat{Y}=U\,\mathrm{diag}(\rho_{1},\dots,\rho_{k})\,V^{\top},
$$

where $k=\min(k_{x},k_{y})$ and $\rho_{j}\in[0,1]$ are canonical correlations.

### A.5 CCA Canonical-correlation Spectra and Original-space Aligned Energy

We visualize the PCA-whitened CCA canonical-correlation spectra and report how much *original-space* feature energy lies in highly aligned CCA directions. Importantly, the “CCA-aligned subspace” here is a post-hoc linear construct and should not be conflated with the learned *shared/unique* factors of the SAE.

##### Canonical-correlation spectra.

Figure 5 shows the canonical correlations $\{\rho_{j}\}$ for backbone-level features (28 PCA-whitened dimensions) and DiT-level features (78 PCA-whitened dimensions). DiT yields a much larger set of near-perfectly aligned directions, consistent with increased decision-level isomorphism.

![[cca_backbone.png|Refer to caption]]

(a) Backbone (28 dims).

##### Original-space aligned-subspace energy (thresholded by ρ>τ\\rho>\\tau).

Let $X\in\mathbb{R}^{n\times d_{x}}$ and $Y\in\mathbb{R}^{n\times d_{y}}$ be centered features. After PCA truncation and whitening (Appendix A.4), we obtain whitened features $\tilde{X}=XU_{X}\Lambda_{X}^{-\frac{1}{2}}$ and $\tilde{Y}=YU_{Y}\Lambda_{Y}^{-\frac{1}{2}}$. Running CCA on $(\tilde{X},\tilde{Y})$ yields canonical directions $A,B$ and canonical correlations $\rho_{1}\geq\cdots\geq\rho_{k}$.

For a threshold $\tau$ (we use $\tau=0.8$), define the index set

$$
\mathcal{I}_{\tau}=\{j:\rho_{j}>\tau\}.
$$

Map the selected canonical directions back to the original feature coordinates:

$$
Q_{X}=U_{X}\Lambda_{X}^{-\frac{1}{2}}A_{\mathcal{I}_{\tau}},\qquad Q_{Y}=U_{Y}\Lambda_{Y}^{-\frac{1}{2}}B_{\mathcal{I}_{\tau}}.
$$

Let $\mathrm{orth}(\cdot)$ return an orthonormal basis for the column span (e.g., via QR), and define $\bar{Q}_{X}=\mathrm{orth}(Q_{X})$ and $\bar{Q}_{Y}=\mathrm{orth}(Q_{Y})$. We then measure the fraction of *original-space* energy captured by the CCA-aligned subspace as

$$
E_{X}^{\mathrm{full}}=\frac{1}{n}\|X\|_{F}^{2},\quad E_{X}^{\mathrm{align}}(\tau)=\frac{1}{n}\|X\bar{Q}_{X}\|_{F}^{2},\quad\mathrm{Frac}_{X}^{\mathrm{align}}(\tau)=\frac{E_{X}^{\mathrm{align}}(\tau)}{E_{X}^{\mathrm{full}}},
$$

and analogously for $Y$ using $\bar{Q}_{Y}$.

##### Numerical summary (used in Table ).

Backbone level (28 dims): top-10 $\rho$ are 0.94, 0.91, 0.87, 0.85, 0.82, 0.77, 0.73, 0.71, 0.71, 0.69 (mean@10 $=0.800$). With $\tau=0.8$, the aligned-energy fractions are $28.6\%$ (VLM) and $55.6\%$ (ViT).

DiT level (78 dims): top-10 $\rho$ are 0.996, 0.994, 0.986, 0.983, 0.980, 0.975, 0.968, 0.954, 0.946, 0.941 (mean@10 $\approx 0.972$). With $\tau=0.8$, the aligned-energy fractions are $53.4\%$ (VLM) and $77.1\%$ (ViT).

### A.6 Shared–Unique SAE: Full Objective and Regularizers

This section specifies the Shared–Unique SAE used in RQ1. SAE is trained in standardized feature space (Appendix A.2).

#### A.6.1 Encoders and additive linear decoders

For a minibatch $\{(x^{(i)},y^{(i)})\}_{i=1}^{B}$,

$$
\displaystyle z_{s}^{x}
$$
 
$$
\displaystyle=f_{s}^{x}(x)\in\mathbb{R}^{B\times d_{s}},
$$
$$
\displaystyle z_{u}^{x}
$$
 
$$
\displaystyle=f_{u}^{x}(x)\in\mathbb{R}^{B\times d_{u}},
$$
$$
\displaystyle z_{s}^{y}
$$
 
$$
\displaystyle=f_{s}^{y}(y)\in\mathbb{R}^{B\times d_{s}},
$$
$$
\displaystyle z_{u}^{y}
$$
 
$$
\displaystyle=f_{u}^{y}(y)\in\mathbb{R}^{B\times d_{u}}.
$$

We use MLP encoders (ReLU) and additive linear decoders

$$
\displaystyle\hat{x}
$$
 
$$
\displaystyle=W_{s}^{x}z_{s}^{x}+W_{u}^{x}z_{u}^{x}+\mathbf{1}(b^{x})^{\top},
$$
$$
\displaystyle\hat{y}
$$
 
$$
\displaystyle=W_{s}^{y}z_{s}^{y}+W_{u}^{y}z_{u}^{y}+\mathbf{1}(b^{y})^{\top},
$$

where $W_{s}^{x}\in\mathbb{R}^{d\times d_{s}}$, $W_{u}^{x}\in\mathbb{R}^{d\times d_{u}}$ (and similarly for $y$), and $b^{x},b^{y}\in\mathbb{R}^{d}$.

##### Shared-only reconstructions (self).

$$
\displaystyle\hat{x}_{s}
$$
 
$$
\displaystyle=W_{s}^{x}z_{s}^{x}+\mathbf{1}(b^{x})^{\top},\qquad\hat{y}_{s}=W_{s}^{y}z_{s}^{y}+\mathbf{1}(b^{y})^{\top}.
$$

##### Shared-only reconstructions (cross).

$$
\displaystyle\hat{x}_{s\leftarrow y}
$$
 
$$
\displaystyle=W_{s}^{x}z_{s}^{y}+\mathbf{1}(b^{x})^{\top},\qquad\hat{y}_{s\leftarrow x}=W_{s}^{y}z_{s}^{x}+\mathbf{1}(b^{y})^{\top}.
$$

##### Mixed reconstructions (cross-shared + self-unique).

$$
\displaystyle\hat{x}_{\mathrm{mix}}
$$
 
$$
\displaystyle=W_{s}^{x}z_{s}^{y}+W_{u}^{x}z_{u}^{x}+\mathbf{1}(b^{x})^{\top},
$$
$$
\displaystyle\hat{y}_{\mathrm{mix}}
$$
 
$$
\displaystyle=W_{s}^{y}z_{s}^{x}+W_{u}^{y}z_{u}^{y}+\mathbf{1}(b^{y})^{\top}.
$$

#### A.6.2 Loss terms

The total objective is

$$
\mathcal{L}=\lambda_{\mathrm{rec}}\mathcal{L}_{\mathrm{rec}}+\lambda_{\mathrm{sh}}\mathcal{L}_{\mathrm{sh}}+\lambda_{\mathrm{cross}}\mathcal{L}_{\mathrm{cross}}+\lambda_{\mathrm{vic}}\mathcal{L}_{\mathrm{vic}}+\lambda_{\mathrm{ort}}\mathcal{L}_{\mathrm{ort}}+\lambda_{\mathrm{sp}}\mathcal{L}_{\mathrm{sp}}.
$$

##### (1) Full reconstruction.

$$
\mathcal{L}_{\mathrm{rec}}=\frac{1}{Bd}\|\hat{x}-x\|_{F}^{2}+\frac{1}{Bd}\|\hat{y}-y\|_{F}^{2}.
$$

##### (2) Shared-only reconstruction (self).

$$
\mathcal{L}_{\mathrm{sh}}=\frac{1}{Bd}\|\hat{x}_{s}-x\|_{F}^{2}+\frac{1}{Bd}\|\hat{y}_{s}-y\|_{F}^{2}.
$$

##### (3) Cross shared-only reconstruction (interchangeability).

$$
\mathcal{L}_{\mathrm{cross}}=\frac{1}{Bd}\|\hat{x}_{s\leftarrow y}-x\|_{F}^{2}+\frac{1}{Bd}\|\hat{y}_{s\leftarrow x}-y\|_{F}^{2}.
$$

#### A.6.3 VICReg-style anti-collapse on shared latents

We apply VICReg-style constraints on $(z_{s}^{x},z_{s}^{y})$:

$$
\mathcal{L}_{\mathrm{vic}}=\alpha\,\mathcal{L}_{\mathrm{inv}}+\beta\left(\mathcal{L}_{\mathrm{var}}(z_{s}^{x})+\mathcal{L}_{\mathrm{var}}(z_{s}^{y})\right)+\gamma\left(\mathcal{L}_{\mathrm{cov}}(z_{s}^{x})+\mathcal{L}_{\mathrm{cov}}(z_{s}^{y})\right).
$$

##### Invariance.

Let $\mathrm{BN}(\cdot)$ standardize each dimension within the minibatch (zero mean, unit std). We use

$$
\mathcal{L}_{\mathrm{inv}}=\frac{1}{B}\left\|\mathrm{BN}(z_{s}^{x})-\mathrm{BN}(z_{s}^{y})\right\|_{F}^{2}.
$$

##### Variance (hinge).

Let $\mathrm{Std}(Z)\in\mathbb{R}^{d_{s}}$ denote per-dimension standard deviation across the batch. With margin $\nu>0$,

$$
\mathcal{L}_{\mathrm{var}}(Z)=\frac{1}{d_{s}}\sum_{j=1}^{d_{s}}\max\bigl(0,\nu-\mathrm{Std}(Z)_{j}\bigr)^{2}.
$$

##### Covariance (decorrelation).

Let $Z_{c}=Z-\frac{1}{B}\mathbf{1}\mathbf{1}^{\top}Z$ be batch-centered and

$$
\mathrm{Cov}(Z)=\frac{1}{B-1}Z_{c}^{\top}Z_{c}.
$$

Then

$$
\mathcal{L}_{\mathrm{cov}}(Z)=\frac{1}{d_{s}}\left\|\mathrm{OffDiag}\bigl(\mathrm{Cov}(Z)\bigr)\right\|_{F}^{2},
$$

where $\mathrm{OffDiag}(\cdot)$ zeros the diagonal entries.

#### A.6.4 Shared–unique separability (orthogonality)

We penalize cross-covariance between shared and unique latents within each branch. For batch-centered $A_{c},B_{c}$,

$$
\mathrm{Cov}(A,B)=\frac{1}{B-1}A_{c}^{\top}B_{c}.
$$

Then

$$
\mathcal{L}_{\mathrm{ort}}=\left\|\mathrm{Cov}(z_{s}^{x},z_{u}^{x})\right\|_{F}^{2}+\left\|\mathrm{Cov}(z_{s}^{y},z_{u}^{y})\right\|_{F}^{2}.
$$

#### A.6.5 Sparsity on unique latents

We encourage compact residual coding via $\ell_{1}$ sparsity:

$$
\mathcal{L}_{\mathrm{sp}}=\frac{1}{B}\|z_{u}^{x}\|_{1}+\frac{1}{B}\|z_{u}^{y}\|_{1}.
$$

### A.7 SAE Metrics

All $R^{2}$ scores are computed in standardized feature space. For any reconstruction $\hat{x}$ of $x$:

$$
\displaystyle R^{2}(\hat{x};x)
$$
 
$$
\displaystyle=1-\frac{\mathrm{MSE}(\hat{x},x)}{\mathrm{Var}(x)},
$$
$$
\displaystyle\mathrm{MSE}(\hat{x},x)
$$
 
$$
\displaystyle=\frac{1}{Bd}\|\hat{x}-x\|_{F}^{2},\qquad\mathrm{Var}(x)=\frac{1}{Bd}\|x-\bar{x}\|_{F}^{2},
$$

where $\bar{x}$ is the per-dimension mean computed consistently with the standardization protocol (Appendix A.2).

We report:

$$
R^{2}_{\mathrm{full}}(x)=R^{2}(\hat{x};x),\qquad R^{2}_{\mathrm{sh}}(x)=R^{2}(\hat{x}_{s};x),\qquad R^{2}_{\mathrm{cross}}(x)=R^{2}(\hat{x}_{s\leftarrow y};x),
$$

(and analogously for $y$), and define the self–cross gap

$$
\Delta_{\mathrm{cross}}(x)=R^{2}_{\mathrm{sh}}(x)-R^{2}_{\mathrm{cross}}(x),\qquad\Delta_{\mathrm{cross}}(y)=R^{2}_{\mathrm{sh}}(y)-R^{2}_{\mathrm{cross}}(y).
$$

### A.8 Output-Space Variance Attribution

With additive decoder contributions

$$
x_{s}=W_{s}^{x}z_{s}^{x},\qquad x_{u}=W_{u}^{x}z_{u}^{x},\qquad\varepsilon_{x}=x-(x_{s}+x_{u}+\mathbf{1}(b^{x})^{\top}),
$$

we decompose (in standardized space)

$$
\mathrm{Var}(x)=\mathrm{Var}(x_{s})+\mathrm{Var}(x_{u})+2\,\mathrm{Cov}(x_{s},x_{u})+\mathrm{Var}(\varepsilon_{x}),
$$

(and similarly for $y$). We report all four components; shared/unique percentages do not necessarily sum to $100\%$ when covariance/residual terms are non-zero.

### A.9 Shuffled-Pair Control for Shared-Space Saturation

Shared-space similarity (e.g., cosine similarity or CKA computed on the SAE shared representations) often saturates by design because the SAE objective explicitly enforces invariance between paired shared latents. To verify that this saturation is not due to a trivial solution, we perform a *shuffled-pair control*: we randomly permute pairings $(x_{i},y_{i})$ while keeping the marginals of $x$ and $y$ unchanged, retrain SAE with the same hyperparameters, and re-compute shared-space and original-space similarity.

We expect two qualitative outcomes: (i) shared-space similarity should decrease under shuffled pairing, and (ii) original-space CKA should collapse toward zero, confirming that high shared-space alignment relies on correct pairings and is not a trivial artifact.

### A.10 SAE Hyperparameter Sweep

We sweep SAE settings over use\_raw\_mse $\in\{$ False, True $\}$ and cross\_weight $\in\{0.0,0.1,0.2,0.5,1.0\}$, and report original-space alignment and interchangeability metrics in standardized space.

Table 6: SAE sweep results (standardized space metrics). We report reconstruction quality (full/shared-only), shared-space alignment (CKA <sub>shared</sub>), and interchangeability via cross reconstruction and the self–cross gap.

<table><tbody><tr><th>Feature</th><td>use_raw_mse</td><td>cross_weight</td><td><math><semantics><mrow><msubsup><mi>R</mi> <mi>full</mi> <mn>2</mn></msubsup> <mo></mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow> <annotation>R^{2}_{\mathrm{full}}(x)</annotation></semantics></math></td><td><math><semantics><mrow><msubsup><mi>R</mi> <mi>full</mi> <mn>2</mn></msubsup> <mo></mo><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mrow> <annotation>R^{2}_{\mathrm{full}}(y)</annotation></semantics></math></td><td><math><semantics><mrow><msubsup><mi>R</mi> <mi>shared</mi> <mn>2</mn></msubsup> <mo></mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow> <annotation>R^{2}_{\mathrm{shared}}(x)</annotation></semantics></math></td><td><math><semantics><mrow><msubsup><mi>R</mi> <mi>shared</mi> <mn>2</mn></msubsup> <mo></mo><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mrow> <annotation>R^{2}_{\mathrm{shared}}(y)</annotation></semantics></math></td><td>CKA <sub>shared</sub></td><td><math><semantics><mrow><msubsup><mi>R</mi> <mi>cross</mi> <mn>2</mn></msubsup> <mo></mo><mrow><mo>(</mo><mrow><mi>x</mi> <mo>←</mo> <msubsup><mi>z</mi> <mi>s</mi> <mi>y</mi></msubsup></mrow><mo>)</mo></mrow></mrow> <annotation>R^{2}_{\mathrm{cross}}(x{\leftarrow}z_{s}^{y})</annotation></semantics></math> / <math><semantics><mrow><msubsup><mi>R</mi> <mi>cross</mi> <mn>2</mn></msubsup> <mo></mo><mrow><mo>(</mo><mrow><mi>y</mi> <mo>←</mo> <msubsup><mi>z</mi> <mi>s</mi> <mi>x</mi></msubsup></mrow><mo>)</mo></mrow></mrow> <annotation>R^{2}_{\mathrm{cross}}(y{\leftarrow}z_{s}^{x})</annotation></semantics></math></td><td><math><semantics><mrow><msub><mi>Δ</mi> <mi>cross</mi></msub> <mo></mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow> <annotation>\Delta_{\mathrm{cross}}(x)</annotation></semantics></math> / <math><semantics><mrow><msub><mi>Δ</mi> <mi>cross</mi></msub> <mo></mo><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mrow> <annotation>\Delta_{\mathrm{cross}}(y)</annotation></semantics></math></td></tr><tr><th colspan="10">Backbone features (CKA <sub>orig</sub> =0.2125)</th></tr><tr><th>backbone</th><td>False</td><td>0.0</td><td>0.787</td><td>0.901</td><td>0.641</td><td>0.784</td><td>0.982</td><td>0.492 / 0.559</td><td>0.149 / 0.226</td></tr><tr><th>backbone</th><td>False</td><td>0.1</td><td>0.786</td><td>0.902</td><td>0.634</td><td>0.784</td><td>0.981</td><td>0.537 / 0.623</td><td>0.098 / 0.160</td></tr><tr><th>backbone</th><td>False</td><td>0.2</td><td>0.784</td><td>0.900</td><td>0.629</td><td>0.768</td><td>0.982</td><td>0.541 / 0.634</td><td>0.087 / 0.134</td></tr><tr><th>backbone</th><td>False</td><td>0.5</td><td>0.779</td><td>0.897</td><td>0.626</td><td>0.768</td><td>0.981</td><td>0.582 / 0.687</td><td>0.045 / 0.081</td></tr><tr><th>backbone</th><td>False</td><td>1.0</td><td>0.780</td><td>0.896</td><td>0.623</td><td>0.772</td><td>0.981</td><td>0.598 / 0.715</td><td>0.025 / 0.057</td></tr><tr><th>backbone</th><td>True</td><td>0.0</td><td>0.785</td><td>0.900</td><td>0.633</td><td>0.781</td><td>0.984</td><td>0.522 / 0.602</td><td>0.111 / 0.180</td></tr><tr><th>backbone</th><td>True</td><td>0.1</td><td>0.783</td><td>0.901</td><td>0.623</td><td>0.774</td><td>0.983</td><td>0.541 / 0.627</td><td>0.082 / 0.146</td></tr><tr><th>backbone</th><td>True</td><td>0.2</td><td>0.781</td><td>0.900</td><td>0.625</td><td>0.772</td><td>0.984</td><td>0.554 / 0.648</td><td>0.072 / 0.124</td></tr><tr><th>backbone</th><td>True</td><td>0.5</td><td>0.779</td><td>0.892</td><td>0.618</td><td>0.763</td><td>0.981</td><td>0.579 / 0.685</td><td>0.039 / 0.078</td></tr><tr><th>backbone</th><td>True</td><td>1.0</td><td>0.779</td><td>0.894</td><td>0.622</td><td>0.761</td><td>0.982</td><td>0.595 / 0.710</td><td>0.026 / 0.051</td></tr><tr><th colspan="10">DiT features (CKA <sub>orig</sub> =0.5369)</th></tr><tr><th>dit</th><td>False</td><td>0.0</td><td>0.801</td><td>0.926</td><td>0.617</td><td>0.825</td><td>0.988</td><td>0.534 / 0.754</td><td>0.071 / 0.083</td></tr><tr><th>dit</th><td>False</td><td>0.1</td><td>0.799</td><td>0.922</td><td>0.617</td><td>0.826</td><td>0.986</td><td>0.546 / 0.763</td><td>0.063 / 0.071</td></tr><tr><th>dit</th><td>False</td><td>0.2</td><td>0.800</td><td>0.924</td><td>0.614</td><td>0.822</td><td>0.986</td><td>0.552 / 0.774</td><td>0.048 / 0.061</td></tr><tr><th>dit</th><td>False</td><td>0.5</td><td>0.800</td><td>0.922</td><td>0.613</td><td>0.822</td><td>0.986</td><td>0.566 / 0.791</td><td>0.031 / 0.047</td></tr><tr><th>dit</th><td>False</td><td>1.0</td><td>0.802</td><td>0.922</td><td>0.622</td><td>0.831</td><td>0.984</td><td>0.580 / 0.811</td><td>0.021 / 0.042</td></tr><tr><th>dit</th><td>True</td><td>0.0</td><td>0.800</td><td>0.923</td><td>0.612</td><td>0.816</td><td>0.987</td><td>0.537 / 0.754</td><td>0.062 / 0.075</td></tr><tr><th>dit</th><td>True</td><td>0.1</td><td>0.801</td><td>0.922</td><td>0.614</td><td>0.816</td><td>0.988</td><td>0.548 / 0.766</td><td>0.050 / 0.066</td></tr><tr><th>dit</th><td>True</td><td>0.2</td><td>0.796</td><td>0.921</td><td>0.606</td><td>0.814</td><td>0.986</td><td>0.549 / 0.770</td><td>0.044 / 0.057</td></tr><tr><th>dit</th><td>True</td><td>0.5</td><td>0.798</td><td>0.922</td><td>0.609</td><td>0.821</td><td>0.987</td><td>0.565 / 0.790</td><td>0.032 / 0.044</td></tr><tr><th>dit</th><td>True</td><td>1.0</td><td>0.797</td><td>0.920</td><td>0.609</td><td>0.823</td><td>0.985</td><td>0.576 / 0.805</td><td>0.018 / 0.033</td></tr></tbody></table>

### A.11 Rule-based Representation-only Gating from SAE Energy Decomposition

We build handcrafted gating rules from the Shared–Unique SAE decomposition (Appendix A.6). For each branch, let the additive decoder contributions (in standardized feature space) be

$$
x_{s}=W_{s}^{x}z_{s}^{x},\quad x_{u}=W_{u}^{x}z_{u}^{x},\qquad y_{s}=W_{s}^{y}z_{s}^{y},\quad y_{u}=W_{u}^{y}z_{u}^{y}.
$$

We define squared- $\ell_{2}$ “energy” in shared/unique subspaces as

$$
E^{s}_{\mathrm{vlm}}=\|x_{s}\|_{2}^{2},\quad E^{u}_{\mathrm{vlm}}=\|x_{u}\|_{2}^{2},\qquad E^{s}_{\mathrm{vit}}=\|y_{s}\|_{2}^{2},\quad E^{u}_{\mathrm{vit}}=\|y_{u}\|_{2}^{2},
$$

and $E^{\mathrm{total}}=E^{s}+E^{u}$. We use a small constant $\epsilon>0$ for numerical stability.

##### Indicators.

We derive four indicators (symmetrically for both branches):

$$
r^{u}=\frac{E^{u}}{E^{\mathrm{total}}+\epsilon},\quad r^{s}=\frac{E^{s}}{E^{\mathrm{total}}+\epsilon},\quad u=\frac{E^{u}}{E^{s}+\epsilon},\quad d=\frac{E^{s}}{E^{s}+E^{u}+\epsilon},\quad\bar{d}=\frac{d_{\mathrm{vlm}}+d_{\mathrm{vit}}}{2}.
$$

##### Decision convention.

Each strategy produces a signed score $s$ (positive favors VLM; negative favors ViT), and outputs the decision

$$
\text{choose VLM if }s>0,\ \text{ otherwise choose ViT}.
$$

##### (i) More-unique wins.

We compare uniqueness strength:

$$
s_{1}=u_{\mathrm{vlm}}-u_{\mathrm{vit}}.
$$

##### (ii) Shared-dominant conditional (hard regime).

Given a shared-dominance threshold $\tau$,

$$
s_{2}(\tau)=\begin{cases}r^{s}_{\mathrm{vlm}}-r^{s}_{\mathrm{vit}},&\bar{d}>\tau,\\
u_{\mathrm{vlm}}-u_{\mathrm{vit}},&\text{otherwise}.\end{cases}
$$

##### (iii) Smoothed shared-dominance (sigmoid regime).

To reduce sensitivity near the threshold, we replace the hard indicator by a sigmoid weight

$$
w(\bar{d};\tau)=\sigma\!\bigl(\kappa(\bar{d}-\tau)\bigr)=\frac{1}{1+\exp\bigl(-\kappa(\bar{d}-\tau)\bigr)}\in(0,1),
$$

where we fix $\kappa=5$ (corresponding to SOFT\_LABEL\_SCALE=5). We then define

$$
s_{3}(\tau)=w(\bar{d};\tau)\,(r^{s}_{\mathrm{vlm}}-r^{s}_{\mathrm{vit}})+\bigl(1-w(\bar{d};\tau)\bigr)\,(u_{\mathrm{vlm}}-u_{\mathrm{vit}}).
$$

##### (iv) ViT-prior fallback.

We default to ViT and switch to VLM only when the scenario is strongly shared-dominant:

$$
\text{choose VLM if }\bar{d}>\tau_{\mathrm{strong}}\ \text{and}\ s_{3}(\tau)>0;\quad\text{else choose ViT}.
$$

This strategy is intentionally conservative to avoid over-switching. (*To reproduce this variant, $\tau_{\mathrm{strong}}$ must be specified.*)

##### Threshold sweep and main setting.

We sweep the shared-dominance threshold over

$$
\tau\in\{0.5,\,0.6,\,0.7,\,0.8,\,0.9\},
$$

and select the best-performing value under the same evaluation protocol used for Table 2. The main text reports results for $\tau=0.7$, which achieves the highest score among the tested thresholds. For the smoothed variant, we use the same sweep with fixed $\kappa=5$.

### A.12 Rule-based Gating: Threshold Sweep

We report the performance of the shared-dominance threshold sweep for the rule-based gates. The main text uses $\tau=0.7$.

Table 7: Threshold sweep for rule-based gating. Fill with the same evaluation metrics used in Table 2.

| $\tau$ | navtest PDMS(%) |
| --- | --- |
| 0.5 | 89.73 |
| 0.6 | 89.75 |
| 0.7 | 89.92 |
| 0.8 | 89.90 |
| 0.9 | 89.89 |

### A.13 Learned Representation-only Gating Models and t-SNE Diagnostics

We formulate gating as supervised learning. For each scenario, we construct inputs from both branches’ representations at either the backbone level or the DiT level. Common feature constructions include concatenation $[x;y]$, difference $(x-y)$, and their combination $[x;y;x-y]$.

##### Labels.

The binary label indicates which branch yields better closed-loop performance for that scenario (VLM-better vs. ViT-better), computed from the evaluation score used in Table 2. Ties can be discarded or broken deterministically.

##### Model families.

We evaluate:

- Tree-based models: Random Forest; Gradient Boosting / GBDT variants.
- MLP gate: multi-layer fully-connected network with BatchNorm and Dropout, trained with binary cross-entropy, sigmoid output.
- Token-aware attention gate: self-attention encoder over the VLM token sequence; cross-attention using the vision-only global feature as a query over VLM keys/values; MLP head for classification. This explicitly uses the long VLM token sequence rather than pooled features alone.

##### t-SNE diagnostic.

To qualitatively assess separability, we apply t-SNE on backbone-level and DiT-level representations and color points by the binary label (VLM-better vs. ViT-better). Poor class separation in both spaces (Fig. 6) is consistent with the difficulty of representation-only gating, though t-SNE is used only as a visualization tool rather than a definitive test.

![[tsne2.png|Refer to caption]]

Figure 6: t-SNE of backbone- and DiT-level features colored by whether VLM outperforms ViT for the scenario. The classes are not separable, suggesting intrinsic difficulty for representation-only gating.((Left: backbone-level features; Right: DiT-level features)

## Appendix B Offline Trajectory-quality Score in NAVSIM (PDMS v1 and EPDMS v2)

We use the planning-oriented NAVSIM benchmark and adopt the official Predictive Driver Model Score (PDMS) from NAVSIM v1 as our primary offline trajectory-quality score $s(\cdot,\cdot)$; higher is better. PDMS is a pseudo closed-loop metric that holistically assesses safety, comfort, and progress via multiplicative penalties and a weighted average:

$$
\mathrm{PDMS}\;=\;\mathrm{NC}\times\mathrm{DAC}\times\left(\frac{5\cdot\mathrm{EP}+5\cdot\mathrm{TTC}+2\cdot\mathrm{C}}{12}\right),
$$

where $\mathrm{NC}$ denotes no at-fault collisions, $\mathrm{DAC}$ drivable-area compliance, $\mathrm{EP}$ ego progress, $\mathrm{TTC}$ time-to-collision within bound, and $\mathrm{C}$ comfort.

##### NAVSIM v2: Extended PDMS (EPDMS).

NAVSIM v2 extends PDMS to improve coverage and fairness of open-loop planning evaluation. Compared to NAVSIM v1, EPDMS introduces additional weighted subscores (lane keeping and extended comfort variants), additional multiplier penalties (driving direction compliance and traffic light compliance), and a false-positive penalty filtering scheme.

Table 8: EPDMS composition in NAVSIM v2 (new metrics relative to v1 are highlighted).

| Metric | Weight | Range |
| --- | --- | --- |
| No at-fault Collisions (NC) | multiplier | $\{0,\tfrac{1}{2},1\}$ |
| Drivable Area Compliance (DAC) | multiplier | $\{0,1\}$ |
| Driving Direction Compliance (DDC) | multiplier | $\{0,\tfrac{1}{2},1\}$ |
| Traffic Light Compliance (TLC) | multiplier | $\{0,1\}$ |
| Ego Progress (EP) | 5 | $[0,1]$ |
| Time to Collision (TTC) within bound | 5 | $\{0,1\}$ |
| Lane Keeping (LK) | 2 | $\{0,1\}$ |
| History Comfort (HC) | 2 | $\{0,1\}$ |
| Extended Comfort (EC) | 2 | $\{0,1\}$ |

##### False-positive penalty filtering.

To reduce false-positive penalties, NAVSIM v2 disables a penalty when the human agent is also responsible for the corresponding violation. Formally, for a metric $m$, define

$$
\mathrm{filter}_{m}(\mathrm{agent},\mathrm{human})=\begin{cases}1.0,&\text{if }m(\mathrm{human})=0,\\
m(\mathrm{agent}),&\text{otherwise.}\end{cases}
$$

Intuitively, if the human baseline also triggers the violation, the metric is neutralized (set to $1.0$) rather than penalizing the planner.

##### EPDMS definition.

With the above filtering, EPDMS is defined as

$$
\mathrm{EPDMS}\;=\;\left(\prod_{m\in\{\mathrm{NC},\mathrm{DAC},\mathrm{DDC},\mathrm{TLC}\}}\mathrm{filter}_{m}(\mathrm{agent},\mathrm{human})\right)\cdot\left(\frac{\sum_{m\in\{\mathrm{TTC},\mathrm{EP},\mathrm{HC},\mathrm{LK},\mathrm{EC}\}}w_{m}\cdot\mathrm{filter}_{m}(\mathrm{agent},\mathrm{human})}{\sum_{m\in\{\mathrm{TTC},\mathrm{EP},\mathrm{HC},\mathrm{LK},\mathrm{EC}\}}w_{m}}\right),
$$

where $w_{m}$ are the weights listed in Table 8.

##### Pseudo closed-loop aggregation in NAVSIM v2.

NAVSIM v1 computes metrics after a 4-second non-reactive simulation rollout (background actors follow recorded futures; ego follows the planned trajectory via a controller). NAVSIM v2 uses a two-stage aggregation to better approximate closed-loop behavior while remaining open-loop: (i) a first-stage score is computed on an initial 4-second scene; (ii) multiple follow-up scenes (precomputed rollouts starting from the same initial scene but with different end states) are also scored, and then aggregated with weights given by a Gaussian kernel based on how close each follow-up scene’s start state is to the submitted planner’s first-stage end state. Finally, the first-stage score and the aggregated second-stage score are multiplied to obtain the final aggregated EPDMS score.

## Appendix C RQ2 Win Counting Protocol and Thresholds

For scenario-level complementarity, we compare policies via the per-scenario advantage

$$
\Delta_{r,i}=s(\mathrm{VLM},r,i)-s(\mathrm{ViT},r,i),
$$

where $s(m,r,i)$ is the NAVSIM v1 PDMS score of policy $m$ on scenario $i$ under random seed $r\in\{1,2,3\}$ (Appendix B). A per-seed significant win is defined as $|\Delta_{r,i}|>\tau$.

## Appendix D Qualitative Case Gallery (VLM vs. ViT vs. Expert)

![[Uncaptioned image]](https://arxiv.org/html/2602.10719v1/case/6.png)

\[Uncaptioned image\]

![[1.png|Refer to caption]]

Figure 7: Qualitative case gallery. Each panel contains a front-camera view and a BEV visualization. Red denotes the VLM trajectory, blue denotes the ViT trajectory, and green denotes the human (expert) trajectory used as imitation-learning supervision. Cases 1–5 primarily highlight longitudinal / speed-profile differences, while Cases 6–10 highlight lateral / path and lane-level preference differences.

[^1]: Is a 3d-tokenized llm the key to reliable autonomous driving?. arXiv preprint arXiv:2405.18361. Cited by: §2.

[^2]: Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: Table 5.

[^3]: Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. arXiv preprint arXiv:2412.18607. Cited by: Table 4.

[^4]: Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (11), pp. 12878–12895. Cited by: Table 4, Table 5.

[^5]: DriveLM: driving with graph visual question answering. arXiv preprint arXiv:2312.14150v3. External Links: [Link](https://www.arxiv.org/abs/2312.14150v3) Cited by: §2.

[^6]: ARTEMIS: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. arXiv preprint arXiv:2504.19580. Cited by: Table 5.

[^7]: ORION: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755. Cited by: §2.

[^8]: IPad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: Table 4.

[^9]: Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: Table 4.

[^10]: Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2.

[^11]: Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §2.

[^12]: Driving on registers. arXiv preprint arXiv:2601.05083. Cited by: §6.1.

[^13]: ImagiDrive: a unified imagination-and-planning framework for autonomous driving. arXiv preprint arXiv:2508.11428. Cited by: Table 4.

[^14]: DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: Table 4.

[^15]: End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941. Cited by: Table 4.

[^16]: Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: Table 4, Table 5.

[^17]: Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 4, Table 5, Table 5.

[^18]: DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. arXiv preprint arXiv:2411.15139. Cited by: Table 4.

[^19]: Gpt-driver: learning to drive with gpt. arXiv preprint arXiv:2310.01415. Cited by: §2.

[^20]: A language agent for autonomous driving. arXiv preprint arXiv:2311.10813. Cited by: §2.

[^21]: Reason2Drive: towards interpretable and chain-based reasoning for autonomous driving. arXiv preprint arXiv:2312.03661. External Links: [Link](https://www.arxiv.org/abs/2312.03661) Cited by: §2.

[^22]: A generalized solution of the orthogonal procrustes problem. Psychometrika 31 (1), pp. 1–10. Cited by: Figure 1, Figure 1.

[^23]: Lmdrive: closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15120–15130. Cited by: §2.

[^24]: A survey on vision-language-action models for autonomous driving. arXiv preprint arXiv:2506.24044. External Links: [Link](https://www.arxiv.org/abs/2506.24044) Cited by: §2.

[^25]: Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §2.

[^26]: Vision-language-action models for autonomous driving: past, present, and future. arXiv preprint arXiv:2512.16760v2. External Links: [Link](https://www.arxiv.org/abs/2512.16760v2) Cited by: §2.

[^27]: Omnidrive: a holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. arXiv preprint arXiv:2405.01533. Cited by: §2.

[^28]: Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, pp. 24824–24837. Cited by: §2.

[^29]: PARA-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 4.

[^30]: WAM-diff: a masked diffusion vla framework with moe and online reinforcement learning for autonomous driving. arXiv preprint arXiv:2512.11872. Cited by: Table 4.

[^31]: Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601. Cited by: Table 4.

[^32]: Fine-grained evaluation of large vision-language models in autonomous driving. arXiv preprint arXiv:2503.21505v1. External Links: [Link](https://www.arxiv.org/abs/2503.21505v1) Cited by: §2.

[^33]: AutoDriDM: an explainable benchmark for decision-making of vision-language models in autonomous driving. arXiv preprint arXiv:2601.14702v1. External Links: [Link](https://www.arxiv.org/abs/2601.14702v1) Cited by: §2.

[^34]: WiseAD: knowledge augmented end-to-end autonomous driving with vision-language model. arXiv preprint arXiv:2412.09951. Cited by: §2.

[^35]: Sce2DriveX: a generalized mllm framework for scene-to-drive learning. arXiv preprint arXiv:2502.14917. Cited by: §2.

[^36]: AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: Table 4.

[^37]: DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: Table 4, Table 5.