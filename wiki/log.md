---
title: Activity Log
type: comparison
sources: []
related: [wiki/index.md]
created: 2026-04-05
updated: 2026-09-11
confidence: high
---

# Activity Log

## 2026-05-01 - Ingest: Policy World Model

**Source**: `raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md`
**arXiv**: 2510.19654v1
**Authors**: Zhida Zhao, Talas Fu, Yifan Wang, Lijun Wang, Huchuan Lu
**Confidence**: high - local markdown includes method text, local figures, all six main tables, two appendix tables, and implementation details

**Pages created**:
- `wiki/sources/policy-world-model.md` - full source summary covering action-free future forecasting, context-guided 28-token frame compression, dynamic focal loss, all local figures, all main/appendix tables, relationships, and limitations

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added PWM as inference-time action-free future-frame forecasting used as planning rationale
- `wiki/concepts/navsim-benchmark.md` - added 88.1 PDMS NAVSIM-v1 row and caveat
- `wiki/concepts/nuscenes-waymo-evals.md` - added PWM's nuScenes L2/collision results and safety-vs-L2 interpretation
- `wiki/concepts/foundation-backbones-for-ad.md` - added Show-o/PWM as a unified generation-understanding backbone role

**Index/README updated**: added Policy World Model row; paper count now 43.

**Key facts**:
- Core method: pretrain an autoregressive world model on unlabeled, action-free driving videos, then generate future frame tokens before action prediction during planning.
- Tokenizer: frozen high-resolution first-frame branch plus trainable low-resolution branch; each future frame is represented by 28 tokens.
- DFL upweights temporally changing tokens; full DFL + pretraining improves nuScenes Avg L2/Col to 0.78/0.07 and NAVSIM PDMS to 88.1.
- nuScenes with ego status: 0.41 average L2 and 0.04 average collision, prioritizing safety over best L2.
- NAVSIM-v1: 88.1 PDMS with one front camera, matching DiffusionDrive's Camera+LiDAR score in the paper's table but below current wiki frontier methods.
- Caveat: no NAVSIM-v2/EPDMS, navhard, Bench2Drive, HUGSIM, or Waymo result; 10-frame forecasting adds 0.28s latency and slightly reduces ego progress versus no forecasting.

---

## 2026-05-01 - Ingest: FeaXDrive

**Source**: `raw/papers/FeaXDrive_ Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving.md`
**arXiv**: 2604.12656v2
**Authors**: Baoyun Wang, Zhuoren Li, Ran Yu, Yu Che, Xinrui Zhang, Ming Liu, Jia Hu, Lv Chen, Bo Leng
**Confidence**: high - local markdown includes method text, all six local figures, all five tables, and implementation/latency details

**Pages created**:
- `wiki/sources/feaxdrive.md` - full source summary covering trajectory-centric diffusion, adaptive curvature regularization, drivable-area SDF guidance, feasibility-aware GRPO, all six figures, all five tables, relationships, and limitations

**Concept pages updated**:
- `wiki/concepts/diffusion-planner.md` - added FeaXDrive as clean-trajectory-parameterized diffusion with feasibility guidance
- `wiki/concepts/rl-for-ad.md` - added feasibility-aware GRPO and score-vs-feasibility trade-off
- `wiki/concepts/navsim-benchmark.md` - added FeaXDrive 90.0 PDMS row and caveat
- `wiki/concepts/inference-time-safety.md` - added drivable-area SDF guidance as gradient-based reverse-sampling safety

**Index/README updated**: added FeaXDrive row; paper count now 42.

**Key facts**:
- Core method: predict clean trajectory `x0` directly at each diffusion step, making feasibility constraints act in trajectory space rather than noise space.
- Training-time curvature regularization uses differentiable curvature estimation, `kappa_geo=0.166 m^-1`, and `a_lat_max=6 m/s^2`.
- Inference-time drivable-area guidance builds a local SDF and samples the vehicle footprint corners, not just trajectory center points.
- NAVSIM-v1: FeaXDrive-IL reaches 88.7 PDMS; FeaXDrive with FA-GRPO reaches 90.0 PDMS.
- Standard GRPO reaches higher PDMS (90.56) but raises curvature violation from 0.88% to 5.79%; FA-GRPO keeps it to 2.40%.
- Latency: 348.73 ms total median; VLM backbone 245.33 ms, planner 82.96 ms, SDF build 16.03 ms, guidance 4.41 ms.
- Caveat: NAVSIM-only evaluation; no NAVSIM-v2/EPDMS, navhard, Bench2Drive, HUGSIM, nuScenes, or Waymo result.

---

## 2026-05-01 - Ingest: Drive-JEPA

**Source**: `raw/papers/Drive-JEPA_ Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving.md`
**arXiv**: 2601.22032v1
**Authors**: Linhan Wang, Zichong Yang, Chen Bai, Guoxiang Zhang, Xiaotong Liu, Xiaoyin Zheng, Xiao-Xiao Long, Chang-Tien Lu, Cheng Lu
**Confidence**: high - raw markdown includes the method text, all six local figures, all eight tables, and appendix details for the pseudo-teacher threshold and input resolution

**Pages created**:
- `wiki/sources/drive-jepa.md` - full source summary covering V-JEPA driving-video pretraining, proposal-centric planning, MTD, momentum-aware selection, all six figures, all eight tables, relationships, and limitations

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added Drive-JEPA as JEPA latent predictive video pretraining for planning representations
- `wiki/concepts/selection-based-planning.md` - added Drive-JEPA as simulator-distilled online proposals rather than fixed-vocabulary inference
- `wiki/concepts/navsim-benchmark.md` - added NAVSIM-v1 and NAVSIM-v2 rows plus comparison-scope caveat
- `wiki/concepts/bench2drive.md` - added Drive-JEPA's 64.52 DS / 36.82 SR result and frontier comparison
- `wiki/concepts/foundation-backbones-for-ad.md` - added V-JEPA as a self-supervised video encoder role

**Index/README updated**: added Drive-JEPA row; paper count now 41.

**Key facts**:
- V-JEPA 2 initialization plus 208h of curated driving videos, 8-frame clips, 512 x 256 front-view images, sampled at 2 Hz.
- Perception-free simple decoder reaches 89.0 PDMS in Table 1, +3 over Epona's 86.1/86.2 comparison number.
- Full Drive-JEPA reports 93.3 PDMS on NAVSIM-v1, 87.8 EPDMS on NAVSIM-v2, and 64.52 DS on Bench2Drive.
- MTD ablation: baseline 84.1 EPDMS; V-JEPA init 85.8; driving video pretraining 86.1; +MTD reaches 40% diversity but drops EC to 47.9 and EPDMS to 84.5; +momentum-aware selection restores EC to 84.8 and EPDMS to 87.8.
- Pseudo-teacher count is shallow: `N_pseudo=1` and `N_pseudo=4` both reach 87.8 EPDMS; `N_pseudo=8` drops to 87.5.
- Vision pretraining table: Epona 86.2, ImageNet ResNet34 76.0, DINOv2 76.1, SigLIP 83.4, V-JEPA 2 86.1, Drive-JEPA 89.0; MAE and DepthAnything did not converge.
- Caveat: the paper's NAVSIM SOTA claim is comparison-scope sensitive in the wiki because DriveSuprim reports 93.5 PDMS on v1 and WAM-Diff/Vega-BoN/Latent-WAM exceed 87.8 on v2.

---

## 2026-05-01 - Ingest: Latent-WAM

**Source**: `raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md`
**arXiv**: 2603.24581v1
**Authors**: Linbo Wang, Yupeng Zheng, Qiang Chen, Shiwei Li, Yichen Zhang, Zebin Xing, Qichao Zhang, Xiang Li, Deheng Qian, Pengxuan Yang, Yihang Dong, Ce Hao, Xiaoqing Ye, Junyu Han, Yifeng Pan, Dongbin Zhao
**Confidence**: high - local markdown includes method text, main figures, all seven tables, and supplementary captions; missing supplementary images 6-15 were downloaded from the arXiv HTML into `raw/assets/latent-wam-x6.png` through `raw/assets/latent-wam-x15.png`

**Pages created**:
- `wiki/sources/latent-wam.md` - full source summary covering SCWE, WorldMirror geometric distillation, DLWM, trajectory decoder, all 15 figures, all 7 tables, relationships, and limitations

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added Latent-WAM as compact latent world-status prediction without pixel decoding
- `wiki/concepts/perception-for-planning.md` - added geometric distillation as a training-time spatial-supervision path
- `wiki/concepts/navsim-benchmark.md` - added 89.3 EPDMS NAVSIM-v2 row and caveat
- `wiki/concepts/hugsim-benchmark.md` - added Latent-WAM zero-shot HUGSIM RC/HDS row
- `wiki/concepts/foundation-backbones-for-ad.md` - added DINOv2-Base plus WorldMirror/VGGT teacher role

**Index/README updated**: added Latent-WAM row; paper count now 40.

**Key facts**:
- Core method: 16 learnable scene queries per camera compress DINOv2 image patch tokens into compact scene tokens, then a causal latent world model predicts future world status from historical scene and ego-status tokens.
- Geometric distillation from frozen WorldMirror improves EPDMS from 88.3 to 89.3; direct geometry-feature concatenation degrades to 88.0.
- Full component ablation improves from 87.9 baseline to 89.3 EPDMS; compression alone slightly hurts to 87.7, while geometry, DLWM, and ego-status supervision are complementary.
- NAVSIM-v2: 89.3 EPDMS, EC 87.3, 104M inference parameters, 107ms inference latency on A100.
- HUGSIM zero-shot: 45.9 RC / 28.9 HD-Score average; best average RC in the paper's table and tied best average HD-Score with UniAD.
- Caveat: no NAVSIM-v1 result, no RL/RFT, and the NAVSIM-v2 comparison table omits several wiki leaders such as WAM-Diff, Vega BoN-6, ExploreVLA, DriveDreamer-Policy, DreamerAD, and HAD.

---

## 2026-05-01 - Ingest: HAD

**Source**: `raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md`
**arXiv**: 2604.03581v1
**Authors**: Wenhao Yao, Xinglong Sun, Zhenxin Li, Shiyi Lan, Zi Wang, Jose M. Alvarez, Zuxuan Wu
**Confidence**: high - raw markdown includes the method text, all seven local figures, and all eleven result/ablation tables

**Pages created**:
- `wiki/sources/had.md` - full source summary covering hierarchical diffusion, structure-preserved polar trajectory expansion, metric-decoupled policy optimization, offline reward retrieval, all figures, all tables, relationships, and limitations
- `wiki/concepts/hugsim-benchmark.md` - closed-loop HUGSIM benchmark note with RC/HDS metrics and HAD-L result

**Concept pages updated**:
- `wiki/concepts/diffusion-planner.md` - added HAD as hierarchical diffusion with top-K coarse anchors, polar local expansion, and real-time inference
- `wiki/concepts/rl-for-ad.md` - added HAD's metric-decoupled policy optimization and offline reward retrieval
- `wiki/concepts/navsim-benchmark.md` - added HAD to NAVSIM-v1 and NAVSIM-v2 tables with comparison caveats
- `wiki/concepts/selection-based-planning.md` - added HAD as a hybrid of reward-cache vocabulary lookup and diffusion local refinement
- `wiki/concepts/navhard-ood-evaluation.md` - added HAD-L NavHard result and 3DGS-noise caveat
- `wiki/concepts/hugsim-benchmark.md` - added HUGSIM as a separate closed-loop benchmark concept

**Index/README updated**: added HAD row; paper count now 39; concept count now 21.

**Key facts**:
- Core method: 20 coarse diffusion trajectories -> top-2 coarse intentions -> 5x5 polar local expansion during training and 7x7 at inference -> local diffusion refinement
- MDPO trains metric-specific heads instead of a single coupled PDMS/EPDMS reward; decoupled metrics reach 88.6 EPDMS vs. 87.8 for coupled PDMS reward
- Offline reward retrieval reduces per-sample reward lookup from 0.2449s to 0.0042s and training time from 64.4h to 13.6h
- NAVSIM-v1: HAD reaches 90.2 PDMS; camera-only HAD-L reaches 89.9
- NAVSIM-v2: HAD reaches 88.6 EPDMS; HAD-L reaches 88.5
- HUGSIM: HAD-L reports 47.5 RC / 30.8 HDS overall, with 39.1 RC / 22.5 HDS on the Extreme split
- NavHard: HAD-L reaches 32.3 EPDMS, above DiffusionDrive and LTF but below DriveSuprim and GTRS-Dense in the paper's table
- Caveat: HAD is a strong non-VLM real-time planner, but it is not the absolute NAVSIM-v1/v2 frontier in the broader wiki.

---

## 2026-05-01 - Ingest: OneVL

**Source**: `raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md`
**arXiv**: 2604.18486v1
**Team**: Xiaomi Embodied Intelligence Team
**Confidence**: high - raw markdown preserves the core method and local figures; table rows recovered from arXiv HTML because the local markdown jumps into references after Table 1

**Pages created**:
- `wiki/sources/onevl.md` - full source summary covering latent tokens, dual auxiliary decoders, prefill inference, staged training, all 10 tables, available local images, relationships, and limitations

**Concept pages updated**:
- `wiki/concepts/chain-of-thought-for-ad.md` - added OneVL as latent CoT with world-model-grounded compression
- `wiki/concepts/world-model-for-ad.md` - added training-only visual decoder as a world-model compression target
- `wiki/concepts/vlm-domain-adaptation.md` - added Qwen3-VL latent-token adaptation with staged auxiliary supervision
- `wiki/concepts/navsim-benchmark.md` - added 88.84 PDMS row and caveat
- `wiki/concepts/foundation-backbones-for-ad.md` - added Qwen3-VL-4B OneVL role

**Index/README updated**: added OneVL row; paper count now 38.

**Key facts**:
- Core method: 4 visual latent tokens plus 2 language latent tokens are trained with visual and language auxiliary decoders
- Visual auxiliary decoder predicts future-frame visual tokens at +0.5s and +1.0s, functioning as a training-time world-model objective
- Inference: auxiliary decoders are discarded and latent tokens are prefilled, giving 4.46s NAVSIM latency vs. 4.49s for answer-only AR
- NAVSIM-v1: 88.84 PDMS under SFT, above AR CoT+Answer 88.29 in the paper's controlled Qwen3-VL-4B setup
- Deployment variant: MLP regression head reaches 86.83 PDMS at 0.24s
- Caveat: no NAVSIM-v2/navhard/Bench2Drive, no RL, AR trajectory decoding remains the latency bottleneck, and broad NAVSIM frontier methods exceed 90 PDMS

---

## 2026-05-01 - Ingest: OneDrive

**Source**: `raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md`
**arXiv**: 2604.17915v1
**Authors**: Yiwei Zhang, Xuesong Chen, Jin Gao, Hanshi Wang, Fudong Ge, Weiming Hu, Shaoshuai Shi, Zhipeng Zhang
**Confidence**: high - raw markdown includes all 3 figures and all 10 tables

**Pages created**:
- `wiki/sources/onedrive.md` - full source summary covering the attention-vs-FFN transfer diagnostic, single causal decoder architecture, mixed decoder layers, three-stage training, all figures, all 10 tables, relationships, and limitations

**Concept pages updated**:
- `wiki/concepts/dual-system-vla.md` - added OneDrive as a single-decoder alternative to dual-system and MoT designs
- `wiki/concepts/vlm-domain-adaptation.md` - added attention-transfer / FFN-replacement adaptation lesson
- `wiki/concepts/perception-for-planning.md` - added single-decoder structured-query perception/planning pattern
- `wiki/concepts/diffusion-planner.md` - added unified causal planning queries to the action-decoder design space
- `wiki/concepts/navsim-benchmark.md` - added 86.8 PDMS row and caveat
- `wiki/concepts/nuscenes-waymo-evals.md` - added OneDrive's 0.28 L2 / 0.18 collision open-loop result
- `wiki/concepts/foundation-backbones-for-ad.md` - added InternVL/Qwen attention-transfer diagnostic
- `wiki/concepts/action-tokenization.md` - added planning-query tokens pattern

**Index/README updated**: added OneDrive row; paper count now 37.

**Key facts**:
- Core architecture: image tokens, detection queries, lane queries, planning queries, and text tokens in one causal VLM decoder
- Diagnostic result: pretrained attention transfers to structured driving prediction; pretrained language FFNs can degrade performance
- nuScenes: 0.28 average L2 and 0.18 average collision rate
- NAVSIM-v1 navtest: 86.8 PDMS under SFT, improving over a 85.0 query-decoder baseline but below current wiki frontier methods
- Latency: 156ms on NAVSIM vs. ReCogDrive 263ms; 513ms on nuScenes vs. ColaVLA 727ms
- Caveat: no RL/RFT, no NAVSIM-v2/navhard/Bench2Drive, and NAVSIM uses planning queries only rather than the full perception-query stack

---

## 2026-05-01 - Lint Fix Pass

**Issues fixed**:
- Added schema frontmatter to `wiki/index.md` and `wiki/log.md`.
- Repaired 33 broken image references across `dreameraD.md`, `drivefine.md`, `hermes.md`, `linkvla.md`, `orion.md`, and `vega.md`.
- Updated stale NAVSIM leaderboard wording around DiffusionDriveV2, DriveFine, DriveDreamer-Policy, and Best-of-N interpretation.
- Created concept pages for high-frequency missing concepts: action tokenization/codebooks, GSPO vs. GRPO, PDM-Lite, nuScenes/Waymo evaluations, foundation backbones, and Navhard/OOD evaluation.
- Updated `README.md` and `wiki/index.md` concept catalogs.

---

Append-only log of all wiki operations.

---

## 2026-04-28 - Ingest: SpanVLA

**Source**: `raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md`
**arXiv**: 2604.19710v1
**Authors**: Zewei Zhou, Ruining Yang, Xuewei (Tony) Qi, Yiluan Guo, Sherry X. Chen, Tao Feng, Kateryna Pistunova, Yishan Shen, Lili Su, Jiaqi Ma
**Confidence**: high - local markdown was clipped after Table 1, but missing tables/figures/limitations were recovered from arXiv HTML and missing figure assets were downloaded locally

**Pages created**:
- `wiki/sources/spanvla.md` - full source summary covering sparse-KV action bridging, flow-matching from historical initialization, mReasoning, negative-recovery GRPO, all 5 tables, main and supplementary figures, NAVSIM/navhard results, relationships, and limitations

**Concept pages updated**:
- `wiki/concepts/diffusion-planner.md` - added sparse-KV flow matching from history and action-policy latency comparison
- `wiki/concepts/rl-for-ad.md` - added negative-recovery GRPO reward design and comparison table row
- `wiki/concepts/chain-of-thought-for-ad.md` - added adaptive CoT with continuous action expert
- `wiki/concepts/vlm-domain-adaptation.md` - added mReasoning and negative-recovery adaptation section
- `wiki/concepts/navsim-benchmark.md` - added SpanVLA to NAVSIM-v1, NAVSIM-v2, and navhard tables with comparison caveat

**Index/README updated**: added SpanVLA row; paper count now 36.

**Key facts**:
- Architecture: Qwen2.5-VL-3B VLM backbone plus sparse-KV action bridge and continuous flow-matching action expert
- Action generation: FM starts from historical trajectory initialization rather than Gaussian noise; 5 FM steps; 0.67s total latency for 10 or 50 action points
- Dataset: mReasoning has 30K complex reasoning samples plus 3K negative and 3K recovery real-world samples
- RFT: GRPO with PDMS/EPDMS driving reward, negative-trajectory proximity penalty, recovery proximity reward, CoT length penalty, and action-reasoning alignment penalty
- NAVSIM-v1: 90.3 PDMS post-RFT vs. 82.1 one-shot
- NAVSIM-v2 navtest: 86.4 EPDMS post-RFT vs. 79.4 one-shot
- NAVSIM-v2 navhard: 40.1 EPDMS reported for both Stage 1 and Stage 2
- Caveat: not absolute SOTA in the wiki; v1 and v2 comparison tables omit several stronger contemporary methods

---

## 2026-04-28 - Ingest: DynVLA

**Source**: `raw/papers/DynVLA_ Learning World Dynamics for Action Reasoning in Autonomous Driving.md`
**arXiv**: 2603.11041v1
**Authors**: Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, Zhaoxiang Zhang, Tieniu Tan
**Confidence**: high - all 10 figures and 8 tables available in source

**Pages created**:
- `wiki/sources/dynvla.md` - full source summary covering Dynamics CoT, Dynamics Tokenizer, SFT/RFT pipeline, NAVSIM/Bench2Drive/in-house results, all ablation tables, qualitative/failure figures, implementation details, relationships, and limitations

**Concept pages updated**:
- `wiki/concepts/chain-of-thought-for-ad.md` - added Dynamics CoT as a compact world-dynamics-token CoT substrate
- `wiki/concepts/world-model-for-ad.md` - added dynamics-token world-model pattern and comparison to future-image/video world models
- `wiki/concepts/rl-for-ad.md` - added DynVLA GRPO RFT reward/design row and section
- `wiki/concepts/navsim-benchmark.md` - added DynVLA to NAVSIM-v1 table and SOTA ordering/caveat
- `wiki/concepts/bench2drive.md` - added DynVLA to Bench2Drive SOTA progression and cross-benchmark comparison

**Index/README updated**: added DynVLA row; paper count now 35.

**Key facts**:
- Core idea: generate compact ego/environment dynamics tokens before action tokens instead of text CoT or future-image CoT
- Dynamics Tokenizer: decoupled ego-centric and environment-centric VQ codebooks, action regularization, and image+BEV reconstruction
- Default Dynamics CoT: K=2 horizon, 16 dynamics tokens total, 4 ego + 4 environment tokens per transition
- Training: Dynamics Tokenizer pretraining, Dynamics CoT SFT, then GRPO RFT with PDMS + format reward
- NAVSIM-v1: 91.7 PDMS; strong but below DriveSuprim 93.5 and HybridDriveVLA 92.1 in the wiki
- Bench2Drive: 88.34 DS / 72.73 SR / 72.23 multi-ability mean; below LinkVLA 91.01 DS but above AutoMoT 87.34 DS
- CoT design ablation: Dynamics CoT 0.37s / 87.2 PDMS vs. no CoT 0.20s / 85.6, scene-description CoT 3.04s / 85.3, future-image CoT 2.29s / 86.3
- Caveats: no NAVSIM-v2 / EPDMS result, in-house 700k-frame dataset is not public, front-view-only setup, and the NAVSIM comparison table omits several stronger contemporary methods

---

## 2026-04-28 — Ingest: ELF-VLA

**Source**: `raw/papers/Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures.md`
**arXiv**: 2603.01063v1
**Authors**: Yuechen Luo, Qimao Chen, Fang Li, Shaoqing Xu, Jiaxin Liu, Ziying Song, Zhi-xin Yang, Fuxi Wen
**Confidence**: high — all 11 figures and 10 tables available in source

**Pages created**:
- `wiki/sources/elf-vla.md` — full source summary covering the persistent-failure problem, two-stage SFT, GRPO with teacher feedback, policy shaping, reward design, data curation, all NAVSIM/high-level-planning/ablation tables, and all figures (`introv6.png`, `mainv4.png`, `methodv3.png`, `rollout_ratio.png`, `visual*.jpg`, prompts, feedback, meta-action labels)

**Concept pages updated**:
- `wiki/concepts/rl-for-ad.md` — added "ELF-VLA: Teacher-Guided Learning from Persistent Failures"; added ELF-VLA row to GRPO reward/design table; frontmatter updated
- `wiki/concepts/chain-of-thought-for-ad.md` — added failure-diagnostic CoT pattern and ELF-VLA row in CoT design table; frontmatter updated
- `wiki/concepts/vlm-domain-adaptation.md` — added feedback-conditioned adaptation section; frontmatter updated
- `wiki/concepts/navsim-benchmark.md` — added ELF-VLA to NAVSIM-v1 and NAVSIM-v2 tables; added comparison-scope caveat; updated SOTA ordering language; frontmatter updated

**Index/README updated**: added ELF-VLA row.

**Key facts**:
- Base model: InternVL3-8B; teacher: Qwen3-VL-32B
- Persistent-failure mechanism: teacher diagnoses failed rollouts below threshold `s=0.8`, student re-rolls from feedback input, and `k=1` better refinement is injected into the GRPO batch
- Reward: PDMS trajectory reward + format reward + endpoint goal reward
- Policy shaping: `f(x)=x/(x+gamma)`, `gamma=0.1`, needed because feedback-generated outputs are low-probability under base input conditioning
- Data curation: 85k NAVSIM entries filtered to 24k high-value difficult/ambiguous scenarios; curated 24k reaches 91.0 PDMS vs. 89.1 for full 85k
- NAVSIM-v1: 91.0 PDMS, +2.0 over standard GRPO and +3.6 over SFT
- NAVSIM-v2: 87.1 EPDMS, EC=87.2
- High-level planning: 80.3% overall accuracy vs. 79.3% for GRPO
- Caveat: v1 table omits DriveSuprim, HybridDriveVLA, FLARE, DiffusionDriveV2, WAM-Diff, ExploreVLA; v2 table omits WAM-Diff, ExploreVLA, DriveDreamer-Policy, DreamerAD, Vega BoN-6
- Main contribution is training-time failure distillation, not current absolute NAVSIM SOTA

---

## 2026-04-23 — Ingest: ExploreVLA

**Source**: `raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md`
**arXiv**: 2604.02714v1
**Authors**: Zihao Sheng, Xin Ye, Jingru Luo, Sikai Chen, Liu Ren
**Confidence**: high — all tables and figures available in source

**Pages created**:
- `wiki/sources/explorevla.md` — full source summary covering all 6 figures (x1 25.png–x6 18.png), both NAVSIM tables (v1 full, v2 full), dense supervision ablation (Table 3), reward component ablation (Table 4), nuScenes comparison (Table 5), method equations, training strategy, qualitative analysis

**Concept pages updated**:
- `wiki/concepts/rl-for-ad.md` — added "ExploreVLA: World Model Uncertainty as Intrinsic Exploration Reward" section (entropy formula, safety-gated reward, RL ablation table, contrast with DreamerAD); added ExploreVLA row to GRPO comparison table; frontmatter updated
- `wiki/concepts/world-model-for-ad.md` — added Pattern 12: Dual-Role World Model (dense supervisor + intrinsic entropy reward); updated World Model vs. VLA table; frontmatter updated
- `wiki/concepts/navsim-benchmark.md` — added ExploreVLA to v1 SOTA table (90.4 single / 93.7 BoN-6); added to v2 SOTA table (88.8 EPDMS, EC=86.8); added caveat paragraph; updated BoN ranking summary; frontmatter updated
- `wiki/concepts/best-of-n.md` — added ExploreVLA BoN-6 (93.7, 2nd in wiki) to NAVSIM-v1 table; updated Key Observations #1; frontmatter updated

**Index updated**: added ExploreVLA row (37th paper).

**Key facts**:
- Architecture: Show-o (Phi-1.5 LLM + MAGVIT-v2 8192-codebook); omni-attention (causal for text/ego, full for image); 2-frame input (current + 0.5s history)
- Stage 1a pre-train (10 epochs, image gen only) → Stage 1b SFT (15 epochs, joint action+image) → Stage 2 GRPO LoRA (5 epochs, G=8, rank 32, 4×H200)
- Dense supervision ablation: RGB only 87.9, depth only 87.8, both 88.5 (from 86.2 baseline)
- Reward design: safety-gated entropy R_i = PDMS_i + 0.5·f(H) if PDMS_i > 0.9 else PDMS_i; image reward alone +0.03, PDMS alone +1.69, combined +1.86
- NAVSIM v1: 90.4 single / 93.7 BoN-6 — 2nd highest BoN in wiki (after Curious-VLA 94.8)
- NAVSIM v2: 88.8 EPDMS (2nd in wiki after WAM-Diff 89.7); EC=86.8 (strong); comparison table omits WAM-Diff, DDP, DreamerAD
- nuScenes (Stage 1 only): avg collision 0.10% (ties OpenDriveVLA for best); avg L2 0.44m
- Comparison scope caveat: v1 table omits WAM-Diff, FLARE, DiffusionDriveV2, HybridDriveVLA, DriveFine — single-sample 90.4 is below all; BoN-6 vs. DriveSuprim single-sample not a fair comparison
- First wiki method to use world model prediction *uncertainty* (not predictions) as an RL reward

---

## 2026-04-23 — Ingest: DriveVA

**Source**: `raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md`
**arXiv**: 2604.04198v1
**Org**: University of Twente (+ multiple affiliations)
**Confidence**: medium — Table 1 (NAVSIM sub-scores and comparison methods) is truncated in the source file

**Pages created**:
- `wiki/sources/driveva.md` — full source summary covering abstract, method (all equations, tokenization, video continuation, flow-matching loss), available figures (x1 24.png, x2 22.png, x3 22.png truncated), key ablation (action-only 71.4 → DriveVA 90.9 PDMS, +19.5), zero-shot results, limitations

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 11: Joint Video-Action DiT from Video Generation Backbone (DriveVA); coupling mechanism comparison table (6 wiki methods); zero-shot generalization data; added DriveVA to World Model vs. VLA table; added 2 open questions (backbone scale, zero-shot ceiling); frontmatter bumped to 2026-04-23
- `wiki/concepts/navsim-benchmark.md` — added DriveVA (90.9 PDMS v1) to SOTA table with truncation caveat; added DriveVA caveat paragraph (medium confidence); frontmatter bumped

**Index updated**: added DriveVA row (36th paper).

**Key facts**:
- Backbone: Wan2.2-TI2V-5B (5B params) — largest backbone in wiki; same family as DDP (1.3B) and UniUGP (Wan2.1)
- Joint generative target: single DiT denoises [future_video_latents ‖ action_tokens] simultaneously — deepest video-action coupling in wiki
- Critical ablation: action-only 71.4 → joint video+action 90.9 PDMS (+19.5) — strongest single-component gain in wiki for any technique
- 2 flow-matching steps sufficient for near-optimal NAVSIM performance
- Zero-shot nuScenes (trained on NAVSIM only): −78.9% avg L2, −83.3% collision rate vs. PWM
- Zero-shot Bench2Drive (real→sim): −52.5% avg L2, −52.4% collision rate vs. PWM
- 90.9 PDMS slots between WAM-Diff (91.0) and DriveFine (90.7) in wiki; no RL stage
- No NAVSIM-v2 / EPDMS results; no latency numbers; video required at every inference step
- Table 1 truncated — cannot verify comparison set or sub-metric breakdown

---

## 2026-04-23 — Ingest: DriveSuprim

**Source**: `raw/papers/DriveSuprim_ Towards Precise Trajectory Selection for End-to-End Planning.md`
**arXiv**: 2506.06659v1
**Org**: Fudan University + NVIDIA

**Pages created**:
- `wiki/sources/drivesuprim.md` — full source summary with all 6 figures (x1 23.png, x2 21.png, x3 21.png, trajectories_ori_vs_rotated.png, x4 20.png, x5 20.png), all 11 tables (oracle top-K, NAVSIM-v1 comparison, NAVSIM-v2 comparison, module ablation, coarse-to-fine evolution, refinement settings, soft-label threshold, FOV settings, inference coefficients v1, inference coefficients v2, turning scenario breakdown), and full method descriptions
- `wiki/concepts/selection-based-planning.md` — new concept page: fixed-vocabulary scoring paradigm; three failure modes (hard negatives / directional bias / hard binary labels); oracle ceiling (98.7 PDMS top-256); DriveSuprim coarse-to-fine mechanism; rotation augmentation; comparison table of selection-based methods (Hydra-MDP, HydraMDP++, DriveSuprim, DreamerAD, HybridDriveVLA); relationship to stochastic BoN

**Concept pages updated**:
- `wiki/concepts/navsim-benchmark.md` — added DriveSuprim (93.5 PDMS v1, 87.1 EPDMS v2) to both SOTA tables; updated SOTA summary (DriveSuprim 93.5 is now highest non-BoN result in wiki); added DriveSuprim caveat paragraph; frontmatter bumped to 2026-04-23
- `wiki/concepts/best-of-n.md` — added "Fixed-Vocabulary Oracle Selection" section; DriveSuprim oracle study (top-1=91.9, top-4=94.5, top-16=96.1, top-256=98.7 vs. human 94.8); explains why vocabulary ceiling (98.7) exceeds stochastic BoN ceiling (94.8); frontmatter bumped to 2026-04-23

**Index updated**: added DriveSuprim row (35th paper); added Selection-Based Planning concept row (14th concept).

**Key facts**:
- Selection-based (non-VLM): scores 8192 fixed candidate trajectories; picks argmax
- Oracle study: top-4 of 8192 → 94.5 PDMS (nearly matches human GT 94.8); top-256 → 98.7 PDMS — ceiling far above stochastic BoN
- Coarse-to-fine: Stage 1 selects top-256 → Stage 2 re-scores only those 256; key ablation: adding decoder layers gives +0, trajectory filtering gives +0.8 EPDMS
- Rotation augmentation: only 8% NAVSIM GT trajectories turn >30°; pseudo-panoramic view from 3 cameras + angle-crop + GT rotation gives uniform direction distribution; +0.7 EPDMS overall, +2.9/+2.0 EPDMS on left/right turning scenarios
- EMA self-distillation: teacher EMA (momentum 0.992→0.998); soft labels clipped within ±0.15 of hard GT; optimal threshold δ=0.15; +1.5 EPDMS
- 93.5 PDMS NAVSIM-v1 (ViT-L, 3-cam, no LiDAR, no VLM) — highest non-BoN result in wiki, surpassing DiffusionDriveV2 (91.2 C+L) and HybridDriveVLA (92.1 ensemble)
- 87.1 EPDMS NAVSIM-v2 (ViT-L) — below DreamerAD (87.7) and WAM-Diff (89.7); EC=78.6 (middle range)
- Limitations: multi-stage (>2) provides no gain; inference speed suboptimal (no latency numbers given); NAVSIM-only evaluation

---

## 2026-04-21 — New Concept Page: Mixture of Experts for AD

**Page created**: `wiki/concepts/mixture-of-experts.md`

**Sources**: WAM-Diff, DriveFine, DriveVLA-W0, AutoMoT, UniDriveVLA

**Key content**:
- 4-type taxonomy: sparse LoRA MoE (WAM-Diff), block-level task MoE (DriveFine), Mixture-of-Transformers (AutoMoT/UniDriveVLA), lightweight side expert (DriveVLA-W0)
- Comparison table across all 5 wiki papers
- MoE + RL routing instability problem and GSPO as solution
- Catastrophic forgetting evidence table (AutoMoT)
- Bottleneck-to-solution mapping across papers

**Index updated**: added MoE to concepts table.

---

## 2026-04-21 — Lint Pass

**Stale claims fixed (9 files)**:
- `wiki/index.md`: removed "SOTA" from WAM-Flow row; removed "SOTA" from Senna-2 row; removed "(new SOTA)" from DriveDreamer-Policy row
- `wiki/sources/senna2.md`: updated NAVSIM-v2 SOTA note (lines 181, 227) to reflect supersession by DriveFine/WAM-Diff (89.7)
- `wiki/sources/drivedreamer-policy.md`: updated NAVSIM-v2 SOTA cross-reference to "(now superseded)"
- `wiki/sources/wam-flow.md`: removed "SOTA" from one-line summary; updated connections note to list all superseding methods
- `wiki/sources/drivefine.md`: updated "current single-sample SOTA" to "most broadly-verified"
- `wiki/sources/recogdrive.md`: removed "SOTA" from one-line summary

**Concept pages updated**:
- `wiki/concepts/navsim-benchmark.md`: updated v1 closing sentence to include WAM-Diff (91.0) in ordering
- `wiki/concepts/discrete-flow-matching.md`: added WAM-Diff and DriveFine to frontmatter sources/related; added both to Applications in Literature; updated DFM vs. Masked Diffusion comparison table

**Orphan pages**:
- `wiki/overview.md`: deleted (empty 1-line file; no inbound links)

**No new orphan source pages** — all 30 source pages referenced in concept frontmatter or body text.

**Concepts mentioned but lacking dedicated pages** (candidates for future work):
- Mixture of Experts (MoE) — heavily used across DriveFine, WAM-Diff, AutoMoT, UniDriveVLA
- nuScenes Benchmark — referenced for L2/collision metrics but no wiki page
- WaymoE2E Benchmark — referenced in NoRD, HERMES but no wiki page
- BEV Representation — mentioned across multiple sources but covered only inline

---

## 2026-04-21 — Ingest: WAM-Diff

**Source**: `raw/papers/WAM-Diff_ A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving.md`
**arXiv**: 2512.11872v1
**Org**: Fudan University + Yinwang Intelligent Technology Co., Ltd

**Pages created**:
- `wiki/sources/wam-diff.md` — full source summary with all 14 figures (teaser.png, main_arch.png, scheduler.png, gspo.png, x1 22.png–x8 10.png, fc2.png, gspo2grpo.png), all 10 tables (NAVSIM-v1 comparison, NAVSIM-v2 comparison, nuScenes comparison, MoE config ablation, GSPO ablation, component ablation, reward ablation, decoding schedule ablation, CFG ablation, training hyperparameters), and full architecture/training/inference descriptions

**Concept pages updated**:
- `wiki/concepts/rl-for-ad.md` — added "WAM-Diff: GSPO — Sequence-Level RL for MoE Policies" section: GSPO motivation (MoE routing instability under token-level GRPO), full formulation (length-normalized sequence likelihood ratio + clipped PPO at sequence level), comparison table GSPO vs. GRPO, ablation results (+4.4 PDMS), architecture-specific RL comparison table (GSPO vs. DiffusionDriveV2 vs. FLARE); added WAM-Diff row to GRPO reward comparison table; updated closing note; frontmatter bumped
- `wiki/concepts/diffusion-planner.md` — added "WAM-Diff: MoE Masked Diffusion with Flexible Decoding and GSPO" section: hybrid tokenization, flexible decoding schedules table (random/causal/reverse-causal), LoRA MoE scaling, GSPO reference, three-way comparison table (ReflectDrive/DriveFine/WAM-Diff); added WAM-Diff row to design space table; frontmatter bumped
- `wiki/concepts/navsim-benchmark.md` — added WAM-Diff (91.0 PDMS v1, 89.7 EPDMS v2) to both SOTA tables; added caveat paragraph; updated single-sample NAVSIM-v2 SOTA note; frontmatter bumped

**Index updated**: added WAM-Diff row (33rd paper).

**Key facts**:
- First VLA combining masked diffusion + sparse MoE + online RL (GSPO) for AD
- GSPO is sequence-level GRPO variant — solves MoE routing instability that token-level GRPO causes
- Reverse-causal decoding schedule (+2.0 PDMS): resolves far-future tokens first, then refines near-term — best for car-following and oncoming scenarios
- GSPO is the single largest contribution: +4.4 PDMS (86.6 → 91.0); LoRA MoE +1.9 PDMS; CFG +2.4 PDMS; decoding schedule +2.0 PDMS
- 91.0 PDMS NAVSIM-v1 (comparison includes ReCogDrive 90.8, DriveVLA-W0 90.2, DiffusionDrive 88.1; excludes FLARE 91.4, DiffusionDriveV2 91.2, HybridDriveVLA 92.1)
- 89.7 EPDMS NAVSIM-v2 (comparison excludes DDP 88.7, DreamerAD 87.7, Senna-2 86.6); potential new single-sample SOTA if on comparable scorer to prior methods
- EC = 78.5 (below DDP 79.4, DiffusionDrive 87.7 — masked diffusion doesn't optimize extended comfort)
- nuScenes: 0.28% avg collision (best VLA, matches UniAD) under UniAD protocol
- Limitations: front-view only (no surround), no temporal history (single frame)
- 8.4B params total; +0.5B MoE (only ~0.05B activated at inference)
- Training: 4 stages on Ascend 910B; 668K nuPlan + 800K VQA SFT → 103K NAVSIM adaptation → GSPO

---

## 2026-04-19 — Ingest: HybridDriveVLA / DualDriveVLA

**Source**: `raw/papers/From Representational Complementarity to Dual Systems_ Synergizing VLM and Vision-Only Backbones for End-to-End Driving.md`
**arXiv**: 2602.10719v1
**Venue**: Machine Learning, ICML

**Pages created**:
- `wiki/sources/hybriddriveVLA.md` — full source summary with all 4 figures (aligned_feature_kde.png, cka_backbone_vs_dit.png, x1 21.png, Dual.png), all quantitative tables (RQ1 CKA/SAE, RQ2 BoN/complementarity, RQ3 ablation, NAVSIM-v1/v2 comparison), method descriptions for HybridDriveVLA and DualDriveVLA

**Concept pages updated**:
- `wiki/concepts/dual-system-vla.md` — added "Representational Complementarity" section with full exposition of 3-RQ framework; HybridDriveVLA/DualDriveVLA mechanism (style-axis interpolation, trajectory scorer, fast–slow deployment); comparison table expanded with new column; frontmatter bumped
- `wiki/concepts/best-of-n.md` — added "Cross-Model BoN" section establishing cross-model diversity as richer than within-model sampling (oracle cross-model best-of-2: 93.58 vs. within-model BoN-6: 91.95); HybridDriveVLA as second deployable BoN variant; frontmatter bumped
- `wiki/concepts/navsim-benchmark.md` — added HybridDriveVLA (92.1 PDMS v1, 85.5 EPDMS v2) and DualDriveVLA (91.0 PDMS) to SOTA tables; added HybridDriveVLA caveat (ensemble method, DAC weakness, limited v2 comparison set); frontmatter bumped

**Index updated**: added HybridDriveVLA / DualDriveVLA row.

**Key facts**:
- 3-RQ analysis on RecogDrive: VLM (InternVL-2B) vs. ViT-large in same DiT planner
- Backbone CKA ~0.22 → DiT CKA ~0.54: policy learning compresses heterogeneous visual signals
- Representation-only gating ceiling: 90.96 PDMS (vs. VLM baseline 90.80, oracle 93.58)
- VLM is faster/more aggressive in ~66% of scenarios; each side decisively wins on ~2–3% long-tail scenarios
- Cross-model oracle best-of-2: 93.58 PDMS (+2.78 over VLM single) >> within-model BoN-6 VLM: 91.95
- HybridDriveVLA: 11-candidate set (2 endpoints + 9 interpolations) + DrivoR-style trajectory scorer → **92.10 PDMS** NAVSIM-v1
- DualDriveVLA: 15% VLM invocations → **91.00 PDMS** at **3.2× throughput**
- NAVSIM-v2: 85.5 EPDMS; DAC = 92.2 (lowest in its comparison table — interpolated trajectories occasionally exit drivable area)
- Comparison table in paper includes DiffusionDriveV2 (91.2) and iPad (91.7) — relatively fair

---

## 2026-04-17 — Ingest: Epona

**Source**: `raw/papers/Epona_ Autoregressive Diffusion World Model for Autonomous Driving.md`
**arXiv**: 2506.24113
**Org**: Horizon Robotics, Tsinghua, PKU, NJU, HKUST, NTU, Tencent
**Venue**: ICCV 2025

**Pages created**:
- `wiki/sources/epona.md` — full source summary with 7 figures (Figure 1 URL-only, not locally available), all tables, complete architecture description (MST + TrajDiT + VisDiT), chain-of-forward training formulation, and full ablation data

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` — expanded Pattern 2 stub (Autoregressive WM + Diffusion Planner) into full section covering MST architecture, chain-of-forward, shared latent ablation, inference modes, and DreamerAD relationship; linked Epona to FID/planning tables; bumped frontmatter
- `wiki/concepts/navsim-benchmark.md` — added Epona (86.2 PDMS) to SOTA v1 table; added caveat paragraph noting pre-2025-baseline-only comparison; bumped frontmatter

**Source pages updated**:
- `wiki/sources/dreameraD.md` — added `sources/epona.md` to related frontmatter

**Index updated**: added Epona row to Sources table (inserted above DreamerAD).

**Key facts**:
- FVD 82.8 NuScenes (SOTA at time; −7.4% vs Vista 89.4); generation horizon 120s / 600 frames (vs Vista 15s)
- 86.2 PDMS NAVSIM-v1 camera-only; comparison table excludes all VLA-era methods (DriveFine 90.7, WAM-Flow 90.3, etc.)
- Joint video+trajectory training critical: disabling VisDiT → PDMS 86.2 → 78.1 (−8.1)
- Chain-of-forward training: 1-step velocity estimate prevents autoregressive drift in long-horizon generation
- Real-time planning (20 Hz) only with VisDiT disabled; full generation is ~2.3s/frame
- Epona is DreamerAD's base model; DreamerAD adds latent RL → 88.7 PDMS (+2.5)
- Figure 1 in source file is a URL reference to arxiv — not saved locally as an asset

---

## 2026-04-16 — Ingest: DiffusionDriveV2

**Source**: `raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md`
**arXiv**: 2512.07745
**Org**: HUST (EIC + AI Institute), Horizon Robotics, Wuhan University
**Venue**: December 2024

**Pages created**:
- `wiki/sources/diffusiondrive-v2.md` — full source summary with all 8 figures, all tables, and complete method description including Intra/Inter-Anchor GRPO formulations and multiplicative noise derivation

**Concept pages updated**:
- `wiki/concepts/rl-for-ad.md` — added "DiffusionDriveV2: Anchored Truncated GRPO" section; updated GRPO comparison table with DiffusionDriveV2 row; bumped frontmatter `updated` and `sources`/`related`
- `wiki/concepts/diffusion-planner.md` — added DiffusionDriveV2 subsection under DiffusionDrive section; updated comparison table to include V2 row; bumped frontmatter
- `wiki/concepts/navsim-benchmark.md` — added DiffusionDriveV2 to NAVSIM v1 SOTA table (91.2) and v2 SOTA table (85.5 EPDMS); added caveat paragraph; bumped frontmatter

**Index updated**: added DiffusionDriveV2 row to Sources table.

**Key facts**:
- 91.2 PDMS NAVSIM-v1 (strong non-VLM diffusion result at ingest time; +3.1 over DiffusionDrive; later superseded by DriveSuprim); 85.5 EPDMS NAVSIM-v2
- EC = 91.0 on NAVSIM-v2 (highest extended comfort in wiki)
- NAVSIM-v2 caveat: 85.5 EPDMS below DriveDreamer-Policy (88.7), DreamerAD (87.7), Senna-2 (86.6); those methods excluded from V2's comparison table
- Intra-Anchor GRPO prevents mode collapse from cross-intent advantage comparison (+0.9 PDMS ablation)
- Inter-Anchor Truncated GRPO provides global collision penalty floor (+0.6 PDMS ablation)
- Multiplicative exploration noise preserves trajectory smoothness (+0.4 PDMS ablation)

---

## 2026-04-15 — Lint + New Concept Pages

**Lint fixes applied**:
- `wiki/concepts/world-model-for-ad.md` — Pattern 8 (FLARE) was displaced after the SOTA tables (after Patterns 9 and 10); moved to its correct position between Pattern 7 (DriveVLA-W0) and Pattern 9 (DreamerAD). `updated` bumped to 2026-04-15.
- `wiki/concepts/navsim-benchmark.md` — Added disambiguation dagger to ReCogDrive SOTA row: 89.6 (arXiv) vs. 90.8 (NeurIPS camera-ready as cited by DreamerAD).
- `wiki/concepts/inference-time-safety.md` — Added DriveFine Block-MoE to taxonomy table with training-time vs. inference-time contrast; added `sources/drivefine.md` and `sources/diffusiondrive.md` to `related` and `sources` frontmatter; `updated` bumped to 2026-04-15.

**New concept pages created**:
- `wiki/concepts/best-of-n.md` — Oracle BoN sampling; NAVSIM-v1 saturation at BoN-6 (94.8 = human GT); DreamerAD vocabulary sampling as deployable variant; implications for benchmark interpretation
- `wiki/concepts/bench2drive.md` — CARLA V2 closed-loop benchmark; DS + SR metrics; full SOTA table (LinkVLA 91.01 DS current SOTA); PDM-Lite caveat; contrast with NAVSIM
- `wiki/concepts/chain-of-thought-for-ad.md` — Text/visual/self-reflection CoT taxonomy; 3 annotation strategies (frontier VLM, GT-grounded, LRM-as-critic); adaptive CoT (AdaThinkDrive); NoRD as reasoning-free counterpoint; efficiency tradeoff table

**Index updated**: added 3 new concept rows.

---

## 2026-04-15 — Ingest: DiffusionDrive

**Source**: `raw/papers/DiffusionDrive_ Truncated Diffusion Model for End-to-End Autonomous Driving.md`
**arXiv**: 2411.15139v1
**Org**: HUST (Institute of AI + School of EIC); Horizon Robotics
**Venue**: pre-VLA era (November 2024)

**Pages created**:
- `wiki/sources/diffusiondrive.md`

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added full DiffusionDrive section: two failure modes of vanilla diffusion (mode collapse 11% diversity, 7 FPS), truncated diffusion policy (20 K-Means anchors, T_trunc=50/1000, 2 DDIM steps), cascade diffusion decoder (deformable spatial+agent/map cross-attention, 2 layers shared params, 60M/-39% params), progression table (Transfuser→DD), DiffusionDrive vs. VLA-era comparison; added DiffusionDrive as first row of design space table; updated DFM comparison to include source link; updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — filled in DiffusionDrive table row note (truncated diffusion, 20 anchors, 2 steps, ResNet-34, 45 FPS); added DiffusionDrive caveat (comparison scope limited to Transfuser-era; 88.1 PDMS SOTA at publication, superseded by VLA methods); added DiffusionDrive to sources/related frontmatter
- `wiki/index.md` — added DiffusionDrive row

**Assets embedded**:
- `x1 19.png` — paradigm comparison (single-mode / vocab / vanilla diffusion / truncated diffusion)
- `x2 17.png` — mode diversity visualization (vanilla diffusion mode collapse)
- `x4 16.png` — truncated vs. vanilla diffusion schedule illustration
- `x5 17.png` — overall DiffusionDrive architecture

**Key findings**:
- Vanilla diffusion policy applied to driving: 11% mode diversity (near-complete collapse), 7 FPS — both unacceptable
- Truncated diffusion: start from anchored Gaussian (20 K-Means clusters), truncate forward schedule to 50/1000 steps, denoise in 2 steps → 74% diversity, 27 FPS
- Cascade decoder (deformable BEV/PV + agent/map cross-attention, 2 layers shared, 60M) beats UNet (101M) by +2.4 PDMS at −39% params → 45 FPS
- Spatial cross-attention is critical: removing it collapses PDMS from 87.1 to 55.1 (−32 PDMS)
- Inference flexibility: N_infer decoupled from N_anchor — dynamically scale trajectory hypotheses
- 88.1 PDMS was SOTA at publication; now the canonical non-VLM diffusion baseline, superseded by ReCogDrive (89.6), WAM-Flow (90.3), DriveFine (90.7), FLARE (91.4)
- nuScenes: 0.57m avg L2 / 0.08 collision (beats VAD by −20.8% L2, −63.6% collision, 1.8× faster)

---

## 2026-04-15 — Ingest: NoRD

**Source**: `raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md`
**arXiv**: 2602.21172v1
**Org**: Applied Intuition; Texas A&M University; UC Berkeley

**Pages created**:
- `wiki/sources/nord.md`

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added NoRD section: difficulty bias identification (polarized reward distribution), GRPO attenuation mechanism (std normalization kills high-variance gradients), Dr. GRPO formulation (remove std, DAPO asymmetric clipping, no KL), reward design (format+length+PDMS), sub-metric comparison table, position in GRPO landscape; added NoRD row to GRPO reward comparison table
- `wiki/concepts/vlm-domain-adaptation.md` — added NoRD section: reasoning-free hypothesis, adaptation design (no CoT at any stage), data efficiency finding, contrast with FLARE and AutoVLA; added NoRD row to strategy comparison table
- `wiki/concepts/navsim-benchmark.md` — added NoRD (85.6 PDMS, no reasoning, no LiDAR, 3C, 80K samples) and NoRD-BoN (92.4, BoN-6, surpasses AutoVLA-BoN); added caveat on comparison scope and data efficiency framing
- `wiki/index.md` — added NoRD row

**Assets embedded** (all in raw/assets/):
- `x1 18.png` — training pipeline comparison (existing vs. NoRD)
- `difficulty_plot.png` — reward distribution for NoRD-base
- `grpo_steps.png` — GRPO training step analysis
- `comparison_figure.png` — GRPO vs. Dr. GRPO qualitative (sharp turn + lane change)
- `nord.png` — model architecture
- `x2 16.png` — NAVSIM Pareto frontier
- `navsim_examples.png` — qualitative NAVSIM results
- `waymo_results.png` — qualitative WaymoE2E results
- `nord_efficient.png` — token and runtime efficiency
- `contour_plots.png` — training improvement per variance group
- `x4 15.png` — training and validation curves
- `prompt_example.png` — inference example

**Key findings**:
- Standard GRPO fails on weak SFT policies because high intra-group variance attenuates GRPO advantages: +0.67% gain only
- Dr. GRPO (remove std normalization + DAPO asymmetric clipping, no KL) achieves +11.68% from same weak base
- Reasoning annotations are not the bottleneck: NoRD matches AutoVLA-BoN (92.4 vs. 92.1) with no CoT and 60% less data
- WaymoE2E: best ADE@3 (1.2504) with 6–17× less training data than SOTA; 3rd RFS (7.709) without reasoning or ensembling
- First identification of difficulty bias failure mode in autonomous driving domain
- Connection: Curious-VLA identified advantage collapse (policy too narrow → $\sigma_R \to 0$); NoRD identifies advantage attenuation (policy too weak → $\sigma_R$ too large); both starve GRPO from opposite distributional extremes

---

## 2026-04-08 — Ingest: Vega

**Source**: `raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md`
**arXiv**: 2603.25741v1
**Org**: Tsinghua University + GigaAI

**Pages created**:
- `wiki/sources/vega.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 10: Instruction-Conditioned World Model (Vega); updated World Model vs. VLA table to include NL instruction following row; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added Vega section on instructional driving paradigm; added InstructScene annotation pipeline; updated final strategy comparison table (now 15 rows); updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — added Vega to NAVSIM-v1 (87.9 / 89.8 BoN-6) and NAVSIM-v2 (86.9 / 89.4 BoN-6) SOTA tables; updated SOTA note (Vega BoN-6 likely new NAVSIM-v2 wiki SOTA but no direct head-to-head); added Vega caveat note; updated frontmatter
- `wiki/index.md` — added Vega row

**Key concepts**:
- Instructional driving: open-ended NL instruction → different trajectory in same scene (vs. imitation driving with fixed expert target)
- InstructScene: 100K automated instruction annotations via Qwen2.5-VL-72B two-stage pipeline (scene understanding → instruction formulation) + rule-based ego-motion labels
- Dense supervision bridge: future image prediction resolves instruction-to-action gap; action-only SFT fails catastrophically (51.8 PDMS); world modeling enables it (77.9→86.9 EPDMS)
- Integrated AR+Diffusion transformer (Bagel-7B, MoT): all transformer params duplicated per understanding/generation module; no information bottleneck (vs. external diffuser)
- Duplicate latent trick: noisy copy for denoising + clean copy for conditioning → joint multi-task training in single forward pass
- Lightweight action expert (hidden=256): separate from understanding/generation modules; diffusion as action planner fails catastrophically (19.7 PDMS)
- CFG: text/ViT/action tokens dropped during training → instruction guidance strength at inference
- NAVSIM-v2: 86.9 EPDMS (single, no RL) / 89.4 BoN-6; NAVSIM-v1: 87.9 PDMS / 89.8 BoN-6; 1 camera only
- EC = 76.3 (single) / 84.5 (BoN) — improved by instruction-consistent planning but not best-in-class

---

## 2026-04-08 — Ingest: DreamerAD

**Source**: `raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md`
**arXiv**: 2603.24587v1
**Org**: Chongqing Chang'an Technology Co., Ltd.

**Pages created**:
- `wiki/sources/dreameraD.md`

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added DreamerAD section (latent world model RL; SF-WM + AD-RM + Gaussian vocab sampling); added DreamerAD row to GRPO comparison table; clarified DreamerAD's unique position as only method using latent features (not simulator) as RL reward source; updated frontmatter sources/related
- `wiki/concepts/world-model-for-ad.md` — added Pattern 9: Latent World Model as RL Reward Source; resolved open question "can world model provide RL rewards?" with DreamerAD evidence; updated frontmatter sources/related
- `wiki/concepts/navsim-benchmark.md` — added DreamerAD to NAVSIM-v1 SOTA table (88.7 PDMS); added DreamerAD to NAVSIM-v2 SOTA table (87.7 EPDMS, EC=72.4); added DreamerAD caveat note; added contextual note that DreamerAD becomes second in wiki behind DDP; updated frontmatter
- `wiki/index.md` — added DreamerAD row

**Key concepts**:
- First latent-space RL framework for AD: rewards from learned AD-RM on denoised Video DiT features, not PDM simulator (at RL time)
- Shortcut Forcing (SF-WM): recursive multi-resolution teacher-student distillation; 100→1 step, 80× speedup, 0.03s/frame, no EPDMS degradation
- PCA finding: denoised latent features show structured spatial/semantic coherence → sufficient for reward learning without decoding
- AD-RM data efficiency: 20% training data ≈ 100% reward model performance; high-quality latent representations simplify reward learning
- Safety-first log-sigmoid reward: collisions force log(σ(r)) → −∞, dominating total reward without manual safety weights
- Gaussian vocabulary sampling: Mahalanobis ranking over 8192→256 filtered trajectories; avoids WorldRFT dynamic discontinuity and Flow-GRPO SDE mismatch
- NAVSIM-v2 87.7 EPDMS: +2.6 over Epona; safety metrics NC +0.9, DAC +1.5, TTC +1.1; EP −0.8 (safety-efficiency tradeoff)
- NAVSIM-v1 88.7 PDMS: best within world-model encoder class; below VLA SOTA (FLARE 91.4, RecogDrive 90.8) using stronger encoders

---

## 2026-04-07 — Ingest: FLARE

**Source**: `raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md`
**arXiv**: 2601.05611v2
**Org**: OpenDriveLab + Li Auto

**Pages created**:
- `wiki/sources/flare.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 8: DINOv2 semantic feature prediction as self-supervised auxiliary loss; action-conditional FFP; prediction target ablation; contrast with DriveVLA-W0 (VAE latents) and FSDrive (visual CoT); updated open questions
- `wiki/concepts/rl-for-ad.md` — added FLARE's BC-regularized GRPO section; BC vs. KL comparison; updated GRPO reward comparison table (now 10 methods)
- `wiki/concepts/vlm-domain-adaptation.md` — added annotation-free adaptation section; positioning table (FLARE vs. DriveVLA-W0 as only annotation-free methods); updated strategy table (now 13 rows)
- `wiki/concepts/navsim-benchmark.md` — added FLARE SFT (86.9) and RFT (91.4) rows; updated EPDMS table (86.3, EC=87.5); updated SOTA statement (FLARE 91.4 best single-sample VLM-based, caveat: no head-to-head with DriveFine/WAM-Flow)
- `wiki/index.md` — added FLARE row

**Key concepts**:
- Annotation-free paradigm: no VQA/CoT needed; DINOv2 patch features as dense self-supervision
- Future Feature Predictor (FFP): spatial queries modulated by action vector z → cross-attention on visual latents → predict DINOv2 patches of next frame
- Action-conditional prediction: FFP conditioned on z simulates how planned action changes the scene
- Prediction target hierarchy: spatial DINO (86.9) > global DINO (85.9) > pixels (84.7) > none (83.4)
- BC regularization instead of KL divergence in GRPO Stage 2 (motivated by DriveFine reward hacking finding)
- NAVSIM-v1: 86.9 SFT (best VLM SFT, no external data), 91.4 RFT (best single-sample VLM)
- NAVSIM-v2: 86.3 EPDMS (comparison scope excludes Senna-2 86.6 and DDP 88.7); EC=87.5 (healthy)
- Two-stage MAP fusion: visual MAP → N_v latents; ego-state-conditioned action MAP → single decision vector z

---

## 2026-04-07 — Ingest: UniDriveVLA

**Source**: `raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md`
**arXiv**: 2604.02190v1
**Org**: HUST + Xiaomi EV + University of Macau

**Pages created**:
- `wiki/sources/unidrivevla.md`

**Pages updated**:
- `wiki/concepts/dual-system-vla.md` — added MoT as third structural paradigm; UniDriveVLA + AutoMoT design comparison; Masked Joint Attention pattern; MoT ablation table; updated master comparison table (now 5 methods)
- `wiki/concepts/perception-for-planning.md` — added sparse query-based perception section; cosine similarity collapse evidence; perception–reasoning conflict diagnosis; updated comparison table (now 6 approaches including UniDriveVLA)
- `wiki/concepts/vlm-domain-adaptation.md` — added UniDriveVLA section: interference diagnosis, MoT fix, 3-stage progressive training, general VQA degradation data; updated strategy comparison table (now 12 approaches)
- `wiki/index.md` — added UniDriveVLA row; updated perception-for-planning description

**Key concepts**:
- Perception–reasoning conflict: cosine similarity → 1 in shared-weight decoder = feature collapse
- MoT: decoupled und/per/act experts; und causally masked from per/act; per reads und; act reads both
- Sparse perception: K-Means instance banks; 5-task unified decoder (det/map/ego/motion/occ); two-pass enrichment via masked joint attention
- 3-stage training: full VLM SFT → LoRA + 0.5× LR joint → VLM frozen specialization
- MoT ablation: +14.4pp General VQA, +4.1pp DriveBench, −0.108m L2 vs. shared-weight
- General VQA after adaptation still −19.7pp vs. base Qwen3-VL (MoT reduces but doesn't eliminate forgetting)
- Bench2Drive: 78.37 DS best w/o PDM-Lite; 11.78 comfort (lowest in table)
- nuScenes: 0.51m avg L2 no-ego (Large, best shown); with-ego 0.42m (FSDrive at 0.28m is better)

---

## 2026-04-07 — Ingest: DriveVLA-W0

**Source**: `raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md`
**arXiv**: 2510.12796v1
**Org**: CASIA + Yinwang Intelligent Technology

**Pages created**:
- `wiki/sources/drivevla-w0.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 7: training-time-only world modeling for data scaling; supervision deficit framing; VQ AR vs. diffusion WM design; generalization and scaling findings
- `wiki/concepts/navsim-benchmark.md` — added DriveVLA-W0 (90.2★ anchor-based, 93.0 BoN-6); v2 table updated (86.1 EPDMS, EC=58.9); added caveat on anchor-based 90.2
- `wiki/concepts/diffusion-planner.md` — added action decoder scaling reversal section (FM vs. AR vs. query-based at 103k vs. 70M frames)
- `wiki/index.md` — added DriveVLA-W0 row

**Key concepts**:
- Supervision deficit: sparse action signal wastes VLA capacity; future image prediction as dense self-supervision
- AR world model (VQ/Emu3): predicts current frame tokens; diffusion WM (ViT/Qwen2.5-VL): predicts future frame $I_{t+1}$
- World modeling unlocks cross-dataset generalization; action-only VLAs overfit and degrade (VLA-VQ: −9.5%)
- 70M-frame scaling: WM adds +28.8% ADE (VQ), +15.9% collision (ViT) vs. action-only at scale
- Action decoder reversal: FM > AR at 103k frames; AR > FM at 70M frames
- 90.2 PDMS uses trajectory anchors (not single-sample); single-sample query-based = 88.4 PDMS
- EC = 58.9 on NAVSIM-v2 (lowest in wiki)

---

## 2026-04-07 — Ingest: DriveDreamer-Policy

**Source**: `raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md`
**arXiv**: 2604.01765v1
**Org**: GigaAI + University of Toronto + CUHK MMLab

**Pages created**:
- `wiki/sources/drivedreamer-policy.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 6: geometry-grounded causal WAM (depth→video→action); added NAVSIM FVD table; updated Open Questions
- `wiki/concepts/navsim-benchmark.md` — added DDP (89.2 PDMS, 88.7 EPDMS new SOTA); updated NAVSIM-v2 SOTA table with DDP; added caveat on comparison scope and ReCogDrive IL-only baseline issue
- `wiki/index.md` — added DriveDreamer-Policy row

**Key concepts**:
- Causal depth→video→action ordering: single LLM forward pass, no iterative cross-branch refinement
- Depth as geometric scaffold: reduces FVD −18.6% for video; +0.5 PDMS for planning alone
- Depth+video combined: +1.2 PDMS over action-only baseline (88.0→89.2)
- NAVSIM-v2 88.7 EPDMS (EC=79.4): new apparent SOTA, surpasses Senna-2 (86.6) by +2.1
- No RL; single-stage multi-task training; pseudo-label depth from DA3
- Comparison gaps: excludes DriveFine (90.7), WAM-Flow (90.3), uses IL-only ReCogDrive (86.5)

---

## 2026-04-07 — Ingest: FutureSightDrive

**Source**: `raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md`
**arXiv**: 2505.17685v3
**Org**: Xi'an Jiaotong University + Amap (Alibaba Group)

**Pages created**:
- `wiki/sources/futuresightdrive.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 5: Visual CoT as Planning Intermediate (FSDrive); updated nuScenes FID and planning SOTA tables; updated Open Questions
- `wiki/concepts/vlm-domain-adaptation.md` — added FSDrive section (vocabulary expansion, visual CoT modality, modality gap ablation); updated CoT design space table (8 rows); updated strategy comparison table (11 rows)
- `wiki/concepts/navsim-benchmark.md` — added FSDrive (85.1 PDMS) to SOTA table; added caveat on comparison scope
- `wiki/index.md` — added FutureSightDrive row

**Key concepts**:
- Visual spatio-temporal CoT: generated unified future frame (red lane dividers + 3D boxes) as planning intermediate
- Dual-role VLA: world model (generates visual CoT) + inverse dynamics model (plans from current obs + visual CoT)
- Vocabulary expansion: VQ-VAE tokens appended to text vocabulary, no architectural change, ~0.3% of prior methods' data
- Progressive generation: lane dividers → 3D boxes → full frame enforces physical laws before appearance
- CoT ablation: visual ST-CoT reduces collision 31% vs. no CoT; text CoT only 8.6%
- 85.1 PDMS NAVSIM (pre-2025 comparisons only); 0.96m L2 nuScenes (no ego status, 2B); FID 10.1

---

## 2026-04-06 — Ingest: AdaThinkDrive

**Source**: `raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md`
**arXiv**: 2509.13769v1
**Orgs**: Xiaomi EV, Tsinghua University

**Pages created**:
- `wiki/sources/adathinkdrive.md` — full source summary with all figures and Tables I–VI (NAVSIM comparison, Think/NonThink SFT/RL comparison, inference time, per-level analysis, training ablation, reward ablation)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added AdaThinkDrive section: empirical CoT-hurts-simple finding, 4-component GRPO reward, Adaptive Think Reward Algorithm 1 (dynamic scene relabeling via rollout comparison, T=0.9 threshold), ablation results, AdaThinkDrive vs. AutoVLA comparison table; updated overall reward comparison table to 9-method version; removed stale statement that AutoVLA "is the only" efficiency approach; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added AdaThinkDrive section: scene complexity categorization (3 levels, CIPO-1/2/Motion Interaction), dual-mode SFT (same-query Think+NonThink vs. AutoVLA's separate fast/slow), comparison table; updated strategy comparison table to 9 rows; updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — added AdaThinkDrive (90.3) and BoN-4 (93.0) to SOTA table; added AdaThinkDrive caveat (no WAM-Flow/Curious-VLA head-to-head, Hydra-NeXt reference baseline is non-VLM); updated sources/related frontmatter

**Index updated**: yes

**Key findings**:
- Empirical proof that CoT hurts in simple scenarios (first paper in wiki to establish this rigorously with 3-level complexity analysis)
- Adaptive Think Reward achieves +2.0 vs. Never-Think RL and +1.4 vs. Always-Think RL — adaptive beats both fixed modes on all levels
- 84% Non-Think in simple scenes, 96% Think in challenging scenes — clean behavioral confirmation
- Three papers (WAM-Flow, Curious-VLA, AdaThinkDrive) independently claim 90.3 PDMS on NAVSIM-v1 with no head-to-head comparison; DriveFine (90.7) remains the single-sample SOTA

---

## 2026-04-05 — Ingest: Alpamayo-R1

**Source**: `raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md`
**arXiv**: 2511.00088v1
**Org**: NVIDIA

**Pages created**:
- `wiki/sources/alpamayo-r1.md` — full source summary with all figures and Tables 6–13 (open-loop CoC ablation nominal+challenging, closed-loop AlpaSim, RL ablation, LingoQA backbone comparison, FM vs. AR decoding, vision encoding comparison, inference latency breakdown)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added Alpamayo-R1 section: LRM-as-critic (generation-verification gap rationale), binary CoC-action consistency reward, trajectory quality reward (L2+collision+jerk), critical Table 9 finding (reasoning-only RL hurts ADE+consistency), Boltzmann RL data curation, GRPO formulation, full 8-method reward comparison table; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added Alpamayo-R1 section: Cosmos-Reason Physical AI backbone (LingoQA comparison, complementarity with AutoMoT finding), CoC dataset (3 desiderata, hybrid 2-stage human + GPT-5 labeling, +132.8% causal score), CoT comparison table with causal locality and decision grounding flags, updated 8-row strategy comparison table; updated sources/related frontmatter
- `wiki/concepts/diffusion-planner.md` — added unicycle dynamics control representation, FM action expert formulation (OT path, Euler integration), dual representation rationale (discrete training for GRPO + FM for inference), FM vs. AR decoding Table 11 (97% vs. 44% comfort), UniUGP FM comparison; added AR1 row to design space table; updated sources/related frontmatter

**Index updated**: yes

**Key findings**:
- Reasoning-only RL hurts action quality (ADE 2.12→2.19m) — consistency reward is essential for grounding reasoning to executable behavior
- Cosmos-Reason Physical AI pre-training (+6.4% LingoQA vs. Qwen2.5-VL-7B) without catastrophic forgetting — supports domain-aligned pre-training over general VLM fine-tuning
- Flow matching dominates AR trajectory decoding on comfort (97% vs. 44%) and closed-loop safety (1.27 vs. 0.59 AlpaSim at-fault)
- No NAVSIM/nuScenes/Bench2Drive results — internal NVIDIA dataset only; direct comparison with wiki peers is not possible

---

## 2026-04-05 — Ingest: AutoDrive-R²

**Source**: `raw/papers/AutoDrive-R²_ Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2509.01944v1
**Assets read**: x2 11.png (pipeline overview), x3 12.png (qualitative comparison)

**Pages created**:
- `wiki/sources/autodrive-r2.md` — full source summary with all figures and 4 tables (nuScenes L2, Waymo zero-shot L2, ablation study, group size ablation)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added AutoDrive-R² physics-grounded reward section (4-component MSE formulas + ablation), SFT cold-start necessity empirical confirmation, GT-based GRPO comparison table; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added AutoDrive-R² self-reflection CoT section (nuScenesR²-6K dataset, 4-step chain, self-reflection backward-check, "aha moment", CoT comparison table); updated sources/related frontmatter

**Index updated**: yes

**Note**: AutoDrive-R² 7B achieves 0.19m avg L2 on nuScenes with only 6K training samples — substantially better than EMMA+ (0.29m, ~103K). The self-reflection step (4th CoT stage: backward-checking physical feasibility before emitting answer) is unique across all wiki papers. The RL-only ablation (0.33m vs. SFT-only 0.27m vs. full 0.19m) independently confirms the Curious-VLA finding that SFT cold-start quality is prerequisite for effective RL. The physics reward ablation shows spatial alignment is indispensable (removal → 0.53m near-collapse); temporal smoothness is the second most critical component.

---

## 2026-04-05 — Ingest: AutoMoT

**Source**: `raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2603.14851v1
**Venue**: ICML
**Assets read**: x1 13.png (four-paradigm comparison), x2 10.png (architecture overview), x3 11.png (attention pattern visualization)

**Pages created**:
- `wiki/sources/automot.md` — full source summary with all figures and 8 tables (Bench2Drive, nuScenes open-loop, reasoning benchmarks, VLM boundary ablation, sync vs. async planning, sync vs. async decision, component ablation, Senna decision benchmark)

**Pages updated**:
- `wiki/concepts/dual-system-vla.md` — added AutoMoT layer-wise KV cache async pattern; updated comparison table to include AutoMoT vs. Senna/Senna-2/ReCogDrive; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added AutoMoT frozen VLM empirical evidence section (catastrophic forgetting table); added AutoMoT to strategy comparison table; updated sources/related frontmatter

**Index updated**: yes

**Note**: AutoMoT's primary contribution is the **catastrophic forgetting finding** — AD fine-tuning of VLMs gives marginal scene understanding gain (+0.2 LingoQA) while destroying general reasoning (TallyQA −35%, InfoVQA −44%). This is the only paper in the wiki with systematic evidence against VLM fine-tuning. Bench2Drive 87.34 DS is best among VLM-augmented methods in its comparison table, but LinkVLA (91.01) is absent and likely supersedes it. No NAVSIM results — cannot compare with recent PDMS leaders.

---

## 2026-04-05 — Ingest: AutoVLA

**Source**: `raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md`
**arXiv**: https://arxiv.org/html/2506.13757v1
**Assets read**: x1 12.png (overview), x2 9.png (4-paradigm comparison), x3 10.png (training pipeline), x4 9.png (data scaling), x5 9.png (RFT results), x6 8.png (Waymo E2E), x7 8.png (action codebook), x8 5.png (reasoning annotation pipeline), x9 1.png (Waymo reasoning examples), x10 1.png (nuPlan reasoning examples), x11 2.png (system prompt)

**Pages created**:
- `wiki/sources/autovla.md` — full source summary with all 11 figures and 3 tables (NAVSIM, Bench2Drive, action tokenization ablation)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added AutoVLA adaptive reasoning via CoT length penalty section; reward table comparing all GRPO reward designs in wiki; updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — added AutoVLA 89.11 PDMS (post-RFT) and 92.12 BoN to SOTA table; added comparison-scope caveat; updated sources/related frontmatter
- `wiki/concepts/diffusion-planner.md` — added AR over physical codebook (AutoVLA) as 10th paradigm in design space table; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added AutoVLA dual-mode SFT section (data scaling finding, GT-hint annotation, adaptive reasoning via RFT); updated sources/related frontmatter

**Index updated**: yes

**Note**: AutoVLA post-RFT 89.11 is below current SOTA (DriveFine 90.7, Curious-VLA 90.3). The primary contribution is not SOTA performance but the CoT length penalty mechanism — the only approach in the wiki that explicitly optimizes for reasoning *efficiency* rather than just quality. The physical action codebook ablation (59.24 → 80.54 PDMS for text waypoint vs. physical tokens) is a strong argument against text waypoint representations. The data scaling finding (CoT < action-only at < 50k samples) is a useful calibration for choosing when CoT training is worth the cost.

---

## 2026-04-05 — Ingest: Curious-VLA

**Source**: `raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md`
**arXiv**: https://arxiv.org/html/2603.06049
**Assets read**: x1 11.png (behavioral diagnostics quantitative), x3 9.png (overall pipeline), x4 8.png (horizon scale mismatch visualization), x5 8.png (qualitative BEV comparison)

**Pages created**:
- `wiki/sources/curious-vla.md` — full source summary with all 4 figures and 6 tables (NAVSIM v1, NAVSIM v2, nuScenes, behavioral diagnostics, FTE ablation, RL ablation)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added Narrow Policy analysis (3 root causes, advantage collapse formula), Behavioral Diagnostics framework, FTE (DE+CoT+SN), ADAS (Bernoulli filter), SDR (focal-loss reward); updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — added Curious-VLA 90.3 PDMS (v1) and 94.8 BoN-6 to SOTA table; added 85.3 EPDMS (v2) with comparison-scope caveat; updated sources/related frontmatter

**Index updated**: yes

**Note**: Curious-VLA (90.3, 1xC, 3B) ties AdaThinkDrive (8B) and is slightly below DriveFine (90.7, 1xC). DriveFine remains single-sample SOTA. The BoN-6 result of 94.8 is the most significant finding: it matches human GT (94.8) and validates that FTE+DARL successfully unlocks exploration potential. Critical negative findings: (1) difficulty-based RL sampling causes training collapse (35.2 PDMS) — not hard scenarios, but diverse-outcome scenarios are needed for GRPO; (2) DE alone without SN hurts performance (85.2 < 85.6 baseline) — diversity expansion must be paired with step-wise normalization to be effective.

---

## 2026-04-05 — Ingest: DriveFine

**Source**: `raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md`
**arXiv**: https://arxiv.org/html/2602.14577v1
**Assets read**: x1 10.png (decoding comparison), x2 8.png (RFT reward hacking finding), x3 8.png (irreversible decoding failures), x4 7.png (architecture overview), x5 7.png (hybrid RL pipeline), x6 6.png (before/after refinement), x7 6.png (PDMS-latency trade-off)

**Pages created**:
- `wiki/sources/drivefine.md` — full source summary with all 7 figures and 7 tables (NAVSIM v1 PDMS, v2 EPDMS, Navhard EPDMS, component ablation, PDMS/EPDMS robustness, refinement block count, group size sensitivity)

**Pages updated**:
- `wiki/concepts/navsim-benchmark.md` — added DriveFine 90.7/91.8 PDMS (v1) and 89.7 EPDMS (v2, bug-fixed); added Navhard benchmark section; updated SOTA summary; updated sources/related frontmatter
- `wiki/concepts/rl-for-ad.md` — added DriveFine reward-hacking finding (diffusion planners lose EPDMS under PDMS GRPO); added hybrid offline+online RL for refinement expert; updated sources/related frontmatter
- `wiki/concepts/diffusion-planner.md` — added block-MoE refinement section; DriveFine vs. ReflectDrive comparison table; updated design space table (9th paradigm); updated sources/related frontmatter

**Index updated**: yes

**Note**: DriveFine (90.7, 1xC) is now the broadly-verified single-camera NAVSIM-v1 SOTA, surpassing WAM-Flow (90.3). DriveFine* (91.8) requires an additional trained scorer. NAVSIM-v2 89.7 EPDMS uses a bug-fixed scorer not comparable to prior results. The reward-hacking finding (diffusion planners degrade EPDMS under PDMS GRPO while token-based VLAs do not) is a practically important negative result for the field.

---

## 2026-04-05 — Ingest: HERMES

**Source**: `raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2602.00993v1
**Assets read**: x1 9.png (paradigm comparison), x2 7.png (architecture overview), x3 7.png (Intent Modulator), x4 6.png (Risk Planning Cross-Attention), x5 6.png (nighttime rain qualitative), x6 5.png (low-visibility residential qualitative), x7 5.png (construction zone qualitative), x8 4.png (urban intersection qualitative)

**Pages created**:
- `wiki/sources/hermes.md` — full source summary with all 8 figures and 4 tables (overall performance, category-wise RFS, ablation, prompt design)

**Pages updated**:
- `wiki/concepts/vlm-domain-adaptation.md` — added HERMES offline VLM annotation / teacher-student distillation section; comparison table of VLM adaptation strategies; long-tail as an adaptation axis

**Index updated**: yes

**Note**: HERMES is the only paper in the wiki targeting WOD-E2E (Waymo real-world open-loop). Not comparable to NAVSIM or Bench2Drive papers. The offline-annotator-only pattern (VLM never runs at inference) is new to the wiki. Caveat: baseline fairness is questionable — LightEMMA is zero-shot, HERMES trains end-to-end on the full training split. Historical motion state is the most critical component (−1.30 RFS without it), outweighing semantic embeddings (−0.60 RFS).

---

## 2026-04-05 — Ingest: LinkVLA

**Source**: `raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2603.01441v1
**Assets read**: x1 8.png (latency-performance overview), x3 6.png (architecture), x4 5.png (bidirectional objective), x5 5.png (qualitative instruction following), x6 4.png (uniform vs. log grid), x7 4.png (additional qualitative)

**Pages created**:
- `wiki/sources/linkvla.md` — full source summary with all 7 figures and 9 tables (Bench2Drive, latency, instruction following, DriveLM VQA/commentary, closed-loop ablation, soft-label, navigation modality, codebook size, σ ablation)

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added shared codebook C2F paradigm (8th in design space table); full section on unified token space, bidirectional objective, and C2F decoder; updated sources/related frontmatter

**Index updated**: yes

**Note**: LinkVLA sets new Bench2Drive SOTA (91.01 DS, 74.55% SR), surpassing ORION (77.74) and SimLingo (85.07). Evaluated only on Bench2Drive/CARLA — no NAVSIM comparison possible. CoT latency is excluded from the reported 48ms. The bidirectional action-captioning objective is the most principled alignment contribution: no new data required, works by enriching shared token embeddings through the inverse task.

---

## 2026-04-05 — Ingest: ORION

**Source**: `raw/papers/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation.md`
**arXiv**: https://arxiv.org/html/2503.19755v1
**Assets read**: x1 7.png (4-paradigm comparison), x2 6.png (full ORION pipeline), x3 5.png (QT-Former architecture), x4 4.png (qualitative results), x5 4.png (paradigm ablation bar chart), x6 3.png (Chat-B2D annotation pipeline)

**Pages created**:
- `wiki/sources/orion.md` — full source summary with all 6 figures and 7 tables (Bench2Drive closed-loop, Multi-Ability, VAE vs. diffusion, QT-Former ablation, history query ablation, VQA+planning joint training, nuScenes open-loop)

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added ORION VAE-based reasoning-action alignment section; updated design space table to include VAE+GRU as 6th paradigm; updated sources/related frontmatter

**Index updated**: yes

**Note**: ORION is evaluated on Bench2Drive (CARLA V2) only — not NAVSIM. nuScenes open-loop 0.34 avg L2 is competitive but below Senna (0.22). VAE clearly outperforms diffusion as the generative planner (77.74 vs. 71.97 DS). Weakness: Merging (25%) and Give Way (30%) Multi-Ability scores — lane-changing decisions remain hard for VLM causal reasoning.

---

## 2026-04-05 — Ingest: Senna-2

**Source**: `raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md`
**arXiv**: https://arxiv.org/abs/2603.11219
**Assets read**: x1 3.png (consistency gap motivation), x2 2.png (architecture), x3 2.png (three-stage training recipe), x4 2.png (speed control qualitative), x5 2.png (collision scenario qualitative), x6 1.png (training curves), x7 1.png (additional qualitative)

**Pages created**:
- `wiki/sources/senna2.md` — full source summary with all figures and tables
- `wiki/concepts/dual-system-vla.md` — dual-system VLM + E2E architecture pattern; decision adapter; kinematic mapping; consistency alignment methods; HRL contrast

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added Senna-2 HRL section: 3DGS environments, bottom-up hierarchical RL, longitudinal penalties, contrast with NAVSIM GRPO
- `wiki/concepts/vlm-domain-adaptation.md` — added Senna-2's consistency-oriented adaptation, kinematic mapping, selective open-loop alignment
- `wiki/concepts/navsim-benchmark.md` — updated NAVSIM-v2 SOTA table; Senna-2 now leads at 86.6 EPDMS

**Index updated**: yes

---

## 2026-04-05 — Ingest: UniUGP

**Source**: `raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md`
**arXiv**: https://arxiv.org/abs/2512.09864
**Assets read**: x1 2.png (architecture), x2 1.png (data pipeline), x3 1.png (world model ablation), x4 1.png (trajectory-controllable generation), x5 1.png (long-tail QA), x8 1.png (CoT reasoning examples)

**Pages created**:
- `wiki/sources/uniugp.md` — full source summary
- `wiki/concepts/world-model-for-ad.md` — world model integration with VLA planning; FID/FVD metrics; coupling patterns

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added UniUGP's continuous FM planning + MoT architecture; world model co-training signal
- `wiki/concepts/vlm-domain-adaptation.md` — added UniUGP's staged training, CoT integration, instruction following via data, long-tail data approach

**Index updated**: yes

---

## 2026-04-05 — Ingest: WAM-Flow

**Source**: `raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md`
**arXiv**: https://arxiv.org/abs/2512.06112

**Pages created**:
- `wiki/sources/wam-flow.md` — full source summary
- `wiki/concepts/discrete-flow-matching.md` — DFM theory, CTMC dynamics, geometry-aware Gibbs paths, metric-aligned numerical tokenizer

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added DFM vs. diffusion comparison table
- `wiki/concepts/rl-for-ad.md` — added WAM-Flow GRPO section; contrasts DFM-GRPO with diffusion-chain MDP approach
- `wiki/concepts/navsim-benchmark.md` — updated SOTA tables (v1 + v2); WAM-Flow now leads both

**Index updated**: yes

---

## 2026-04-05 — Ingest: Percept-WAM

**Source**: `raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2511.19221v1
**Assets read**: x1 6.png (motivation), x2 5.png (architecture), x3 4.png (IoU confidence dataset), x5 3.png (grid query tokens), x6 2.png (trajectory decoder), x7 2.png (confidence calibration scatter), x10.png (PV perception qualitative), x11 1.png (BEV perception qualitative), x12.png (trajectory planning qualitative)

**Pages created**:
- `wiki/sources/percept-wam.md` — full source summary with all 9 figures and 7 tables (PV perception, BEV perception, trajectory planning, IoU confidence ablation, BEV ablation, decoding efficiency, dataset tasks)
- `wiki/concepts/perception-for-planning.md` — new concept: World-PV/BEV tokens; grid-conditioned parallel AR; IoU-aware confidence calibration; four-query modality-aligned decoder; comparison of perception integration approaches

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added Percept-WAM's four-query MLP decoder section; attention-masked modality alignment; reuse of perception tokens
- `wiki/concepts/navsim-benchmark.md` — added Percept-WAM\* 90.2 PDMS with caveat (LiDAR-assisted, limited comparison scope)

**Index updated**: yes

---

## 2026-04-05 — Ingest: Reasoning-VLA

**Source**: `raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2511.19912v1
**Assets read**: x1 5.png (framework + training pipeline), x2 4.png (action module cross-attention), 8data.png (dataset distribution), x3 3.png (qualitative across 8 datasets), x4 3.png (dataset construction pipeline)

**Pages created**:
- `wiki/sources/reasoning-vla.md` — full source summary with all 5 figures and 9 tables (nuScenes, NeuroNCAP, NAVSIM, generalized, zero-shot, ablation, efficiency, unified/nuScenes comparison, closed-loop comparison)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added Reasoning-VLA GT-based GRPO section; physics reward table; contrast with NAVSIM GRPO (simulator vs. trajectory-only)
- `wiki/concepts/vlm-domain-adaptation.md` — added Reasoning-VLA unified 8-dataset corpus section; CoT pipeline; zero-shot generalization findings
- `wiki/concepts/diffusion-planner.md` — added learnable action queries paradigm section; design space comparison table (5 paradigms)
- `wiki/concepts/navsim-benchmark.md` — added Reasoning-VLA 91.7 PDMS claim with caveat (comparison scope limited to old baselines only)

**Index updated**: yes

**Note**: Reasoning-VLA claims 91.7 PDMS on NAVSIM but compares only against TransFuser/UniAD/Para-Drive (~84 PDMS). No head-to-head vs. WAM-Flow (90.3) or ReCogDrive (89.6). WAM-Flow remains the last verified SOTA.

---

## 2026-04-05 — Ingest: ReflectDrive

**Source**: `raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2509.20109v1
**Assets read**: x1 4.png (framework overview), x2 3.png (safety-guided regeneration pipeline), goodcase.png (DAC + TTC violation correction), easy_case.png (1-step easy cases), medium_case.png (1–3 step medium cases), hard_case.png (1–5 step hard cases)

**Pages created**:
- `wiki/sources/reflectdrive.md` — full source summary with all figures; architecture, two-phase reflective inference, three model variants, limitations
- `wiki/concepts/inference-time-safety.md` — gradient-free inference-time safety; taxonomy vs. diffusion guidance/RL/anchors; scoring functions; inpainting-as-repair mechanism; limitations

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added ReflectDrive masked discrete diffusion section; 3-way comparison table (continuous diffusion / DFM / masked diffusion)
- `wiki/concepts/discrete-flow-matching.md` — added DFM vs. masked discrete diffusion distinction table; clarified WAM-Flow (CTMC) vs. ReflectDrive (BERT-style) are distinct paradigms despite both being "discrete diffusion"
- `wiki/concepts/navsim-benchmark.md` — added ReflectDrive to v1 SOTA table (>89.1 claimed, exact number missing from markdown)

**Index updated**: yes

**Note**: Table 1 (NAVSIM closed-loop results) was not rendered in the paper's markdown conversion. Exact NC/DAC/TTC/Comf/EP sub-metrics unavailable.

---

## 2026-04-05 — Ingest: ReCogDrive

**Source**: `raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2506.08052v1

**Pages created**:
- `wiki/sources/recogdrive.md` — full source summary
- `wiki/concepts/diffusion-planner.md` — diffusion-based trajectory planning
- `wiki/concepts/rl-for-ad.md` — RL for autonomous driving (incl. GRPO/diffusion RL)
- `wiki/concepts/vlm-domain-adaptation.md` — VLM adaptation for driving domain
- `wiki/concepts/navsim-benchmark.md` — NAVSIM benchmark and PDMS metric

**Index updated**: yes
---

## 2026-06-11 — Ingest: CLEAR

**Source**: `raw/papers/CLEAR_ Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2606.06219v1
**Assets read**: `framework.png` (architecture), `training_plot.png` (drift training dynamics)

**Pages created**:
- `wiki/sources/clear.md` — full source summary with two figures, NAVSIM-v1/v2 tables, ablation table, relationships, and limitations
- `wiki/concepts/adaptive-routing.md` — new concept for scene-conditioned candidate budget/diversity routing and learned trajectory scoring

**Pages updated**:
- `wiki/index.md` — added CLEAR source and Adaptive Routing concept; updated DriveSuprim wording
- `wiki/concepts/navsim-benchmark.md` — added CLEAR v1/v2 leaderboard entries; updated non-BoN frontier note and caveat
- `wiki/concepts/diffusion-planner.md` — added single-step VAE latent drift as a diffusion alternative
- `wiki/concepts/best-of-n.md` — distinguished CLEAR's learned adaptive routing from oracle BoN
- `wiki/concepts/selection-based-planning.md` — clarified DriveSuprim remains strongest fixed-vocabulary selector, while CLEAR is adjacent but not fixed-vocabulary
- `wiki/concepts/foundation-backbones-for-ad.md` — added Qwen hidden-state router/scorer role

**Index updated**: yes

---

## 2026-06-18 - Ingest: Understanding R1-Zero-Like Training

**Source**: `raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md`
**arXiv**: https://arxiv.org/html/2503.20783v2
**Authors**: Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin
**Confidence**: high - local markdown includes the main text, all 15 local figures, benchmark tables, GRPO bias derivation, and hyperparameter table; Table 5 is referenced but not rendered as table rows

**Pages created**:
- `wiki/sources/understanding-r1-zero-like-training.md` - full source summary covering base-model/template analysis, GRPO bias, Dr. GRPO, benchmark results, all figures, reproduced tables, AD relevance, and limitations
- `wiki/concepts/r1-zero-like-training.md` - concept page for base-model RL, template/base-prior confounds, Dr. GRPO, and AD interpretation rules

**Concept pages updated**:
- `wiki/concepts/gspo-vs-grpo.md` - added Dr. GRPO as a correction to GRPO normalization, distinct from GSPO's MoE sequence stabilization
- `wiki/concepts/rl-for-ad.md` - linked NoRD's difficulty-bias fix back to the original Dr. GRPO analysis
- `wiki/concepts/chain-of-thought-for-ad.md` - added caveat that self-reflection-like language can preexist RL and does not guarantee higher accuracy
- `wiki/concepts/foundation-backbones-for-ad.md` - added Qwen base-prior/template caveat for interpreting RL gains

**Index updated**: added source and R1-Zero-Like Training concept.

**Key facts**:
- Qwen2.5-Math base models perform best with no template, suggesting pretraining/template confounds in R1-Zero-like replication claims.
- DeepSeek-V3-Base already shows "Aha moment" examples before RL tuning.
- Standard GRPO has response-length bias and question-level difficulty bias from response-length and reward-std normalization.
- Dr. GRPO removes both normalizers and improves token efficiency while preserving reasoning performance.
- Oat-Zero-7B reports 43.3 AIME24 and 51.4 average across AIME24, AMC, MATH500, Minerva, and OlympiadBench under the paper's 3k-budget comparison.
- Main limitation: evidence is from verifiable math RL, so transfer to multimodal AD planning is methodological rather than direct.

---

## 2026-06-18 - Ingest: All Roads Lead to Rome

**Source**: `raw/papers/All Roads Lead to Rome_ Incentivizing Divergent Thinking in Vision-Language Models.md`
**arXiv**: https://arxiv.org/html/2604.00479v1
**Authors**: Xinyu Tian, Shu Zou, Zhaoyuan Yang, Mengqi He, Peter Tu, Jing Zhang
**Confidence**: high - local markdown includes main method text, all nine local figures, four tables, benchmark setup, and implementation details

**Pages created**:
- `wiki/sources/all-roads-lead-to-rome.md` - source summary covering GRPO diversity collapse, MUPO, figures, reconstructed benchmark tables, AD relevance, and limitations
- `wiki/concepts/divergent-thinking-in-vlms.md` - concept page for reasoning-strategy diversity, MUPO, and parallel test-time scaling

**Concept pages updated**:
- `wiki/concepts/gspo-vs-grpo.md` - added MUPO as a multi-group GRPO variant for reasoning diversity
- `wiki/concepts/best-of-n.md` - added candidate-diversity caveat for BoN and acc@k scaling
- `wiki/concepts/chain-of-thought-for-ad.md` - added sequential-depth vs parallel-breadth caution for CoT
- `wiki/concepts/r1-zero-like-training.md` - added GRPO diversity-collapse caveat for RL-from-base interpretation

**Index updated**: added All Roads Lead to Rome source and Divergent Thinking in VLMs concept.

**Key facts**:
- RL VLMs can outperform base models at `acc@1` while underperforming under multi-sample `acc@k` because their reasoning strategies collapse.
- Base models often retain broader alternative reasoning paths, improving their chance of success with parallel sampling.
- MUPO partitions responses into `K` reasoning groups, computes local advantages, and adds an accuracy-gated diversity reward.
- MUPO-Thinker-7B reports 51.6/58.8 average math `acc@1/acc@4`, improving over Vision-R1-7B's 49.1/52.8.
- On general benchmarks, MUPO reports 65.6/72.4 average `acc@1/acc@4`, above the listed 7B RL baselines.
- Main limitation: this is a VLM reasoning paper, not an AD evaluation; transfer to driving is methodological.

## 2026-06-18 - Ingest: Plan-R1

**Source**: `raw/papers/Plan-R1_ Safe and Feasible Trajectory Planning as Language Modeling.md`

**Pages created**:
- `wiki/sources/plan-r1.md` - source summary covering motion-token pretraining, dual-model reactive rollout, rule-based rewards, VD-GRPO, figures, tables, nuPlan/interPlan results, and limitations

**Concept pages updated**:
- `wiki/concepts/rl-for-ad.md` - added Plan-R1 as VD-GRPO safety-critical reward normalization case
- `wiki/concepts/action-tokenization.md` - added Plan-R1 motion-token language modeling
- `wiki/concepts/gspo-vs-grpo.md` - added VD-GRPO alongside Dr. GRPO and MUPO
- `wiki/concepts/inference-time-safety.md` - contrasted Plan-R1 training-time safety alignment with inference-time repair methods
- `wiki/concepts/r1-zero-like-training.md` - added Plan-R1 as a planning-as-language-modeling R1-style analogy with a GRPO caveat

**Index updated**: added Plan-R1 source and refreshed affected concept descriptions.

**Key facts**:
- Plan-R1 pretrains a 1024-token-per-category motion-token predictor, then fine-tunes the ego planner with rule-based rewards.
- Dual-model rollout keeps a frozen pretrained model as a reactive world model for surrounding agents.
- Standard GRPO improves overall score but lowers collision avoidance; VD-GRPO removes per-group std normalization to avoid downweighting rare unsafe groups.
- Plan-R1 reports 88.98/87.69 Val14 NR/R, 77.45/77.20 Test14-hard NR/R, and 91.23/90.04 Test14-random NR/R on nuPlan without post-processing.
- With post-processing, Plan-R1* reports 72.33 on interPlan, above the listed postprocessed baselines.

**Limitations**:
- Evaluation is simulation-only.
- Reward priority structure is manually designed.
- Some cited appendix tables (Table 3 and Table 4) are referenced in the raw markdown but their bodies are absent; the wiki records the textual numbers available around those references.

## 2026-06-23 - Ingest: PlannerRFT

**Source**: `raw/papers/PlannerRFT_ Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning.md`

**Pages created**:
- `wiki/sources/plannerrft.md` - full source summary covering adaptive guided denoising, PPO/GRPO dual-branch training, survival reward, nuMax, all 16 figures, and all ten paper tables
- `wiki/concepts/nuplan-benchmark.md` - nuPlan reactive/non-reactive protocol, split definitions, scoring caveats, representative learning-only results, and nuMax limitations

**Concept pages updated**:
- `wiki/concepts/diffusion-planner.md` - added learned exploration distributions and training-only guidance
- `wiki/concepts/rl-for-ad.md` - added PPO-guided exploration plus GRPO and survival reward
- `wiki/concepts/gspo-vs-grpo.md` - distinguished PlannerRFT group construction/reward shaping from Dr. GRPO and VD-GRPO normalization fixes

**Index updated**: added PlannerRFT and the nuPlan Closed-Loop Planning Benchmark concept.

**Key facts**:
- PlannerRFT learns scene-conditioned Beta distributions over lateral and longitudinal denoising guidance using PPO, then fine-tunes the DiT with GRPO.
- Uniform exploration is most diverse (39.78%) but unstable and worse-performing; adaptive exploration reaches 72.21 Test14-hard reactive with 25.34% diversity.
- Against the matched five-step DDIM baseline, PlannerRFT improves Test14-hard reactive from 68.18 to 72.21 and Test14-random reactive from 82.63 to 85.80.
- Survival reward preserves relative signal in groups whose candidates eventually collide or leave the road.
- The deployment model removes the reference/guidance modules and runs five-step DDIM at a reported 34.27 ms.
- nuMax is reported as up to 10x faster than native nuPlan through JAX/XLA simulation and fixed-shape caching.

**Limitations**:
- Structured abstract inputs only; no camera-based visuomotor evaluation.
- One benchmark family, no real-world validation, and no multiple-seed uncertainty.
- Training uses 40M environment steps on eight H100 GPUs.
- nuMax uses log-replay traffic during training, has representation-specific static caches, and is a calibrated reimplementation rather than the official simulator.

## 2026-06-23 - Ingest: PaIR-Drive

**Source**: `raw/papers/Fine-tuning is Not Enough_ A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving.md`

**Pages created**:
- `wiki/sources/pair-drive.md` - full source summary covering parallel IL/RL, tree-structured residual sampling, GRPO, RWM inference, all seven figures, and all six tables
- `wiki/concepts/parallel-il-rl.md` - concept page comparing one-shot, iterative, and parallel IL/RL and defining modularity/evaluation requirements

**Concept pages updated**:
- `wiki/concepts/rl-for-ad.md` - added reusable RL refinement as an alternative to fine-tuning
- `wiki/concepts/gspo-vs-grpo.md` - added tree-structured group construction upstream of GRPO normalization
- `wiki/concepts/best-of-n.md` - separated PaIR-Drive single-plan and Best-of-6 results
- `wiki/concepts/selection-based-planning.md` - added residual-tree generation plus RWM selection
- `wiki/concepts/navsim-benchmark.md` - added human-bad splits, Best-of-6 caveat, and aggregate-metric comfort regressions

**Index updated**: added PaIR-Drive and Parallel Imitation and Reinforcement Learning.

**Key facts**:
- PaIR-Drive trains independent IL and RL branches; the RL branch learns intention-conditioned residual trees around human trajectories with GRPO.
- At inference, the RL branch is centered on an IL proposal and an RWM selects the final trajectory.
- Without Best-of-N, TransFuser improves from 84.0 to 89.7 PDMS and 79.7 to 86.6 EPDMS; DiffusionDrive improves from 88.1 to 91.2 and 84.3 to 87.9.
- Best-of-6 raises the PaIR-Drive + DiffusionDrive results to 94.0 PDMS and 89.6 EPDMS.
- Tree-structured residual sampling reaches 93.3/88.5 PDMS/EPDMS versus 88.8/81.6 for unstructured residual sampling in the reported Best-of-6 ablation.
- Refining recorded human trajectories improves the paper's low-human-score subsets by +1.6 PDMS and +10.8 EPDMS.

**Limitations**:
- Human-reference training and IL-reference inference introduce unquantified distribution shift.
- The RWM architecture, training targets, calibration, selection details, and inference cost are largely unspecified.
- Plug-and-play reuse is demonstrated only for TransFuser and DiffusionDrive on NAVSIM.
- Peak and ablation scores use Best-of-6; no latency, seed variance, or non-NAVSIM validation is reported.
- Aggregate gains hide substantial Comfort/Extended Comfort regressions.

## 2026-06-23 - Ingest: DIAL

**Source**: `raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md`

**Pages created**:
- `wiki/sources/dial.md` - full summary covering intent-CFG, multi-intent GRPO, reward-hacking-aware RFS, all four tables, four embedded figures, and the missing Figure 4 extraction gap
- `wiki/concepts/intent-conditioned-planning.md` - concept page for intent variables as proposal-support controls, ontology design, diversity metrics, and evaluation requirements

**Concept pages updated**:
- `wiki/concepts/rl-for-ad.md` - added preference contrast through intent-balanced rollout groups
- `wiki/concepts/gspo-vs-grpo.md` - added semantic group composition as a GRPO design axis
- `wiki/concepts/best-of-n.md` - added WOD-E2E proposal-support ceilings versus deployable performance
- `wiki/concepts/diffusion-planner.md` - added intent-CFG support expansion for continuous flow policies
- `wiki/concepts/nuscenes-waymo-evals.md` - added RFS protocol, split leakage, oracle selection, and open-loop caveats

**Index updated**: added DIAL and Intent-Conditioned Trajectory Planning.

**Key facts**:
- DIAL conditions a continuous flow action head on eight rule-derived intents with classifier-free guidance.
- GRPO groups contain two samples from every intent, holding total group size at 16 while guaranteeing maneuver-level contrast.
- The controlled Waymo-only experiment improves held-out RFS from 7.696 to 8.211, above all single-intent variants.
- Pre-RL eight-intent pooling reaches oracle Best-of-128 RFS 9.14, above the logged trajectory at 8.13.
- DIAL preserves an RFS diversity dividend of +2.04 at Best-of-16, close to the SFT initialization's +2.23.
- Label-softmax rater aggregation plus dense 1–5 s anchors improves resistance to RFS reward hacking.

**Limitations**:
- The eight-intent ontology and labels are hand-engineered and incomplete for long-tail/composite behavior.
- RL uses 338 labeled validation sequences; the 100-sequence held-out partition is used for checkpoint/hyperparameter selection and is not an untouched test.
- Evaluation is open-loop WOD-E2E RFS only; no reactive, closed-loop, or cross-dataset validation is reported.
- Intent-classifier accuracy, inference latency, final-checkpoint variance, and multiple-seed uncertainty are absent.
- Best-of-128 is oracle-selected and not deployable.
- Referenced Figure 4 is absent from the raw extraction; its numeric sweep is reconstructed in the source summary.

## 2026-06-23 - Ingest: DisCO

**Source**: `raw/papers/DisCO_ Reinforcing Large Reasoning Models with Discriminative Constrained Optimization.md`

**Source condition**: the local markdown ends after Proposition 1 at line 96. The official arXiv v5 PDF was used to recover the missing method, algorithm, experiments, six tables, figure findings, conclusion, and appendices. `raw/` remained unchanged.

**Pages created**:
- `wiki/sources/disco.md` - summary of GRPO difficulty weighting, DisCO-b/DisCO objectives, hard-negative DRO, KL-constrained optimization, all six paper tables, the one available local figure, PDF-recovered findings, and limitations
- `wiki/concepts/discriminative-policy-optimization.md` - concept page for positive/negative policy scoring, hard-negative emphasis, trust-region constraints, and requirements for AD transfer

**Concept pages updated**:
- `wiki/concepts/gspo-vs-grpo.md` - added DisCO as an objective replacement and clarified residual difficulty bias in Dr. GRPO
- `wiki/concepts/r1-zero-like-training.md` - added the expected-objective analysis of GRPO/Dr. GRPO question weights
- `wiki/concepts/rl-for-ad.md` - added cautious methodological relevance for binary safety rewards and all-failed-group caveats

**Index updated**: added DisCO and Discriminative Policy Optimization.

**Key facts**:
- Binary-reward GRPO weights each question's discriminative objective by `sqrt(p(1-p))`; Dr. GRPO retains weight `p(1-p)`.
- DisCO directly increases positive rollout scores and decreases negative rollout scores without clipping.
- Full DisCO uses a DRO/partial-AUC objective to emphasize high-scoring hard negatives.
- A squared-hinge KL penalty is active only when old-to-new KL exceeds the trust threshold.
- On Qwen 1.5B with 8k responses, DisCO log-L averages 0.533 across six math tasks versus 0.457 GRPO, 0.443 Dr. GRPO, and 0.473 DAPO.
- DisCO also leads same-base comparisons on Qwen 7B and Llama 8B and generalizes to DAPO-Math-17K.

**Limitations**:
- Binary rewards and mixed positive/negative groups are required; all-wrong/all-correct groups provide no pairwise signal.
- Evidence is math-only and does not directly validate autonomous-driving policies or continuous rewards.
- Best checkpoints are reported without seed variance; full DAPO dynamic sampling is omitted.
- Compute is high and trust-region hyperparameters are empirically selected.
- Only Figure 1 exists locally; Figures 2–6 and Tables 1–6 are absent from the raw extraction, so missing figure findings are textual and tables are reconstructed from the official PDF.

## 2026-06-23 - Ingest: DAPO

**Source**: `raw/papers/DAPO_ An Open-Source LLM Reinforcement Learning System at Scale.md`

**Pages created**:
- `wiki/sources/dapo.md` - source summary covering the four-part DAPO recipe, algorithm, DAPO-Math-17K transformation, all seven figures, the progressive result table, two extracted reflective cases, training dynamics, and limitations

**Concept pages updated**:
- `wiki/concepts/gspo-vs-grpo.md` - added DAPO as a systems-scale GRPO recipe and contrasted degenerate-group filtering with signal-recovery methods
- `wiki/concepts/r1-zero-like-training.md` - added the full-recipe reproducibility lesson and cautioned against interpreting reflection anecdotes as proof of capability emergence
- `wiki/concepts/discriminative-policy-optimization.md` - contrasted DAPO's clipped, dynamically sampled system with DisCO's objective-level redesign
- `wiki/concepts/rl-for-ad.md` - added transferable monitoring/reward lessons and safety caveats for dynamic filtering, Clip-Higher, token weighting, and KL removal

**Index updated**: added DAPO.

**Key facts**:
- DAPO raises the upper PPO ratio clip from 0.2 to 0.28 while retaining a 0.2 lower clip.
- Dynamic sampling oversamples and discards all-correct/all-wrong groups until the effective prompt batch is full.
- Token-level loss divides by total generated tokens rather than averaging each response equally.
- Overlong filtering and soft punishment reduce noisy incorrect labels caused only by truncation.
- The cumulative recipe improves Qwen2.5-32B AIME24 avg@32 from 30 under naive GRPO to 50.
- The released setup uses 512 prompts × 16 responses, 20,480-token maximum generation, exact integer-answer verification, and no reference KL penalty.

**Limitations**:
- Evidence is math-only and primarily one 32B base model.
- Progressive ablations are cumulative/order-dependent and no training-seed variance is reported.
- Dynamic sampling changes the prompt distribution and may incur substantial extra rollout cost.
- Removing KL allows unconstrained policy drift; token-level loss intentionally gives longer responses more total influence.
- All-failed-group filtering is unsafe to transfer directly to driving because it can discard the most critical failure scenes.
- Reflective behavior evidence is qualitative and anecdotal.

## 2026-08-17 - Ingest: DriveWAM

**Source**: `raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md`

**Pages created**:
- `wiki/sources/drivewam.md` - source summary covering the chunked autoregressive video-action formulation, inverse-dynamics action flow, scene-evolving VLM guidance, selective KV memory, all six local figures, Tables 1-6, dataset curation, and limitations
- `wiki/concepts/physicalai-av-benchmark.md` - new concept page for the NVIDIA PhysicalAI-Autonomous-Vehicles benchmark (1,700h, 306,152 clips, ADE/FDE), its reported results, and the self-curated-test-subset caveat

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added Pattern 18 (chunked AR video-action policy with VLM guidance), the DriveVA/DriveWAM controlled contrast, a coupling-table row, a KV-cache entry under computational cost, and four open questions
- `wiki/concepts/navsim-benchmark.md` - added the DriveWAM SOTA-table row and a comparison-scope caveat noting the omission of DriveVA and the wiki frontier
- `wiki/concepts/foundation-backbones-for-ad.md` - added two backbone roles (video backbone as policy core; frozen advisory VLM), the DriveWAM backbone section, and the objective-retention takeaway
- `wiki/concepts/dual-system-vla.md` - added the inverted dual-system section (video model plans, VLM advises) with a bridge-type comparison against Senna-2 and AutoMoT
- `wiki/concepts/nuscenes-waymo-evals.md` - extended the open-loop caveats to PhysicalAI-AV

**Source pages updated**:
- `wiki/sources/driveva.md` - added DriveWAM to the contrast table and a same-backbone counterpart note
- `wiki/sources/alpamayo-r1.md` - corrected the stale "no public benchmark" limitation; the underlying data is now the public PhysicalAI-AV benchmark with third-party Alpamayo-1.5 numbers

**Index updated**: added DriveWAM and the PhysicalAI-AV benchmark concept.

**Key facts**:
- Wan2.2-TI2V-5B video diffusion transformer is the policy core; video and action streams share one transformer under a joint flow-matching objective with beta_a = 1.0.
- Actions are generated by inverse dynamics: the future video latent is sampled first, then the action chunk is sampled conditioned on it (clean latent under teacher forcing, generated latent at inference).
- A frozen Qwen3-VL-8B emits two-sentence guidance per 4-second chunk from causally available context only; a block-diagonal text mask prevents chunk k+1 from attending to later-step guidance.
- Selective KV memory scores cached tokens by relevance minus redundancy (lambda = 0.07) with separate pools of 448 video and 160 action tokens; training-free and inference-only.
- NAVSIM v1: 90.1 PDMS single front camera, with the table's best DAC 98.1 and EP 84.3, and comfort 100.0.
- PhysicalAI-AV curated 1,000-clip subset: 0.47/1.35 ADE/FDE@3s and 0.83/2.47 ADE/FDE@4s, versus Alpamayo-1.5 at 0.80/2.31 and 1.44/4.18.
- Backbone ablation: pretrained init without video supervision (1.23 ADE@4s) is worse than from-scratch with video supervision (1.10); the full configuration reaches 0.83.
- Selective KV memory reaches 0.89 ADE@4s versus 0.83 full caching and 1.40 FIFO, cutting 300s-rollout memory from 3.07 GB to 0.25 GB and attention from 17.37 to 1.44 GFLOPs.
- Data scaling from 4k to 20k to 100k clips at fixed 50k iterations improves monotonically and has not saturated.
- The 5-step-action variant totals about 871 ms per 4-second chunk on one H20, comparable to Alpamayo-1.5's 900 ms while also producing generated future video.

**Limitations**:
- The NAVSIM table omits the entire wiki frontier (CLEAR 93.7 down through DriveFine 90.7), and anchor-based DriveVLA-W0 at 90.2 still leads DriveWAM inside its own table.
- DriveVA uses the same Wan2.2-TI2V-5B backbone and reports 90.9 PDMS but is never cited or compared; the two papers' many other differences make the 0.8 gap unattributable.
- The PhysicalAI-AV test subset is curated by the authors' own VLM tagging pipeline; VaVAM is capped at 3s by its released checkpoint and Alpamayo-1.5 is evaluated under a protocol DriveWAM chose.
- All large-benchmark evidence is open-loop ADE/FDE and NAVSIM is non-reactive; no Bench2Drive, HUGSIM, or other closed-loop reactive evaluation.
- Selective KV memory accuracy is measured only on 20s clips while cost is profiled at 300s, and bounded pools at inference are a deliberate mismatch with full-history attention at training.
- The route command is derived from ground-truth ego yaw change over the upcoming chunk, leaking coarse directional future at training and evaluation time.
- 5B DiT plus 8B frozen VLM with per-chunk latency near 0.87-1.26s; no RL/RFT stage; the guidance ablation compares only against a fixed global prompt.
- Figure 3 (KV retention visualization) and Figure 5 (data-scaling plot) images are absent from the raw clipping, and the body text labels both the backbone and KV ablations "Table 5".

## 2026-08-17 - Ingest: SimWAM

**Source**: `raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md`

**Pages created**:
- `wiki/sources/simwam.md` - source summary covering the isolated attention mask, joint video-action flow matching, Flow-GRPO SDE reinforcement, all four figures, Tables 1-10, and limitations

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added Pattern 19 (video backbone as training-time-only prior) and a new synthesis section "Does Test-Time Future Imagination Help?" collecting the imagine-then-act vs. training-time-only evidence; updated the coupling table, the world-model/VLA capability table, and three open questions
- `wiki/concepts/navsim-benchmark.md` - added the SimWAM row at 91.5 PDMS, a comparison-scope caveat, and reordered the single-pass frontier list
- `wiki/concepts/foundation-backbones-for-ad.md` - added the controlled video-prior swap section, a swappable-prior backbone role, and the scale-versus-quality takeaway
- `wiki/concepts/rl-for-ad.md` - added the SimWAM RL section (SDE vs. random noise, LoRA-only updates, hard-subset selection) and flagged the tension with DreamerAD's characterization of Flow-GRPO
- `wiki/concepts/nuscenes-waymo-evals.md` - added the zero-shot NAVSIM-to-nuScenes WAM comparison table and noted that DriveVA's absolute numbers are now available

**Source pages updated**:
- `wiki/sources/drivewam.md` - added a "superseded by SimWAM on NAVSIM" section and flagged the unverified nuScenes row
- `wiki/sources/driveva.md` - added SimWAM as the third same-backbone entry and partially resolved the missing-absolute-numbers limitation

**Index updated**: added SimWAM.

**Key facts**:
- A pretrained Wan2.2-5B video expert and a 1.02B action DiT are co-trained under joint flow matching with lambda = 1; they share no parameters and interact only through a shared attention stream.
- The isolated attention mask lets future-frame tokens and action tokens each attend to the current-observation latents while remaining mutually invisible, so the future branch is dropped at inference.
- Component analysis: action-only 86.6 PDMS, plus video co-training 90.3, plus RL 91.5.
- Mask ablation: bidirectional 90.2, action-to-video 90.1, isolated 90.3 - future-token access gives no measurable benefit.
- Video prior swap under a fixed planner: LTX-Video 88.7, Wan2.1-1.3B 90.2, Wan2.2-5B 90.3, Cosmos-Predict2.5 90.4.
- Action expert scaling 0.21B to 1.02B moves PDMS only 89.9 to 90.3.
- RL replaces the flow ODE with a marginal-preserving SDE, samples G = 8, and updates only rank-32 LoRA adapters on the action expert's attention projections, training on navtrain scenes below 90 PDMS and peaking at 15k steps.
- Sampler comparison: random noise 91.3 PDMS with best EP 88.0 but NC 97.7; SDE 91.5 with NC 98.4 and TTC 95.5.
- Future-video target: 4s horizon matters more than frame density (2s costs 0.4 PDMS; halving frame rate at 4s costs 0.1).
- Latency: 518 ms at 384x672 with 10 sampling steps; 297 ms at 5 steps for 90.1 PDMS; one step collapses to 68.9.
- Zero-shot nuScenes without fine-tuning: 0.96 avg L2 and 0.04 avg collision, the lowest collision rate in its table.

**Limitations**:
- The end-to-end SOTA claim is comparison-scope limited; Table 1 omits CLEAR 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, DynVLA 91.7, and DiffusionDriveV2 91.2.
- The isolated-mask conclusion rests on a 0.2 PDMS spread with no seed variance, supporting "unnecessary" rather than "harmful".
- 518 ms is efficient for a 5B-video-backbone WAM but far above DiffusionDrive, HAD, and OneDrive in absolute terms.
- Table 6 attributes zero-shot nuScenes results to DriveWAM, but the DriveWAM v1 clipping in this wiki contains no nuScenes evaluation.
- NAVSIM-v1 only; no NAVSIM-v2/EPDMS, no navhard, no Bench2Drive or HUGSIM.
- RL depends on early stopping at 15k steps and a hand-set below-90 difficulty threshold, with the peak selected on the evaluation benchmark.
- Ablation tables omit the comfort column even though it enters PDMS.
- The video expert's actual inference cost is unquantified, and latency scaling with sampling steps suggests the shared stack is re-entered per step.
- Six Table 1 baselines are not ingested in this wiki (SGDrive, UniWorldVLA, DriveLaW, SeerDrive, ImagiDrive, WorldRFT).

## 2026-08-17 - Ingest: SGDrive

**Source**: `raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md`

**Pages created**:
- `wiki/sources/sgdrive.md` - source summary covering the scene-agent-goal world query hierarchy, the three supervision heads, the block-wise structured attention mask, the DiT planner with learned-prior initialization, all nine figures, Tables 1-6, and limitations

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added Pattern 20 (structured symbolic state forecasting) with a world-model-target taxonomy table; linked it from the existing occupancy world-model pattern; added the capability-table row
- `wiki/concepts/navsim-benchmark.md` - added the SGDrive v1 row and v2 EPDMS row plus a comparison-scope caveat covering both claims
- `wiki/concepts/perception-for-planning.md` - added the ego-relevance filtering section contrasting task-level filtering with representational sparsity
- `wiki/concepts/intent-conditioned-planning.md` - added continuous goal pose as a third intent mechanism and a section on what continuity buys and costs
- `wiki/concepts/foundation-backbones-for-ad.md` - added the VLM-as-frozen-world-model role and the 2B-beats-8B takeaway
- `wiki/concepts/rl-for-ad.md` - added the SGDrive section using its reuse of ReCogDrive's RL config as a controlled read on what RL adds

**Source pages updated**:
- `wiki/sources/simwam.md` - SGDrive is now an ingested and independently confirmed Table 1 entry; added it as the nearest competitor

**Index and README updated**: added SGDrive; README count 55 to 56; known-gaps list refreshed with DriveLaW as the top next ingest.

**Key facts**:
- Learnable <world> queries are appended to the VLM token stream and split into five subqueries: three for current-world knowledge and two for future forecasting.
- Three supervision heads: occupancy geometry via a VAE decoder with resampled CE plus BCE, safety-critical agent detection via DETR bipartite matching with lambda_cls = 10, and a short-term goal pose about 4s ahead under L1.
- Scene geometry is deliberately semantic-free; agents are filtered by ego-trajectory relevance and front-camera frustum visibility rather than detecting everything.
- A block-wise structured mask forbids attention between scene/agent/goal blocks, allows temporal attention within a block, and leaves cross-attention to visual and text tokens open.
- The <world> query hidden states condition the DiT directly and are never decoded at inference; the diffusion prior is initialized from those queries plus the historical ego trajectory rather than pure Gaussian noise.
- Two-stage training: stage 1 SFT on VQA plus the three heads with lambda_agent = 0.1; stage 2 freezes the VLM and trains the planner alone for 220 epochs.
- Backbone is InternVL3-2B; 3.1M QA pairs of domain adaptation following ReCogDrive, then 85k trajectory QA pairs; 32 H20 GPUs.
- NAVSIM v1: 87.4 PDMS SFT and 91.1 PDMS RFT, with best NC and TTC in both blocks; NAVSIM v2: 86.2 EPDMS.
- SFT at 2B beats ReCogDrive-8B (86.8) and beats plain InternVL3-8B and QwenVL2.5-8B (both 83.3) by 4.1.
- Table 3 (stage 1, text trajectories): base 82.2, plus current-state hierarchy 84.7, plus future forecasting 85.5 - structured present-state perception carries most of the gain.
- Table 4 (with planner): scene 86.0, plus agent 86.3, plus goal 87.0, plus future 87.4; the goal subquery's gain is concentrated in Ego Progress.
- Structured versus causal mask is worth only +0.3 PDMS, entirely in EP, with TTC slightly regressing.
- RL reuses ReCogDrive's configuration exactly, giving a comparable +3.7 SFT-to-RFT delta.

**Limitations**:
- The SOTA claim is scoped to camera-only VLM methods in its own table; 91.1 sits below CLEAR, DriveSuprim, Drive-JEPA, HybridDriveVLA, DynVLA, SimWAM, FLARE, and DiffusionDriveV2.
- The v2 claim is weaker still: 86.2 EPDMS is compared against seven baselines ending at DiffusionDrive and lands mid-table in the wiki.
- DAC 94.3 and EC 85.9 on v2 are the weakest modern entries in its own table.
- Occupancy labels (or LiDAR to derive them) and 3D boxes are required at training, so it is camera-only at inference but not annotation-free - unlike FLARE or SimWAM.
- The structured mask, presented as a core contribution, is the smallest measured effect at +0.3 PDMS with no seed variance.
- Stage 2 freezes the VLM, so the world heads never receive gradient from trajectory quality and joint fine-tuning is untested.
- SGDrive reports DiffusionDrive at 84.3 EPDMS where the wiki records 84.5 from other sources.
- Front-camera-only input causes lane-change errors under extreme turns, acknowledged in the failure cases.
- No latency numbers despite an efficiency argument for feeding query hidden states directly to the planner.
- Compute is substantial for a 2B model: 32 H20 GPUs, 3.1M QA pairs, and 220 planner epochs.

## 2026-08-17 - Ingest: DriveLaW

**Source**: `raw/papers/DriveLaW_ Unifying Planning and Video Generation in a Latent Driving World.md`

**Pages created**:
- `wiki/sources/drivelaw.md` - source summary covering the chained generation-planning design, the high-compression spatiotemporal VAE with hybrid pixel-space decoding, noise reinjection, the three-stage curriculum, all six figures, Tables 1-11, and limitations

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added Pattern 21 (mid-denoising latents as the planning state); substantially extended the test-time-imagination synthesis with DriveLaW's denoising-step sweep as independent corroboration of SimWAM; refreshed the nuScenes generation table with DriveLaW at FID 4.6 and corrected the claim added during the 2026-08-17 lint that no post-April world-model entry reports FID/FVD
- `wiki/concepts/navsim-benchmark.md` - added the DriveLaW row and a caveat noting NC 99.0 / TTC 96.7 are the highest in the wiki and that DriveVLA-W0 appears there as a flow-matching reimplementation at 87.2
- `wiki/concepts/foundation-backbones-for-ad.md` - added the controlled representation comparison (video latents 89.1 > VLM hidden states 86.5 > BEV 84.1) and the pretraining-data versus model-size distinction against SimWAM
- `wiki/concepts/diffusion-planner.md` - added the section on conditioning a planner on another diffusion model's latents, including the brittleness of the conditioning timestep

**Source pages updated**:
- `wiki/sources/epona.md` - added DriveLaW as its main challenger, noting Epona still wins at the 100-frame horizon and on trajectory-only latency
- `wiki/sources/simwam.md` - DriveLaW now independently confirms SimWAM's transcription and, unexpectedly, its thesis

**Index and README updated**: added DriveLaW; README count 56 to 57; open thread #1 rewritten around two independent results; known-gaps list refreshed with Hydra-MDP as the top remaining gap.

**Key facts**:
- Chained rather than parallel: the Action DiT cross-attends to per-block latents cached from the Video DiT during its first denoising step, so the generator's internal state is the planning representation.
- DriveLaW-Video is a 2B LTX-Video-initialized DiT; DriveLaW-Act is a 133M vanilla DiT trained with flow matching.
- The spatiotemporal VAE uses 32x32x8 downsampling with 128 channels, a 1:192 compression ratio (1:8192 pixel-to-token), and a causal 3D encoder; the final rectified-flow step is executed by the decoder in pixel space.
- Noise reinjection perturbs only high-frequency regions, identified by a Laplacian response on a decoded grayscale preview thresholded at beta times the standard deviation.
- Three-stage curriculum: 740x352x121 for long-horizon motion, then 1280x704x25 for spatial detail, then chaining latents into the planner.
- NAVSIM: 89.1 PDMS with no RL and no learned scorer; NC 99.0 and TTC 96.7 are the highest recorded in this wiki, but EP is 81.3.
- nuScenes generation: FID 4.6 and FVD 81.3, the best FID in the wiki; UniUGP retains the best FVD at 75.9.
- Representation ablation under a fixed planner: BEV features 84.1, VLM hidden states 86.5, video latents 89.1.
- Denoising-step ablation: t=1 gives 89.1, t=5 gives 86.9, t=10 collapses to 23.2 with comfort 0.
- Video pretraining scaling: 0, 76k, 3.8M, 7.6M samples give 85.9, 87.0, 87.8, 89.1 PDMS.
- Noise reinjection is worth 1.5 FID and 20.8 FVD; dropping the first curriculum stage costs 28.0 FVD.
- Speed: video generation is about 5x faster than Epona at matched resolution, but trajectory planning is slower (0.71s versus 0.42s on H20).

**Limitations**:
- The NAVSIM record is scoped to its own table; 89.1 sits below ten wiki entries, and SimWAM cites this exact figure while beating it by 2.4.
- The policy is safety-skewed: leading NC and TTC but mediocre EP, with no RL or scorer stage to recover progress, and the nuScenes 1s collision rate regresses versus Epona.
- The t=10 collapse is reported without diagnosis; a 66-point PDMS drop suggests a distribution or scaling pathology rather than merely redundant information.
- The high-compression VAE introduces motion artifacts in high-motion scenes, acknowledged in Appendix D.1 and only partly mitigated by noise reinjection.
- Long-horizon generation degrades past about 80 frames, where Epona overtakes it.
- Planning latency is worse than Epona's despite much faster video generation.
- DriveVLA-W0 appears at 87.2, a third distinct value for that method, because it is a flow-matching reimplementation rather than the published configuration.
- No NAVSIM-v2/EPDMS, no navhard, no closed-loop reactive benchmark.
- The claim of gradient isolation between generator and planner sits awkwardly with the statement that stage 3 updates both the Video DiT and the Planning DiT.

**Process note**: a PowerShell read-modify-write round-trip corrupted UTF-8 characters in `README.md` and `wiki/concepts/diffusion-planner.md` (84 mojibake sequences). Both files were restored from HEAD and the edits reapplied with the editing tool. Future frontmatter edits should not use `Get-Content -Raw` piped to `Set-Content` on files containing non-ASCII characters.

---

## 2026-08-24 - Ingest: How Can Driving World Models Do Counterfactual Prediction?

**Source**: `raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md`
**arXiv**: 2608.11601v1
**Authors**: Jiaru Zhang (corresponding), Can Cui, Yi Xu, Xin Ye, Ruqi Zhang, Ziran Wang (Purdue University + Bosch Center for Artificial Intelligence)
**Confidence**: high - the local markdown includes the full method text, all four figures, all five tables (main results, component ablation, benchmark composition, stage checkpoints, evidence-source controls), and appendices A-D

**Pages created**:
- `wiki/sources/driving-wm-counterfactuals.md` - source summary covering the causal analysis, the four-stage Abduce/Transport/Complete/Combine pipeline, the three-arm CARLA benchmark, both metrics, all four figures, all five tables, cost and implementation details, and limitations
- `wiki/concepts/counterfactual-prediction.md` - new concept page on Pearl's ladder applied to driving, the four distinct senses of "counterfactual" in the AD literature, the abduction requirement, matched-ground-truth benchmark construction, and the recovered-fraction metric

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - corrected the long-standing intro claim that world models "enable counterfactual reasoning" (it is interventional, rung 2); added the "Action-Conditioned != Counterfactual" section to the test-time-imagination synthesis; added LPIPS-vs-matched-reference and recovered fraction to the world-model quality metrics table; rewrote the test-time-imagination open question and added a new one on whether a world model can do abduction

**Source pages updated**:
- `wiki/sources/simwam.md` - the counterfactual escape hatch its mask ablation left standing is now partly closed
- `wiki/sources/drivelaw.md` - orthogonal evidence pointing the same way: the generator's value is its representation, not its rendered futures

**Index and README updated**: added the source and concept rows; README counts 57 to 58 papers and 29 to 30 concept pages; Open Thread #1 rewritten around three independent results; Known Gaps notes that Vista and DrivingWorld now have added priority as the two evaluated backbones.

**Key facts**:
- The core claim is a conditioning argument, not a capability argument: the counterfactual and the direct prediction integrate the same mechanism and differ only in the posterior over the world, p(w | H, F+) versus p(w | H). Scaling the generator cannot close the gap.
- Direct action-conditioned prediction is rung 2 at best because the alternative action is specified by the query rather than observed, so it does not update the posterior over the world beyond the history.
- Benchmark: 186 CARLA cases from 72 placements across Town01 (60), Town03 (72), Town10HD (54); three scenario types (side street 60, lead cuts in 45, lead brake 81) and three action edits (accelerate 1.6x, brake 0.4x, full stop 0x displacement scaling); 25 frames at 10 fps, 576x320, 15-frame shared history and 10-frame prediction window.
- Each case is three replays of one placement: F (executed action, event occurs), P (target action, event occurs, the counterfactual ground truth), U (target action, event never triggered, the null reference).
- Recovered fraction rescales the preference for P over U so that Rec(U)=0, Rec(P)=1, and 0.5 means no preference. Encoders DINOv2 ViT-B/14 and CLIP ViT-L/14. LPIPS (AlexNet) is measured against P.
- Main result: direct prediction reaches 0.38 (Vista) and 0.31 (DrivingWorld) overall recovered fraction under DINOv2, below 0.5 in every cell but Vista's lead brake (0.50). Evidence transport raises this to 0.70 and 0.67 and cuts LPIPS from 0.423 to 0.169 and from 0.291 to 0.211.
- The internal gradient matters more than the averages: direct prediction is worst on side street (0.29 / 0.25), where the event is revealed only in the factual continuation, and best on lead brake (0.50 / 0.37), the confounded control where an already-visible lead looms under acceleration.
- Component ablation: transport plus multi-frame filling alone reaches 0.68 / 0.67 with no completion at all. Complete and Combine add at most 0.02 recovered fraction and buy fidelity instead (LPIPS 0.195 to 0.169 on Vista, 0.238 to 0.211 on DrivingWorld).
- Stage checkpoints: on DrivingWorld the recovered fraction drops from 0.67 to 0.52 after completion and returns to 0.67 after Combine, a direct measurement of VQ encode/decode loss on evidence-bearing pixels. Vista, completing in a continuous latent space, barely dips.
- Evidence-source controls: matching evidence 0.66 / 0.67; five frames earlier 0.40 / 0.41; final history frame 0.35 / 0.36 (indistinguishable from direct prediction); different case 0.62 / 0.64 but LPIPS 0.564 / 0.556.
- Method components: Depth Anything V2 Small for depth, forward splatting with a depth buffer for transport, SDEdit-style mid-schedule start plus RePaint-style evidence restoration for Vista (25 EDM steps from schedule index 14), token fixing at 60 percent patch coverage for DrivingWorld, and a feathered alpha blend with temporally smoothed color correction for Combine.
- Cost: about 90 s per Vista case and 108 s per DrivingWorld case versus 47 s and 45 s for direct prediction on one A100. The pipeline computes the direct prediction once because it fills the unsupported regions of the Complete stage's input video.

**Limitations**:
- Scripted agents are what make the matched reference obtainable, but transported evidence preserves behavior the counterfactual action would have changed (the paper's example: a pedestrian who would have stopped keeps walking). Beyond about a second the method fails confidently rather than by omission.
- Both backbones are real-world-trained and evaluated on CARLA renders. The scenario-type gradient argues against pure domain shift as the explanation, but a world model trained in the benchmark's render domain would settle it.
- The method is retrospective by construction, since it reads the factual continuation, which exists only after the episode is recorded. It therefore says nothing about what a planner could do at decision time, where no factual continuation exists.
- The frozen world model contributes almost none of the recovery; geometric transport does the work. The paper demonstrates that evidence closes the gap, not that world models can be made to abduce.
- The recovered fraction is category-sensitive rather than identity-sensitive: evidence from a different episode of the same scenario type scores 0.62-0.64 against 0.66-0.67 for the matching episode, while LPIPS more than doubles. It must not be reported alone.
- Lead brake is 81 of 186 cases and is both the confounded control and the type with the smallest metric denominator, so it is simultaneously the least informative and the noisiest third of the benchmark.
- 19 of 186 cases fail the post-hoc image check and are retained; the claim that excluding them changes nothing is asserted without numbers.
- The ceiling is 0.70, with no decomposition of the residual between depth error, splatting artifacts, completion quality, and encoder noise.
- Only two open backbones. The industrial claims that motivate the paper (Waymo's world model, Genie 3) are untestable, and Drive-WM, whose counterfactual claim is quoted directly, is not evaluated.
- Ten pedestrian-crossing cases were collected and dropped, which removes exactly the case where the scripted-agent assumption is least defensible and the liability application most consequential.
- No stated release of benchmark, code, or data.

**Cross-page effect**: this is the first ingested paper that constrains a capability claim rather than proposing a method, and it forced a correction in `world-model-for-ad.md`, which had listed counterfactual reasoning among the reasons world models help planning since 2026-04-05.
---

## 2026-09-02 - Ingest: Auto-JEPA

**Source**: `raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md`
**arXiv**: 2607.29031v1
**Code**: https://github.com/NoctYang/Auto-JEPA
**Authors**: Jiwei Yang, Zhengxian Chen, Chaosheng Huang, Jun Li (Tsinghua University, School of Vehicle and Mobility)
**Confidence**: high - the local markdown includes the full method text, all five figures, all five tables (NAVSIM v1, NAVSIM v2, component ablation, candidate-pool sweep, hyperparameters), and appendices A-G

**Pages created**:
- `wiki/sources/auto-jepa.md` - full source summary: four-stage pipeline, all loss definitions, all five figures embedded, all five tables reproduced, the semantic occlusion protocol, relationships to seven other wiki entries, and a four-part limitations section

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added Pattern 22 "The Ego Trajectory as the Prediction Target"; added a third position to the test-time-imagination synthesis (predicts at inference, prediction is load-bearing, but the target is an action rather than a world state) and reframed the surviving generalization as "future-prediction objectives are valuable; instantiated future world states at decision time are not"; added an Auto-JEPA row to the World Model vs. VLA table; noted that FID/FVD are undefined rather than merely unreported for this method; two new open questions (is the JEPA objective load-bearing on the trajectory side; does occlusion selectivity predict driving quality)
- `wiki/concepts/selection-based-planning.md` - added "latent retrieval" as a fifth paradigm row; new section contrasting retrieval against fixed-vocabulary selection, including the point that DriveSuprim's 98.7 oracle ceiling does not transfer because retrieval introduces a recall failure mode fixed vocabularies do not have; methods table row
- `wiki/concepts/perception-for-planning.md` - new "Emergent Ego-Relevance Without Perception" section documenting the semantic occlusion protocol as a reusable methodology, its two missing controls, and a table comparing it against the page's three other evidence types (cosine collapse, detection ablation, attention visualization); two new open questions
- `wiki/concepts/intent-conditioned-planning.md` - new "Full-Trajectory Latent as Intent" section with a three-way table separating discrete class labels (DIAL/PaIR-Drive), continuous terminal pose (SGDrive), and full-trajectory latent (Auto-JEPA); corrected the stale "Two Uses in the Wiki" heading to "Four Uses"
- `wiki/concepts/navsim-benchmark.md` - v1 SOTA row, two v2 rows (matched and updated protocol), a long caveat covering both the stale v1 comparison set and the v2 protocol mismatch, and a new "Evaluator Drift Is Now a First-Class Confound" section tabulating DriveFine (+2.6) and Auto-JEPA (+3.5) same-checkpoint swings under different evaluators
- `wiki/concepts/foundation-backbones-for-ad.md` - new backbone role (frozen off-the-shelf video encoder); new "Freezing the Encoder, Moving the Objective" section comparing Drive-JEPA's and Auto-JEPA's opposite allocations of the same V-JEPA 2 starting point; takeaway putting an upper bound of 2.0 PDMS on driving-domain encoder adaptation
- `wiki/concepts/counterfactual-prediction.md` - added consequence #5: a world model that is on no rung of the ladder for the environment, by its own admission, and what that trade buys

**Index and README updated**: source row added to both; README counts 58 to 59 papers; Open Thread #1 reframed around the axis Auto-JEPA exposes; CLOVER added to Known Gaps.

**Key facts**:
- Prediction target is the latent of the future ego trajectory, not any scene state. A trajectory autoencoder (4 Transformer blocks, 1024-d, 16 heads, 8 Fourier bands) is trained first with coordinate/endpoint/velocity/acceleration losses (weights 1 / 2.0 / 0.5 / 0.2), then its decoder is discarded and the encoder frozen.
- Eight waypoints over 4 s at 0.5 s intervals, coordinates normalized by 64, encoded to 8x1024. The eight tokens describe one continuous realization, explicitly not eight maneuver classes.
- Visual predictor: frozen V-JEPA 2 encoder (no driving adaptation), history and command encoders, 24-layer / 16-head / 1024-d Transformer with eight learnable future-time query tokens. Input is four 256x256 front-camera frames, four ego positions, a 4-d command.
- Objective is 0.1 L_feat + 2.0 L_cos + L_NCE with tau=0.07; no ADE/FDE supervision. InfoNCE is the anti-collapse term and its negative set is bounded by batch size x GPU count (8/GPU, 1-2 GPUs).
- Memory is 110,335 GT trajectory-latent pairs from NAVSIM training, navtest excluded. Retrieval is flat cosine over the whole memory, top-K = 300.
- Scorer is initialized from the released CLOVER checkpoint and re-optimized with L_comp + 0.5 L_rank on labels from the NAVSIM/CLOVER get_sub_score evaluator (batched navsim_v1_style path, per-proposal two-way rollout disabled). DAC gate is BCE + 0.3 rank with positive-class weight 8, threshold 0.2, candidate-set self-attention; all-rejected falls back to ungated ranking.
- NAVSIM v1: 91.3 PDMS (NC 98.4, DAC 98.3, TTC 95.0, C 100.0, EP 87.1). EP is second only to Curious-VLA in the paper's own table and close to the human 87.5.
- NAVSIM v2: 85.6 EPDMS original evaluator, 89.1 with the updated official implementation and human-behavior filtering. The entire 3.5-point gap is TL 97.2 to 99.7 and LK 84.0 to 94.7; every other submetric moves at most 0.1.
- Component ablation (read conditionally, not sequentially): no-intent 52.6, no-scorer 87.6, no-gate 91.0, full 91.3. Scorer is worth +3.7 given the gate; gate is worth +0.3 PDMS and +0.4 DAC given the scorer.
- Candidate pool: K=1 gives 87.6, K=200 gives 91.1, K=300 gives 91.3. K=1 is pure retrieval with no selection and is the cleanest measure of the JEPA component alone.
- Semantic occlusion over 15,364 validation scenes: dynamic-agent masking gives mean 1-cos of 0.080 vs. 0.027 for equal-area random masking, a 2.97x ratio, larger on the agent arm in 71.1% of scenes. Both arms hold ego history and command fixed. Seed 42.
- One deterministic full-navtest evaluation of one checkpoint; no seed variance. Intent-predictor checkpoint selected as "epoch 10", not by a validation criterion. A100-SXM4 80 GB, Python 3.12, BF16.

**Limitations**:
- The 89.1 EPDMS headline requires an evaluator no baseline in its own table used. Under the matched protocol the score is 85.6, which ties HydraMDP++ and falls below DriveSuprim 87.1 and DriveWorld-VLA 86.8 within that same table, and far below the wiki frontier.
- LK 84.0 under the original evaluator is a 12-point outlier against every other camera method, and the paper does not explain it or why a retrieval planner would be uniquely exposed.
- NAVSIM v1 Table 1 tops out at 90.3 and omits CLEAR 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, DriveFine 91.8, DynVLA 91.7, SimWAM 91.5, iPad, and GoalFlow.
- EC 75.2/75.4 is near the bottom of the wiki's v2 entries. Retrieval has no frame-to-frame continuity mechanism; consecutive frames can land on different memory entries with nothing penalizing the jump. Drive-JEPA hit exactly this and needed a momentum-aware selector (EC 47.9 to 84.8); Auto-JEPA does not discuss it.
- The scorer is inherited from CLOVER rather than built, and contributes +3.7 of the 91.3. No row isolates CLOVER's scorer on CLOVER's own candidates versus Auto-JEPA's.
- "No perception annotations" is true; "no privileged supervision" is not. Scorer and gate train on NAVSIM evaluator-derived labels - the same simulator-label distillation Hydra-MDP-style methods use.
- The no-intent ablation substitutes a scene-independent codebook medoid, so it shows scene conditioning matters at all, not that this representation is good. The missing control is retrieval keyed by a regressed trajectory encoded through the same frozen encoder - which would isolate whether the JEPA objective or the shared retrieval space is doing the work.
- Retrieval cannot synthesize a maneuver the memory lacks, and the K=200 to 300 saturation (+0.2) suggests enlarging the pool is not the fix. No retrieval-recall study exists.
- Memory inherits NAVSIM's forward-heavy trajectory distribution (only ~8% of GT trajectories turn more than 30 degrees per DriveSuprim). No rotation augmentation analogue and no turn-vs-straight breakdown.
- No latency, FPS, or memory-footprint number anywhere, despite a flat cosine scan over 110,335 x 8 x 1024 latents per frame (~1.8 GB at BF16). Every competing wiki method reports latency.
- All-rejected fallback firing rate is never reported, so the gate's measured +0.3 is an average over two different systems. Gate threshold 0.2 has no sensitivity sweep.
- NAVSIM only - no Bench2Drive, HUGSIM, navhard, nuScenes, or Waymo. This matters more than usual for a retrieval planner: a 4 s non-reactive horizon is exactly the regime where a memory of recorded human trajectories should look best.
- 256x256 input, well below the 1024x256 used by TransFuser, HydraMDP++, DriveSuprim, and GoalFlow, with no resolution ablation.
- Occlusion controls match area but not shape, contiguity, or placement. Agent masks are object-shaped and road-level; random masks may land on sky or periphery. The missing arm is equal-area masks on the drivable surface.
- 28.9% of scenes respond more to random masking, and that tail is uncharacterized. Absolute dependence is modest: 1-cos of 0.080 means the intent stays 92% aligned after removing every visible dynamic agent from all four frames.
- The link from latent change to behavioral change is shown on three hand-picked scenes only. No dataset-level statistic connects delta-intent to a changed selected trajectory or to PDMS.
- By the paper's own admission the representation gives no scene-level forecasts, so interactive simulation and counterfactual environment generation are out of scope by construction.

**Cross-page effect**: this is the first ingested method whose world-model target is the ego action rather than the environment, which required a new architecture pattern on the world-model page and a reframing of the test-time-imagination question from "does predicting the future help at inference" to "what should be predicted". It also introduced retrieval as a distinct planning paradigm alongside fixed-vocabulary selection, and supplied the wiki's first interventional protocol for testing whether a planner's representation actually depends on the agents we assume matter.

**Naming hazard recorded**: Auto-JEPA and Drive-JEPA are different papers by different groups, both using V-JEPA 2 on NAVSIM. The source page opens with an explicit warning and a difference table.
---

## 2026-09-02 - Ingest: WA-JEPA

**Source**: `raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md`
**arXiv**: 2608.20974v1
**Code**: https://github.com/AFARI-Research/WA-JEPA
**Authors**: Xinlin Wang, Yujiao Xiang, Yuheng Zhou, Jingqi Wang, Minqing Huang (corresponding), Jiajie Huang, Dongxu Wei (project lead), Tingguang Zhou, Xiyang Wang, Gong Chen, Zhi Xu, Feiyang Tan, Hangning Zhou, Mu Yang - Afari Intelligent Drive, UESTC, Southeast University, BUPT, Tianjin University
**Confidence**: high - full method text, five of six figures, all seven tables, and appendices A-E. Figure 4 (temporal-collapse bar chart) is missing its image in the local conversion; both plotted values appear in the body text.

**Pages created**:
- `wiki/sources/wa-jepa.md` - two-stage method with all loss definitions, five figures embedded, all seven tables reproduced (NAVSIM-v2 with both EPDMS columns, HUGSIM main + aggregation + per-dataset, NAVSIM-v1, three ablations, temporal metrics, seed variance), a three-way JEPA comparison table, and a five-part limitations section

**Concept pages updated**:
- `wiki/concepts/navsim-benchmark.md` - **substantially rewritten around evaluator drift**. The old two-row "Evaluator Drift" section became a full section documenting the devkit commit 359c7f7 correction, an eight-method delta table (+2.0 to +3.8), and a three-way partition of the wiki's own v2 entries into corrected / pre-fix / unclassified cohorts. Added a new seed-variance section. Added v1 and two v2 rows for WA-JEPA plus a scope caveat. Softened the WAM-Diff SOTA claim and narrowed the Auto-JEPA caveat.
- `wiki/concepts/hugsim-benchmark.md` - **rewritten**. New current-snapshot table (436 scenarios, commit ead17f2), aggregation-robustness and per-dataset tables, an explicit two-era warning that HAD-L and Latent-WAM are not comparable to WA-JEPA, and four open questions. Confidence stays medium.
- `wiki/concepts/world-model-for-ad.md` - added Pattern 23 with an `{#objective-form}` anchor covering the FM-vs-regression result and the temporal-collapse diagnosis; added the entropy-of-the-target framing that reconciles it with Auto-JEPA; noted WA-JEPA does not test SimWAM's control; World-Model-vs-VLA row; two open questions.
- `wiki/concepts/foundation-backbones-for-ad.md` - new section putting Drive-JEPA's and WA-JEPA's encoder ablations side by side as two independent confirmations, plus the shared image-vs-video confound neither paper breaks; new backbone-role row.
- `wiki/concepts/diffusion-planner.md` - new section on flow matching over latents rather than trajectories, the asymmetric stop-gradient coupling, and the FM-vs-regression table read alongside SpanVLA's L1-head result.

**Source pages updated**:
- `wiki/sources/auto-jepa.md` - corrected yesterday's over-strong claim that 89.1 is incomparable to everything; it belongs to the corrected cohort where it ranks ninth. Added a WA-JEPA relationship entry on target entropy.
- `wiki/sources/drive-jepa.md` - added WA-JEPA's direct critique, the corroborating encoder ablation, and the note that WA-JEPA's v1 table cites Drive-JEPA's 89.0 perception-free baseline rather than 93.3. Added the Auto-JEPA cross-reference.

**Index and README updated**: rows added to both; README counts 59 to 60 papers; NAVSIM-v2 leader line changed from WAM-Diff 89.7 to WA-JEPA 91.7 with the protocol caveat; new HUGSIM leader line; Open Thread #1 extended with the objective-form axis; Known Gaps now leads with the four un-ingested 89.9-90.4 EPDMS methods.

**Key facts**:
- Three-part critique of V-JEPA for planning: random spatiotemporal masking is a completion objective with no future-directed component; deterministic regression cannot generate unseen future tokens; V-JEPA 2's action-conditioned variant needs a goal image plus MPC, which is not online planning.
- Stage 1 pretrains on multi-view nuPlan video with no action supervision. History tokens always visible; masking applies only to future tokens under two patterns. Full-mask makes every future token a target (strictly causal); Patch-mask keeps a subset visible. Visible future tokens are scattered back into a sequence of learnable mask tokens via a mask-aware fill-and-scatter operation. An EMA target encoder supplies clean targets.
- Flow matching uses linear interpolation Z_t = (1-t) eps + t Z*, x-prediction (clean-endpoint) parameterization, MSE loss against the stop-gradiented target, MMDiT-style predictor doing joint self-attention between context and future scene tokens.
- Stage 2 uses Full-mask only. Actions are noised in a normalized space in parallel; noisy future actions, historical actions, and ego state are separately encoded and concatenated into action tokens. One MMDiT predictor emits both streams.
- Asymmetric stop-gradient: the scene stream reads action tokens but its loss cannot update them; the action stream reads differentiable scene tokens. Action supervision shapes the world representation, world modeling never perturbs the policy. Never ablated. Lambda weights never reported.
- Inference: 12 sampling steps, both future scene latents and actions initialized from Gaussian noise, clean endpoints converted to velocities and integrated. No future images, no GT actions.
- Inputs: 4 historical frames, 4 cameras (left/front/right/rear), 256x512. Outputs 8 actions at 2 Hz including heading phi_k, which most NAVSIM planners do not predict.
- NAVSIM-v2: 91.7 EPDMS corrected, 88.0 EPDMS* pre-fix. NC 99.4, DAC 98.2, DDC 99.7, TLC 99.9, EP 87.8, TTC 98.9, LK 98.3, HC 98.3, EC 88.1. NC/TTC/LK best in its table.
- NAVSIM-v1: 91.8 PDMS with NC 99.5, the highest NC in the wiki.
- HUGSIM zero-shot: HD-Score 0.4462, RC 0.5689, NC 0.6856, DAC 0.9635, TTC 0.6120, Comf 0.6620. Baselines rescored by the authors at commit ead17f2 including PR #57's heading-order fix: DrivoR 0.3252, UniAD 0.3124, LTF 0.2310, VAD 0.1393. Best on all four source datasets separately.
- Encoder ablation, all trained directly in Stage 2 with no Stage 1: V-JEPA 2 89.5, MAE 83.8, DINOv3 83.8, SigLIP2 83.1.
- Stage 1 masking: no Stage 1 89.5, Patch-mask 91.0, Full-mask 91.3, both 91.7.
- Stage 2 components: cascaded historical-latents baseline 89.9, separate FM future predictor cross-attended 90.8, joint without future supervision 91.1, joint + regression 90.7, joint + FM 91.7.
- Temporal metrics on the K=64 most dynamic locations, F=4 future token steps: directional-similarity collapse gap 0.30 (Reg) to 0.10 (FM); change-magnitude ratio 0.45 (Reg) to 0.80 (FM). For FM the metric uses the one-step x-prediction at the sampled training flow time, not the multi-step sampler.
- Seed variance over 10 seeds: mean 91.7014, std 0.0531, SE 0.0168, 95% CI [91.6634, 91.7393], median 91.6960, range [91.6294, 91.8070]. First NAVSIM seed-variance table in the wiki.
- Six methods in Table 1 report both EPDMS columns, making the correction measurable: DiffusionDriveV2 +2.0, DriveFuture +3.5, CoWorld-VLA +3.8, SparseDriveV2 +3.4, Discrete-WAM +3.4, WA-JEPA +3.7. With DriveFine's +2.6 and Auto-JEPA's +3.5 that is eight measured deltas.
- Hardware: 64 A800 for Stage 1, 32 A800 for Stage 2, batch 4/GPU, AdamW, bf16, DeepSpeed ZeRO-2. LRs encoder 1e-5, scene projector 1e-4, joint predictor 1.5e-4, weight decay 0.04.

**Limitations**:
- The NAVSIM-v1 table cites Drive-JEPA at 89.0, its perception-free baseline, rather than its 93.3 full planner - a 4.3-PDMS understatement of the nearest methodological competitor. Submetrics confirm the identification. The v2 table does cite Drive-JEPA's strong configuration, flagged with a dagger.
- The v1 table also omits CLEAR 93.7, DriveSuprim 93.5, HybridDriveVLA 92.1, DynVLA 91.7, SimWAM 91.5, and DriveFine, so 91.8 PDMS is mid-frontier in this wiki.
- The abstract's +1.6 and +1.3 margins are corrected-column figures. In the pre-fix column WA-JEPA leads Drive-JEPA-dagger by 0.2. Both claims are internally sound but the two columns tell different stories about the size of the lead.
- WA-JEPA loses the HUGSIM Extreme tier to DrivoR, 0.1362 vs 0.1407, on 103 of 436 scenarios. Every method is near the floor there.
- HUGSIM comfort is 0.6620 against about 0.95 for LTF, DrivoR, and VAD. A 12-step noise-initialized sampler has no mechanism enforcing kinematic consistency across closed-loop timesteps, and nothing in the method addresses it.
- HUGSIM baselines are LTF, DrivoR, UniAD, and VAD - three of them pre-2024 architectures. No world-model or VLA baseline. DrivoR's number is the authors' reproduction, since its published scores used the 345-scenario release.
- No ablation separates the future-prediction training objective from the inference-time generation; every row that removes one removes both. This is exactly the control SimWAM ran and found empty. The live hypothesis is that the 12-step scene denoising at inference contributes nothing.
- The encoder ablation confounds JEPA objective with video pretraining: MAE, DINOv3, and SigLIP2 are all image-level, and no video-pretrained non-JEPA control is included. Drive-JEPA's Table 7 has the identical hole.
- The asymmetric stop-gradient is described at length in Appendix C and never ablated. Lambda_future and lambda_act are never given values.
- 64 + 32 A800 GPUs with no wall-clock or GPU-hour total. No latency, FPS, or parameter count anywhere, against Latent-WAM's 107 ms and SimWAM's 518 ms.
- No scaling study on either axis. Stage 1 data volume is never quantified beyond "multi-view driving videos from nuPlan"; no encoder-size or predictor-size sweep.
- Stage 1 pretrains on nuPlan and NAVSIM navtest derives from OpenScene which derives from nuPlan. The paper never states whether navtest scenes are excluded from the pretraining corpus. No action labels are used in Stage 1, so this is not label leakage, but visual familiarity with the eval scenes would still inflate results.
- The 10-seed study covers only the main experiment; ablations are presumably single-seed, so the 0.3-0.4 gaps within Table 4(b) sit near the measured noise floor.
- NAVSIM and HUGSIM only. No Bench2Drive, navhard, nuScenes, or Waymo.

**Cross-page effect**: two structural corrections to the wiki. First, NAVSIM-v2's EPDMS column has been mixing pre-fix and corrected evaluator results for several ingests, and WA-JEPA's two-column table makes the partition legible for the first time - WAM-Diff is no longer the v2 leader, the correction is worth more than the spread between the top six entries, and it does not preserve ranking. Second, HUGSIM has the same problem independently: the wiki's HAD-L and Latent-WAM scores predate a scenario-set expansion and a controller heading-order fix, so they cannot be read against WA-JEPA's. Protocol drift is now tracked as a first-class confound on both benchmark pages.

**New synthesis**: the FM-vs-regression result opens an axis the wiki had not tracked - the *form* of the future-prediction objective, not just its presence. Regression on multi-view scene latents is worse than no future prediction at all (90.7 vs 91.1) because it collapses to a temporal mean, measured directly. Reconciled with Auto-JEPA via target entropy: deterministic objectives suffice for low-entropy targets (a single ego trajectory) and fail on high-entropy ones (a four-camera scene), which leaves Drive-JEPA in the awkward middle and puts DeepSight's, FLARE's, and Latent-WAM's deterministic scene-level objectives under a testable question.

**Naming hazard, updated**: the wiki now holds three V-JEPA 2 papers evaluated on NAVSIM - Drive-JEPA, Auto-JEPA, and WA-JEPA - by three different groups, sharing almost nothing beyond the backbone. All three source pages now open with or contain an explicit disambiguation, and `wiki/sources/wa-jepa.md` carries the three-way comparison table.
---

## 2026-09-02 - Ingest: GeoWAM

**Source**: `raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md`
**arXiv**: 2608.23486v2 (published 2026-08-25)
**Project page**: https://yiren-lu.com/project_pages/geowam/
**Authors**: Yiren Lu, Xin Ye (corresponding), Jiaming Liu, Philip Jacobson, Jin Yao, Yi-chung Chen, Liam Merino, Dhruva Dixith Kurra, Min Cai, Tom Lampo, Yu Yin (corresponding), Danhua Guo, Burhan Yaman (project lead) - Uber AV Labs + Case Western Reserve University
**Confidence**: high - full method text, all three figures, all three tables, complete implementation details. The paper contains no ablation section.

**Pages created**:
- `wiki/sources/geowam.md` - two-stage method with all loss definitions, three figures embedded, all three tables reproduced, a dedicated section documenting the protocol incommensurability, eight relationship entries, and a five-part limitations section

**Concept pages updated**:
- `wiki/concepts/navsim-benchmark.md` - **added "GeoWAM Breaks the Two-Protocol Model"**, which retracts the clean corrected/pre-fix partition written during the WA-JEPA ingest. Transfuser has digit-for-digit identical submetrics in WA-JEPA and GeoWAM but EPDMS 76.7 vs 84.0; DiffusionDrive differs by at most 0.3 on submetrics but 84.5 vs 88.2 on EPDMS. GeoWAM's own table is internally mixed - its DriveSuprim and Hydra-MDP++ rows are byte-identical to Drive-JEPA's including EPDMS while Transfuser and DiffusionDrive are recomputed. Added a navtest row and a caveat; reframed the cohort partition as WA-JEPA's attribution rather than ground truth.
- `wiki/concepts/navhard-ood-evaluation.md` - **rewritten**. Was three thin rows; now has the combined-EPDMS leaderboard from GeoWAM's Table 3 (ten methods), an explicit warning about the two incompatible reporting conventions (combined vs per-stage), a Stage-2 collapse table showing every method losing about half its lane keeping, the per-stage reports section, and four open questions. Confidence stays medium.
- `wiki/concepts/world-model-for-ad.md` - added Pattern 24 "Metric Geometry as the World-Model State Space" with an updated prediction-target table including annotation cost and coordinate-frame alignment; World-Model-vs-VLA row; three open questions including the navtest/navhard asymmetry.
- `wiki/concepts/perception-for-planning.md` - new "Metric Geometry Without Annotation" section positioning GeoWAM as a fourth answer to the label-cost question alongside SGDrive (annotation), Latent-WAM (distillation), and Auto-JEPA (nothing); comparison-table row.
- `wiki/concepts/foundation-backbones-for-ad.md` - new section establishing visual geometry models (DUSt3R/CUT3R/VGGT/MapAnything, specialized as DVGT/DVGT-2) as a third backbone family alongside language/VLM and video-generation; notes GeoWAM credits a strong initialization and a novel objective together; new backbone-role row and takeaway.
- `wiki/concepts/diffusion-planner.md` - new "A Deterministic Counterexample" section: GeoWAM tops two tables with single-trajectory L1 regression and no sampling, its EC of 86.8 against the 77-band is the likely mechanism, and the honest reading is that it never varies the head so it is evidence about representation quality rather than head design.

**Index and README updated**: rows added to both; README counts 60 to 61 papers; at-a-glance now says v2 EPDMS is not comparable across papers and adds a navhard leader line; Known Gaps now leads with DVGT-2; new Open Thread on whether navtest is the wrong benchmark for world models.

**Key facts**:
- Thesis: pixels encode geometry and motion indirectly, entangled with appearance/texture/illumination, so a video world model can produce plausible futures via photometric regularities without recovering 3D transformations. Geometry is native because point clouds explicitly encode structure and because scene geometry and ego trajectories live in the same 3D coordinate frame.
- Stage 1: DVGT-2 geometry encoder produces multi-level geometry tokens X and ego tokens E. A 6-layer, 1024-d, 16-head future decoder takes learned queries with time/view/2D-sinusoidal embeddings, applies causal temporal self-attention across F future steps then cross-attends to historical memory. A shared Point DPT head decodes to dense point maps plus per-pixel confidence, one 3D point per pixel in the ego frame at t+k.
- Supervision is hybrid: L_feat is cosine alignment to features from pushing future images through the same encoder with stop-gradient (JEPA-like, but a plain shared encoder rather than an EMA target), plus L_point_future = reg + conf + multi-scale surface normal, plus L_point_current anchoring the encoder. Future images never reach the forecasting branch or inference.
- Stage 2: N_e learned ego-query seeds, causal temporal self-attention, cross-attention to both historical memory and stop-gradiented predicted future geometry. Trajectory loss cannot propagate into the geometry - the paper's inverse-dynamics-like reading. Action head concatenates historical and predicted future ego tokens, refines with a causal temporal transformer, and a learned trajectory query plus regression head emits ONE trajectory (x, y, theta) with no anchors, mode classification, or sampling.
- L_plan = L_pre + 5*L_traj + 5*L_pose, both loss weights 5, auxiliary L1 on relative poses between historical frames.
- The stop-gradient points opposite to WA-JEPA's: GeoWAM protects the world model from the policy, WA-JEPA lets action supervision shape the world model. Neither ablates it.
- Pretraining: OpenScene, nuScenes, Bench2Drive, Waymo, KITTI, Argoverse 2, DDAD. 3 historical frames, F=8 future frames at 2 Hz, 2-8 camera views dynamically sampled, 161 epochs, AdamW wd 0.05, future decoder LR 1e-4, pretrained components 2e-5, 5% warmup + cosine, bf16. Planning: 40 epochs on navtrain with 8 camera views.
- Future geometry (nuScenes val, ray depth): mean Abs Rel 0.257 vs Epona+DVGT 0.274, VGGT-World 0.325, Cosmos3+DVGT 0.376. Mean delta<1.25 0.754 vs 0.655 / 0.544 / 0.503. At 1s Epona+DVGT is better on delta<1.25 (0.732 vs 0.708); GeoWAM leads only from 2s. GeoWAM's own delta<1.25 is non-monotone (0.708 at 1s, 0.769 at 2s), unexplained.
- NAVSIM v2 navtest: 90.2 EPDMS, EC 86.8. DVGT-2 89.6, EponaV2 88.9, DriveLaW 88.6, PWM 88.2, DiffusionDrive 88.2, WoTE 87.7, DriveVLA-W0 86.9.
- navhard combined: GeoWAM 36.6, EponaV2-dagger 36.1, NavFormer-dagger 34.1, LTFv6-dagger 31.9, DVGT-2 31.7, DriveLaW 30.6, LTF 25.1, DriveVLA-W0 24.4, Ego MLP 14.1, CV 11.4. Dagger marks RL or PDMS-score supervision, which GeoWAM does not use.
- The gain over DVGT-2 - GeoWAM's own initialization and already a geometry model - is +0.6 on navtest and +4.9 on navhard.
- Stage 2 of navhard collapses everything: lane keeping falls from about 96 to about 48 for every learned planner and from 78.6 to 47.9 even for constant velocity; NC falls from about 97 to about 80. Stage 2 LK spread across learned methods is under 5 points against over 12 in Stage 1.

**Limitations**:
- The navtest number cannot be placed. GeoWAM scores Transfuser at 84.0 from nine submetrics digit-for-digit identical to the ones WA-JEPA, Drive-JEPA, and the wiki score at 76.7, and DiffusionDrive at 88.2 against WA-JEPA's 84.5 from submetrics differing by at most 0.3. Its own DriveSuprim 83.1 and Hydra-MDP++ 81.4 rows are byte-identical to Drive-JEPA's including EPDMS, so the table mixes conventions internally and there is no way to tell which convention 90.2 belongs to.
- The gain over DVGT-2 is +0.6 EPDMS on navtest, and DVGT-2 is the paper's own initialization. The abstract's claim of "substantially stronger driving policies than image-based alternatives" rests entirely on cross-paper comparisons where backbone, data, and training all differ.
- There are no ablations in the paper. No geometry-pretraining vs none, no L_feat vs L_point decomposition, no stop-gradient ablation, no horizon or camera-count sweep. Most importantly the paper never trains its own architecture with a pixel-prediction objective, which is the experiment its geometry-beats-pixels thesis requires.
- The action head is deterministic and unimodal. It buys EC 86.8 and costs Best-of-N, proposal diversity, and any handle for a scorer or preference optimizer. Not discussed.
- Point-map targets are pseudo-labels from geometry foundation models, and the encoder is initialized from DVGT-2, so supervision is bounded by that family's biases. "Requires only RGB" is presented as a pure advantage without discussing the dependency.
- L_feat uses a plain shared encoder with stop-gradient rather than an EMA target, the classic collapse risk. The dense point objectives presumably prevent it, but no representation-health metric is reported.
- "Inverse-dynamics-like" is loose: a learned decoder produces ego tokens conditioned on predicted geometry, which is conditioning with a stop-gradient, not inversion.
- Geometry prediction is evaluated on nuScenes validation while nuScenes is in the pretraining mix, and OpenScene is a pretraining dataset while NAVSIM navtest derives from OpenScene. Neither exclusion is stated. Same undisclosed overlap flagged for WA-JEPA's nuPlan pretraining; it now looks systemic.
- Two of three geometry baselines are two-stage video-then-reconstruct pipelines whose errors compound by construction, so part of the margin is pipeline depth rather than representation choice.
- Bench2Drive is in the pretraining mix but no Bench2Drive result is reported. No HUGSIM, no NAVSIM-v1.
- No compute reported at all - no GPU count, no hours - for 161 pretraining epochs across seven datasets plus 40 finetuning epochs at 8 camera views. No latency, FPS, or parameter count. No seed variance.

**Cross-page effect**: this ingest retracts a claim made one ingest earlier. The WA-JEPA ingest established a clean corrected/pre-fix partition of NAVSIM-v2 EPDMS and used it to re-rank the wiki's v2 table. GeoWAM shows that partition is not reliable: identical submetrics produce EPDMS values 7.3 points apart across papers, GeoWAM's own table mixes conventions, and at least one of "more than two variants exist", "WA-JEPA's attribution is partly wrong", or "papers copy baseline rows without recomputing" must be true. The defensible position is now stronger and simpler - NAVSIM-v2 EPDMS is not comparable across papers, and cross-paper v2 rankings including this wiki's own table should be read as indicative only.

**New synthesis**: the navtest/navhard asymmetry. GeoWAM's future-geometry forecasting is worth +0.6 EPDMS open-loop and +4.9 under the reactive protocol, the same architectural change worth eight times more where errors compound. This is what a world-model thesis predicts and it is the first measurement of it in the wiki. It also means the benchmark nearly every world-model paper optimizes may be the one where world modeling matters least. Recorded as an open question on both the world-model and navhard pages and as a new README open thread.

**Second synthesis**: geometry joins language/VLM and video generation as a third backbone family, and the wiki now has three families each with papers reporting 89-92 on NAVSIM-v2 and no experiment holding the planner fixed across families. DriveLaW ran the controlled comparison for BEV vs VLM hidden states vs video latents; geometry was not in it, and pixels are not in GeoWAM's.
---

## 2026-09-02 - Ingest: DA-WAM

**Source**: `raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md`
**arXiv**: 2608.19085v2
**Code**: https://github.com/LeapWM/da-wam
**Authors**: Ruiguo Zhong, Benshan Ma, Xiaolong Chen, Lang Zhang, Mingyue Feng, Yaonong Wang, Pei Liu, Jun Ma - HKUST (Guangzhou), Leapmotor, HKUST
**Confidence**: high - full method text, all four figures (present locally in raw/assets under generic filenames), all six tables including the notation appendix, complete ablations

**Pages created**:
- `wiki/sources/da-wam.md` - method with all loss definitions, four figures embedded, all six tables reproduced, a four-way JEPA comparison table, six relationship entries, and a six-part limitations section

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` - added Pattern 25 "One Future Per Candidate" with DA-WAM's four-design taxonomy; **substantially revised the test-time-imagination synthesis** with a new subsection "DA-WAM Supplies the Missing Variable"; World-Model-vs-VLA row; two open questions on whether the 31 unsupervised futures are futures and whether the effect survives a realistic horizon
- `wiki/concepts/navsim-benchmark.md` - v1 row (ties CLEAR at 93.7) and v2 row; DA-WAM's TransFuser 76.7 makes the protocol tally three papers to one against GeoWAM's 84.0; documented DA-WAM's table as the second provably mixed-convention table; scope-and-attribution caveat
- `wiki/concepts/selection-based-planning.md` - new section on scoring candidates against their own predicted futures, the retrieval-based hard-negative construction contrasted against DriveSuprim's filtering-based one, the anti-pooling scorer design, and the candidate-count comparison against Auto-JEPA; methods-table row
- `wiki/concepts/counterfactual-prediction.md` - DA-WAM added to sense D with a note that it is the wiki's purest instance and uses "counterfactual" throughout for a strict rung-2 computation, while being unusually careful about the underlying data problem
- `wiki/concepts/foundation-backbones-for-ad.md` - new section "LoRA vs. Full Fine-Tuning: Two Papers, Opposite Answers" reconciling Latent-WAM's LoRA collapse (89.3 to 68.5) against DA-WAM's LoRA win (92.98 vs 92.62) via distance between pretrained representation and target; V-JEPA 2.1 backbone-role row; the EMA target-policy ablation

**Index and README updated**: rows added to both; README counts 61 to 62 papers; NAVSIM-v1 leader line now shows CLEAR 93.7 = DA-WAM 93.7; Open Thread #1 substantially rewritten around the shared-vs-per-candidate variable; Known Gaps now leads with DrivoR, Centaur, SparseDriveV2, iPad and the six cited-but-uningested related methods.

**Key facts**:
- Thesis: the planning value of a world model is bounded by how directly its predictions influence candidate-level scoring. Figure 1 taxonomy: (a) trajectory-only, (b) loosely coupled fusion with a single proposal, (c) one future shared across candidates, (d) DA-WAM's one future per candidate.
- Online encoder is V-JEPA 2.1 with LoRA on selected layers, base frozen, LoRA updated by both future-prediction and planning gradients. EMA target encoder with stop-gradient supplies Z_{t+delta} from the observed future frame, training only.
- Action-conditioned prediction: a_i = E_tau(tau_i), then Zhat_i = P_phi(Q=a_i, K=Z_t, V=Z_t). The predictor is shared across all candidates deliberately, so differences among Zhat_i come from the action queries rather than per-candidate parameters.
- Expert matching: i_exp = argmin_i ADE(tau_i, tau_exp). L_pred is applied token-wise ONLY to the expert-matched candidate. The other N-1 latents get no feature-level supervision and are shaped solely by scoring losses.
- Scorer cross-attends (Z_t, Zhat_i, a_i) and explicitly avoids pooling futures into a proposal-invariant vector. Factorized heads emit NC/DAC/EP/TTC/Comfort, then a utility head aggregates h_i and qhat_i into a scalar. Factors supervised by simulation-derived or rule-based metrics.
- Hard negatives retrieved from an offline trajectory bank under dual constraints: d_traj < eps_geo (geometrically close to expert) and Delta_safety > eps_safety (substantially worse safety). Appended to the candidate set, given their own future latent, excluded from expert matching and L_pred, upweighted in ranking pairs.
- L = lambda_pred L_pred + lambda_factor L_factor + lambda_score L_score + lambda_rank L_rank. Pairwise ranking uses cross-entropy over sigmoid score differences.
- Implementation: 2 historical frames, front camera only, 32 candidates of 8 poses each, prediction horizon 0.5 seconds, 20 epochs on 8 GPUs at batch 8/GPU.
- NAVSIM-v1: 93.7 PDMS (NC 99.1, DAC 98.9, TTC 96.8, Comfort 99.8, EP 90.0). Ties CLEAR for the highest non-BoN result in the wiki.
- NAVSIM-v2: 87.7 EPDMS with ViT/L, claimed to exceed the strongest comparison by 0.2.
- Table 3 matched ablation: no future 93.31, shared global future 92.81, current-latent 93.25, action-conditioned 93.46, action-conditioned + hard negatives 93.68. The shared-future row improves NC (99.02) and TTC (96.54) while EP collapses 91.36 to 88.68.
- Table 4: frozen+2.0+frozen 91.26, frozen+dense+frozen 91.95, LoRA+2.0+frozen 92.74, LoRA+dense+frozen 92.98, full-ft+dense+frozen 92.62, LoRA+dense+separate 93.10, LoRA+dense+shared 93.34, LoRA+dense+EMA 93.68. LoRA beats full fine-tuning by 0.36; EMA target beats frozen by 0.70.
- Table 5 candidate count: 1 -> 87.11, 8 -> 90.76, 16 -> 91.89, 32 -> 93.68, 64 -> 93.68.
- Introduces V-JEPA 2.1 (arXiv 2603.14482, dense features in video SSL) to the wiki. Its dense objective is worth +0.69 PDMS frozen and +0.24 under LoRA.
- Fourth JEPA paper in the wiki and the first to cite two others (Drive-JEPA and Auto-JEPA), both described accurately.

**Limitations**:
- The headline mechanism is worth +0.15 PDMS. The no-future ablation baseline is already 93.31, which would rank third in this wiki on its own. Hard negatives add +0.22 and are not a world-model contribution. The representation choices add +2.42. The mechanism the paper is named for is the smallest measured effect in it.
- Single run, no seed variance. WA-JEPA measured 0.053 EPDMS seed std for a stochastic sampler and training-seed variance is typically larger, so a 0.15 gap is not demonstrated to exceed noise.
- The predicted future horizon is 0.5 seconds while candidates span 8 future poses. Collisions, lane departures, and rule violations - the action-specific consequences the introduction promises the scorer will exploit - mostly occur beyond 0.5 s. Never discussed, never ablated.
- 31 of 32 predicted futures receive no feature-level supervision. Nothing shows they encode anything future-like; no decoding, no cross-candidate divergence statistics, no check that a braking candidate's latent differs from an accelerating one's as physics requires. WA-JEPA's temporal-collapse metrics are the right instrument and are not applied.
- The paper calls its per-candidate latents "counterfactual" throughout, including a section heading, for a computation with no abduction step where the alternative action is specified rather than observed. Strict rung 2. The engineering is appropriate; the terminology is what the counterfactual-prediction page exists to disambiguate.
- The NAVSIM-v2 table mixes conventions: TransFuser 76.7, SparseDriveV2 86.7, DriveSuprim ViT/L 86.0, and ARTEMIS 83.1 match WA-JEPA's pre-fix column while DiffusionDriveV2 87.5 matches its corrected column. The "+0.2 over the strongest comparison" claim is measured against that one corrected row. DA-WAM's own 87.7 cannot be assigned to either convention.
- ARTEMIS's EC is reported as 98.3, duplicating its HC, where WA-JEPA and GeoWAM both report "-". A transcription error carried into a published comparison.
- NAVSIM-v1 table omits CLEAR 93.7, Drive-JEPA 93.3, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7, SimWAM 91.5 - though it is better populated than most, including DrivoR, Centaur, SparseDriveV2, iPad, and DIVER.
- The EMA momentum mu is never given despite the EMA target being the single largest isolated design gain (+0.70). Also unreported: all four lambda weights, the per-factor lambda_k, eps_geo, eps_safety, LoRA rank, which layers get LoRA, M, and D.
- No latency, FPS, or parameter count, despite running the predictor 32 times per frame plus a token-level scorer over each candidate's full future latent.
- Only 2 historical frames from a single front camera, against WA-JEPA's 4 frames x 4 cameras and GeoWAM's 3 frames x 8 cameras.
- NAVSIM only. No navhard, HUGSIM, Bench2Drive, nuScenes, or Waymo - a notable gap for a paper claiming future prediction sharpens safety-critical discrimination.
- Factor heads are supervised by simulation-derived metrics, so this is Hydra-MDP-style privileged distillation, the same caveat as Auto-JEPA's CLOVER-initialized scorer.

**Cross-page effect**: this ingest reorganizes Open Thread #1, which has been the wiki's central dispute since the SimWAM ingest. The question had been framed as "does conditioning on a generated future help at inference," with SimWAM's mask ablation and DriveLaW's denoising sweep both answering no. DA-WAM shows the question was underspecified: those experiments varied whether a *shared* future reaches the planner, and DA-WAM measures that exact configuration at 0.50 PDMS *below* not modelling the future at all, with a visible mechanism (an averaged future cannot attribute a hazard to a candidate, so the scorer becomes uniformly cautious and ego progress collapses). Only per-candidate futures help. The synthesis now reads: shared future conditioning is useless to harmful, only per-candidate futures help, and then by little. The negative half of DA-WAM's result is better supported than the positive half.

**Second cross-page effect**: the LoRA contradiction. Latent-WAM found LoRA collapsed geometric distillation from 89.3 to 68.5 EPDMS and concluded low-rank adaptation is too restrictive; DA-WAM finds LoRA beats full fine-tuning by 0.36 PDMS for JEPA latent adaptation. Reconciled on the backbones page by the distance between the pretrained representation and the target - geometric distillation asks a semantic encoder for metric features (a large move LoRA cannot express), JEPA adaptation asks a video-predictive encoder to keep predicting video (where the risk is destroying a prior that is already close). Predicts that GeoWAM's DVGT-2 fine-tuning should also favour LoRA, which it does not test.

**Protocol note**: DA-WAM reports TransFuser at 76.7, agreeing with WA-JEPA and Drive-JEPA against GeoWAM's 84.0 from identical submetrics. The tally is three to one. It is also the second table provably mixing conventions, which supports the "papers copy baseline rows without recomputing" explanation over the "more than two aggregation variants" one.

**Figure note**: the source markdown references figures as remote arXiv URLs rather than local wikilinks, but all four assets are present in raw/assets under the generic filenames pipeline_compare.png, overview2.png, counterfactual_trajectory_supervision2.png, and camera_bev_score_comparison_32.png. The source page uses standard local wikilinks.
---

## 2026-09-02 - Lint (post four-paper JEPA/WAM batch)

Run after ingesting Auto-JEPA, WA-JEPA, GeoWAM, and DA-WAM.

**Structure**: 62 source pages, 30 concept pages, 62 raw papers - counts consistent with README. **Zero orphan pages** and **zero frontmatter violations** across all 92 wiki pages (all seven required keys present everywhere). Zero broken wikilinks or asset references.

**Contradictions found**:
- **ARTEMIS extended comfort is unsourced and contradicted three ways.** The master NAVSIM-v2 table carried EC 89.1 with no provenance; every ingested paper that reproduces ARTEMIS's v2 row (WA-JEPA, GeoWAM, ExploreVLA, DiffusionDriveV2) reports EC as "-", and DA-WAM reports 98.3, which duplicates ARTEMIS's own HC of 98.3 and is almost certainly a transcription error. Marked the cell with a warning and the full provenance rather than deleting a number whose origin is unknown.
- The TransFuser 76.7-vs-84.0 and DiffusionDrive 84.5-vs-88.2 discrepancies were already documented during the GeoWAM ingest; DA-WAM's TransFuser row (76.7) makes the tally three papers to one.
- The Latent-WAM vs DA-WAM disagreement on LoRA is documented and reconciled on the backbones page, so it is a resolved tension rather than an open contradiction.

**Stale claims corrected**:
- `wiki/index.md` - SimWAM described as "highest WAM in wiki" at 91.5 PDMS; superseded by WA-JEPA 91.8 and DA-WAM 93.7. Now scoped to "at ingest" with the successors named.
- `wiki/concepts/navsim-benchmark.md` - Latent-WAM "has the best EC among world-model-style entries" (87.3); WA-JEPA's 88.1 passed it. Rewritten as an at-ingest claim.
- `wiki/concepts/navsim-benchmark.md` - CLEAR "is the highest non-BoN result in the wiki"; DA-WAM ties it at 93.7. Rewritten as a shared claim.
- `wiki/concepts/best-of-n.md` - "The strongest non-BoN wiki result is CLEAR (93.7)"; same fix.
- `README.md` - the Known Gaps mention counts were badly out of date (Hydra-MDP 59 -> 85, Vista 12 -> 72) and omitted every method surfaced by the last four ingests. Replaced the prose list with a table of the top 16 by mention count plus a tail listing, and added an explicit highest-value-next-ingests line.

**Orphan-adjacent**: `wiki/concepts/divergent-thinking-in-vlms.md` has zero inbound `[[concepts/...]]` wikilinks and is reachable only from the index's markdown link. Not a structural orphan but substantively disconnected; worth linking from `concepts/best-of-n.md` or `concepts/gspo-vs-grpo.md` on the next relevant ingest.

**Concepts recurring across many pages with no page of their own** (page spread, log.md excluded):
- **Evaluator / protocol drift (23 pages)** - the strongest candidate. Currently split across three benchmark pages (navsim-benchmark, hugsim-benchmark, navhard-ood-evaluation), each documenting a different instance of the same phenomenon. A single page could hold the general rule, the four measured NAVSIM-v2 deltas, the HUGSIM scenario-set change, and the navhard combined-vs-per-stage convention split.
- **Latency and deployment cost (59 pages)** - mentioned nearly everywhere, tracked nowhere. Most recent ingests report no latency at all (Auto-JEPA, WA-JEPA, GeoWAM, DA-WAM all omit it), while older entries report figures that are never compared. A page tabulating who reports what would make the omission visible.
- **Seed variance and single-run reporting (15 pages)** - WA-JEPA supplied the wiki's only measurement (std 0.053 over 10 seeds); every other paper is single-run, and several ablation deltas the wiki quotes are smaller than that.
- **Pretraining data overlap (13 pages)** - WA-JEPA (nuPlan -> navtest) and GeoWAM (OpenScene -> navtest, nuScenes -> nuScenes val) both leave the exclusion unstated. This now looks systemic rather than incidental.
- **Hard negatives (12 pages)** and **EMA target / stop-gradient (12 pages)** - both currently housed inside larger pages (selection-based-planning, world-model-for-ad) and adequately covered there for now.

**Questions to investigate next**, in rough order of how much they would change the picture:
1. Does world modeling buy open-loop accuracy or closed-loop robustness? GeoWAM's +0.6 navtest / +4.9 navhard split is the only measurement and it implies the benchmark the whole field optimizes is the wrong one for the mechanism it is testing.
2. Is DA-WAM's shared-vs-per-candidate result real at a realistic horizon? Its per-candidate futures reach 0.5 s while trajectories span 8 poses, and the positive effect is +0.15 PDMS single-run.
3. Are the 31 unsupervised per-candidate futures in DA-WAM actually futures? WA-JEPA's temporal-collapse metrics are the right instrument and nobody has applied them.
4. Which deterministic latent predictors are leaving performance on the table? WA-JEPA measures regression on scene latents as worse than no prediction; DeepSight, FLARE, and Latent-WAM all use deterministic objectives on scene-level targets.
5. Geometry vs pixels vs video latents under a fixed planner. DriveLaW ran the controlled comparison without geometry; GeoWAM argued for geometry without the control.
6. Can anything move navhard Stage 2 or HUGSIM's Extreme tier? Every method collapses to near-indistinguishable on both.

---

## 2026-09-04 - Ingest: See Tomorrow, Act Today: Foresight-Driven Autonomous Driving (ForeSight)

**Source**: `raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md` (arXiv 2605.07195v1)
**Orgs**: Fudan University (School of Data Science) + Shanghai Innovation Institute + Imperial College London + University of Surrey. Code announced at github.com/LogosRoboticsGroup/ForeSight.
**Page created**: `wiki/sources/foresight.md`

**What it is**: the maximal statement of the imagine-then-act paradigm. A frozen 2.5B Epona is declared the planner's *primary* visual encoder, run forward at inference to a finished imagined future; a 52M TransFuser current-frame branch (multi-view + LiDAR + ego status) is explicitly labelled "an additional supplement"; a 21M state-based action decoder consumes both.

**Method**:
- WM encoder: F_wm = WM^(t_d)(I, F_cond), shape T_wm x C_wm x H x W, sampled at an adjustable denoising step t_d whose value is never reported.
- WM-QFormer: spatiotemporal Transformer, N_wm learnable queries per generated frame, compressing to T_wm x N_wm x C. Stated purpose is to strip fine-grained texture and noise from generated frames before the planner sees them - the same pathology DriveLaW diagnosed at t=10, answered by filtering the finished future rather than reading an earlier one.
- Time state queries Q_s in R^{M x T_f x C}, one per future timestep, from the authors' own BridgeAD.
- Factorized attention: CrossAttn(Q_s, F_cur) then CrossAttn(Q_s + E_s, F'_wm + E_wm), sinusoidal position embeddings so T_wm and T_f may differ.
- Two-phase training: 80 epochs action pretraining without the WM, then 20 epochs post-training with the WM frozen. L = lambda_1 L_bev + lambda_2 L_traj, weights unreported.
- 8x H100. NAVSIM 3 cams + LiDAR at 1024x256, 4s/8 steps/20 modes. nuScenes 6 cams, images only, 640x360, 3s/6 steps/6 modes. Epona finetuned nuPlan 5 Hz to 2 Hz, then frozen.

**Results**:
- NAVSIM navtest: 89.3 PDMS (NC 98.8, DAC 97.2, TTC 94.8, Comf 100, EP 83.5). Best in its own "planning with world model" group: +0.4 over SeerDrive 88.9, +1.0 over WoTE 88.3. Correctly concedes ReCogDrive 90.8 and GoalFlow 90.3 beat it.
- nuScenes val: 0.62 avg L2 / 0.18 avg collision. Wins no column; beaten on both by World4Drive (0.50 / 0.16), the other world-model entry in its own table.
- Efficiency: 2.5B + 52M + 21M, **900 ms on an H100 with ~870 ms in the world model (96.7%)**. Slowest NAVSIM planner in the wiki.
- Table 3 components: 86.8 baseline, then 87.1 (+WM, vanilla attention), 87.9 (+WM-QFormer), 88.5 (+state queries), 88.2 (+factorized, without state queries), 89.3 full.
- Table 4: w/o WM 86.8, Simple WM 87.5, Foundation WM 89.3.
- Table 5 denoising steps: 25 gives 88.0, 50 gives 88.3, 75 gives 89.2, 100 gives 89.3. Paper recommends 75; headline uses 100.
- Table 6 generation: Epona FVD10 50.77, ForeSight 54.63 on nuPlan after the 2 Hz finetune.
- Table 7 w/o current encoder: 88.2 PDMS (DAC 96.3, TTC 95.4, EP 81.7) vs 89.3 full.
- Table 8 nuScenes with Vista instead of Epona: 0.64 L2 / 0.27 collision vs 0.62 / 0.18.
- Failure case (a): the WM correctly predicts a right turn and the post-turn scene, but the action decoder produces an overly conservative trajectory. The paper's own conclusion is that the WM and action model "should be more tightly coupled."
- Failure case (b): the WM degrades on a fast winding road but the trajectory stays accurate, credited to the current-frame encoder.

**Limitations**:
- The headline mechanism is worth +0.3 PDMS. Table 3 row 1 to row 2 adds the entire frozen 2.5B foundation world model under vanilla attention for +0.3; the remaining +2.2 comes from a compression-and-routing stack, one component of which (state queries, a BridgeAD mechanism) has no intrinsic world-model dependency and is never ablated without one. The causal story is not separable from "a better action decoder."
- 900 ms for 89.3 PDMS, against SimWAM's 91.5 at 518 ms with inference-time generation removed entirely. The last 25 of 100 denoising steps buy +0.1.
- Table 4's "Simple WM" baseline (87.5) is a reimplementation scoring below the published WoTE (88.3) and SeerDrive (88.9) in ForeSight's *own* Table 1. The argued +1.8 for foundation over simplified world models is +0.4 against real published numbers, at roughly 50x the parameters.
- nuScenes is a loss framed as a draw ("competitive performance"), in a paragraph that also hedges that nuScenes metrics are "not entirely comprehensive."
- Epona is finetuned on nuPlan and navtest is a nuPlan subset; whether navtest scenes were excluded is never stated. Sharper than the same gap flagged for WA-JEPA and GeoWAM, because here the finetuned generator is the primary encoder rather than a pretraining initialization.
- Table 5 is uninterpretable without t_d: it varies the schedule length while the extraction point is an unreported free parameter.
- The 2 Hz finetune degrades generation (FVD 50.77 to 54.63), reported as "nearly the same."
- The Vista swap loses on 6 of 8 nuScenes columns and raises average collision 50%. Architecture-agnosticism shown as tolerance, not benefit.
- NAVSIM-v1 only. No EPDMS, no navhard, no Bench2Drive, no HUGSIM, no reactive closed loop - the most consequential gap for a method whose thesis is anticipation in interactive scenarios.
- Single runs, no seed variance, against deltas as small as +0.1 and +0.3.
- No RL, acknowledged in the paper's own limitations.
- Unreported: t_d, lambda_1, lambda_2, N_wm, C, C_wm, WM-QFormer depth and parameter count, FPS.

**Protocol note (positive)**: every NAVSIM-v1 baseline row matches this wiki's canonical values - UniAD 83.4, TransFuser 84.0, PARA-Drive 84.0, DiffusionDrive 88.1, LAW 84.6, Epona 86.2, WoTE 88.3. This is the first table ingested in several papers with no evaluator-drift problem to record.

**Cross-page effect 1 - the test-time-imagination synthesis gets its price tag.** Open Thread #1 has accumulated evidence that a *shared* generated future does not help at inference (SimWAM's isolated mask, DriveLaW's denoising sweep, DA-WAM's -0.50 for configuration (c)). ForeSight is the same configuration measured a third time, in a third architecture, at +0.3. The two matched measurements now straddle zero, which supports a weaker and better-founded claim than either paper makes on its own: a shared generated future is worth approximately nothing at inference. What ForeSight adds that nobody else supplied is the cost side - 870 of 900 ms - and one experiment nobody else ran: Table 7 shows a planner driven by generated futures *alone*, with no current-frame perception at all, reaches 88.2, roughly what a competent BEV world model (WoTE 88.3) delivers for two orders of magnitude less compute.

**Cross-page effect 2 - the first dissent on denoising depth.** DriveLaW's Table 6 (t=1 gives 89.1, t=10 gives 23.2) has been the wiki's main evidence for "borrow the generator's representation, not its imagination." ForeSight's Table 5 runs the other way: 25 steps 88.0, 100 steps 89.3, monotone. These are different variables - DriveLaW moves the extraction point along a fixed schedule, ForeSight changes the schedule length with the extraction point unreported - so this is logged as an unresolved tension, not a contradiction. Publishing t_d, or running DriveLaW's extraction sweep inside ForeSight at fixed schedule length, resolves it in one run. Added as a new open question on the world-model page.

**Pages updated**: `wiki/concepts/world-model-for-ad.md` (new Pattern 26; new synthesis subsection "ForeSight Prices the Paradigm - and Disputes DriveLaW's Sweep"; imagine-then-act camp roster; computational-cost challenge; "what survives across all seven papers"; two open questions), `wiki/concepts/navsim-benchmark.md` (SOTA row + caveat), `wiki/concepts/nuscenes-waymo-evals.md` (new section on hedging the benchmark while reporting on it), `wiki/concepts/foundation-backbones-for-ad.md` (new section: a diffusion world model as the whole visual encoder; 35:1 frozen-to-trained ratio; the lossy 2 Hz finetune), `wiki/concepts/perception-for-planning.md` (new item 8: how much perception a generated future replaces), `wiki/index.md`, `README.md` (63 papers).

**Figure note**: all five figures (cmp.png, pipeline.png, visual.png, visual_supp.png, visual_fail.png) and all eight tables are present in the source markdown and reproduced on the wiki page.

---

## 2026-09-04 - Ingest: BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving

**Source**: `raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md` (arXiv 2608.12854v2)
**Orgs**: NLPR, Institute of Automation, Chinese Academy of Sciences (CASIA) + Li Auto Inc. Bing Zhan and Shuyao Shang equal contribution; Jiahao Gu project lead.
**Page created**: `wiki/sources/brainwam.md`

**What it is**: a VLA branch (Qwen3-VL-4B) and a WAM branch (Wan2.2-TI2V-5B) run in parallel, each compressed to 8 action tokens, coordinated only through those tokens by CAB (bidirectional zero-init gated cross-attention at layers 9 and 18, 16.8M) and CIF (2-layer Transformer with AdaLN on the action timestep, then element-wise mean, 49.3M). Both branches frozen in stage 3. 89.5 PDMS NAVSIM v1, 89.6 EPDMS NAVSIM v2, 475-644 ms on an H20.

**The result that carries the paper - modality competition in Tri-MoT**:
- Tri-MoT (VLM tokens + VGM tokens + action tokens in one shared attention pool) scores 87.8 PDMS, BELOW the paper's own WAM-only branch at 88.1, with strictly more information and comparable parameters.
- Diagnosis: action tokens attend more strongly to VLM tokens than VGM tokens across most layers (Fig. 2), because VLM tokens are clean large-scale-pretrained abstractions while VGM tokens are mid-denoising and low-signal. The easier modality wins the shared attention budget.
- Two controls rule out simpler explanations. The VGM tokens are not uninformative: disabling video denoising drops PDMS to 79.3. And Tri-MoT is not capacity-limited: it has strictly more information than WAM-only and still loses.
- This is the wiki's first measurement of a VLM actively degrading a world-model planner.

**Method details**:
- WAM branch: Wan2.2-TI2V-5B + lightweight rectified-flow action expert, Dual-MoT coupling (shared self-attention, modality-specific FFNs). Independent rectified-flow timesteps t_v and t_a for video and action. L_WAM = L_vid + lambda_pred * L_pred.
- VLA branch: Qwen3-VL-4B encodes multi-view images and instructions into semantic tokens U, ego history into state tokens E; action expert denoises x_a into A_sem via Dual-MoT.
- CAB: L=8 tokens per stream at hidden dim 1024, 8 heads x 128, no bias on QKVO, separate norms for query and context, tanh gates zero-initialised (Flamingo-style) so CAB starts as identity.
- CIF: project both streams to 1024 with learnable source embedding, concat, 2-layer Transformer, element-wise average, decode to action velocity. Only L_fuse is supervised in stage 3.
- Training: 3 stages x 100K steps, 8x H20, batch 6/GPU, AdamW lr 5e-5 cosine, 200 warmup, wd 0.01, bf16, DeepSpeed ZeRO-2. Inference: 3-step action sampling, 1-3 video steps.

**Results**:
- NAVSIM v1: 89.5 PDMS (NC 98.1, DAC 97.5, TTC 94.9, C 100.0, EP 83.8). DAC and EP lead its own table; NC and TTC mid-table. Camera only, no LiDAR.
- NAVSIM v2: 89.6 EPDMS (NC 98.1, DAC 97.5, DDC 99.6, TLC 99.9, EP 88.2, TTC 97.4, LK 97.6, HC 98.4, EC 85.8). EC 85.8 is unusually strong for a generative planner (DriveVLA-W0 58.9, DriveDreamer-Policy 79.4) and is not investigated.
- Table 3: VLA-only 86.1, WAM-only 88.1, Tri-MoT 87.8, BrainWAM 89.5. Coordination is worth +1.4 over WAM-only.
- Table 4: CAB only 88.7, CIF only 88.5, both 89.5.
- Table 5 asynchronous video denoising (H20): 0 steps 382 ms / 79.3 PDMS / 75.8 EPDMS; 1 step 475 ms / 89.3 / 89.4; 2 steps 565 ms / 89.5 / 89.6; 3 steps 644 ms / 89.4 / 89.6.
- Table 6 CAB block count (10-step joint denoising reference, 89.3): 1 -> 88.9, 2 -> 89.3, 3 -> 89.2, 5 -> 89.3, 28 -> 89.3. Saturates at two.
- Table 7 CIF fusion: MLP 88.8, Gate 89.1, Transformer 89.3.
- Table 8 CIF depth: 1 -> 89.0, 2 -> 89.3, 3 -> 89.3.
- Table 9 stage-3 strategy: full-model fine-tuning 88.8, frozen branches + CAB/CIF/decoder only 89.5. Supporting datum: VLA-only reaches 86.1 at 54K steps, WAM-only needs 81K steps for 88.1.

**Limitations**:
- 89.5 is mid-frontier; the v1 table tops out at AutoVLA 89.1 and DriveLaW 89.1 and omits CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7, SimWAM 91.5, FLARE 91.4, DiffusionDriveV2 91.2, SGDrive 91.1, DriveVA 90.9.
- DriveSuprim appears in the v2 table but not the v1 table, where its 93.5 would beat BrainWAM by 4.0. The authors evidently know the method. No stated criterion excludes it.
- Three v1 baseline rows are weaker configurations presented unqualified, each verifiable against wiki records. DynVLA 87.2 is its SFT-only score (submetrics 98.6/95.3/95.5/100/80.6 match the wiki's DynVLA SFT row exactly) against a published post-RFT 91.7. DriveVLA-W0 87.2 (98.4/95.3/95.2/100/80.9) is DriveLaW's flow-matching reimplementation row, which DriveLaW marks with a dagger and BrainWAM does not. ReCogDrive 86.5 is the IL-only variant against 89.6 with RL. None is fabricated; all three flatter the comparison.
- The 0-step video ablation does not show what it is said to show. With zero denoising steps the video stream is pure Gaussian noise, not absent, fed into a pathway trained on partially-denoised features. That is a distribution-shift test. The clean version is SimWAM's isolated attention mask, which BrainWAM does not run. Only the 1-vs-2-vs-3 rows are informative.
- 475-644 ms on H20, acknowledged as not deployable. Two large backbones (5B video + 4B VLM) resident at inference.
- The video branch is never evaluated as a generator - no FVD, no FID, no future frames shown. Predictive dynamics is established purely by ablation, so it cannot be told apart from a well-initialised visual encoder.
- Fig. 2 is the core evidence and is a single visualisation: no numeric ratios in text, no seed or checkpoint variance, no layer indices behind "most layers", and no causal test such as re-weighting attention toward VGM tokens to see whether Tri-MoT recovers.
- Tri-MoT is the authors' own baseline, not a published method. "Comparable parameter counts" is asserted, never tabulated. A weak instantiation would inflate everything downstream and there is no way to check.
- The neuroscience framing does no work: CAB is Flamingo-style zero-init gated cross-attention, CIF is a 2-layer Transformer plus a mean, and Tables 6-8 show both saturate at minimal depth.
- Single runs, no seed variance, against deltas of +0.8, +0.7, +0.2, +0.1.
- NAVSIM only. No navhard, Bench2Drive, HUGSIM, nuScenes, or reactive closed loop - notable given the qualitative claims centre on interactive negotiation.
- No RL; all three stages are supervised flow matching.
- Pretraining overlap for Wan2.2 and Qwen3-VL against OpenScene/nuPlan navtest is never discussed.

**Cross-page effect 1 - the denoising-depth question is now 2 against 1.** BrainWAM's decoupled timesteps make "video steps executed before caching" a free parameter, the axis closest to DriveLaW's extraction-point sweep. One step delivers 89.3 of an achievable 89.5; steps 2 and 3 add 0.2 then nothing for 169 ms. That reproduces DriveLaW's t=1 result in a completely different architecture (Wan2.2-5B + Qwen3-VL-4B versus LTX-Video DiT + 133M action DiT). ForeSight's monotone 25-to-100-step sweep is now the outlier, and it is also the only one of the three that never reports its extraction step t_d. The world-model page's working position is updated: the useful signal lives in the generator's early internal state, with ForeSight's sweep more likely confounded by a shifting extraction point than reflecting a real preference for finished futures.

**Cross-page effect 2 - a boundary condition for MoT.** BrainWAM's Tri-MoT is the first MoT design in the wiki that loses to its own single-branch ablation. Cross-referencing against the MoT designs that work gives a clean boundary: stream count is not the culprit (UniDriveVLA runs three streams successfully), and joint video-action attention is fine on its own (SimWAM bidirectional 90.2 vs isolated 90.3, DriveVA 90.9, DriveWAM 90.1 - none with a VLM in the pool). What separates failure from success is a clean pretrained semantic stream competing with an iteratively-denoised stream under symmetric unmasked attention. Every successful design breaks at least one of those conditions, usually the third. New section added to the MoE page with the comparison table, plus the untested middle ground: nobody has tried Tri-MoT with UniDriveVLA-style asymmetric masking, which would keep raw-token access while removing the competition.

**Cross-page effect 3 - freeze-then-coordinate now has a mechanism.** Table 9 (frozen 89.5 vs full fine-tuning 88.8) plus the convergence-rate measurement (VLA 54K steps, WAM 81K steps) is the wiki's clearest quantitative argument for freezing pretrained branches in a two-backbone planner. Three papers now converge on this from three different motivations: AutoMoT (catastrophic forgetting), ForeSight (capacity imbalance), BrainWAM (convergence-rate mismatch).

**Cross-page effect 4 - Wan2.2-TI2V-5B is now a four-way natural experiment.** SimWAM (training-time only, 91.5, 518 ms), DriveVA (joint denoising, 90.9), DriveWAM (chunked inverse dynamics, 90.1, 871-1262 ms), BrainWAM (8-token action bridge, 89.5, 475-644 ms). The ordering is inverse to how much video computation happens at inference. BrainWAM is also the only one with a VLM inside the model rather than outside the attention path, and it scores lowest of the four.

**Protocol notes**:
- ARTEMIS EC = 89.1 now has a published source. The last lint flagged this value as unsourced and contradicted; BrainWAM reports it. Tally is now one paper for 89.1, one (DA-WAM) for 98.3, four abstaining with "-". The wiki's warning marker is kept but rewritten.
- BrainWAM is the third v2 table provably mixing conventions, and mixes differently again: TransFuser 76.7, ARTEMIS 83.1, DriveVLA-W0 86.1 match the pre-fix cohort, while HydraMDP++ 81.4 and DriveSuprim 83.1 match neither (wiki carries 84.1 and 87.1). DriveSuprim's submetrics differ throughout, so that row is a different configuration rather than a recomputation.
- That is a fourth failure mode beyond the three already enumerated on the benchmark page: a baseline row silently drawn from a non-headline configuration. BrainWAM's v1 table does it three times. Practical consequence recorded on the page: submetric agreement, not just the aggregate, must be checked before treating two rows as commensurable.

**Pages updated**: `wiki/concepts/world-model-for-ad.md` (new Pattern 27; denoising-depth tally rewritten with a three-paper table; imagine-then-act roster; "what survives across all eight papers"; computational-cost section gains the asynchronous-truncation technique; two open questions rewritten/added), `wiki/concepts/navsim-benchmark.md` (v1 SOTA row + four-paragraph caveat; ARTEMIS EC provenance resolved; fourth failure mode added to the evaluator-drift analysis), `wiki/concepts/mixture-of-experts.md` (BrainWAM under MoT pattern 3; new "Where MoT Breaks" boundary section; comparison-table row), `wiki/concepts/dual-system-vla.md` (new BrainWAM section; freeze-then-coordinate; four-way two-backbone interface comparison), `wiki/concepts/foundation-backbones-for-ad.md` (new Wan2.2 four-coupling-strategy section), `wiki/index.md`, `README.md` (64 papers).

**Figure note**: all six figures (teaser 1.png, tri-mot.png, framework 1.png, training_pipeline.png, qualitative 1.png, appendix.png) and all nine tables are present in the source markdown and reproduced on the wiki page.

---

## 2026-09-04 - Ingest: Adaptive-WAM: Quality-Guided Early-Exit Planning from Intermediate Video-Diffusion Features

**Source**: `raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md` (arXiv 2608.06008v1)
**Orgs**: Institute for AI Industry Research (AIR), Tsinghua University + USTC + Beihang. Sining Ang, Yuguang Yang, Yan Wang. Code announced, not released.
**Page created**: `wiki/sources/adaptive-wam.md`

**What it is**: six independent ReCogDrive-style 5-step trajectory diffusion heads attached to blocks {5, 9, 15, 18, 22, 30} of a LoRA-adapted Wan2.2-TI2V-5B, fed by a SINGLE conditional forward at a fixed noise index (no denoising loop, no CFG unconditional branch, no VAE video decode). A fine-tuned DINOv2-Small verifier predicts the six NAVSIM components from the current image plus a candidate trajectory and terminates execution once the best plan accumulated so far clears a threshold. 90.8 PDMS NAVSIM v1, 89.9 EPDMS v2, 170 ms end-to-end on an A100.

**The central diagnostic - two axes the field had been conflating**:
- Video noise index, five values {1,9,17,25,32} of a 40-step schedule, single forward: block 15 gives {86.44, 86.56, 86.57, 86.55, 86.50}, range 0.13; block 18 range 0.15; 64-proposal diagnostics range 0.11 and 0.14. Essentially nothing.
- DiT readout depth, six exits sharing architecture / optimizer / batch size / epochs / head capacity: IL 81.94 / 83.60 / 86.56 / 84.14 / 83.62 / 80.71; post-RL 86.02 / 87.56 / 90.62 / 88.92 / 87.42 / 85.82. Spread 5.85 IL and 4.80 RL. Block 15 of 30 is best at both stages and the FULL-DEPTH block 30 is worst.
- Depth is worth roughly forty times what noise level is worth, and no prior wiki paper reports which layer it reads from.
- Planner-only RL lifts every exit by 3.80-5.11 points without changing the depth ordering.

**Why route rather than just pick block 15**: post-RL Jaccard overlap of per-exit high-quality scene sets (PDMS >= 90) runs 0.69-0.82 off-diagonal. Directional large-advantage counts (>= 50 point gaps, mean over ten paired runs): block 15 beats block 30 on 598.64 scenes but block 30 beats block 15 on 422.41. No fixed depth dominates scene-wise. Maximum cell-wise std falls 182.82 -> 84.94 after planner RL.

**Method details**:
- h_l = F_1:l(I, d(o); s*) with s* = 17. Text condition generated programmatically from deployment-available attributes (map metadata, discretised ego speed, a maneuver derived from past ego poses only, traffic density) - never a future-trajectory label.
- Scorer inputs are only the current front image and the flattened 8x3 trajectory; no ego state, no navigation command. Six independent two-layer MLP heads for NC, DAC, DDC, TTC, EP, Comf, composed through the normalised PDMS formula as Q = 100 * Gamma.
- Equal-weight soft-label BCE on un-binarised evaluator components, NO rank loss, because >95% of diagnostic scenes contain candidate groups that are jointly perfect, jointly zero, or tied at the top. Framed explicitly as an exit-quality verifier rather than a total-order ranker.
- Controller keeps the best trajectory accumulated across attempted exits; rejected exits execute only the unevaluated blocks, reusing cached hidden states and scores.
- Training: LoRA on Wan with L_actor = lambda_vid L_vid + sum_l lambda_l L_traj^l, video and trajectory sharing one s* forward; scorer trained in alternation on stop-gradient trajectories so it cannot shift the actor distribution; then planner-only DiffGRPO with backbone and scorer frozen, full five-step denoising chain as one action, NAVSIM evaluator reward.
- Layer-wise statistics use the validation-best checkpoint per seed aggregated over TEN seeds.

**Results**:
- NAVSIM v1: 90.8 PDMS single-trajectory (NC 98.6, DAC 97.9, TTC 95.6, Comf 100, EP 85.1). EP 85.1 leads every world-model entry in its own table. Front camera only, no LiDAR.
- Auxiliary fixed-B22 64-proposal model: 92.6 PDMS (NC 99.8, DAC 98.3, TTC 98.3, Comf 100, EP 86.6).
- NAVSIM v2: 89.9 EPDMS (NC 98.5, DAC 98.0, DDC 99.5, TLC 99.8, EP 87.6, TTC 97.4, LK 95.4, HC 98.2, EC 75.5). Human Agent in the same table is 90.3 - the closest approach to the human reference recorded here on v2.
- Zero-shot nuScenes: 0.88 m avg L2, 0.08% avg collision (DriveVA 0.84 / 0.06). Horizon breakdown shows a crossover: Adaptive-WAM leads at 2 s (0.71 vs 0.76) and trails at 3 s (1.58 vs 1.43).
- Routing sweep on A100 batch 1: fixed B15 90.62 @ 190 ms; eta=70 88.49 @ 112 ms; eta=80 90.64 @ 143 ms; eta=90 90.79 @ 170 ms; eta=95 90.75 @ 284 ms; full path 85.82 @ 320 ms. 94.1% of scenes exit within the first three blocks at eta=90.
- Wan adaptation ladder (single / 64-prop): frozen 84.20 / 89.91; separate LoRA then cache 84.95 / 90.80; joint LoRA 90.62 / 92.59; full fine-tuning 90.64 / 92.54.
- Visual backbone: ViT-S 83.91, ViT-B 85.62, ViT-L 88.88, Wan intermediate 90.62 single-trajectory; the same comparison narrows to 92.17 / 92.21 / 92.31 / 92.59 with 64 proposals.
- Scorer backbone diagnostic on a fixed candidate pool: best Wan exit (B22) 92.62, DINO-Small 92.59, DINO-Base 92.54, ResNet-50 92.55, ResNet-34 92.19, ViT-S/B 91.17 / 91.20. Wan buys 0.03 for a full world-model forward per attempted exit.
- Scorer reliability on 12,146 scenes: exact top-score selection 91.2%, within 5 points 94.4%, >= 20-point failures 0.57%, >= 50-point failures 0.42% (51 scenes).
- Full video-generation profile on the same A100: 40-step CFG rollout = 80 DiT forwards = 13.22 s (12.05 s denoising, 0.27 s VAE encode, 0.90 s VAE decode), 31.19 GiB peak. Conditional DiT 149.40 ms/step, unconditional 147.80 ms/step. VAE image encoding alone ~50 ms.

**Limitations**:
- 90.8 is mid-frontier. Table 2's caption says "baselines follow DriveVA", so it inherits that comparison set and omits CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7, SimWAM 91.5, FLARE 91.4, DiffusionDriveV2 91.2, SGDrive 91.1. The "SOTA among compared front-view video world-model planners" phrasing is accurate and the hedge is load-bearing. It also re-inherits DriveVLA-W0 at 87.2, the same non-headline value now propagating through DriveLaW, BrainWAM, and this paper.
- 92.6 is a different model: fixed block-22 exit, 64 proposals, no adaptive controller, non-diffusion four-block refinement decoder, and CLOVER-derived pseudo-expert targets scored with the TRUE NAVSIM evaluator using training-time map and future occupancy. Privileged supervision of the same class flagged for Hydra-MDP distillation, Auto-JEPA, and DA-WAM. Belongs in the selection-based family, where it does not lead DriveSuprim 93.5 or CLEAR 93.7.
- Provenance issue on that headline: Table 13's "fixed-exit 64-proposal" rows carry values identical to Table 19's "Wan-based scorer pretest", which Table 20 explicitly says "measure the true score of the selected candidate and are not end-to-end planner PDMS". One caption must be wrong, and 92.59 (Table 21) / 92.62 (Table 20) / 92.6 (Table 2) all sit in the same neighbourhood.
- "Validation-best checkpoint over ten seeds" is a max-over-validation selection rule, not variance reporting; it is optimistic relative to single-run numbers. No PDMS standard deviation is given anywhere - only cell-wise std for the pairwise-advantage matrices. The wiki still has exactly one measured PDMS/EPDMS seed std (WA-JEPA, 0.053). The 10-seed protocol is nonetheless better discipline than almost every paper here.
- The routing gain over the best fixed exit is +0.17 PDMS with no variance, smaller than WA-JEPA's measured std would suggest is resolvable. The defensible claim is the latency one (190 -> 170 ms). The "47% below the 320 ms fixed full-depth planner" headline compares against a configuration that is also 4.80 PDMS worse and would never be deployed.
- The verifier is trained on evaluator-provided component targets, so the deployed controller is distilled from the benchmark's own metric.
- Latency is a mean, not a bound: at eta=95 routing costs 284 ms, worse than the 190 ms fixed baseline. Relevant for a safety-critical scheduler.
- The noise-robustness result is measured on a SINGLE forward pass and does not directly refute DriveLaW's t=10 collapse, whose latents went through ten actual denoising iterations and carry different activation statistics. The paper scopes this correctly.
- Five noise indices, one scheduler, one backbone family, one head type. Whether "~50% depth" generalises is untested.
- NAVSIM and zero-shot nuScenes only. No navhard, Bench2Drive, HUGSIM, or reactive closed loop - notable because the routing thesis (spend more on harder scenes) is exactly what a hard/OOD split would test.
- Six trajectory heads whose combined parameter count is never reported. Front camera only.
- Figures 2 and 3 have captions in the source markdown but no image files; the data exists only as Appendix G tables.

**Cross-page effect 1 - the denoising thread is reframed, not just extended.** Open Thread #1's sub-dispute has been "how denoised should the conditioning latent be", with DriveLaW (t=1 best), BrainWAM (1 step enough), and ForeSight (100 steps better) disagreeing. Adaptive-WAM shows that was three questions wearing one name: noise index (<= 0.15 PDMS), denoising iterations (1 step suffices per BrainWAM and DriveLaW), and readout depth (4.80 PDMS, never previously varied). DriveLaW's t=1, BrainWAM's one step, and Adaptive-WAM's single conditional forward are the same operation - one pass through the video DiT - so three architectures now converge on it. ForeSight remains the outlier. New subsection "Adaptive-WAM Shows the Axis Was Wrong" on the world-model page with the three-axis table; the old two-paper tally is folded into it.

**Cross-page effect 2 - readout depth is a new design axis for the backbones page.** Every other entry there implicitly reads the final layer. Adaptive-WAM's controlled sweep (identical heads, identical budgets) shows the mid-network exit beating the final block by 4.80. That also puts a caveat on DriveLaW's representation sweep and SimWAM's four-way prior swap: if depth is worth 4.8 within one backbone, a cross-backbone comparison at unmatched relative depth may be partly measuring the readout point.

**Cross-page effect 3 - the strongest evidence yet against frozen generative priors.** Table 21: frozen Wan 84.20, separate LoRA then cached features 84.95, joint LoRA 90.62, full fine-tuning 90.64. A 6.42-point gap between frozen and jointly-adapted. ForeSight freezes Epona entirely and uses it as the primary encoder; DriveLaW caches Video-DiT features. Neither architecture is tested here so it is a prior rather than a refutation, but it is the most direct measurement the wiki has. Also the third LoRA-vs-full-FT data point: full FT adds 0.02, agreeing with DA-WAM against Latent-WAM, and consistent with the existing reconciliation (LoRA is safe when the pretrained representation is close to the target).

**Cross-page effect 4 - the tie problem is named and measured.** >95% of scenes contain candidate groups that are jointly perfect, jointly zero, or tied at the top. That is why rank losses and rank correlations are poor instruments for trajectory scorers, why BoN oracle ceilings saturate, and it retro-explains DA-WAM's refusal to pool futures and DriveSuprim's coarse-to-fine filtering. New section on the selection-based page with the tie-aware diagnostic table. Also recorded there: Wan beats ViT-L by 1.74 with one trajectory but only 0.28 with 64 proposals, so multi-proposal scoring masks representation quality and selection leaderboards are poor encoder comparisons.

**Cross-page effect 5 - adaptive routing gains a second, orthogonal knob.** CLEAR routes candidate count and diversity, decided once before generation from VLM hidden states. Adaptive-WAM routes backbone depth, decided incrementally after each decoded plan from the plan itself. The first reallocates compute, the second reduces it. Comparison table added to the routing page, plus the observation that the two knobs partly substitute (the feature-quality advantage shrinks 1.74 -> 0.28 as proposals grow), so a joint scheduler has a real trade-off to learn. Nobody has built one.

**Gap filled - DriveVA's truncated table.** The DriveVA page has carried medium confidence because its NAVSIM sub-scores were truncated in the source clipping. Adaptive-WAM reproduces the row: NC 99.2, DAC 97.5, TTC 98.7, Comf 100, EP 83.5. TTC 98.7 is now the highest in the wiki (above DriveLaW's 96.7) and NC 99.2 is second only to WA-JEPA's 99.5 - DriveVA is markedly safety-skewed, which was invisible while truncated. It also disambiguates the headline: DriveVA is 90.9 with mixed data and 90.5 on NAVSIM alone, and Adaptive-WAM compares against 90.5. DriveVA page confidence raised medium -> high.

**Protocol note**: the v2 table mixes conventions again - DiffusionDrive 84.5 matches the corrected cohort while ReCogDrive 83.6 and DriveVLA-W0 86.1 match the pre-fix cohort. Fourth ingested table shown to mix, after GeoWAM, DA-WAM, and BrainWAM. 89.9 EPDMS cannot be placed against the wiki's v2 leaderboard.

**CLOVER note**: the first author is CLOVER's first author (arXiv 2605.15120). CLOVER supplies Auto-JEPA's scorer initialisation and this paper's pseudo-expert target protocol, and is now referenced by two ingested papers while remaining un-ingested. It stays on the Known Gaps list with a stronger case than before.

**Pages updated**: `wiki/concepts/world-model-for-ad.md` (new Pattern 28; new "Adaptive-WAM Shows the Axis Was Wrong" subsection replacing the two-paper tally; training-time-only camp roster; "what survives across all nine papers"; computational-cost section; two open questions rewritten), `wiki/concepts/navsim-benchmark.md` (two SOTA rows + three-part caveat), `wiki/concepts/foundation-backbones-for-ad.md` (Wan table row + reframing; new "Readout Depth" section; new "Frozen Is Not Good Enough" adaptation ladder), `wiki/concepts/selection-based-planning.md` (table row + new "The Tie Problem, Finally Named" section), `wiki/concepts/adaptive-routing.md` (new depth-routing section with the CLEAR comparison), `wiki/concepts/nuscenes-waymo-evals.md` (extended zero-shot WAM cluster with horizon breakdown), `wiki/sources/driveva.md` (truncation gap filled, confidence medium -> high), `wiki/index.md`, `README.md` (65 papers).

**Figure note**: three figures embedded (paradigm_comparison.png, main_architecture.png, supp_early_vs_deep.png) and all relevant tables reproduced. Figures 2 and 3 are referenced by caption in the source but have no image files; their data is carried as Appendix G tables on the wiki page.

---

## 2026-09-04 - Ingest: GeoWorldAD: Geometry World Action Model for Autonomous Driving

**Source**: `raw/papers/GeoWorldAD_ Geometry World Action Model for Autonomous Driving.md` (arXiv 2607.17521v2)
**Orgs**: Nanyang Technological University + Xiaomi EV + Zhejiang University. Songyan Zhang et al., Chen Lv senior.
**Page created**: `wiki/sources/geoworldad.md`

**What it is**: the wiki's second geometry world-action model, arriving independently of GeoWAM (Uber AV Labs) from a different group, with the same DVGT-2 ancestor, a near-identical NAVSIM-v2 score, and no mutual citation. Three components: EgoStreamVGGT (a re-parameterised StreamVGGT), a Q-Former geometry world model producing latent future tokens supervised by future depth, and a geometry-oriented action model that refines 64 trajectory proposals over five stages. 91.0 PDMS NAVSIM v1, 90.4 EPDMS v2, camera-only, no map/box/occupancy supervision.

**Method details**:
- Multi-scale geometry tokens from layers {4, 11, 17, 23} of StreamVGGT's 24-block decoder, fed to DPT heads for point map, depth, and camera parameters.
- EgoStreamVGGT: each point map expressed in the ego-camera coordinate system of ITS OWN timestep, camera poses as adjacent-frame relative transforms, instead of StreamVGGT's anchor-frame convention. Loss form unchanged (L_camera Huber + confidence-weighted L1 + gradient matching on depth and point maps); only the target frame changes.
- Geometry world model: Q_fut in R^{K x M x C} with K=4 chunks over 2 s, M=64 tokens per chunk, learnable temporal embedding per chunk. Ego status (velocity, steering, command) MLP-projected and concatenated with geometry tokens. Four aggregation stages, one per selected layer, each cross-attending future tokens to present geometry then applying causal self-attention across chunks. Future geometry tokens produced by CrossAttn(G_t^l, Q_fut^k) and decoded to future depth through the SAME DPT head, with the future-depth loss NOT updating that head.
- Action model: Q_traj in R^{64 x 8 x 1024}. Five refinement stages - four present-geometry (one per layer) plus one future-geometry - each supervised with min-over-proposals L1, exponentially down-weighted for earlier stages. Proposal scoring head trained with BCE against the NAVSIM simulator's own PDMS composition.
- L = L_traj + L_score + L_recon + L_wm. Reconstruction and future-depth decoders are not needed at planning inference.
- Training: 32x H20, global batch 64, AdamW, lr 1e-4 (stages 1-2) / 1e-5 (stage 3), cosine. Stage 1 EgoStreamVGGT 23K steps on OpenScene + nuScenes + ParallelDomain + RealDriveSim (10:10:1:1). Stage 2 world model 47K steps on OpenScene, planner 32K steps on NAVSIM navtrain (= GeoAD). Stage 3 full model +64K steps, future-geometry block zero-initialised so it starts as identity.

**Results**:
- NAVSIM v1: 91.0 PDMS (NC 99.0, DAC 97.8, TTC 95.8, Comf 99.9, EP 85.9). Claim scoped to "best among perception-free methods"; iPad 91.7 is in the same table and beats it but uses map and box supervision.
- NAVSIM v2: 90.4 EPDMS (NC 99.0, DAC 97.8, DDC 99.6, TL 99.7, EP 89.1, TTC 98.6, LK 97.6, HC 98.0, EC 82.2). EP and TTC lead the table; EC 82.2 is strong for a world-model method (DVGT-2 and EponaV2 both ~77, DriveVLA-W0 58.9).
- Table 3: GeoAD (present geometry only) 89.3 PDMS / 87.6 EPDMS -> GeoWorldAD 91.0 / 90.4. Deltas: NC +0.1/+0.1, TTC +0.1/+0.3, EP +3.3/+2.8, aggregate +1.7/+2.8.
- Table 4: Scratch 84.2; StreamVGGT + 4D recon 84.8; EgoStreamVGGT no aux 87.3; EgoStreamVGGT + 4D recon 89.3.
- Table 6: 24 layers / 1 iteration 87.6; 1 layer (final) / 4 iterations 88.2; 4 layers / 4 iterations 89.3.
- Tables 5 and 7 (identical, printed twice) video depth: StreamVGGT -> EgoStreamVGGT AbsRel OpenScene 0.236 -> 0.141, nuScenes 0.265 -> 0.117, KITTI 0.173 -> 0.077; delta<1.25 65.6 -> 86.5, 58.2 -> 88.5, 72.2 -> 95.5.
- Table 8 camera pose: nuScenes ATE 14.79 -> 5.78, RPE trans 1.77 -> 0.63, RPE rot 0.47 -> 1.31 (WORSE); OpenScene ATE 8.66 -> 4.07, RPE trans 1.00 -> 0.39, RPE rot 1.53 -> 0.92.

**Limitations**:
- The +1.7 future-geometry ablation is NOT compute-matched: GeoAD has 32K planner steps, GeoWorldAD has 96K. The zero-init future block means the two are identical at the start of Stage 3, so the delta is attributable to Stage 3 - but Stage 3 varies the mechanism and triples the planner budget together. A GeoAD trained a further 64K steps is the missing row and it is one run.
- "RGB representations are redundant and provide limited geometric guidance" is asserted and never tested. Like GeoWAM, the geometry-beats-pixels thesis is argued only against other papers' methods, never against its own architecture with a pixel future target. Two independent geometry papers, same missing experiment.
- 91.0 is below CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7. iPad 91.7 beats it inside its own table; the perception-free scoping is honest but "state-of-the-art" in the abstract does more work than the table supports.
- The proposal scorer is trained on NAVSIM-simulator PDMS labels - privileged Hydra-MDP-class distillation - and the headline uses 64 proposals, so it is not a single-trajectory result.
- 32x H20 across four training stages on four datasets including two synthetic ones. NO latency, FPS, or parameter count reported anywhere, for a model running a 24-block geometry decoder plus a Q-Former plus five refinement stages.
- nuScenes rotational RPE regresses ~3x (0.47 -> 1.31) under the change most likely to affect it; the prose says "trajectory-level and translational pose errors", excluding rotation by careful wording, and never discusses it.
- Tables 5 and 7 are the same table printed twice; the depth/pose comparison conflates ego-alignment with driving-domain fine-tuning (EgoStreamVGGT is fine-tuned on four datasets, StreamVGGT is off the shelf). Table 4 is the clean instrument.
- No navhard, HUGSIM, Bench2Drive, or nuScenes planning. navhard is the conspicuous gap: GeoWAM's navhard result is its strongest and its navtest/navhard split (+0.6 vs +4.9) is the wiki's only evidence that geometry world modelling pays off more under reactive protocols. GeoWorldAD tests only the protocol where that effect was smallest.
- Single runs, no seed variance, against sub-metric deltas of +0.1.
- Paper's own limitation: the planner operates on fixed-length clips despite a streaming backbone; KV caching for streaming trajectory inference is future work.
- Does not cite GeoWAM.
- The v2 table mixes evaluator conventions by the wiki's partition (Transfuser 76.7 pre-fix, DiffusionDrive 84.5 corrected) - fifth ingested table to do so.

**Cross-page effect 1 - the shared-future rule is now scoped rather than general.** The wiki's standing claim, assembled from DA-WAM (-0.50), SimWAM (~0), and ForeSight (+0.3), was "shared future conditioning is useless to harmful; only per-candidate futures help." Every one of those measurements used a photometric or feature-space target. GeoWorldAD measures the same structural configuration - one shared future for all 64 proposals - with a GEOMETRIC target and gets +1.7 PDMS / +2.8 EPDMS, with EP up 3.3 and safety flat. That is the exact mirror of DA-WAM's shared-future row, where NC and TTC rose while ego progress collapsed 91.36 -> 88.68. Two readings are open and the paper does not separate them: either the target matters (an averaged photometric future carries no candidate-discriminative signal, while a shared geometric future says where free space will be and licenses commitment), or it is the 64K unmatched training steps. The blockquote rule on the world-model page is now qualified to "shared PHOTOMETRIC future conditioning", with a new subsection laying out the four-way comparison. DA-WAM's page carries a matching scoping note.

**Cross-page effect 2 - coordinate frame is a first-class backbone variable.** Table 4 is the cleanest result in the paper and new to the wiki: an off-the-shelf StreamVGGT with 4D reconstruction supervision beats a from-scratch planner by only 0.6 PDMS AND lowers NC, DAC, and TTC, buying nothing but ego progress. Re-expressing the same model's point maps in per-timestep ego frames - pure re-parameterisation, no added capacity - is worth +2.5, with gains on every metric; joint 4D recon supervision adds +2.0 more. This is the first measurement of the argument GeoWAM makes rhetorically (geometry's advantage is sharing a coordinate frame with the action) and it shows the advantage is conditional on doing the alignment rather than automatic from choosing a geometric target. New section on the backbones page, placed beside Adaptive-WAM's frozen-vs-joint-LoRA ladder, which has the same shape.

**Cross-page effect 3 - readout depth confirmed on a second backbone, with a sharper answer.** Adaptive-WAM showed which single layer you read is worth up to 4.80 PDMS on Wan2.2 and that a mid-network layer beats the final one. GeoWorldAD's Table 6 runs the analogous study on StreamVGGT and decomposes it: iterating on the final layer alone buys progress (EP 81.5 -> 82.9, collision metrics flat); adding multi-scale buys safety (DAC 95.5 -> 97.2, NC 98.6 -> 98.9, EP flat); and feeding all 24 layers into one interaction stage is the WORST of the three at 87.6 despite the most information. So the field's default of reading the last layer is wrong (both papers agree), but the fix is several depths consumed progressively, not one well-chosen depth, and naive concatenation fails.

**Cross-page effect 4 - the GeoWAM protocol warning is narrowed from a table to two rows.** GeoWorldAD's v2 table reports DVGT-2 at 89.6 and EponaV2 at 88.9 - digit-for-digit identical to GeoWAM - while reporting Transfuser 76.7 and DiffusionDrive 84.5, the values GeoWAM gives as 84.0 and 88.2. A second independent paper reproducing GeoWAM's headline anchors alongside the standard baseline values makes "two anomalous rows" a far more economical explanation than "a third aggregation protocol". Tally on Transfuser is now four papers to one. Practical consequence: GeoWAM's 90.2 and GeoWorldAD's 90.4 are comparable to each other, both anchored on DVGT-2 89.6, so GeoWAM's honest +0.6 attribution and GeoWorldAD's +0.8 sit on the same scale. This does NOT rescue the global corrected/pre-fix partition - GeoWorldAD's own table mixes - and the general rule stands. New section on the benchmark page; the warning banner and headline criticism on the GeoWAM page rewritten.

**Cross-page effect 5 - GeoWAM's missing ablations, two of three supplied.** The GeoWAM page's headline criticism has been "There are no ablations. Not one." GeoWorldAD supplies the with/without-future comparison and the coordinate-frame comparison. Neither paper supplies the third and most important: no version of either architecture trained with a pixel or video future target under an otherwise fixed planner. Recorded on both pages.

**New methods for the gap list**: DVGT-2 (arXiv 2604.00813) is now cited as a baseline by two ingested papers and is the direct ancestor of both geometry WAMs - the strongest un-ingested candidate in the wiki. Also EponaV2 (2605.14696, 90.4 v1 / 88.9 v2), iPad (2505.15111, 91.7 v1), WorldDrive (2603.14948, 89.0), LFG (CVPR'26, 85.2), StreamVGGT (2507.11539).

**Baseline-hygiene note (positive)**: GeoWorldAD's v1 table matches this wiki's canonical values throughout and cites DriveVLA-W0 at its 90.2 headline rather than the 87.2 reimplementation row that DriveLaW, BrainWAM, and Adaptive-WAM all propagate. DriveSuprim appears at 89.9, the widely-circulated non-ViT-L figure.

**Pages updated**: `wiki/concepts/world-model-for-ad.md` (new Pattern 29; new "GeoWorldAD Reopens the Shared-Future Question" subsection; blockquote rule scoped to photometric futures; imagine-then-act roster), `wiki/concepts/navsim-benchmark.md` (SOTA row + caveat; new "GeoWorldAD Narrows the GeoWAM Anomaly" section; Transfuser tally three -> four), `wiki/concepts/foundation-backbones-for-ad.md` (GeoWorldAD's aggregation study added to the readout-depth section; new "Coordinate Frame Beats the Foundation Model" section), `wiki/concepts/selection-based-planning.md` (table row), `wiki/sources/geowam.md` (warning banner rewritten; new "The Sibling Paper" section), `wiki/sources/da-wam.md` (shared-future verdict scoped), `wiki/sources/adaptive-wam.md` (depth confirmed on a second backbone), `wiki/index.md`, `README.md` (66 papers).

**Figure note**: all nine figures embedded (teaserv3_4hist.png, frameworkv8_4hist.png, supp_planning_geo.png, recon_vis_0-3.png, wm1.png, wm2.png) and all eight distinct tables reproduced (Tables 5 and 7 are duplicates in the source).

---

## 2026-09-04 - Ingest: WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving

**Source**: `raw/papers/WCog-VLA_ A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving.md` (arXiv 2607.08375v1)
**Orgs**: Tongji University + Nanyang Technological University. Xuerun Yan, Zhexi Lian, Nuoheng Zhang, Shiyu Fang, Haoran Wang, Chen Lv, Jia Hu, Binyang Song. (Chen Lv is also senior author on GeoWorldAD.)
**Page created**: `wiki/sources/wcog-vla.md`

**What it is**: a 2B VLA whose world model forecasts **the future trajectories of surrounding agents, jointly with the ego's** - a target no other paper in the wiki predicts. Two levels: semantic (agent tokens from BEVFormer + TrackFormer injected into InternVL3-2B, with a world head decoding current 3D boxes and future agent trajectories) and generative (ADDT, a decoupled diffusion transformer synthesising the joint multi-agent rollout). Plus Game-CoT, an 85k Stackelberg-game reasoning dataset. 92.9 PDMS NAVSIM v1, 85.9 EPDMS v2.

**Architecture**:
- VLM: InternVL3-2B (300M InternViT + Qwen2.5). Inputs: 6 surround views, navigation instruction, ego state (v, a, 2 s history at 2 Hz).
- 3D perception: off-the-shelf BEVFormer BEV encoder + UniAD-style TrackFormer producing N_a sparse agent tokens.
- Role-decoupled hidden states: LLM([T_vision, T_text, T_agent]) -> O_agent routed to a world head (current 3D boxes + future agent trajectories); O_vision and O_text to the language head for Game-CoT text.
- ADDT: 16-block DiT, 8-block condition encoder + 8-block generation decoder. Encoder input F_at = concat(E_act(x_t), E_his(tau_his), mean-pooled F_VLM); t and ego state S via AdaLN; full F_VLM via cross-attention; outputs self-condition feature z_t. Decoder takes t and z_t via AdaLN plus F_VLM cross-attention. Joint action noise x_t in R^{N_m x H x 3}. Agent-specific loss mask W with separate alpha_ego and alpha_surr.
- Representation alignment: cosine loss between the 6th encoder block feature and a latent from a GenAD-style VAE (MLP encoder, GRU decoder) pretrained to reconstruct multi-agent trajectories. Stated purpose is stability of z_t across denoising timesteps, not fidelity.
- Game-CoT: Qwen3-VL-Plus pipeline, four steps (scene description, critical object analysis, game-theoretic reasoning as a Stackelberg game with ego as leader, payoff evaluation). GT actions supplied as hints.
- Training: 4 stages on 4x A100 40GB. S1 3D perception 1 epoch. S2 VLM 1 epoch on 158k open VQA then 3 epochs joint with world heads on 170K NAVSIM-tailored (85k trajectory VQA + 85k Game-CoT). S3 ADDT 200 epochs DDPM, VLM frozen. S4 DiffGRPO 10 epochs, group size 6, reward r = r_PDMS - lambda_surr * L1(tau_surr).

**Results**:
- NAVSIM v1 (after all four stages): 92.9 PDMS (NC 99.4, DAC 98.8, TTC 98.5, Comf 100, EP 87.1). Fourth-highest in the wiki behind CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3; ahead of HybridDriveVLA 92.1. NC and TTC are both second-highest in the wiki (behind WA-JEPA 99.5 and DriveVA 98.7); DAC trails only DriveVLA-W0's 99.1. Best PDMS-per-parameter entry tracked here at 2B.
- NAVSIM v2 (after three-stage SFT only, NO RFT): 85.9 EPDMS (NC 98.8, DAC 96.6, DDC 99.3, TLC 99.8, EP 85.8, TTC 98.2, LK 96.4, HC 98.3, EC 86.3).
- Table 3 four stages: S2 only 84.4; +S1 85.5; +S3 89.3; +S4 92.9. RFT +3.6, ADDT +3.8, 3D perception pretraining +1.1.
- Table 4 dual-level cognition (all three-stage SFT): neither 86.5; +Cur 87.0; +Fut 87.2; +both semantic 88.1; +generative only 87.4; all 89.3.
- Table 5 ADDT: VLM text no reasoning 85.0 @ 1.131 s; VLM text with Game-CoT 85.5 @ 9.896 s; SDT-5 87.4 @ 0.105; SDT-20 88.5 @ 0.388; DDT-5 87.9; DDT-20 88.7; ADT-5 88.6; ADT-20 89.1; ADDT-5 89.3 @ 0.106; ADDT-20 89.6 @ 0.383.
- Table 6 VQA data: Traj only 86.7; +Drive 88.2; +CoT 87.5; all 89.3.
- Table 7 3D perception: without 86.0, with 89.3.

**Limitations**:
- The AutoVLA-3B baseline is listed at 92.1, which is AutoVLA's ORACLE Best-of-6 score (92.12). Its single-sample post-RFT score is 89.11. Unmarked, in a table of single-sample results, and the paper's "surpasses ReCogDrive and AutoVLA by at least 0.8 PDMS" claim is measured against it. Real margin against a deployable AutoVLA is 3.8.
- v1 table omits CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, WA-JEPA 91.8, SimWAM 91.5, FLARE 91.4, DiffusionDriveV2 91.2.
- v1 and v2 report DIFFERENT models: 92.9 includes RFT, 85.9 does not, and RFT is worth +3.6. Stated in a caption, never flagged as a caveat.
- The v2 baselines are a fourth distinct set. TransFuser 77.8 with submetrics matching neither the wiki's 76.7 row nor GeoWAM's 84.0 - and its NC/DAC/EP/TTC (97.7 / 92.8 / 79.2 / 92.8) are identical to its own NAVSIM-v1 row in Table 1, with LK 67.6 against 92.7 everywhere else. HydraMDP++ 80.6 (third value after 84.1 and 81.4), DiffusionDrive 84.3 (against 84.5), ARTEMIS HC 100 (against 98.3 elsewhere). Its strongest v2 baseline is DiffusionDrive 84.3, so the v2 "SOTA" claim is scoped to a weak field.
- RFT is the largest single contribution and its reward is the benchmark's own PDMS - benchmark distillation of the Hydra-MDP class.
- Heavily supervised: 3D boxes, per-agent future trajectories, BEV supervision, 328k VQA/CoT samples. Not comparable to annotation-free lines.
- Game-CoT annotations use GT actions as hints, so the traces are post-hoc rationalisations of a known answer rather than independent reasoning. The paper is candid about it.
- Latency scope is undefined. ADDT at 0.106 s cannot include an InternVL3-2B forward over six surround views, and certainly excludes Game-CoT generation (9.896 s in the same table). End-to-end deployed latency is unreported and the 10.7x speedup claim compares an action head against a full text-generation pipeline.
- Table 3 (+1.1) and Table 7 (+3.3) disagree on the value of 3D perception, measured at different pipeline points, unreconciled.
- No navhard, Bench2Drive, HUGSIM, nuScenes. Single runs, no seed variance. No ADDT parameter count. No code release mentioned. Figures 6-7 referenced three times in text but absent from the source clipping.
- Paper's own limitation: semantic cognition covers agents only, omitting road geometry and map topology evolution.

**Cross-page effect 1 - a genuinely new world-model target.** Pattern 30 on the world-model page. Every prior pattern forecasts the SCENE (pixels, video latents, features, occupancy, metric point maps, symbolic state) or the EGO's own action latent. WCog-VLA forecasts what the other agents will do. The argument for it is a taxonomy gap rather than a fidelity gap: a scene forecast conditioned on one ego plan cannot express "if I go, they yield", whereas a joint multi-agent rollout can. Its Table 4 row 5 measures generative joint multi-agent synthesis alone, with no semantic world supervision, at +0.9 PDMS.

**Cross-page effect 2 - the shared-future question gains a second non-photometric positive.** The synthesis table now reads: DA-WAM (JEPA scene latents) -0.50, SimWAM (video tokens) ~0.0, ForeSight (generated frames) +0.3, GeoWorldAD (future depth latents) +1.7, WCog-VLA (joint multi-agent trajectories) +0.9. The two positives share a property the four negatives lack: **the forecast is of something the planner cannot read off the current frame**, whereas an averaged future image or scene latent is largely redundant with the observation it was conditioned on. WCog-VLA's is also the better-controlled of the two positives - same three-stage SFT budget across all six rows of its Table 4, against GeoWorldAD's 32K-vs-96K confound - though it is smaller and its ADDT also adds a continuous action head the row-1 baseline lacks.

**Cross-page effect 3 - the price of inference-time textual CoT, measured on one model.** VLM text with Game-CoT reasoning: 85.5 PDMS at 9.896 s. VLM text without: 85.0 at 1.131 s. So reasoning costs 8.8 seconds and buys 0.5 points. Meanwhile the 5-step ADDT head, conditioned on the same VLM hidden states, scores 89.3 at 0.106 s. And the deployed system never generates the reasoning at all - Game-CoT is retained as training data (worth +0.8 alone, +1.1 on top of open driving VQA) while the CoT path is discarded. That is a distinct position for the CoT page: the existing entries ask WHEN to reason (AdaThinkDrive, SpanVLA, DeepSight) or WHETHER reasoning is needed (NoRD); WCog-VLA answers keep the corpus, drop the computation. Two caveats recorded: the 0.5 measurement is on this model's possibly-weak text trajectory head, and no run controls for total token budget without Game-CoT.

**Cross-page effect 4 - architecture substitutes for denoising budget, measured.** Table 5 varies decoupling and alignment at 5 and 20 steps on an identical DiT backbone. The two mechanisms are worth +1.9 PDMS at 5 steps but only +1.1 at 20; conversely the value of going 5 -> 20 steps shrinks from +1.1 (plain SDT) to +0.3 (full ADDT). Added to the diffusion-planner page as the clearest statement in the wiki of why few-step planners can match many-step ones, and connected to the video-side finding (Adaptive-WAM, BrainWAM, DriveLaW) that iterating a denoiser buys nothing - WCog-VLA finds the same for the ACTION denoiser and supplies a reason.

**Cross-page effect 5 - BoN contamination is now a documented failure mode.** New section on the best-of-n page. This is the wiki's first recorded case of an oracle Best-of-N score appearing unmarked in another paper's single-sample comparison table. BoN scores sit 3-7 points above their single-sample counterparts across the wiki (AutoVLA +3.0, DriveVLA-W0 +4.6, ExploreVLA +3.3, Curious-VLA +4.5, NoRD +6.8), so once one paper carries a BoN figure into a single-sample table, downstream copying propagates it - the same mechanism already documented for evaluator drift. Noted on the AutoVLA page too.

**Cross-page effect 6 - a fifth failure mode for the evaluator-drift analysis: cross-version submetric contamination.** WCog-VLA's v2 TransFuser row carries NC/DAC/EP/TTC identical to that method's NAVSIM-v1 row in the same paper's Table 1, with v2-only columns appended and LK at 67.6 against 92.7 everywhere else. That is not an aggregation difference; it is a different evaluation assembled from two sources. The benchmark page's failure-mode list now runs to five.

**Cross-page effect 7 - perception ablation asymmetry.** Removing 3D perception costs 3.3 PDMS with the full ADDT stack (89.3 -> 86.0) but only 1.1 when the planner still emits text tokens (84.4 -> 85.5). Recorded on the perception page as evidence that **explicit 3D structure is worth roughly three times more once there is a continuous action head able to exploit it** - a VLM emitting waypoints as text may simply be unable to use precise spatial input.

**Cross-page effect 8 - the closest comparison in the wiki.** SGDrive and WCog-VLA are both InternVL3-2B, both need 3D supervision at training, both camera-only at inference, both use a ReCogDrive-derived RL recipe. SGDrive 87.4 SFT / 91.1 RFT; WCog-VLA 89.3 / 92.9. The +1.9 and +1.8 gaps are unusually interpretable, and the two candidate explanations are agent-behaviour forecasting versus scene-structure encoding, and joint multi-agent versus ego-only generation. Neither paper cites the other and no controlled experiment separates them; comparison table added to the SGDrive page.

**New methods for the gap list**: LatentVLA-3B (92.4 PDMS v1) and BevDrive (83.8). iPad 91.7 is corroborated against GeoWorldAD's value.

**Pages updated**: `wiki/concepts/world-model-for-ad.md` (new Pattern 30; shared-future table and analysis extended; imagine-then-act roster), `wiki/concepts/navsim-benchmark.md` (SOTA row + caveat; fifth failure mode; ARTEMIS EC tally now 2-vs-1 for 98.3), `wiki/concepts/chain-of-thought-for-ad.md` (new Game-CoT section with the latency table), `wiki/concepts/best-of-n.md` (new BoN-leakage section), `wiki/concepts/diffusion-planner.md` (new ADDT decoupling/alignment section), `wiki/concepts/perception-for-planning.md` (new item 9), `wiki/sources/sgdrive.md` (head-to-head comparison), `wiki/sources/autovla.md` (citation note), `wiki/index.md`, `README.md` (67 papers; leaders line updated; new commensurability bullet).

**Figure note**: all five figures present in the source are embedded (ECCV_intro.png, ECCV_Frame.png, ADDT.png, train_stage.png, compare_with_previous.png) and all seven tables reproduced. The text refers to "Fig. 7" three times but Figures 6 and 7 are absent from the clipping; noted on the page.

## 2026-09-04 - Ingest: LWDrive: Layer-Wise World-Model-Guided Vision-Language Model Planning for Autonomous Driving

**Source**: `raw/papers/LWDrive_ Layer-Wise World-Model-Guided Vision-Language ModelPlanning for Autonomous Driving.md` (arXiv 2606.29879v2)
**Org**: Chongqing University. Chen Yang, Yuhao Wei, Ze Xu, Ziheng Zou, Shuang Liang, Delin Ouyang, Lingfeng Qi, Jie Li, Guofa Li.
**Page created**: `wiki/sources/lwdrive.md`
**Confidence**: high - source markdown contains full method text, all four tables, all four figures, and implementation details.

**What it is**: a VLM planner that treats its own trajectory output as a coarse plan to be refined. Qwen2.5-VL-3B emits an intent-aware coarse trajectory; a Foresight Cascade Planner (FCP) initialises a proposal pool from the pooled action-query latent and runs six refinement stages, each reading hidden states from a different Qwen layer through Bridge Attention and applying BEV-grounded residual updates; a score head distilled from NAVSIM's PDMS composition selects. 92.0 PDMS NAVSIM-v1, 89.6 EPDMS v2, 0.37 m L2 nuScenes ST-P3. **No reinforcement learning of any kind.**

**Architecture**:
- VLM branch: Qwen2.5-VL-3B, front view only at 280x504, 36 action layers. Inputs: image, ego status, navigation instruction, learnable action queries. Adapted on Impromptu before planning training.
- World head (training only): frozen VAE encodes future frame I_{t+delta} to z_{t+delta}; denoiser D_theta predicts the clean latent from a noisy one, conditioned on c_t = [F_t^V ; h_t^A] (final-layer vision + action-query hidden states). L_wm = d(zhat, z). Stage-1 loss weights lambda_wm = 15.0 vs lambda_traj = 1.0.
- FCP: pool init Q0,P0 = TPM(z_AQ, e_t, E_init) with N_p proposals (N_p never stated). Refinement at layers l_r = rM; L=36, R=6 so M=6 and taps are {6,12,18,24,30,36}. Per stage C_r = (H_V^{l_r}, H_A^{l_r}, B_t, s_t). Bridge Attention = single attention over concatenated proposal-self / action-query+ego / VLM-foresight memories. Then BEV cross-attention (ResNet-34 over four camera views) and residual update Y_i^r = Y_i^{r-1} + MLP_ref(Q_i^r).
- Score head: MLP over temporally pooled Q_i^R; BCE against Shat_i = NC*DAC*(5EP+5TTC+2C)/12 obtained by non-reactive log simulation of every candidate.
- Refinement supervision: min-over-N SmoothL1 at every stage, L_ref = sum_r beta^{R-r} min_i l_i^r. beta unreported.
- Training: Stage 1 = VLM + world head + trajectory decoder, 8,000 steps, lr 2e-4, batch 48, ~100 h. Stage 2 = Qwen and action-query embeddings FROZEN, FCP + score head trained, 12 epochs, lr 3e-5, batch 64, ~80 h. GPU count never given.

**Results**:
- NAVSIM v1: 92.0 PDMS (NC 98.8, DAC 98.4, TTC 96.2, C 99.8, EP 87.3). Eighth-highest non-BoN entry in the wiki. EP 87.3 ties ReCogDrive and sits 0.2 below the human driver.
- NAVSIM v2: 89.6 EPDMS (NC 98.8, DAC 98.4, DDC 99.0, TLC 99.7, EP 90.3, TTC 98.6, LK 96.3, HC 97.9, EC 73.3). **EP 90.3 is the highest v2 ego progress in the wiki and exceeds the Human Agent's 87.4 by 2.9.** TTC 98.6 ties GeoWorldAD. EC 73.3 is the third-worst in its own table and is not discussed. 89.6 is 0.7 below the Human Agent's 90.3.
- nuScenes ST-P3: 0.37 m avg L2, 0.16% collision. Best L2 in its table by 0.03 m; RDA-Driver has a better collision rate at 0.10%.
- Table 4 ablation (WM / LW / BA / BR): 84.5 (none) -> 86.3 (+WM) -> 89.3 (no BR) -> 90.0 (no BA) -> 91.8 (final-layer only) -> 92.0 (full).

**Limitations**:
- **The central claim is untested.** The WM column is only ever varied with NO FCP present (84.5 -> 86.3). There is no full-FCP-without-WM row, so "the cascade consumes foresight features rather than merely deep features" is asserted, never measured. Of the +7.5 total: FCP stack +5.5, WM +1.8, layer-wise +0.2.
- **The layer-wise mechanism in the title is worth +0.2 PDMS** and is a trade, not a gain: final-layer-only has HIGHER ego progress (88.1 vs 87.3); layer-wise buys DAC +0.4 and TTC +0.5.
- PDM-Closed row in Table 1 has TTC/C/EP cyclically permuted (89.9/86.9/99.9 against canonical 86.9/99.9/89.9). EP 99.9 would exceed the human driver's 87.5.
- DriveSuprim appears in the v2 table but NOT the v1 table. The whole current v1 frontier is absent (CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, WCog-VLA 92.9, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7, SimWAM 91.5). The results text scopes the claim correctly ("best among the compared methods").
- v2 table mixes evaluator conventions (sixth ingested table to do so): pre-fix TransFuser 76.7 / ARTEMIS 83.1 / DriveVLA-W0 86.1 alongside corrected DiffusionDrive 84.5 / DriveWorld-VLA 86.8, plus DriveSuprim 83.1 and HydraMDP++ 81.4 matching neither cohort.
- Score head trained on NAVSIM's own PDMS - privileged benchmark distillation; 92.0 is a selection result, not a single trajectory. N_p never stated.
- No latency, FPS, parameter count, or GPU count anywhere. Training hours quoted without device counts.
- Stage-2 freezing never ablated; the FCP consumes cached foresight features, which is the losing rung of Adaptive-WAM's adaptation ladder on a video backbone (though a different backbone family).
- nuScenes protocol unstated - zero-shot vs separately trained is unrecoverable from the text; ego status is consumed, so the 0.03 m margin carries no planning signal.
- Unreported: N_p, beta, lambda_ref, lambda_score, Delta (future-frame offset), world-head denoising steps. lambda_wm = 15 is reported but never ablated.
- No navhard, Bench2Drive, HUGSIM, or reactive evaluation. Single runs, no seed variance.
- Four cameras for BEV against one 280x504 front view for the VLM; the mismatch is unexplained and untested.
- Text/table cross-references wrong throughout (v1 results paragraph says "Table 2"; every footnote says "Cited by: Table 2").
- Code "will be made publicly available" - not released.

**Cross-page effect 1 - the readout-depth axis gets its third backbone family, and it nearly vanishes.** New subsection on the foundation-backbones page. LWDrive's rows 5-6 are the closest available analogue to GeoWorldAD's multi-scale-vs-final comparison: six refinement stages either way, only the tap depth changing. The sub-metric DIRECTION reproduces GeoWorldAD's exactly (multi-scale buys DAC and TTC, costs EP) - two independent confirmations that multi-scale readout is a safety mechanism. The MAGNITUDE does not: +4.80 on a Wan2.2 video DiT (Adaptive-WAM), +1.1 on a StreamVGGT geometry decoder (GeoWorldAD), **+0.2 on a Qwen2.5-VL** (LWDrive). Hypothesis recorded as a hypothesis: intermediate readout is worth most where the final layer is most committed to a non-planning output (noise > point maps > language, and this VLM was already tuned to emit trajectory-style responses). Practical conclusion for the wiki: **Adaptive-WAM's +4.80 must not be generalised to VLM backbones** - designs reading a VLM's final hidden state are leaving ~0.2 PDMS on the table, not 5.

**Cross-page effect 2 - nothing above 92.0 PDMS scores its own candidates.** New section on the selection-based-planning page, and a new README bullet. With LWDrive ingested this is a complete statement rather than a tendency: CLEAR 93.7, DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, WCog-VLA 92.9, Adaptive-WAM (aux) 92.6, HybridDriveVLA 92.1, LWDrive 92.0 - **every one trains against the benchmark's own scoring function**, seven as a ranking head and one (WCog-VLA) as an RL reward. The highest entry that does not is WA-JEPA at 91.8. Not an accusation - simulator distillation is declared in all eight - but it bounds what the leaderboard measures: the last ~2 PDMS on NAVSIM-v1 was bought by learning the evaluator, and the ordering among these eight would probably not survive a withheld scoring function.

**Cross-page effect 3 - sixth mixing table, and the copying becomes visible.** LWDrive's v2 table shares four rows with GeoWorldAD's - TransFuser 76.7, DriveSuprim 83.1, DiffusionDrive 84.5, DriveVLA-W0 86.1 - **digit-identical across all nine submetrics in every case**, plus ARTEMIS 83.1 and HydraMDP++ 81.4 from the pool BrainWAM drew on. Two unrelated groups, no mutual citation, identical baseline block. Narrow upside: within that shared block **LWDrive's 89.6 and GeoWorldAD's 90.4 are comparable with each other** (GeoWorldAD +0.8), the same local resolution the page reached for GeoWAM/GeoWorldAD. The block itself still pairs a pre-fix TransFuser with a corrected DiffusionDrive.

**Cross-page effect 4 - a paper finally states why recomputation is impossible.** LWDrive's results section: *"PDMS/EPDMS follow official evaluator: scores are scenario-averaged, while sub-metrics are independently averaged and cannot reconstruct aggregate scores."* First ingested paper to say this explicitly, and it is the justification for the whole evaluator-drift method on the NAVSIM page - provenance can be audited from submetrics, arithmetic cannot.

**Cross-page effect 5 - ARTEMIS EC tally flips decisively.** LWDrive is the third paper to print ARTEMIS EC = 98.3, a duplicate of that method's HC, after DA-WAM and WCog-VLA. BrainWAM remains the only source for 89.1; four papers abstain with "-". Tally now **three to one for 98.3**, which reads as a copied transcription error propagating rather than a real value. The v2 table row on the NAVSIM page was rewritten accordingly.

**Cross-page effect 6 - the constricted-interface rule gets its boundary.** New section on the dual-system page. LWDrive's Figure 1 names the pathology this page circles: (a) direct decoding, (b) inject-once fusion, (c) repeated injection. Structurally System A is Qwen2.5-VL-3B and System B is a six-stage BEV proposal refiner, and the interface is not narrowed but REPEATED - full hidden states read six times. That works (Bridge Attention +2.0, BEV grounding +2.7). Read against BrainWAM's negative result, the two are compatible and jointly sharpen the rule: **what needs constricting is coupling to a GENERATIVE branch, not VLM coupling as such.** BrainWAM's harm came from mixing a clean semantic stream with a denoising one; LWDrive has no denoising stream at inference at all. The "inject once is wrong" complaint is supported by its ablation; the "inject at many depths" prescription is not (+0.2).

**Cross-page effect 7 - a new world-model pattern and a new training-time-only entry.** Pattern 31 on the world-model page. It is the training-time-only counterpart to Pattern 29 (GeoWorldAD): same interleaving of planner refinement with backbone depth, on a VLM, with the future never instantiated. First entry in the training-time-only camp whose future target is supervised **directly on VLM hidden states** rather than on a video backbone's activations (SimWAM, Adaptive-WAM) or a separate feature predictor (FLARE, OneVL). Its 15:1 world-model loss weighting is the most lopsided recorded on that page and is never ablated.

**Cross-page effect 8 - nuScenes row recorded but not placed.** New section on the nuScenes/Waymo page. 0.37 m / 0.16% leads its table by 3 cm, with ego status consumed and Ego-MLP (0.78) and BEV-Planner (0.55) in the same table. The paper never states whether the result is zero-shot from the NAVSIM model or from a separate nuScenes run, and the implementation section describes only the two NAVSIM stages - so the row is deliberately **not** added to the zero-shot WAM cluster table.

**New methods for the gap list**: DriveX-S (ICCV 2025, 84.5), PRIX (2507.17596, 87.8 v1 / 84.2 v2), World4Drive (2507.00603, 85.1), InsightDrive (2503.13047), DiffRefiner (AAAI 2026), Doe-1 (2412.09627), RDA-Driver (2408.13890), DME-Driver (2401.03641). iPad 91.7, DriveWorld-VLA 91.3 / 86.8, and ReCogDrive 90.8 corroborated against existing wiki values.

**Pages updated**: `wiki/concepts/navsim-benchmark.md` (v1 + v2 SOTA rows, two caveat paragraphs, ARTEMIS EC tally rewritten, sixth-mixing-table subsection with the submetric-reconstruction quote), `wiki/concepts/foundation-backbones-for-ad.md` (new backbone-role row; new "third backbone family" subsection under Readout Depth), `wiki/concepts/world-model-for-ad.md` (new Pattern 31; training-time-only roster), `wiki/concepts/selection-based-planning.md` (methods-table row; new "nothing above 92 scores its own candidates" section), `wiki/concepts/dual-system-vla.md` (new LWDrive section + four-way interface table), `wiki/concepts/nuscenes-waymo-evals.md` (new section, row not placed in the zero-shot table), `wiki/index.md`, `README.md` (68 papers; leaders line; six mixing tables; new scorer bullet).

**Figure note**: all four figures in the source are embedded (figure1_pver2.png, 113.png, figure17fix.png, visualization_two_scenarios_wide.png) and all four tables are reproduced in full.

## 2026-09-04 - Ingest: ReWorld: Representation Learning for World Action Models

**Source**: `raw/papers/ReWorld_ Representation Learning for World Action Models.md` (arXiv 2606.27504v2)
**Orgs**: Huazhong University of Science and Technology + Xiaomi EV. Tianze Xia, Lijun Zhou (equal contribution; Lijun Zhou project lead), Kaixin Xiong, Jingfeng Yao, Zhenxin Zhu, Haiyang Sun, Bing Wang, Guang Chen, Wenyu Liu, Hangjun Ye, Xinggang Wang (corresponding).
**Code**: https://github.com/xiaomi-research/ReWorld
**Page created**: `wiki/sources/reworld.md`
**Confidence**: high - source markdown contains full method text, all five main tables plus both two-part ablation tables, all five figures, and complete training configuration.

**What it is**: the **direct sequel to DriveLaW from the same group** - same first author, same project lead, same 2B LTX-Video DiT + 133M Action DiT, same chained latent interface. DriveLaW established that a planner can read a video generator's mid-denoising states; ReWorld observes those states are supervised only through the final generation output and names the gap the **representation bottleneck of WAMs**. Three teacher-free objectives at 1.003x per-step training cost. FVD 81.3 -> 61.9, FID 4.6 -> 4.4, PDMS 89.1 -> 90.4 (no RL, no inference-time scorer), UCF-101 frozen linear probe 68.3 -> 80.2.

**The three objectives**:
- **Stage 1, L_Mid**: lightweight head on Video DiT block 8 of 28 predicts the SAME rectified-flow velocity target (eps_z - z_0) as the final head. L_Video = L_Gen + lambda_Mid * L_Mid. Free because both target and activations already exist. ~2x convergence speedup from scratch. Enables inference-time **self-guidance** v_w = v_i + gamma(v_f - v_i), gamma = 1.4 - structurally the CFG form, adapted from a cited CVPR'26 precedent ("Guiding a diffusion transformer with the internal dynamics of itself"). Explicitly stated not to change the planner-facing latent interface.
- **Stage 2, L_align**: cosine alignment of post-cross-attention Action DiT state a_i^(k) to its **stop-gradient** attended readout r_i^(k) = sum_j alpha_ij v_j, at layer K={12}, lambda=0.05. Video DiT frozen so the target space is stable. Self-distillation of an attention operation; no teacher, no encoder, no cached target.
- **Stage 3, L_RDE**: hard negatives mined offline following BeyondDrive - 64 CFG-diversified candidates per training scene scored by the **NAVSIM PDM simulator**, keep those below delta=0.6, select the one NEAREST the expert. Clean trajectory recovered from the same forward pass and same t_a as L_FM (ahat_0 = a_ta - t_a * v). Delta representation [dx, dy, sin psi, cos psi]. Negative L1 repulsion, lambda=0.04. Video DiT unfrozen, F recomputed without detachment.

**Training**: Stage 1 20k steps batch 64 lr 1e-5 from LTX-Video. Stage 2 6k steps batch 128 from "the DriveLaW checkpoint", Video frozen. Stage 3 10k steps batch 160, both branches joint. Inference: 30 video sampling steps at gamma=1.4, five action flow-matching steps. Video data nuPlan+nuScenes at 8 Hz; trajectories NAVSIM navtrain at 2 Hz.

**Results**:
- nuScenes generation: FID 4.4 / FVD 61.9. **First entry in the wiki to lead both metrics at once** (previously DriveLaW FID 4.6, UniUGP FVD 75.9, no single leader).
- NAVSIM v1: 90.4 PDMS (NC 99.1, DAC 98.2, TTC 97.7, Comf 99.8, EP 82.0). DAC +1.1 and TTC +1.0 carry the +1.3; EP only +0.7 and remains 5+ points below the frontier and below the human driver's 87.5. Comfort regresses 100 -> 99.8.
- UCF-101 frozen linear probe: LTX-Video 66.8, DriveLaW 68.3, ReWorld Stage 1 71.7, ReWorld full 80.2.
- Unified from-scratch protocol (120k steps, 224x224x25, no text): ReWorld 270.4 < Self-Flow 283.3 < SRA2 295.2 < REPA-DINOv2 295.9 < SRA 296.9 < Vanilla Flow 304.1 < REPA-DepthAnything3 319.4 < REPA-VideoMAEv2 328.3 < REPA-V-JEPA2 331.6 < ReDi 421.7.
- Training cost: ReWorld 1.003x vs Self-Flow/SRA ~1.4x, ReDi ~1.6x, REPA ~1.7x.
- Ablations: block sweep 2/8/12/16/20 -> FVD 65.5/61.9/62.7/63.0/64.3. gamma sweep 1.0/1.2/1.4/1.6/1.8 -> FVD 78.9/72.0/61.9/69.7/68.2. Table VII: DriveLaW 89.1, +align 89.5, +RDE 89.8, both 90.4. lambda_align 0.01/0.03/0.05/0.07/0.10 -> 88.8/89.2/89.5/88.2/87.7. lambda_RDE 0.02/0.03/0.04/0.05/0.10 -> 89.4/89.6/90.4/89.7/85.5.

**Limitations**:
- **The title claim is unmeasured.** Table VII has no Stage-1 row; the whole +1.3 PDMS is attributed to two action-side objectives, one of which (RDE) compares two trajectories in delta space and touches no world information at all.
- **Stage 2's initialization contradicts the curriculum description.** Sec III-F describes Stage 2 as following Stage 1; Sec IV-A says Stage 2 "initializes from the DriveLaW checkpoint". Read literally, the planning result never sees the improved video model and the "progressive curriculum" is two independent fine-tunes.
- **88% of the FVD headline is a sampling trick**: representation supervision alone gives 78.9, self-guidance takes it to 61.9. The abstract leads with 61.9 throughout. FID at gamma=1.0 is not reported, so sampling and representation effects on frame fidelity cannot be separated. gamma is sharply non-monotonic and tuned on the reported set; whether DriveLaW's 81.3 baseline is sampling-matched is unstated.
- **RDE is simulator distillation.** "Without reinforcement learning or test-time scoring" is literally true, but 64 candidates per training scene are scored by the NAVSIM PDM simulator to construct the training signal - the same privileged-supervision class as DriveSuprim, Drive-JEPA, DA-WAM, GeoWorldAD, LWDrive, differing only in that the simulator trains a repulsion target rather than a scorer. It is also the larger of the two planning contributions (+0.7 of +1.3).
- **Both objectives are sharply peaked and go below baseline when mis-weighted.** lambda_align at 0.01 (88.8) and 0.10 (87.7) are both under the 89.1 baseline; lambda_RDE at 0.10 costs 4.9 PDMS. The 90.4 headline is the peak of a five-point sweep with 89.6 and 89.7 as neighbours.
- **The +8.5 UCF-101 jump from Stage 3 is unexplained.** Video DiT is frozen in Stage 2, so the entire Stage-1 -> full gap comes from planning-oriented hard-negative gradients. No control, no hypothesis, and the direction it implies (action -> world) reverses the paper's thesis.
- No NAVSIM-v2, no navhard, no Bench2Drive, no HUGSIM, no nuScenes planning. Single runs, no seed variance against 0.4-point deltas.
- No latency, FPS, or inference-cost numbers, though training cost is reported meticulously.
- EP 82.0 is the weak metric; the safety-shaped objective plausibly causes it and this is not discussed.
- Table IV's protocol is 224x224x25 without text, far from the deployed 1280x704 setting, and each teacher gets one untuned configuration.
- Neither lambda_Mid nor |S| > 1 is ablated - Table VI(a) varies which single block, never how many or how strongly.

**Cross-page effect 1 - generation/planning decoupling becomes intra-paper.** New section on the world-model page (`#generation-planning-decoupling`) and a new README bullet. The wiki had assembled this across papers with heavy confounds: DriveLaW leads FID and plans at 89.1; SimWAM plans at 91.5 with generation removed at inference; ForeSight spends 870 ms of a 900 ms budget for 89.3; UniUGP held best FVD and is nowhere near the planning frontier. ReWorld runs both halves in **one architecture, one codebase, one paper** - and connects them nowhere. Stage 1 owns the video results and is never measured for PDMS; Stages 2-3 own the planning results and one of the two objectives uses no world information. **Sharpest available statement: in a chained WAM, the mechanisms that improve future prediction and the mechanisms that improve action selection are separable, and no paper - including the one whose title asserts the link - has shown that improving the first improves the second.** Two counter-readings recorded: Stage 3's planning gradients DO reshape the video branch (UCF-101 71.7 -> 80.2, all of it Stage 3 since Stage 2 freezes the video DiT), an influence running action -> world; and the missing experiment is one run on released code.

**Cross-page effect 2 - a foundation model as an alignment TARGET, and V-JEPA 2 measured below a null baseline.** New section on the foundation-backbones page (`#alignment-targets`). Every prior entry on that page uses a foundation model as an encoder; ReWorld's Table IV uses them as frozen REPA-style alignment targets for a video generator's intermediate features. **Three of four external teachers are worse than no teacher, and the worst is V-JEPA 2 at 331.6 against vanilla flow's 304.1** - a 27.5-point regression, against a wiki thread (Drive-JEPA, Auto-JEPA, WA-JEPA, DA-WAM) that treats V-JEPA 2 as the strongest driving video representation. **The reconciliation is scoped and recorded**: a JEPA encoder is trained to be predictive in latent space, i.e. trained to DISCARD appearance detail - the exact property that makes it a good planning encoder and a bad alignment target for a model that must put pixels back. DepthAnything3 and VideoMAEv2 fail the same way; DINOv2, the only teacher that beats vanilla, retains the most dense appearance structure. **Transferable rule: encoder quality for a discriminative downstream task does not predict target quality for a generative alignment objective, and can anticorrelate with it.** Scope caveats recorded (224x224 without text, one untuned config per teacher).

**Cross-page effect 3 - first entry to lead both nuScenes generation metrics.** The generation-quality table on the world-model page previously recorded split leadership (DriveLaW FID 4.6, UniUGP FVD 75.9) with the note that "the two metrics do not agree on a single leader." 4.4 / 61.9 takes both. The row is added with the decomposition attached (78.9 at gamma=1.0), plus the controlled from-scratch table as the better instrument for method comparison.

**Cross-page effect 4 - hard negatives, third mechanism, and the only one needing no scorer at inference.** Selection-based-planning page, failure-mode 1. DriveSuprim concentrates them by coarse-to-fine filtering; DA-WAM retrieves them as scorer training data (+0.22); ReWorld repels from them with a delta-space L1 loss on a flow-matching planner's own recovered ahat_0 (+0.7), reusing the same forward pass and the same t_a so it costs nothing. The selection rule is DriveSuprim's insight restated - the negative must be the NEAREST low-scoring trajectory, because "random negatives are often distinguishable by geometry alone." All three mine with privileged supervision; only ReWorld's leaves no candidate pool at inference.

**Cross-page effect 5 - two portable auxiliary objectives for flow-matching planners.** New section on the diffusion-planner page. L_align is the more general of the two: a stop-gradient self-distillation of a cross-attention readout, with the frozen conditioning branch and the stop-gradient both load-bearing. Nothing about it is specific to video, driving, or diffusion - it applies to any cross-attention interface where the conditioning signal may be consumed transiently, including the VLM-to-planner interfaces on the dual-system page. Connected to WCog-VLA's ADDT result immediately above it: both are evidence that the marginal return on a diffusion planner now lies in what constrains its intermediate states rather than in its sampler or its capacity.

**Cross-page effect 6 - mid-network again, from the supervision side.** Foundation-backbones readout-depth section. ReWorld sweeps WHERE TO INJECT supervision (block 2/8/12/16/20 of 28 -> FVD 65.5/61.9/62.7/63.0/64.3) where Adaptive-WAM swept where to READ features. Different operations, same shape of answer: **a mid-network block beats both ends**, at ~29% depth for injecting and ~50% for reading. Spread here is only 3.6 FVD, far flatter than Adaptive-WAM's 4.80 PDMS. Two datapoints, one backbone family, no theory.

**Cross-page effect 7 - the cleanest NAVSIM-v1 baseline table on the page.** Every row canonical (VADv2 80.9, UniAD 83.4, TransFuser 84.0, PARA-Drive 84.0, DiffusionDrive 88.1, DrivingGPT 82.4, LAW 84.6, Epona 86.2, WoTE 88.3, PWM 88.1, WorldDrive 89.0, DriveLaW 89.1), and **the first ingested paper to label BOTH configuration-sensitive rows**: "ReCogDrive-IL" at 86.5 and DriveVLA-W0 with a dagger the caption defines. BrainWAM carried both unmarked; Adaptive-WAM carried the second. Comparable to ForeSight, which the navsim page called the one table with no protocol problems, and better on the labelling axis.

**New methods for the gap list**: BeyondDrive (2605.19771, the hard-negative mining source), ReSim (2506.09981, 86.6 PDMS), SRA2 (CVPR'26), Self-Flow (2603.06507), ReDi, "Guiding a diffusion transformer with the internal dynamics of itself" (CVPR'26, the self-guidance precedent), plus the diffusion-representation family REPA / REPA-E / U-REPA / iREPA / RAE / SVG / VA-VAE / VFM-VAE / AlignTok / FAE / Latent Forcing / DiverseDiT / SFD / REG. WorldDrive 89.0 corroborated against GeoWorldAD's value.

**Pages updated**: `wiki/concepts/world-model-for-ad.md` (new Pattern 32; new decoupling section; generation-quality table rewritten with ReWorld leading both metrics plus the controlled from-scratch ordering), `wiki/concepts/navsim-benchmark.md` (SOTA row + caveat paragraph), `wiki/concepts/foundation-backbones-for-ad.md` (new alignment-targets section with the V-JEPA 2 result; supervision-depth sweep added to the readout-depth section), `wiki/concepts/selection-based-planning.md` (three-mechanism hard-negative table under failure mode 1), `wiki/concepts/diffusion-planner.md` (new section on the two auxiliary objectives), `wiki/index.md`, `README.md` (69 papers; new decoupling bullet; open-question 1 extended).

**Figure note**: all five figures in the source are embedded (ReWorld_framework.drawio_compressed.png, ReWorld_infer.drawio.png, contrast.png, qualitive_new.png, navsimvis.png) and all seven tables are reproduced in full, including both two-part ablation tables that appear as unformatted inline text in the clipping.

## 2026-09-04 - Ingest: Unified Driving Tokens: Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning

**Source**: `raw/papers/Unified Driving Tokens_ Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning.md` (arXiv 2606.01935v2)
**Orgs**: Peking University + Xiaomi EV. Ziyang Yao, Zeyu Zhu, YunCheng Jiang, Zibin Guo, Huijing Zhao.
**Pages created**: `wiki/sources/unified-driving-tokens.md`, `wiki/concepts/visual-tokenization.md` (new concept page - the wiki's 31st)
**Confidence**: high for the page; the new concept page is marked medium since it is assembled largely from tokenizer details recorded incidentally on other source pages.

**What it is**: the wiki's **first paper about a tokenizer rather than a planner or a world model**. Premise: in token-based driving pipelines the discrete tokenizer is the "discrete language of the driving world" - simultaneously an AR world model's prediction target and a planner's input - yet nearly all driving tokenizers are inherited from image generation and optimised for pixel reconstruction, leaving "a gap between what is easy to generate and what is useful to decode for driving decisions." Fix: three-way supervision on a VQ-VAE bottleneck. 91.8 PDMS NAVSIM-v1 from a 20M readout head on frozen tokens, single front camera.

**Architecture**:
- Encoder takes BOTH a lightweight RGB detail branch and frozen DINOv3-B patch features, concatenated and projected, then a pre-norm Transformer with RoPE over the patch grid. Most representation-guided tokenizers align to a foundation model at the output only; this one also feeds it in at the input, with the RGB branch present to "recover textures and boundaries weakened by high-level features and quantization."
- Dual-headed decoder: shared post-Transformer feeds D_img (pixels) and D_dino (DINO feature space). The discrete code must be sufficient for both.
- L_tok = lambda_rec(L2 + LPIPS) + lambda_sem(cosine + MSE on DINO) + lambda_gan(GAN) + lambda_vq. None of the lambdas is reported anywhere.
- EMA codebook update rather than gradient, stated reason being to "decouple codeword updates from the competing multiple objectives". Straight-through estimator, commitment term, dead-code reinit, weak orthogonality regularizer.
- Geometry branch (training only): VGGT-style aggregator A_psi alternating frame-wise and cross-frame attention over POST-quantization tokens of two adjacent frames, each with an appended ego token. DPT-style depth head with confidence; relative pose from the ego tokens with L1 translation + sign-invariant quaternion loss. Depth targets are radar-aligned pseudo-labels from DVGT (the predecessor of DVGT-2, the backbone behind GeoWAM and GeoWorldAD).
- Multi-codebook quantization (from UniTok): attention splitter produces M head-specific vectors from one patch embedding, each quantized against its own codebook, merged for the shared decoder. 4 x 4096 vs 1 x 16384; patch grid unchanged so capacity grows combinatorially (4096^4 vs 16384).
- Consumers: (a) planning - tokenizer frozen, 20M readout with learnable registers -> R scene tokens, 11-dim ego status, shallow attention, MLP emitting MULTIPLE trajectories under min-over-N L1, plus a PDM-metric score head trained with BCE against a rule-based evaluator; argmax at inference. (b) world model - tokenizer frozen, 1B GPT next-token Transformer, AdaLN conditioning on actions/ego state, 3 history frames -> 8 generated frames at 2 Hz.
- Data split: tokenizer + world model on OpenScene train; planning decoder on NAVSIM navtrain; all evaluation on NAVSIM test. Resolution 288x512.

**Results**:
- Table 1 tokenizer comparison: LlamaGen (DrivingGPT's tokenizer, 16384) rFID 5.67 / PSNR 23.09 / SSIM 0.652 / dcos 0.0869; Orbis (2x16384) 5.53 / 25.94 / 0.773 / 0.0595; Ours-Rep+Geo (4x4096) 5.14 / 26.33 / 0.769 / 0.0563; **Ours-Rep (16384) 4.15 / 26.51 / 0.774 / 0.0453**. The best tokenizer on every reconstruction column is the one WITHOUT geometry.
- Table 2 roadmap: Naive PSNR 25.96. +Rep PSNR 26.51, ddec_cos 0.0346. +Geo (same 16384 book) AbsRel 0.0640, delta1 0.959, Trans 0.717 m, Rot 2.02 deg, ddec_cos 0.0717, **PSNR 23.90**. +MCB AbsRel 0.0556, delta1 0.965, Trans 0.647, Rot 2.01, ddec_cos 0.0486, PSNR 26.33.
- Table 3 planning: 91.8 PDMS (NC 98.7, DAC 98.2, TTC 95.9, Comf 100.0, EP 87.3). Ties WA-JEPA and DriveFine* in the wiki. Beats the four other frozen-representation planners in its own table: DriveVLA-W0 90.2, DriveLaW 89.1, PWM 88.1, Epona 86.2.
- Table 4 ablation (fixed 20M decoder): none 85.5 -> +Rep 89.4 -> +Rep+Geo 90.9 -> +Rep+Geo+MCB 91.8.
- World model generation: "better FID and FVD than the baseline under the matched setting" - **no numbers anywhere in the body, results in a figure only, baseline unnamed**.

**Limitations**:
- **The unified-interface claim is not tested - it is forked.** Sec 4.1 states planning uses the GEOMETRY-ENHANCED tokenizer and world modelling uses the REPRESENTATION-GUIDED one. Two different tokenizers for the two tasks whose shared token language is the paper's contributions (1) and (4). Table 1 shows why nobody tried one: Ours-Rep beats Ours-Rep+Geo on all five reconstruction metrics. Two unstated reasons the fork is probably necessary, both inferable: MCB emits M=4 indices per patch so an AR world model faces a 4x longer sequence; and ddec_cos stays 40% worse after MCB, which generation quality plausibly tracks. The missing run - geometry tokens into the 1B GPT - uses two artifacts the paper already has.
- The generation result has no numbers and cannot be entered on the wiki's generation-quality tables.
- **Codebook utilization is the central motivation and is never measured.** The intro blames "collapse and low utilization"; Sec 3.2 adds dead-code reinit and an orthogonality regularizer "to improve utilization"; split-and-merge with EMA is called "helpful for balancing codebook utilization". No utilization figure appears anywhere.
- No ablation of M or K_m. Only 1x16384 vs 4x4096, which differ in both. Orbis's 2x8192-shaped configuration is in the comparison table but not the ablation.
- The residual semantic cost is unacknowledged: prose says MCB "largely restores reconstruction quality" without noting ddec_cos stays 0.0486 vs 0.0346, i.e. 40% worse - the quantity the paper's own thesis says matters most for planning.
- The planning readout has a PDM-score head and returns the argmax, so 91.8 is a selection result, not single-trajectory; candidate count R and register count unreported; the scorer is retrained per tokenizer, so "differences primarily reflect the quality of the token representation" is slightly stronger than the design supports.
- No hyperparameters at all (no lambda values, depth loss form deferred to a supplementary not in the clipping). Only d_q=64, K=16384, M=4 are given.
- No latency, throughput, or parameter count for the tokenizer - and a **frozen DINOv3-B forward is required at inference** to tokenize, an unquantified deployment cost.
- Geometry supervision needs DVGT radar-aligned pseudo-depth, so the recipe depends on an external geometry model.
- Adjacent frames only (the paper's own stated limitation). No NAVSIM-v2, no navhard, no nuScenes, single runs.
- Ours-Rep+Geo in Table 1 and +MCB in Table 2 are the same model under two names.

**Cross-page effect 1 - a new concept page, and the lint backlog it clears.** `wiki/concepts/visual-tokenization.md`. The wiki had `action-tokenization.md` for trajectory vocabularies and nothing for visual ones, despite six ingested papers carrying tokenizer details: LlamaGen in DrivingGPT (via UDT's table), MoVQGAN in FutureSightDrive (vocabulary appended to the MLLM's text vocabulary), MAGVIT-v2 K=8192 in ExploreVLA, Emu3-8B discrete tokens in DriveVLA-W0's AR variant, PWM's context-guided two-branch tokenizer at 28 tokens/frame with 8192 codebooks, and DynVLA's dual VQ over DYNAMICS with only 64 ego + 64 environment code types. Page collects those, plus the capacity conflict, the DINO-space diagnostics, the planning-readout evidence, and five open questions. Two of the eight tokenize something other than appearance, which is the page's most transferable observation: **the vocabulary size a driving task needs is set by what the token must support downstream, not by reconstruction fidelity.** Marked medium confidence and cross-linked from action-tokenization as a sibling.

**Cross-page effect 2 - the wiki's cleanest isolation of representation quality from planner capacity.** +6.3 PDMS from tokenizer training objectives alone, planner fixed at 20M. Each component has a distinct signature: Rep +3.9 moves every sub-metric (DAC +2.6, TTC +1.7, EP +4.3); Geo +1.5 is almost entirely EP (+4.2) with TTC DOWN 0.6; MCB +0.9 lands in DAC and EP, consistent with Table 2 where MCB recovers what Geo cost. Set beside DriveLaW's cross-family sweep under a fixed planner (video latents 89.1 > VLM hidden 86.5 > BEV 84.1, spread 5.0), UDT varies only supervision within one family and gets a LARGER spread. Joint statement recorded on the backbones and tokenization pages: **how a representation is trained matters at least as much as which family it comes from.**

**Cross-page effect 3 - geometry buys progress, not safety: third instance, and the first where safety goes backwards.** New table in the world-model page's shared-future section. GeoWorldAD's shared latent future-depth tokens: +3.3 EP, NC/TTC flat. UDT's depth+pose tokenizer supervision: **+4.2 EP, TTC -0.6**. GeoWAM: +4.9 EPDMS navhard vs +0.6 navtest, arguing the case without ablating it. Three papers, three mechanisms - inference-time shared future, training-time tokenizer supervision, and dense point-map forecasting - one sub-metric. The generalization recorded is narrower and more useful than any of their claims: **geometric supervision buys ego progress rather than safety**, plausibly because knowing where free space will be lets a planner commit instead of hedging. Consequence noted: a geometry result reported only as aggregate PDMS hides which half of the trade it bought.

**Cross-page effect 4 - a measured capacity conflict at a DISCRETE bottleneck, and the opposite remedy to BrainWAM's.** Adding geometry to a fixed 16384 codebook costs 2.61 PSNR (26.51 -> 23.90, below even the naive RGB-only tokenizer at 25.96) and doubles ddec_cos (0.0346 -> 0.0717) - the tokens stop carrying the DINO representation they were trained to carry. MCB then improves EVERY column at once (PSNR 26.33, AbsRel 0.0640 -> 0.0556, delta1 0.959 -> 0.965, Trans 0.717 -> 0.647), which is the signature of a capacity limit rather than an objective conflict and supports the paper's reading. **The contrast with BrainWAM is the transferable part**: BrainWAM found that mixing a clean semantic stream with a denoising one in a shared attention pool hurts (87.8 vs 88.1 WAM-only) and fixed it by NARROWING the interface to 8 action tokens; UDT finds that mixing appearance, semantics, and geometry in a shared codebook hurts and fixes it by WIDENING capacity. Opposite remedies for related-looking problems - gradient competition between streams of unequal maturity versus an information budget - and distinguishing the two before reaching for a remedy is the lesson.

**Cross-page effect 5 - a second paper picks DINO from the same family, days after ReWorld measured V-JEPA 2 below a null baseline.** Added to the backbones page's alignment-targets section. ReWorld's Table IV had REPA-with-DINOv2 as the only external teacher beating vanilla flow (295.9 vs 304.1) while V-JEPA 2 was worst at 331.6. UDT uses frozen DINOv3-B as both encoder input and decoding target and that supervision is its largest single component (+3.9 PDMS). Different task, different mechanism, same conclusion: **of the frozen encoders tried as supervision targets in driving, the DINO family is the one that transfers.** Neither paper explains why, and the two candidates (dense appearance retention vs patch-level object structure) are separated by neither. Also recorded: a frozen foundation model in the INPUT path is a deployment cost in a way a training-time-only target is not, and this page has no latency numbers for it from any source.

**Cross-page effect 6 - a reconstruction diagnostic worth adopting.** UDT introduces measuring reconstruction error in a frozen foundation model's feature space rather than in pixels: d_img (reconstructed vs ground-truth image once both pass through frozen DINO) and d_dec (the tokenizer's decoded feature head vs the frozen target). The two come apart from PSNR in exactly the case that matters - the +Geo row holds depth accuracy while PSNR collapses and ddec_cos doubles. Recorded on the new concept page as the recommended tokenizer-comparison metric alongside rFID/PSNR/SSIM.

**Cross-page effect 7 - the best baseline row in any ingested table.** UDT is the **first ingested paper to cite DriveSuprim at its correct 93.5** with submetrics identical to that method's published row (98.6 / 98.6 / 95.5 / 100 / 91.3) - GeoWorldAD used the circulated 89.9. Every other row is canonical too (VADv2 80.9, UniAD 83.4, Para-Drive 84.0, TransFuser 84.0, DRAMA 85.5, DiffusionDrive 88.1, WoTE 88.3, Epona 86.2, PWM 88.1, DriveLaW 89.1, DriveVLA-W0 90.2 at its anchor headline). Consequence: DriveSuprim sits ABOVE Ours in its own table and the claim is scoped honestly to "the highest PDMS among methods that plan from frozen visual tokens." One classification quibble recorded: the dagger group mixes discrete-token conditioning with continuous-latent conditioning (DriveLaW caches Video DiT activations; Epona is AR+diffusion), so "frozen tokenizer" is doing loose work.

**Cross-page effect 8 - the Xiaomi EV cluster.** DriveLaW, ReWorld, GeoWorldAD, and UDT are all Xiaomi EV collaborations ingested within days of each other, and UDT cites DriveLaW. The DVGT lineage (UDT's depth pseudo-labels; DVGT-2 in GeoWAM and GeoWorldAD) connects a fifth. Recorded on the source page as a research-program observation rather than a finding.

**New methods for the gap list**: ResWorld (2602.10884, 89.0 PDMS), Orbis (2507.13162, a 2x16384 driving tokenizer with a published checkpoint, run here under the same eval pipeline), DVGT (2512.16919, depth-pseudo-label source and DVGT-2's predecessor), UniTok (multi-codebook), TokenFlow, LlamaGen, MoVQGAN, and the VFM-as-tokenizer line (NeurIPS 2025).

**Pages updated**: `wiki/concepts/visual-tokenization.md` (NEW), `wiki/concepts/navsim-benchmark.md` (SOTA row + caveat), `wiki/concepts/world-model-for-ad.md` (geometry-buys-progress table, third instance), `wiki/concepts/foundation-backbones-for-ad.md` (DINO corroboration + the inference-cost note in the alignment-targets section), `wiki/concepts/selection-based-planning.md` (methods-table row), `wiki/concepts/action-tokenization.md` (sibling-page pointer), `wiki/index.md` (source row + concept row + frontmatter), `README.md` (70 papers / 31 concepts; new representation bullet).

**Figure note**: all four figures in the source are embedded (pic2.png, pic3.png, pic4.png, pic5.png) and all four tables are reproduced in full. Note that the world-model FID/FVD comparison exists only inside pic5.png and is not transcribable from the clipping.

## 2026-09-11 - Ingest: DriveFuture: Future-Aware Latent World Models for Autonomous Driving

**Source**: `raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md` (arXiv 2605.09701v1)
**Orgs**: Institute of Automation CAS + Beihang + Beijing Jiaotong + Queensland + Macau + NTU + Yanshan/BIT. Yufeng Hong, Xiaotian Zhou, Yingyan Li (equal), Xiangpo Zhou, Lin Liu, Yadan Luo, Shaoqing Xu, Lei Yang, Ziying Song (corresponding).
**Pages created**: `wiki/sources/drivefuture.md`
**Confidence**: high. All seven tables reproduced, all four figures embedded, two reported numbers cross-checked against independent wiki entries.

**Why this one mattered before it was read**: DriveFuture was a named gap in four existing pages - the "89.9 corrected EPDMS, not ingested" row on `navsim-benchmark.md`, and cited-but-unavailable in wa-jepa, auto-jepa, and da-wam. Its row checks out at the primary source.

**What it is**: a role reversal rather than a new predictor. Every latent world model here defines a future-prediction target and a distance to it. DriveFuture defines **none**. Total objective is `L_plan + L_BEV`, both weighted 10; the 16-token foresight latent is shaped only by trajectory-denoising gradients plus a training-time cross-attention grounding against the real future BEV. Appendix B.2 states it: do not ask whether the future latent can be predicted, ask whether it improves denoising of the plan.

**Architecture**:
- V2-99 -> 16x64 BEV anchors -> 64 BEV tokens + 1 ego token = Z_t in R^(65x256). Two temporal frames, two stitched panoramas (front-left/front/front-right and the rear trio), both 2048x512.
- Latent Dynamics Predictor: 4 decoder layers, 16 learnable future queries, cross-attending [Z_t || E_tau]; FFN 2048. The 16-token output against a 64-token future bank is a deliberate bottleneck "to prevent the world model from copying dense future appearance."
- Trajectory tokenizer: normalised finite differences + sin/cos heading + temporal PE, 8 tokens over 4 s at 0.5 s.
- Conditioning-source randomisation: expert / constant-acceleration kinematic rollout / learned null at (0.4, 0.4, 0.2). The null branch exists so CFG has a calibrated baseline rather than a difference of two conditionals.
- **Future Alignment Adapter (training only)**: Z_{t+T} = sg(phi_enc(I_{t+T})) in R^(64xd); one MHA with the prediction as query and the real future as K/V, **no residual from the query side** - so the oracle condition is a pure selection of GT future evidence, indexed by what the model predicted.
- **LatentAlign**: sigmoid anneal alpha(e) = 1 - sigma(beta(e - e0)), e0 = rho_E * E. Early training gets the oracle, late training gets only the self-prediction.
- Planning Decoder: 5-layer DiT, DDPM over (dx, dy, sin th, cos th) in R^(8x4); each block cross-attends scene then future; **zero-init on the future cross-attention output projection** so the module starts as identity.
- **Progressive Foresight Guidance** at inference: three CFG branches (null / kinematic / Tweedie-from-the-current-sample) with cosine decay and cosine rise weights handing over on (0.3, 0.7); w_kin_max 1.5, w_tw_max 2.5. The schedule is *derived*, not tuned: the Tweedie error is amplified by sqrt((1-alpha_bar)/alpha_bar), which diverges at high noise.
- Training: navtrain, Adam 1e-4, batch 16 x 5 accumulation, 100-130 epochs, bf16, 8x RTX 5090, cache-only loading. 100 proposals at inference, GTRS-Dense scorer for leaderboard submissions.

**Results**: 55.5 EPDMS navhard (1st, April 2026), 89.9 corrected / 86.4 pre-fix EPDMS navtest v2, 90.7 PDMS navtest v1.

**The decomposition, which is the ingest's main finding**: every ablation is captioned "without GTRS-Dense scorer"; the best ablation row is 34.6 EPDMS; Table 7's unscored stage-wise row matches that row on all six shared submetrics (NC 82.3, DAC 78.8, TTC 79.6, LK 47.6, HC 97.0, EC 75.9). Therefore **55.5 = 34.6 model + 20.9 scorer**, and **34.6 = 30.9 no-future-frames + 3.7 grounding**. The scorer is worth 5.6x the mechanism the paper is named for, and 85% of the distance from the weakest configuration (30.9) to the headline (55.5). The paper reports Table 7 honestly and discusses the EC trade-off; the abstract and contributions attribute 55.5 to future conditioning.

**Correction made during the ingest**: the ablation's baseline row removes *future frames*, not the world model. KS and GT are the world model's own conditioning-source modes, so row 1 still builds the foresight latent and still feeds it to the DiT. `use_wm` and `use_wm_to_dit` are named in Appendix C.3 and varied in **no reported table** - so +3.7 prices the grounding only, and **the paper's headline mechanism has no measured value in the paper that proposes it.**

**Ablation Table 4** (navhard, no scorer). FF = future frames in training, Impl = implicit constraint, MSE = feature regression, KS/GT = guidance sources:
- no FF, KS+GT: 30.9
- FF + MSE + KS + GT: 32.1
- FF + Impl, no guidance: 32.0
- FF + Impl + KS + GT: **34.6**

Clean single-factor reads: grounding +3.7; **implicit conditioning over MSE +2.5**; dual-source guidance +2.6; MSE over null **+1.2**.

**Sensitivity Table 5**: t_f 0.5/1.0/1.5 -> 30.2/31.2/34.6; q_s 4/16/64 -> 28.2/34.6/33.7; e0 0.75/0.83/0.95 -> 29.2/34.6/28.9. **Three of nine settings fall below the 30.9 no-future baseline.** The e0 span (5.7) exceeds the +3.7 the grounding is worth. t_f = 1.5 is the largest value swept, so "best in group" is a boundary.

**Limitations**: no latency / parameter count / FLOPs / denoising-step count, for a pipeline running 100 proposals x up to 3 denoiser branches x per-proposal world-model forwards + a 100-candidate scoring pass; navhard prose in 4.3 quotes NAVSIM-v1 numbers (99.1 DAC / 95.4 TTC); DriveSuprim cited at the circulated 89.9 on v1 and 83.1 on v2; GoalFlow appears twice on v1 unlabelled; ARTEMIS EC duplicates its HC at 98.3; the only world-model head-to-head is ResNet-34 World4Drive against V2-99 DriveFuture with no backbone ablation; single runs; near-future only (t_f <= 1.5 s of a 4 s horizon, one step not a rollout); no generation evaluation; no code; no closed-loop benchmark beyond navhard; broken cross-references (Table 4.2, Fig. 3/4 mislabelling).

**Cross-page effect 1 - navhard's ceiling moved from 36.6 to 55.5, and the reason is selection.** `navhard-ood-evaluation.md` gains a second, larger leaderboard from DriveFuture's Table 1 (14 entries), which **shares no method with GeoWAM's**. The split is almost perfectly clean: every entry above 42 scores its own candidates (DrivoR 54.6, SimScale 53.2, GTRS-E 49.4, ZTRS 48.1, DiffVLA 45.0, DriveSuprim 42.1), every entry below 35 does not. Recorded as `#scorer-cohort` and `#scorer-price`, with the +20.9 decomposition and its EC cost (76.9 -> 66.2 Stage 1, 75.9 -> 45.6 Stage 2).

**Cross-page effect 2 - the "Stage 2 collapses lane keeping for everyone" claim is wrong.** That page recorded a 45-50 Stage-2 LK band with "under 5 points" between the best and worst learned method, generalised from ten unscored baselines. DriveFuture's table spans **45.4 to 62.3**, and the ordering tracks scoring. **The same checkpoint moves 47.6 -> 58.3 by adding a scorer and changing nothing else.** What survives: every method still loses 35-45 points between stages, and the 3DGS-artifact question is narrowed rather than answered. Recorded as `#lk-correction`.

**Cross-page effect 3 - SpanVLA's 40.1 is resolved.** The wiki had it as an uninterpretable per-stage pair ("identical headline for both stages while submetrics differ"). DriveFuture reproduces SpanVLA's eighteen submetrics exactly and prints a single combined 40.1 in a table whose other rows are verifiably combined. Moved to the combined leaderboard on both `navhard-ood-evaluation.md` and `navsim-benchmark.md`. **One paper reprinting another's submetrics resolved an ambiguity neither paper's own text could** - the same audit technique the protocol-drift section relies on, used constructively for once.

**Cross-page effect 4 - the first cross-validation of the EPDMS correction.** `navsim-benchmark.md` gains `#both-columns`. DriveFuture is the second paper to tabulate both EPDMS columns (DriveFine reported both only for its own model), and it reproduces WA-JEPA's pairs exactly for itself (86.4 / 89.9) and for DiffusionDriveV2 (85.5 / 87.5) - two of the eight measured deltas now have two independent sources. But it **contradicts WA-JEPA's cohort assignment for ReCogDrive (83.6) and DriveVLA-W0 (86.1)**, printing both in the corrected column where WA-JEPA filed them pre-fix; both are now unclassified. It also agrees with GeoWAM and BrainWAM on Hydra-MDP++ 81.4 against WA-JEPA's 84.1. On TransFuser it gives 76.7 pre-fix with the corrected column **blank** - the sixth table to print 76.7 - which reframes the GeoWAM anomaly: **no paper in this wiki has published a corrected TransFuser number at all**, so GeoWAM's 84.0 is unrefuted rather than outvoted; the other five all report the pre-fix value. It is also the fourth table to carry ARTEMIS EC = 98.3, pushing that tally to four-to-one.

**Cross-page effect 4b - the shared baseline block is now labelled by one of its users.** DriveFuture's Transfuser (76.7), DriveSuprim (83.1), DiffusionDrive (84.5) and DriveVLA-W0 (86.1) rows are **digit-identical across all nine submetrics** to GeoWorldAD's - the block LWDrive and BrainWAM also carry. The wiki had *inferred* from row-pairing that the block mixes conventions. DriveFuture sorts it into two labelled columns and states the mapping: TransFuser / Hydra-MDP++ / DriveSuprim / ARTEMIS pre-fix, DiffusionDrive / ReCogDrive / DriveVLA-W0 / DriveWorld-VLA / Latent-WAM corrected. **The block is auditable rather than merely suspect**, and where it disagrees with WA-JEPA's cohort partition (ReCogDrive, DriveVLA-W0) the disagreement is now between a second-hand attribution and a first-hand label.

**Cross-page effect 5 - a navhard bridge between two tables that share no method.** The one connecting row: SpanVLA calls it **LTF 23.1**, DriveFuture calls it **TransFuser 23.1** with submetrics identical to within 0.1, and GeoWAM calls it **LTF 25.1** with the four submetrics this wiki recorded **identical**. Two papers agree on the value and disagree on the name; GeoWAM aggregates the same submetrics 2.0 higher - the same direction as its navtest TransFuser anomaly. Working assumption adopted: the two navhard tables are on the same scale +/- 2 points, under which **GeoWAM's 36.6 is eighth, not the reactive/OOD leader**. Recorded as `#bridge`, and the old leaderboard section carries a scope note.

**Cross-page effect 6 - Pattern 33 on `world-model-for-ad.md`, and the wiki's first annealed privileged signal.** Every prior privileged-supervision case (Hydra-MDP distillation, Auto-JEPA's CLOVER scorer, Adaptive-WAM's pseudo-expert targets, DA-WAM's factor heads) distils a *simulator's labels* into a head that keeps using them. DriveFuture hands the planner a real future *observation* and then withdraws it on a schedule. The withdrawal schedule is the most sensitive hyper-parameter in the system.

**Cross-page effect 7 - DriveLaW's t=10 collapse now has a training-side twin.** Recorded as `#oracle-annealing`. DriveLaW: conditioning on a *generated* future at increasing denoising completion gives 89.1 -> 86.9 -> **23.2**. DriveFuture: conditioning on a *real* future held for an increasing fraction of training gives 34.6 -> **28.9**, which is below its own no-future baseline of 30.9. Different signals, different mechanisms, same shape: **the closer a planner's conditioning gets to a future it cannot produce at inference, the worse it does.** Neither paper cites the other's phenomenon. DriveFuture's `force_alpha_one` switch - named in Appendix C.3, reported nowhere - would settle it directly.

**Cross-page effect 8 - the objective-form axis gains a third arm and loses a generalization.** WA-JEPA measured deterministic regression on future latents at **-0.4 against no prediction**; DriveFuture measures it at **+1.2**. The ordering holds in both (regression is well behind the alternative) but "regression on future latents is actively harmful" does not generalise - it was a property of WA-JEPA's high-entropy multi-view ViT-L target, not of regression. The entropy-of-the-target framing survives both and now explains the discrepancy. The new arm: **no prediction objective at all**, only planning gradients plus attention grounding, which wins by +2.5 in the one paper that tests it. Flow matching and no-objective-plus-grounding are not in competition - the first wins when the latent must stand alone as a prediction, the second when it only has to be a conditioning context. **Nobody has run both on one target.**

**Cross-page effect 9 - DA-WAM's taxonomy is half wrong about this paper.** DA-WAM files DriveFuture under "(b) loosely coupled latent fusion - one proposal, nothing to compare." True for the null and kinematic branches, which are computed once per scene and broadcast across 100 proposals. **False for the Tweedie branch**, which conditions the world model on an estimate recovered from *that proposal's own* noisy sample and therefore produces one future latent per candidate - DA-WAM's own category (d), applied inside the *generator* rather than the scorer. Recorded as `#per-candidate-in-generator`. Neither paper knows what a per-candidate future is worth inside a generator; DriveFuture's `use_dspcfg` switch already exists.

**Cross-page effect 10 - guidance applied one level upstream.** `diffusion-planner.md` gains `#guidance-upstream`. Every prior CFG variant here (DiffusionDrive anchors, PlannerRFT scales, DIAL intents, FeaXDrive SDF) guides a condition the *denoiser* consumes. PFG guides a condition the *world model* consumes, and one of its branches is recursive. The reusable part is the derivation: when a diffusion model's condition is computed from a quantity the model is itself producing, the Tweedie error-amplification factor sqrt((1-alpha_bar)/alpha_bar) tells you when to trust the recursive estimate, and a noise-free heuristic covers the rest. `intent-conditioned-planning.md` gains `#circular-intent` for the dependency itself, plus the observation that **three unrelated objects now share the word "intent"** in this wiki - a maneuver class, a goal/path specification, and a dynamics-model action - and only the first supports the multimodality argument that page is built on.

**New methods for the gap list**: **DrivoR** (54.6 navhard, ViT-S - now the second-strongest navhard entry and already GeoWAM's HUGSIM baseline), **SimScale** (53.2), **GTRS-E** (49.4), **ZTRS** (48.1), **DiffVLA** (45.0), **MindDrive** (2512.04441), **GuideFlow** (2511.18729, shares authors with this paper), **DriveWorld-VLA** (2602.06521, 91.3 PDMS v1 - the one method in DriveFuture's own v1 table it does not beat, and from overlapping authors), **World4Drive** (ICCV'25, its only world-model head-to-head), **WorldRFT** (AAAI'26), **LAW** (2406.08481 - the direct ancestor, still uningested), **GTRS-Dense** (the scorer carrying +20.9 of the headline), **Senna-E2E**, **DIVER**.

**Pages updated**: `wiki/sources/drivefuture.md` (NEW), `wiki/concepts/navhard-ood-evaluation.md` (new leaderboard, scorer price, bridge, LK correction, SpanVLA resolution, open questions, lint rule), `wiki/concepts/navsim-benchmark.md` (v1 + v2 rows, correction-delta verification, `#both-columns`, navhard section), `wiki/concepts/world-model-for-ad.md` (Pattern 33, `#oracle-annealing`, objective-form third arm, `#per-candidate-in-generator`, imagine-then-act camp list), `wiki/concepts/selection-based-planning.md` (`#scorer-price`), `wiki/concepts/diffusion-planner.md` (`#guidance-upstream`), `wiki/concepts/intent-conditioned-planning.md` (`#circular-intent` + table row), `wiki/index.md`, `README.md`.

**Figure note**: all four figures embedded (motivationv13.png, methodv8.png, visv2.png, tu3.drawio.png) and all seven tables reproduced in full on the source page. Figures 3 and 4 carry no quantitative annotation, and the failure-category claim in 4.5 has no counts behind it.

## 2026-09-11 - Ingest: CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving

**Source**: `raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md` (arXiv 2605.10426v3)
**Orgs**: Afari Intelligent Drive + Southeast University + Tianjin University + UESTC + SJTU + BUPT. Minqing Huang, Yujiao Xiang, Zihan Liang, Jiajie Huang, Jingqi Wang (corresponding), Yuheng Zhou, Zhi Xu, Feiyang Tan, Hangning Zhou, Mu Yang, Gong Chen.
**Pages created**: `wiki/sources/coworld-vla.md`
**Confidence**: high. All seven tables reproduced, all seven figures embedded, two reported numbers cross-checked against independent wiki entries.

**Why this one mattered before it was read**: CoWorld-VLA was the last of the four un-ingested corrected-protocol NAVSIM-v2 methods flagged on `navsim-benchmark.md` since the WA-JEPA ingest ("CoWorld-VLA 90.0*, not ingested"). Its row checks out at the primary source: EPDMS* 86.2 / EPDMS 90.0, exactly as WA-JEPA reported it.

**What it is**: the first paper here to question whether a world model should predict **one** thing. Four future-prediction objectives run in parallel against four frozen teachers, all landing as typed expert tokens inside one Qwen3-VL-2B's hidden states, then fused by a four-branch diffusion planner. Single front camera, single frame, no RL, no scorer.

**Architecture**:
- Stage 1: Wan2.2-5B DiT trained on 8 Hz nuPlan, flow matching applied **only to the future segment** with history left noise-free. "Action-conditioned" means **text**: a serialized prompt [Scene]+[Speed]+[Navigation]+[Trajectory] through Wan's frozen UMT5 encoder - speed as a coarse phrase, waypoints as a polyline anchored at the origin, headings omitted. Documented fallbacks when command or future trajectory is missing.
- Stage 2, four expert tokens appended after image+text tokens:
  - H_sem <- pooled frozen **V-JEPA** features of the FUTURE frame, SmoothL1 + cosine
  - H_geo <- pooled frozen **VGGT** features of the FUTURE frame, MSE
  - H_dyn <- **replaces the Stage-1 text condition** of the Wan DiT; flow-matching loss back-propagates into the token. The VLM never decodes pixels: the video model is a **differentiable critic on one latent**.
  - H_traj <- GT waypoints via an MLP head, MSE
  - Weights (w_dyn, w_sem, w_geo, w_traj) = (1.0, 0.1, 0.1, 1.0).
- Stage 3 HMEF: two-stream joint-self-attention denoiser. Clean scene stream = non-action VLM tokens + current-frame V-JEPA + current-frame VGGT, each Perceiver-compressed **separately**. Noisy action stream carries per-expert, per-future-step features (bidirectional Transformer per expert group, averaged within horizon step) plus MLP-encoded ego history/status. Rectified flow, logit-normal tau, normalized (x,y,psi) in [-1,1] from empirical NAVSIM ranges.
- **Every branch regresses the SAME target**, and alpha = softmax(w) are **global scalars**. Expert trajectories are **detached** before the weighted average, so the fusion term updates only w.
- **The Wan model is not used in Stage 3 or at planning inference** (stated in the limitations). V-JEPA and VGGT *are* - frozen, per frame, for the scene stream.
- Compute, disclosed in full: 64xA800 x 74h + 32xA800 x 70h + 16xA800 x 20h = **~7,296 A800-GPU-hours**. Named by the paper as its primary limitation.

**Results**: 90.0 PDMS NAVSIM-v1; 86.2* / **90.0** EPDMS NAVSIM-v2 (4th in the corrected cohort); FVD 32.7 on NAVSIM. Inference: 20 video sampling steps, 10 planning steps.

**Finding 1 - the wiki's first complete representation x planner 2x2, and the interaction is strongly negative.** Table 5 gives all four cells:

|                  | VLM readout | + HMEF | effect of planner |
|------------------|------------:|-------:|------------------:|
| EgoT. token only |        83.7 |   88.9 |             +5.2  |
| Full 4 experts   |        88.7 |   90.0 |             +1.3  |
| effect of repr.  |       +5.0  |  +1.1  | interaction -3.9  |

Each intervention is worth ~5 alone and ~1 on top of the other; additively one would predict 93.9, observed 90.0. The paper calls this "complementary benefits"; the dominant story in its own table is **substitution**. **A diffusion action expert on a bare trajectory token already reaches 88.9**, above every no-RL VLA baseline in its own Table 1 (ReCogDrive 86.5, DriveVLA-W0 87.2, LaST-VLA 87.3, SGDrive 87.4). HMEF over ReCogDrive's action expert in the same pipeline is **+0.9** (90.0 vs 89.1) - the honest size of the architecture's contribution. Caveats recorded: PDMS is bounded so some sub-additivity is mechanical, but 90.0 is well short of the 94.8 human reference; single runs, one architecture.

**Finding 2 - CoWorld-VLA and WA-JEPA are the same lab, and they contradict each other.** Ten of eleven authors shared; same five affiliations; both release under **github.com/AFARI-Research**; each paper's corresponding author is an author of the other. The wiki had recorded this as "four authors," which understated it badly. Two consequences: WA-JEPA's CoWorld-VLA row is a **first-party** number, so it is reliable about CoWorld-VLA and worthless as independent corroboration of the correction's size; and the two papers **place DriveVLA-W0's 86.1 in different columns** - WA-JEPA pre-fix, CoWorld-VLA corrected. Not a copying error between strangers: overlapping author sets with the same pipeline disagreeing about which evaluator produced a value they both publish.

**Finding 3 - objective form is assigned by target entropy, correctly and silently.** Flow matching for the high-entropy future video; deterministic regression for the *pooled* semantic and geometric targets and the single trajectory. That is exactly the rule assembled from WA-JEPA (regression worse than no prediction on multi-view scene latents) and DriveFuture (regression +1.2 on a 16-token BEV compression). **Pooling is the lever nobody else pulled** - it converts a dense high-entropy feature field into a low-entropy summary, at which point a conditional mean is usable. CoWorld-VLA never states the principle.

**Ablation Table 4** (Stage-2 readout, NAVSIM-v1 PDMS): EgoT. 83.7 -> +Geo 85.1 -> +Sem 85.2 -> +Sem+Dyn 87.3 -> +Geo+Sem 87.7 -> all four **88.7**. Marginals: Geo +1.4/+2.5, Sem +1.5, Dyn +2.1/+1.0. **The dynamic expert - the world model the paper is named for - is never measured alone**; the missing EgoT.+Dyn. row is one run. Geometry beats dynamics as the third addition (87.7 vs 87.3), which the learned fusion weights invert. Across the sweep DAC +3.8 and EP +3.8 against NC +0.7 - a fourth instance of geometric/structural supervision buying progress rather than safety.

**Table 6 denoising steps**: 5/10/20 -> 89.5/**90.0**/89.7. Non-monotonic, spread 0.5, action denoiser not video.

**Table 7 inference-seed stability**: six seeds, fixed Stage-3 weights. PDMS mean 89.980 std **0.013**; EPDMS mean 89.985 std **0.014**. Second seed measurement in the wiki after WA-JEPA's 0.053, and the first paper anywhere here to state the gap that matters: *"They do not replace variance estimates over independently trained models, which were not conducted because of the high training cost."* **Every variance number on the navsim page is sampler noise; nobody has measured training-seed variance.**

**Learned expert weights** (from uniform 0.25): dyn 0.35, traj 0.31, sem 0.19, geo 0.15. **Contradicts the paper's own ablation ordering**, which makes geometry the stronger third addition. No uniform-weight control is run, even though all four branches regress an identical target.

**Limitations**: no latency / parameters / FLOPs, with **two frozen foundation encoders (V-JEPA + VGGT) in the inference input path**; the dynamic expert never ablated alone; three v2 numbers published for methods that never reported any (WoTE 87.7, PWM 88.2, DriveLaW 88.6) with no provenance; a third distinct DriveSuprim v2 value (86.0, third submetric set); DriveVLA-W0 carried on v1 at DriveLaW's reimplementation value 87.2 unmarked; global non-adaptive fusion weights; single front camera, single frame; ~7,300 GPU-hours; no RL, no scorer, **no navhard / Bench2Drive / HUGSIM / nuScenes**; WA-JEPA neither cited nor compared (chronology 2605 vs 2608 probably explains it); PDMS and EPDMS both averaging 89.98 across six seeds is a coincidence worth checking; Figures 5 and 7 are multi-panel and only panel (a) of each survives the clipping.

**Cross-page effect 1 - Pattern 34 on `world-model-for-ad.md`.** Patterns 1-33 each pick one prediction target; this is the first to run four and ablate complementarity. The reusable mechanism is the dynamic branch: a **video generator used as a differentiable critic on a single latent**, which generalizes to any conditioning interface a generator already exposes and is strictly cheaper than making the planner generate. Sub-section `#entropy-assignment` records the silent-but-correct objective assignment and names pooling as the lever.

**Cross-page effect 2 - a third position between the two world-model camps.** The Wan DiT is training-time-only (discarded before Stage 3), but the **token it supervised survives into inference** and conditions its own denoising branch. Not "generate a future at decision time," not "shape a shared representation and discard everything" - keep the specific latent the generator shaped. DriveFuture is the closest relative with one such latent instead of four. Recorded in the camp lists and in Pattern 34.

**Cross-page effect 3 - `#factorial` on `foundation-backbones-for-ad.md`, and a correction to a claim this wiki has been repeating.** "The representation is worth several PDMS, whoever holds the planner fixed" was measured by DriveLaW against a 133M DiT and by UDT against a 20M readout - i.e. always in the cell where the other factor is weak, which is the cell where a representation looks most valuable. CoWorld-VLA's 2x2 shows the two are **substitutes over most of this range**. Neither prior number is wrong; the additive reading of them was. The cheap replication is named: DriveLaW already has three representations and a planner.

**Cross-page effect 4 - `#target-role` bounds ReWorld's negative V-JEPA result.** ReWorld measured REPA-with-V-JEPA-2 at **331.6 FVD against vanilla flow's 304.1** - the worst external teacher tested. CoWorld-VLA aligns a VLM token to pooled V-JEPA future features and it is worth **+1.5 PDMS**. Same mechanism, opposite sign, and the distinguishing variable is **what the aligned model produces**: a JEPA encoder discards appearance detail prediction does not need, which makes it a bad target for a model that must put pixels back and a good one for a model that must output a plan. ReWorld supplies the failure case, CoWorld-VLA the success case. VGGT adds a second geometry-as-target data point on the same split.

**Cross-page effect 5 - the protocol-drift section tightens again.** `navsim-benchmark.md`'s `#both-columns` becomes "Three Papers, Both Columns - and Two of Them Share a Lab." The independent count is two groups, not three. **DiffusionDriveV2's 85.5 -> 87.5 is the one genuinely cross-validated delta** (two independent groups). And the intra-lab DriveVLA-W0 disagreement forces the rule from "not comparable across papers" to **"treat a column label as trustworthy only for the paper's own row"** - the attributions are not reliable within a lab either.

**Cross-page effect 6 - a sixth baseline-provenance failure mode.** The catalogue previously held: copied rows under the wrong convention, mislabelled cohorts, non-headline configurations cited unmarked, cross-version submetric contamination, oracle BoN scores in single-sample tables. CoWorld-VLA adds **rows with no attributable origin**: three v2 numbers for methods whose own papers report no v2 at all. Either they are recomputations - the practice the page keeps asking for, worth more than the numbers - or they are not, and the paper does not say. Against that, its "Drive-JEPA (R34)" label is exactly the discipline the BrainWAM rows lacked, so the authors clearly can do it.

**Cross-page effect 7 - the seed question gets its second data point and its real open problem.** Two papers, two architectures: sampler-seed std 0.053 (WA-JEPA, 12-step flow) and 0.013 (CoWorld-VLA, 10-step rectified flow). Sampler noise on NAVSIM is 0.01-0.05, two to three orders below the evaluator shift, so ablation deltas above ~0.2 stay meaningful. **The new part is the gap CoWorld-VLA names**: every variance number here is sampler noise, and training-seed variance - the one that would license the sub-point ablation claims this wiki reads constantly - has never been measured by anyone. Its ~7,300 GPU-hour bill is a fair explanation of why.

**Cross-page effect 8 - `#expert-ensemble` on `diffusion-planner.md`.** HMEF is easy to mis-file: four branches, one per condition, **all regressing the same target**, combined by a frozen global convex weight. It is variance reduction, not multimodal proposal generation - no candidate set, no scorer, one output. Averaging is safe here only because the branches were never trained to disagree. Three reusable details recorded: detached-output fusion-weight training, per-future-timestep expert features, and separate Perceiver compressors per scene source.

**Cross-page effect 9 - `#static-mixture` on `adaptive-routing.md`: the page's natural null, unrun.** CoWorld-VLA computes four fully-denoised trajectories and combines them with a global alpha learned once. Routing that mixture per scene costs **zero extra inference compute** - the trajectories all exist regardless - and the experts are exactly the heterogeneous-competence set routing is for. The paper runs neither a scene-conditioned variant nor a uniform-weight control, and its learned weights disagree with its own ablation about which expert matters, which is what one would expect if a single global weight were averaging over scenes with different needs. Logged as the cheapest untried experiment on that page.

**Cross-page effect 10 - `chain-of-thought-for-ad.md` gains a fifth substrate, on a new axis.** Text, visual, dynamics, single-latent, and now **typed multi-expert latent**. The novelty is not latency: it is that each latent has an assigned external supervisor rather than an emergent meaning. OneVL and DynVLA specify content with purpose-built decoders and tokenizers; CoWorld-VLA specifies it with off-the-shelf frozen foundation models, which is cheaper to assemble and ties the trace's semantics to whatever those teachers encode.

**Cross-page effect 11 - a fifth Wan2.2-5B coupling strategy, at zero inference DiT forwards.** The backbone table now reads SimWAM 0 forwards / 91.5, **CoWorld-VLA 0 / 90.0**, Adaptive-WAM ~0.5 / 90.8, DriveVA 2 / 90.9, BrainWAM 1-3 / 89.5, DriveWAM 3 / 90.1. CoWorld-VLA also inverts BrainWAM's pairing: the VLM is the policy and the video model is the teacher, rather than two branches competing inside one attention pool.

**Cross-page effect 12 - NAVSIM FVD table.** 32.7 leads, with three caveats attached: the paper's own caption disclaims cross-setting comparison, its table omits PWM (85.95) and DriveDreamer-Policy (53.59) which are the wiki's existing NAVSIM entries, and generation resolution is never stated. Also recorded: the Stage-1 -> Stage-2 gain comes from replacing a text condition with a VLM hidden state, and **no Stage-1 FVD number appears anywhere in the paper**, so that gain is unquantified.

**New methods for the gap list**: **Uni-World VLA** (2603.27287, 89.4 PDMS - the strongest method in CoWorld-VLA's own v1 table), **ResWorld** (2602.10884, 89.0, already on the list), **WorldDrive** (2603.14948, 89.0), **LaST-VLA** (2603.01928, 87.3), **TrajDiff** (88.5), **ReSim** (86.6), **VGGT** as a planning-side teacher (already 16 mentions), **DrivingGPT** and **SVD** as NAVSIM FVD baselines.

**Pages updated**: `wiki/sources/coworld-vla.md` (NEW), `wiki/concepts/world-model-for-ad.md` (Pattern 34, `#entropy-assignment`, camp list, NAVSIM FVD table), `wiki/concepts/navsim-benchmark.md` (v1 + v2 rows, `#both-columns` rewritten, correction-delta verification, provenance note, seed-variance section), `wiki/concepts/foundation-backbones-for-ad.md` (`#factorial`, `#target-role`, fifth Wan coupling row), `wiki/concepts/chain-of-thought-for-ad.md` (fifth substrate + design-space row), `wiki/concepts/diffusion-planner.md` (`#expert-ensemble`), `wiki/concepts/adaptive-routing.md` (`#static-mixture` + open question), `wiki/index.md`, `README.md`.

**Figure note**: all seven referenced figures embedded (introduction.png, overview_new.png, case1_main.png, traj_best.png, case1.png, case4.png, traj_2.png) and all seven tables reproduced in full. Figures 5 and 7 are multi-panel in the original and only panel (a) of each is present in the clipping; `case1_baseline.png` exists in raw/assets but is referenced nowhere. All qualitative comparisons in the paper are Stage-1-vs-Stage-2 or Stage-2-vs-Stage-3 - there is no qualitative comparison against any external method.

---

## 2026-09-14 - Ingest: Drive-HWM: Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving

**Source**: `raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md` (arXiv 2609.03572v1)
**Orgs**: not stated in the clipping; funding acknowledges Beihang University (Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing) plus a National S&T Major Project and Beijing NSF grants. Zhaoxin Fan, Tianbao Zhang, Wenjun Wu, Xiaofeng Wang, Yeying Jin, Jian Zhao, Zheng Zhu, Shuicheng Yan.
**Pages created**: `wiki/sources/drive-hwm.md`
**Confidence**: **medium** - and this is the first ingest to carry that rating for a reason internal to the paper rather than to the clipping.

**What it is**: two world models on two clocks, with two different prediction targets. A V-JEPA-family slow model fires every N=8 steps and predicts K=8 future **optical-flow** fields in parallel from one history; the hidden state behind that flow decoder (the Dynamic-Aware Latent) is FiLM-injected into an Emu3-8B fast model that predicts the **next RGB frame plus the immediate action** at every step. Single front camera at 256x144, no RL, no scorer, no anchors, no BoN. Slow branch 25.6 ms amortized to 3.2; fast branch 81.6 ms.

**Finding 1 - the paper reports two mutually inconsistent result sets, and this is the dominant fact about it.** **Every table says 93.8 PDMS / 86.4 EPDMS. Every sentence says 93.3 / 86.2.** Systematic across five passages and both benchmarks, with a complete internally-consistent submetric row behind each. The prose set is recoverable because Sec. IV-B states its v2 results only as deltas against DriveVLA-W0: "+1.4 DDC, +1.7 EP, +1.5 LK, +27.5 EC" reconstructs 99.4 / 88.1 / 94.7 / 86.4 against the tables' 99.4 / 88.5 / 96.5 / 87.5, and the same passage then writes "a comparable EP of 88.1." The v1 set reconstructs identically from "+0.2, 1.0, 1.0, 0.1, 0.2, 0.3."

Two consequences, neither addressed by the paper: **the v1 rank is undecidable** (93.8 would lead the wiki 0.1 above CLEAR/DA-WAM; 93.3 ties Drive-JEPA fourth), and **the paper's own mechanism is worth +0.3 or +0.8** since Table IV's Fast-only row is 93.0 under either reading - a factor of 2.7. The prose picks the smaller figure and calls it validation. Most likely mechanical explanation: tables refreshed after a final run, prose never updated. The abstract makes no numerical claim at all.

**Finding 2 - the target-role split, which is the paper's real and transferable contribution.** It is the first paper here to ablate the prediction target **separately for two branches in one codebase**, and the ordering fully inverts: **slow branch flow > depth ~ BEV > RGB; fast branch RGB 93.8 > depth 93.6 > flow 93.5 > none 93.1.** The principle - far futures are dominated by motion so predict motion, near futures by what is visible so predict appearance - is **orthogonal to the entropy rule** the wiki assembled from WA-JEPA and DriveFuture, which governs the *form* of the loss rather than the *content* of the target. **Target content by horizon; objective form by entropy. Nobody has crossed them**, and the decisive sweep is one run (long-horizon flow under a generative objective versus a regression one). Caveats recorded: the slow-side numbers live in Fig. 4, whose image is missing from the clipping; the fast-side spread is 0.3 across the top three; role is confounded with backbone, horizon, loss form and dataset stage.

**Finding 3 - the central claim cannot be exercised on the only benchmark.** The fast branch exists so "decisions [can] be continuously updated as new observations arrive" - a closed-loop property. **NAVSIM is non-reactive and single-shot**: one observation in, one 4 s / 8-pose trajectory out, no future observation ever delivered. With N=K=8 at that convention the slow model fires **exactly once per evaluated scenario** and the fast model sees one frame. So Table IV measures a conditioning stream, not a schedule. Compounding this, **what one action is is never defined** - L_a is never given - and the two readings are mutually exclusive with different parts of the paper: if A_t is the whole trajectory, the "one-step" framing is an inaccurate description of the model; if A_t is one pose, then steps 2-8 must consume the model's own predicted frames, contradicting "future image generation can be omitted," and the per-trajectory cost is ~678 ms rather than 84.8, making Drive-HWM **5.8x slower** than the baseline it reports beating by 28%. Table III's arithmetic presumes the second reading; its headline comparison presumes the first.

**Finding 4 - Drive-HWM supplies the video-pretrained non-JEPA control that `foundation-backbones-for-ad.md` has been asking for since the WA-JEPA ingest.** That page flags a confound in both encoder sweeps: every alternative WA-JEPA and Drive-JEPA test is *image-level*, so "the JEPA objective transfers" and "video pretraining transfers" are not separated. Drive-HWM's slow-slot swap is **all-video** - V-JEPA 93.8 > WAN 93.2 > CogVideo 93.0 - so it isolates the objective family with temporal pretraining held fixed. **+0.6 to +0.8 for joint-embedding prediction over generation, against the +2.7 to +5.7 the encoder sweeps attribute to V-JEPA 2 over image-level alternatives.** If both hold, most of the larger gap is video pretraining, not JEPA - the reading that page had already flagged as the unsupported part of WA-JEPA's claim. Four caveats recorded: wrong slot (predictor, not encoder), no parameter matching, 0.8 spread on single runs, and the backbone's identity is ambiguous (see below).

**Finding 5 - the conditioning interface outweighs the hierarchy it conditions.** Table VII is the cleanest sweep in the paper: concatenation 92.5 < cross-attention 93.0 < gated cross-attention 93.1 < AdaLN 93.3 < **FiLM 93.8**. A 1.3 spread - larger than the slow-fast hierarchy under either result set, and larger than the fast model's backbone spread (0.5 across Emu3 / LLaVA-OneVision / Qwen). **Both this and BrainWAM put concatenation last and both conclude the pretrained backbone's token layout should not be disturbed**, reaching that by opposite mechanisms (FiLM channel modulation vs. an 8-token gated cross-attention bottleneck). Two independent architectures now agree that a world-model branch should reach a pretrained multimodal policy through a narrow, non-token channel.

**Finding 6 - a seventh baseline-provenance failure mode, plus the first direct evidence that the shared v2 block is copied.** Drive-HWM is the **seventh** user of the shared block and carries all six rows digit-identically in **one unlabelled column**. The new evidence: it prints the DriveSuprim row under the name **"DriveSupervisor"** while citing DriveSuprim's own arXiv entry (2506.06659). A paper that recomputed that row would know what it ran. Six tables of digit-identity had made copying the obvious inference; this is the mechanism. Separately, **two of Table IV's four ablation PDMS values equal DriveVLA-W0's two published Table I rows** - Fast-only 99.3/97.4/93.0 matches the oracle BoN-6 row on all three metrics, Slow-only 90.2 matches the anchors row on PDMS with different submetrics. No ablation anywhere in the paper is stated to be a retrained model. Note the direction: if Fast-only really is the BoN-6 row, the paper **understates** its own mechanism.

**Finding 7 - a within-table audit this wiki had not been running.** Applying the paper's own stated EPDMS formula to its own Table II: peers cluster at +1.3 to +2.5 residual, Drive-HWM's own row is **+5.25** - more than double the largest peer. Without any formula: its submetrics beat DriveVLA-W0's on **all nine**, including **EC by 28.6** (a 2/16-weight metric, so worth about +3.4 alone), and the aggregate moves **+0.3**. This is *not* inflation - the residual points the other way, and the same check on Table I gives it the *smallest* PDMS residual in that table (-0.33 against a -2 to -4 field). What it shows is that **submetrics and aggregate can fail to be mutually consistent inside one published table**, which cross-paper audits cannot see. Logged on `navsim-benchmark.md` as a check worth running on the other six block-derived tables; it is arithmetic on numbers already transcribed.

**Internal contradictions recorded**: Sec. III-D redefines the fast loss as a latent regression one page after Sec. III-C defines it as a token-level NLL, and describes the whole objective without mentioning optical flow - almost certainly residue from an earlier latent-regression version. Sec. III-B says "the slow branch does not output actions or an explicit trajectory"; Table IV reports a Slow-only row at 90.2 PDMS. Fig. 6's failure analysis rolls the next-frame predictor forward four steps under two action conditionings, which is the capability Sec. III-C says can be omitted. Fig. 2's caption says VL-JEPA, Table V says V-JEPA, the prose says V-JEPA, and reference 55 is VL-JEPA (2512.10942) - recorded as **V-JEPA-family, variant undetermined**. Table V's "Qwen2.5-VL" row cites the Qwen3-VL report (2511.21631).

**Unspecified**: parameter counts for every component; lambda_img, lambda_s, lambda_f, w_k, rho, L_a, L_v, d; **which optical-flow estimator produces every slow-model target**; which layers FiLM modulates; DAL token count; GPU-hours; code. Also no FID / FVD / flow EPE / any generation metric despite two generative objectives, and no navhard / HUGSIM / Bench2Drive / nuScenes / reactive evaluation. Emu3-8B is described as a "lightweight multimodal backbone."

**Credit where due**: Table I marks **both** oracle Best-of-6 rows (AutoVLA 92.1, DriveVLA-W0 93.0) with an explicit footnote - the discipline WCog-VLA lacked - so Drive-HWM's unmarked single-sample row standing above two oracle-6 scores is a stronger claim than the paper makes. NC 99.6 (99.5 in prose) would be the wiki's highest v1 NC either way (above WA-JEPA's 99.5), from one front camera at 256x144 with no RL and no scorer; TTC 98.5 ties WCog-VLA for second behind DriveVA's 98.7, and DAC 99.0 is second behind DriveVLA-W0's 99.1. The parallel multi-offset flow formulation is the right structural answer to rollout drift. And **Table VI is the wiki's second linear-probe diagnostic** (after HybridDriveVLA's CKA/CCA/SAE) - though both its probe tasks are motion tasks and the probed latent was trained on motion, which makes the result close to definitional; the informative probe is an off-objective one.

**Also the first user study in this wiki** - 20 raters, 50 paired scenarios, 68% preference over DriveVLA-W0 with higher means on all four axes. No blinding protocol, no inter-rater agreement, no statistical test, authors' own study, and the four axes duplicate PDMS submetrics - so it is closest to a confirmation that the metric gaps are *perceptible*.

**Cross-page effect 1 - Pattern 35 on `world-model-for-ad.md`**, anchor `#two-rates`. The rate axis is new: Patterns 1-34 all run world model and planner on one clock. Adds **dense optical flow as a target family** (annotation-free like GeoWAM's point maps, but *not* in the action's coordinate frame), sub-sections `#target-by-horizon` for the two composable rules and `#flow-vs-geometry` for the tension below, and **a fourth position in the training-time/inference-time taxonomy**: both objectives are training-time-only *and* confined to nuPlan pretraining, yet the whole slow predictor runs at inference with only its flow decoder dropped. CoWorld-VLA keeps a supervised token; Drive-HWM keeps a supervised predictor.

**Cross-page effect 2 - a real tension with the geometry papers, unremarked by anybody.** GeoWorldAD prices the coordinate frame at **+2.5 PDMS on its own** and GeoWAM's whole thesis is that the world-model target should share the trajectory's frame. **Optical flow does not** - it is 2D image-space displacement. Yet Drive-HWM's probe puts its flow latent **first** on future-ego-motion (83.7) and an ego-frame BEV latent **last** (71.2). Three candidate readings recorded, none tested; no paper has compared a flow target against a metric-geometry target under a fixed planner, and it is one sweep in either codebase.

**Cross-page effect 3 - `#temporal-role` on `dual-system-vla.md`, a third axis.** Scene difficulty (AutoVLA, AdaThinkDrive, DeepSight, CLEAR), module cost (DualDriveVLA, DriveWAM, AutoMoT), and now **temporal role at a fixed unconditional period**. More importantly it is **the first entry on that page whose slow system emits no decision** - a forecast, not a meta-action or an intent anchor - which dissolves the VLM-action consistency problem that page is organized around and replaces it with "is the forecast used at all," answered by the DAL-conditioned next-frame loss.

**Cross-page effect 4 - `navsim-benchmark.md`'s non-reactive section gets a sharper statement.** The page said scores "may not reflect performance in fully interactive scenarios." It now says the benchmark is **single-shot**, so any mechanism whose benefit is re-deciding on fresh evidence is **structurally unmeasurable** rather than merely under-measured, with Drive-HWM as the concrete case.

**Cross-page effect 5 - the "mechanism is the smaller term" streak reaches five.** DA-WAM +0.15, DriveFuture +3.7 of 55.5, CoWorld-VLA's dynamic expert never ablated alone, ReWorld's video-side objective never measured for planning, and now Drive-HWM's hierarchy at +0.3-to-+0.8 against 1.3 for its conditioning mechanism and 0.7 for its auxiliary RGB loss.

**New methods/terms for the gap list**: **DriveSupervisor** (a renaming of DriveSuprim - watch for it propagating), **VL-JEPA** (2512.10942, vision-language JEPA, distinct from Meta's V-JEPA 2), **Copilot4D** cited as a US patent application rather than the ICLR paper, and DriveSuprim appearing at 83.1 within the shared block (a value this page already carries).

**Pages updated**: `wiki/sources/drive-hwm.md` (NEW), `wiki/concepts/world-model-for-ad.md` (Pattern 35 `#two-rates`, `#target-by-horizon`, `#flow-vs-geometry`, target-table row, camp list, three open questions), `wiki/concepts/navsim-benchmark.md` (v1 + v2 rows, `#drive-hwm-caveat`, `#block-copying`, `#submetric-residual`, non-reactive section), `wiki/concepts/dual-system-vla.md` (`#temporal-role`, comparison-table row, two open questions), `wiki/concepts/foundation-backbones-for-ad.md` (`#jepa-vs-generative`, `#vjepa-naming-ambiguity`, fast-model swap, SimWAM section retitled), `wiki/index.md`, `README.md`.

**Figure note**: six of seven figures embedded (`firgure_teaser.png`, `framework 2.png`, `vis_new.png`, `vis1_new.png`, `failed.png`, `user.png`) and all eight tables reproduced in full. **Figure 4 - the slow-model prediction-target sweep, the evidence for the paper's most transferable claim - is referenced twice and captioned but has no image in the clipping**, so only the prose ordering survives. Author photograph links are dead external URLs.

---

## 2026-09-18 - Ingest: Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving

**Source**: `raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md` (arXiv 2609.00111v1)
**Orgs**: Qwen Team (Alibaba) + Huazhong University of Science and Technology. Xin Zhou, Zongchuang Zhao, Zhibo Yang, Mingsheng Li, Humen Zhong, Shuai Bai, Dingkang Liang, Xiang Bai, Dayiheng Liu, plus seven contributors.
**Pages created**: `wiki/sources/qwen-drive-1.0.md`, `wiki/concepts/general-capability-retention.md` (NEW), `wiki/concepts/alpasim-benchmark.md` (NEW)
**Confidence**: **high** - the clipping is complete, the tables are internally consistent, and every arithmetic claim checked reproduces.

**What it is**: the wiki's 74th paper and its **first released foundation model rather than a planner** - 4B weights and code are public (`QwenLM/Qwen-Drive-1.0`). Qwen3.5-4B is kept architecturally unchanged (view/frame tags use ordinary vocabulary; no special tokens, no MoT, no FFN replacement) and two external modules attach: a **BEV perception head** reading pre-VLM encoder features (lifted by an unsupervised depth view transform) plus post-VLM image-token features, and a **1.1B 32-layer flow-matching Planning Expert** conditioned on the **cached post-RoPE keys and values of all eight GQA-softmax VLM layers**, four expert layers per cache. Four stages: head-only, joint perception+VQA with the encoder and VLM unfrozen (head LR 20x the VLM's), Planning Expert on frozen representations, then RL on the Planning Expert alone. 5.0B total, 50 waypoints at 10Hz over 5s, 10-step Euler.

**Headline**: 43.95 mAP / 60.99 map mIoU nuScenes and 43.45 / 71.27 OpenScene; driving-VQA avg 69.43 vs. base 63.52; **general VLM avg 66.41 vs. base 67.40**; **90.7 PDMS / 91.4 BoN-6** NAVSIM; **7.91 RFS** WOD-E2E test; closed-loop AlpaSim.

**Finding 1 - the general-capability retention measurement, and it is the contribution.** Fifteen public VLM benchmarks, thirteen models, one near-deterministic protocol (top-k 1, top-p 0.001, T 0.01), "-" scored as zero. Qwen-Drive loses **0.99** on the 10-benchmark knowledge group and **gains 0.97** on the 5-benchmark spatial group. Every specialized comparison model has collapsed: **Alpamayo-1.5-10B 14.65** (MMBench 7.51, OCRBench 3.20), **MiMo-Embodied-7B 26.53** (three unparsable), **UniDriveVLA-8B 48.73**, Cosmos-Reason2-32B 60.93. Across all 15 settings Qwen-Drive is 62.26 against the 32B Cosmos-Reason2's 56.78.

**This forces a scope onto AutoMoT's "do not fine-tune the VLM."** Qwen-Drive fine-tunes *more* - encoder and full VLM, under perception gradients that traverse the language model - and loses 0.99 where AutoMoT's fine-tuned expert lost 35-44% on multi-step reasoning. **The distinguishing variable is the mixture (26% general-purpose VL, 31% after repetition), not the freezing.** That reading is one data point from the lab that owns the base model, and the mixture-ratio sweep that would make it a rule has not been run - logged on the new page as the cheapest high-value experiment available. Recorded confound: Alpamayo-1.5's 7.51 MMBench is a **parse failure**, not a capability score, so the table measures *deployable* general capability - knowledge and interface conflated. The paper's cockpit-integration argument accepts that conflation deliberately, and it is the first **falsifiable** motivation for retention this wiki has seen (one SoC, one model or two) against the untestable OOD-robustness argument everyone else uses.

**Finding 2 - the BEV head is a probe, and it reports that vision-language features do not contain driving 3D structure.** A **converged** head on frozen SigLIP-Qwen features reaches 35.60 mAP against a dedicated BEVFormerV2* on the *same features* at 41.94 - **-6.34 mAP, -6.91 RayIoU**. Unfreezing the encoder and VLM recovers **+10.46 mAP / +9.84 map mIoU**, and the paper closes the obvious alternative explanation ("the head-only model had already converged"). This is the VLM-side entry in the adaptation ladder that previously held Adaptive-WAM (frozen Wan 84.20 -> joint LoRA 90.62) and Latent-WAM (LoRA collapses geometric distillation). **Three backbone families, one answer.** The same table also gives the wiki's cleanest **encoder swap**: replacing ResNet-50 with the Qwen-initialized SigLIP tower is worth **+6.6 to +10.1 mAP** to three separate detectors under a fixed schedule.

**Finding 3 - the paper prices its own driving adaptation at +0.08 RFS, and this is the largest investment-to-effect gap the wiki has recorded.** Table 8 trains a matched Planning Expert on three Stage-2 variants: unadapted Qwen3.5-4B **7.88**, +driving VL **7.91**, +3D perception **7.96**. The same driving VL supervision moves driving QA +6.55 and CoC reasoning +38.39. So 24 public datasets, 3.09M filtered VQA pairs, 1.54M curated samples and a full 3D perception stack buy **+0.03 and +0.05 RFS** for planning, against **+0.50** for RL on the action head alone. Scoped correctly: Stage 3 freezes the VLM, so this bounds what adaptation delivers *through a fixed conditioning interface*. **Driving knowledge and driving competence are separate acquisitions**, and the wiki's "mechanism is the smaller term" streak reaches six - with the twist that here the small term is not the paper's headline, so nobody had an incentive to inflate it. Also visible in the same table: **3D perception supervision costs 0.64 driving-QA and 0.92 general-VQA points.** Explicit 3D is not free and the price is paid in language.

**Finding 4 - the first head-to-head closed-loop reproduction of NAVSIM leaders, and the ordering nearly inverts.** AlpaSim, 916 scenarios on PAI-AV-NuRec v26.02 with novel-view synthesis, six methods, all reproduced by this paper. At-fault AlpaSim score: **Alpamayo-R1 0.58 > DriveWAM 0.53 > Alpamayo-1.5 0.45 > Qwen-Drive-RL 0.37 > SimWAM 0.30.** NAVSIM order: SimWAM 91.5 > Qwen-Drive 90.7 > DriveWAM 90.1, with both Alpamayo variants reporting no NAVSIM number at all. **The wiki's best world-action model is last and the best closed-loop method is not on the leaderboard.** The paper supplies the two behavioural diagnoses that make the table readable: DriveWAM is **degenerately cautious** (35% progress, at-fault score flattered because the metric divides distance by events, 56% all-event close encounters from being struck behind), SimWAM **degenerately aggressive** (62% progress, 22% at-fault, 19% off-road). Rule adopted: an AlpaSim score is uninterpretable without progress and both close-encounter rates. Caveats: third-party reproductions, single runs, protocol chosen by the reproducing party.

**Finding 5 - RL moves closed-loop behaviour in the opposite direction from how it moves NAVSIM.** The same Stage-4 update raises NAVSIM ego progress 82.4 -> 84.8 (the paper: "particularly effective at reducing conservative, low-progress behavior") and in AlpaSim **halves off-road 24 -> 12%** and raises at-fault score 0.27 -> 0.37 **while dropping progress 54 -> 48%** and raising all-event encounters 38 -> 41%. Progress-positive under the non-reactive scorer, progress-negative under the reactive one. The paper does not remark on it. **Open-loop reward shaping does reach closed-loop behaviour; the sign does not carry.**

**Finding 6 - the PhysicalAI-AV ordering reverses.** That page carried DriveWAM's own curated 1,000-clip subset: DriveWAM 0.47 ADE@3s, Alpamayo-1.5 0.80. On the official 644 split reproduced here: **DriveWAM 0.67, Alpamayo-1.5 0.35** - a ~3x swing in relative terms. Different splits, so not a contradiction of one number; it is the "no standard public test protocol" caveat realized. Qwen-Drive also publishes the first **leakage-free 700-frame subset** curated from held-out test clips (the standard split "overlaps with publicly available training data by official construction"), and contamination turns out to be real but small: **0.04-0.06 m uniformly**. Its second data-scaling study on this benchmark (0.17M -> 1.38M, ADE 1.34 -> 1.05, FDE 4.18 -> 3.24) is the **third unsaturated real-log scaling curve** in the wiki.

**Finding 7 - the RL machinery is the most engineered in the wiki and the most reusable.** A deterministic Euler flow has no transition density, so: **(a)** noise is injected only over the last 3 of 10 steps ("perturbations near t=1 affect the emitted trajectory more directly"), **(b)** exploration is confined to the **first 6 orthonormal cosine modes** over 50 waypoints, because "independent waypoint noise primarily introduces high-frequency jitter rather than meaningful maneuver diversity" - about 1.7 m longitudinal and 0.26 m lateral per stochastic step under the 165 m / 25 m normalization, with the likelihood surrogate evaluated in the 6 mode coefficients rather than 150 dimensions, and **(c)** an approximate restoring score derived from the endpoint-substituted Gaussian conditional, which the paper honestly labels approximate because the perturbation is not isotropic. G=8 rollouts share one initial noise and differ **only** by sampled reasoning trace and low-frequency perturbation; gamma=0.6 discounts toward the output; one on-policy update per group, so no importance ratio, no clipping, no KL. **This independently explains SimWAM's random-noise-vs-SDE ablation** (random noise gave the best EP 88.0 but cost 0.7 NC / 1.4 TTC): isotropic waypoint noise is high-frequency content no driving metric rewards. Three papers now shape exploration noise - DiffusionDriveV2 multiplicatively, SimWAM via a marginal-preserving SDE, Qwen-Drive by frequency band - and none has been compared with the others.

**Finding 8 - multi-source RL interference is small, and the shared displacement anchor is a priced trade.** One policy against three scoring functions: NAVSIM-only + ADE reaches 90.8, joint three-source + ADE reaches 90.7. **Cross-dataset interference ~0.1 PDMS when each source carries a task-aligned reward** - the wiki's only measurement of this. The shared ADE term costs **-0.23 RFS for -0.97 m of 5s ADE**; without it preference optimization drifts from recorded motion. The un-anchored variant hits **8.68 RFS in-sample against a human reference of 8.13**, which is the clearest available demonstration that in-sample RFS measures reward-model agreement rather than driving. Also flagged: the NAVSIM **reward** uses sub-weights EP 6 / TTC 4 / Comf 2 while evaluation uses the official protocol.

**Finding 9 - the base model is the story in the driving-VQA table.** **Unadapted Qwen3.5-4B scores 63.52 and beats every driving- or embodiment-specialized model evaluated** - MiMo-Embodied-7B 60.32, Cosmos-Reason2-32B 46.62, UniDriveVLA-8B 46.29, Alpamayo-1.5-10B 33.33 - and Gemma4-12B at 58.15. This bears on the Physical-AI-backbone claim the wiki recorded from Alpamayo-R1 (Cosmos-Reason beat Qwen2.5-VL-7B by 6.4 on zero-shot LingoQA): two generations later, under a third party's protocol, the plain Qwen model leads the Cosmos family by 14-28 points on driving QA. Caveat attached: the judge (Qwen3.5-Plus) shares a family with the winner.

**Finding 10 - the paper doubts its own reasoning gain, in print.** Reasoning is worth **+0.4 PDMS, +0.02 RFS, -0.01 m ADE**, from traces present on only 24.2% of planning samples. The limitations section: "the generated trajectory does not always adhere to the textual rationale... **part of this gain may stem from the additional model-internal information that the self-generated trace contributes to the conditioning context.**" Nothing in the objective requires trajectory-rationale consistency - unlike Alpamayo-R1's binary consistency reward, SpanVLA's alignment penalty, or ORION's latent alignment. **The decisive experiment is one run**: condition on a shuffled trace from a different scene. The second limitation is a schema gap no CoC dataset addresses: a trace names causes but does not order them in time ("a red light 20 m ahead" vs. "a child emerging 5 m ahead").

**Data-recipe results worth keeping.** **5.53M -> 3.09M public driving VQA samples survive a consistency check against their own source annotations - a 55.9% retention rate across 24 datasets.** Nearly half the public corpus is internally inconsistent; this is the measured version of what ReCogDrive and FLARE argue. The pipeline is two models in series (Qwen3.5-Plus rewrites to one schema, Qwen3.5-Flash checks against the source annotation - the step the rewrite cannot perform). CoC traces are written by Qwen3.7-Plus from a **rule-based motion prior** and audited by judges that **answer classification questions rather than emit scalar scores**, with programmatic aggregation, per-factor causal-role classification, and explicit rejection of future-information leakage. A **camera-ordering** task (surround views shuffled, view tags removed, recover the clockwise order) is a zero-annotation cross-view objective nothing else here uses. Perception unification: one head serving two **native** occupancy grids via a single differentiable trilinear sampling on the *predicted* features rather than resampling categorical labels; **the OpenScene RayIoU ray origin is raised 1.84 m** to match nuScenes LiDAR height, a protocol correction others should adopt. And **"we apply a factor-of-four scale correction to the raw accel_x and accel_y metadata"** in WOD-E2E, uncited.

**Limitations recorded**: no latency / throughput / memory / GPU-hours / per-module parameter counts anywhere - in a paper whose central argument is sharing one SoC between cockpit and driving; occupancy is a loss not a win (19.82 mIoU / 37.02 RayIoU vs. a dedicated detector's 25.72 / 43.89) and joint adaptation cannot move OpenScene occupancy at all (<0.3 points); the entire perception comparison is self-reproduced on self-defined 7-class labels with a modified NDS and **no external baseline whatsoever on the OpenScene half**; same-family judging touches LingoQA, PAI-AV-CoC and the data pipeline at once; the NAVSIM table stops at 90.4 (no DriveSuprim 93.5, CLEAR/DA-WAM 93.7, WCog-VLA 92.9, LWDrive 92.0, or SimWAM 91.5 - which it evaluates in closed loop two tables later); no NAVSIM-v2 / navhard / Bench2Drive / HUGSIM / nuScenes planning; **perception and VQA are reported only for the SFT model**, so no row confirms Stage 4 left them untouched; and the **unification thesis is never demonstrated** - each capability works, but the claim that sharing a representation improves all three is measured once, at +0.08 RFS, which the paper declines to attribute.

**Credit where due**: Table 6 separates RL from non-RL blocks, footnotes oracle best-of-6, and publishes **pre-RL AutoVLA 80.5 and SpanVLA 82.1** against their published 89.1 and 90.3 - implying +8.6 and +8.2 for RL, against Qwen-Drive's own +2.5, and independently reproducing the wiki's observation that GRPO gains are inversely related to SFT quality. The in-sample nature of the 8.45 val RFS is stated by the authors, not inferred. The Alpamayo-1.5 CoC score is explicitly not claimed as evidence of weak causal reasoning. Both missing figures' key numbers appear in prose. And the model and weights are actually released.

**Cross-page effect 1 - two new concept pages.** `general-capability-retention.md` (the 15x13 table, the recipe, the data-mixture reading, the knowledge-vs-interface confound, and five open questions of which the first is the unrun mixture sweep) and `alpasim-benchmark.md` (metrics, the ratio-score trap, the two degenerate behaviours, the NAVSIM inversion, and the note that AlpaSim is no longer internal).

**Cross-page effect 2 - `vlm-domain-adaptation.md` gains its counter-position.** AutoMoT's frozen-VLM conclusion and UniDriveVLA's "MoT is necessary but not sufficient" both get scoped: retention is achievable with an **unchanged architecture and a fully unfrozen backbone** if a quarter of the mixture is general data. The 55.9% filter statistic and the classification-question audit are recorded there too, plus a new row in the strategy comparison table.

**Cross-page effect 3 - `perception-for-planning.md`'s central tension gets its missing measurement.** Options A and C both presumed a VLM holds spatial structure that QA supervision fails to express; the probe says the structure largely is not there. Also recorded: a perception head **the planner never reads** (the Planning Expert conditions on VLM attention KV, not on boxes or occupancy), which explains why its planning effect is +0.05 RFS against WCog-VLA's -3.3 PDMS for removing perception the planner *does* read. **Perception supervision helps planning when the planner consumes the perception output**; shaping a shared trunk alone is within noise. The missing arm - same BEV outputs fed to the same expert as explicit conditions - is unrun.

**Cross-page effect 4 - `foundation-backbones-for-ad.md`** gains two backbone-role rows, the encoder-swap table, the VLM-side adaptation ladder, and a note that the Physical-AI-backbone advantage recorded from Alpamayo-R1 has not obviously survived two model generations.

**Cross-page effect 5 - `navsim-benchmark.md`** gains the 90.7 row, a `#qwen-drive` section carrying the pre-RL AutoVLA/SpanVLA values, the paper's own scoring-function quote adopted verbatim, and the AlpaSim inversion as the first multi-method evidence that PDMS rank and closed-loop rank come apart.

**Cross-page effect 6 - `rl-for-ad.md`** gains `#subspace-exploration` and `#multi-source-reward`, a reward-table row, and the **fourth** instance of PDMS-reward RL on an already-safe policy buying progress and compliance rather than collision avoidance.

**Cross-page effect 7 - `diffusion-planner.md`** gains `#cached-kv` (four coupling designs compared, and the observation that a **frozen** backbone makes the KV cache a fixed conditioning interface) and `#x-prediction` - the wiki's first endpoint-parameterized flow planner, argued as robustness to heterogeneous multi-source recordings, with the parameterization ablation named as a one-run experiment nobody has run.

**Cross-page effect 8 - `chain-of-thought-for-ad.md`** gains reasoning-as-an-optional-condition (24.2% of samples, ablatable at inference by construction, reused as RL exploration diversity), the paper's own doubt about whether the trace is a rationale or extra conditioning capacity, and the multi-timescale causation gap.

**Cross-page effect 9 - `physicalai-av-benchmark.md`** is substantially rewritten: the standard split, the leakage-free subset, the ordering reversal, the diversity axis, the scaling curve, and its second open question marked **answered, negatively**.

**Cross-page effect 10 - `world-model-for-ad.md`** gains `#closed-loop-check`. The test-time-imagination thread has repeatedly noted that NAVSIM cannot exercise what a world model is for; the first reactive measurement does not rescue the case, but it does confirm the premise - **a 1.4-point PDMS lead carried no closed-loop information at all.**

**Also updated**: `nuscenes-waymo-evals.md` (a consolidated WOD-E2E RFS table, the in-sample-RFS rule, and a **438-vs-479 discrepancy** in the annotated validation-set count between DIAL and this paper), `best-of-n.md` (the smallest BoN gain in the wiki, +0.7 after RL against +1.1 before it - the inverse relationship seen *within* one method), `wiki/index.md`, `README.md`.

**New methods for the gap list**: **MindVLA-U1** (the strongest WOD-E2E competitor, 7.87 test RFS with RL, 8.20 val), **Cosmos-Reason2** (2B/8B/32B) and **Cosmos3-nano**, **MiMo-Embodied-7B**, **Alpamayo-1.5** (evaluated in four tables here and never ingested), **RAP-DINO**, **UniPlan**, **DiffusionLTF**, **Swin-Trajectory**, **HMVLM**, **dVLM-AD**, **Ego3D-Bench**, **VLADBench**, **SURDS**, **WaymoQA**, **PAI-AV-NuRec**, **AlpaSim** itself, and **nuScenes-OccNet** / **FlashOcc** on the perception side.

**Figure note**: all eleven available figures embedded (`intro.png`, `qwendrive_overview.png`, `head.png`, `training_recipe.png`, `occ_process_vis.png`, `data_analysis.png`, `perception_vis.png`, `drivevqa-vis1.png`, `planning_vis.png`, `navsim-rl.png`, `ood_perception_vis.png`) and all eight tables reproduced in full. **Figures 11 and 12 - the RL reward/mixture ablation and the PAI-AV scaling curve - are captioned with no image in the clipping**, but unlike Drive-HWM's missing Fig. 4 both are largely recoverable from prose and are tabulated on the source page from the stated values. **Table 3 arrives flattened**, its HTML column structure gone; the column mapping is recovered and verified two ways (every row average reproduces to the stated figure, and the prose's "best on MMStar and RealWorldQA" plus "first or second on 6 of 10" hold only under that mapping). Appendix B's six qualitative images are external arXiv URLs and were not downloaded; the prompts and raw responses survive as text and the MiMo-Embodied 4,000-character self-correction loop in Appendix C.1 is recorded.


## 2026-09-27 - Ingest: GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving

**Source**: `raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md` (arXiv 2609.15169v1)
**Authors**: Xiao Liu, Haoyu Li, Jianghao Leng, Lin Wang, Chao Sun. Organizations are not stated; compute is H20.
**Pages created**: `wiki/sources/grava.md`
**Pages updated**: `concepts/navsim-benchmark.md`, `concepts/rl-for-ad.md`, `concepts/chain-of-thought-for-ad.md`, `concepts/action-tokenization.md`, `concepts/perception-for-planning.md`, `index.md`
**Confidence**: **medium**. Tables are intact and the arithmetic checks out (+19.3% / +20.5% relative; 62,111 / 102,861 = 60.4%). However, the reasoning ablation is internal-only, five figures are missing, and the reasoning targets leak the expert trajectory.

**What it is**: the wiki's 75th paper. It is a purely autoregressive VLA (Qwen3-VL-8B, one front camera) that emits box-token-grounded reasoning followed by a primitive-plus-parameters action, which a parameter-free geometric decoder turns into waypoints.

**Headline**:
- **90.48 PDMS** on NAVSIM-v1, single greedy sample; pre-RL 82.10.
- Internal long-tail benchmark: KOC 92.3 / CDS 90.1, against action-only 77.4 / 74.8.
- Held-out QA: 6.86, against GPT-5.4 at 5.39.

**Finding 1 - the most graded CoT ablation in the wiki.** On the internal benchmark: action-only 74.8 → coarse 77.9 → grounded objects 85.5 → full GRA 90.1 CDS, which separates having reasoning, grounding it, and structuring it. Without RL, full GRA scores 73.1, below action-only with RL. Reasoning pays only after RL. The ablation has no NAVSIM counterpart.

**Finding 2 - editing the trace edits the action (0.438 → 0.823 normalized PDMS; win rate 3% → 55%).** This answers Qwen-Drive's doubt for this architecture. **Caveat recorded**: in Fig. 3 and the Appendix E transcript, the ego-decision target is wired from `trajectory.intent`, which is derived from the GT trajectory. Cognition QAs also say "the future trajectory shows…". The intervention therefore shows that the planner obeys the stated maneuver, not that the grounding produced it.

**Finding 3 - Active RL.** DAPO-style GRPO on scenes that are greedy-failing, best-of-8-recoverable and high-variance, re-screened every loop (26,151 → 14,398 → 4,676 cases; 21.8% Jaccard between loops). It is never compared against plain GRPO. RL gives +8.38, **mostly safety** (DAC +6.4, TTC +6.1). This is the mirror image of the four already-safe-policy cases, and the third ~+8 gain from a ~81 pre-RL level (after AutoVLA and SpanVLA).

**Finding 4 - the Executable Planner** (a mode-conditioned parametric vocabulary) beats direct waypoints by +3.25 PDMS. High- and low-reward samples usually share a primitive, so RL mostly tunes continuous parameters.

**Limitations recorded**:
- The +0.18 "best purely AR" margin comes from a table that omits DynVLA 91.7.
- "60% of demonstrations" counts only warm-up cases; RL and self-distillation touch the whole pool, and Table II's 415K planner examples are unreconciled.
- Screening shows a greedy training score of 0.91 against a navtest pre-RL score of 82.1; the checkpoint is unspecified.
- The QA benchmark is in-distribution, uses GRAVA's own taxonomy, and is scored by a text-only judge.
- The showcase transcript contradicts itself (vehicle/pedestrian flip; 15–20 m vs. 24.0 m).
- No latency, no NAVSIM-v2, no public closed loop, single runs, no code.

**Figure note**: 8 images embedded. Figs. 4, 5, 6, the density half of Fig. 7, and S3 have no image. The Fig. 7 caption is attached to the qualitative figure (`nous_qualitative_compact_draft.png`). In-text table and figure numbers are misreferenced ("Table V" is used for Tables III and IV). All tables (I–VI, S1, S2) are reproduced.

**New methods for the gap list**: Neuro-Symbolic Drive (rule-grounded faithful reasoning from executable planners), DriveAgent-R1, OpenDriveVLA, SimLingo, AlphaDrive, Gen-Drive, NuPlanQA.


## 2026-09-27 - Ingest: SUV: Future Scene Understanding as Video Generation for End-to-End Driving

**Source**: `raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md` (arXiv 2608.03084v1)
**Orgs**: Xi'an Jiaotong University, USTC, Yinwang Intelligent Technology, Fudan University. Code announced (`ASH-2046/SUV`).
**Pages created**: `wiki/sources/suv.md`
**Pages updated**: `concepts/world-model-for-ad.md` (new Pattern 36 plus `#navhard-access`), `concepts/navsim-benchmark.md` (v1 and v2 rows, corrected-cohort list), `concepts/navhard-ood-evaluation.md`, `concepts/nuscenes-waymo-evals.md`, `concepts/perception-for-planning.md`, `index.md`
**Confidence**: **medium**. The clipping and tables are complete, the numbers are internally consistent, and the v2 submetric residual is normal (+0.5). However, the baseline tables are selective, one "corrected" table mixes conventions, and everything is single-run.

**What it is**: the wiki's 76th paper. A Wan2.2-5B video expert generates RGB, segmentation, relative-depth and instance-track futures as four native video streams, with SAM 3 and DA3 as offline teachers. A 1B MoT action expert reads their latents during joint flow-matching denoising. One front camera, no RL, no scorer.

**Headline**: 91.0 EPDMS on NAVSIM-v2 (corrected), 36.9 on navhard, 7.94 RFS on WOD-E2E test, 90.8 PDMS on v1. At 2 steps, 288 ms on an RTX 4090.

**Finding 1 - the test-time imagination dispute gets a benchmark-dependent answer.** Future access is worth +0.3 on navtest (reproducing SimWAM's null) and **+4.1 on navhard**; with RGB-only supervision the figures are +0.9 and +4.5. Recorded in world-model-for-ad `#navhard-access` as the live hypothesis: access matters under observation shift. The deciding experiment is SimWAM's isolated mask evaluated on navhard.

**Finding 2 - structured future supervision helps without access** (+1.0 navtest / +2.3 navhard). Per-stream access removal on navhard: tracks −1.5, depth −1.1, segmentation −0.5.

**Finding 3 - native generation vs. generate-then-perceive is split.** Native generation wins on depth and loses on segmentation and tracks, measured against the teachers only.

**Placement**:
- v2 corrected: second, behind WA-JEPA 91.7, which SUV omits.
- navhard unscored: second, behind SpanVLA 40.1, which is omitted.
- WOD-E2E: 7.94 against Qwen-Drive's 7.91, which is omitted.
- v1: 90.8, below the uncited same-backbone SimWAM 91.5 and DriveVA 90.9.

**Limitations recorded**:
- The v2 table is captioned "corrected" but carries DriveSuprim at the pre-fix 87.1, and DriveVLA-W0 at 86.9, which matches no known value.
- The v1 table carries weaker variants of three baselines and has a Drive-JEPA TTC/Comf column swap.
- Future-stream metrics are measured as teacher agreement only; cross-stream consistency is never measured.
- Track IDs are encoded modulo 7 per class, and depth is clip-relative.
- There is no planning result for random initialization, and no latency for the cheap no-access variant.
- Deterministic re-evaluation (90.84 / 91.01 six times) is not training variance. Scene SD is 18.2, so the evaluation SE is about 0.16.

**Figure note**: teaser, pipeline, Fig. 3 and supplementary Figs. 2–3 are embedded. Supplementary Fig. 1 is an external URL. Supplementary Figs. 4–5 (horizon curves) have no image. All 8 main and 7 supplementary tables are reproduced. Prose table references are off by one ("Table 2" means Table 1; "Table 7" means Table 6).

**New methods for the gap list**: Metis (WAM, 89.5 / 90.3 top-6 v2), EponaV2 (88.9 v2, 36.1 navhard), SparseDriveV2 (90.1 v2), Fast-WAM, GeoSem-WAM, X-WAM, WAM4D, ImagiDrive, AutoDrive-P³ (90.6 v1), IRL-VLA, Poutine, Fast-dDrive, HMVLM.


## 2026-09-27 - Ingest: Metis: A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation

**Source**: `raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md` (arXiv 2606.15869v1)
**Orgs**: Fudan, Shanghai Innovation Institute, HKU, Tongji, Li Auto, HUST, Imperial College London, University of Surrey. Code: `LogosRoboticsGroup/Metis`.
**Pages created**: `wiki/sources/metis.md`
**Pages updated**: `concepts/world-model-for-ad.md` (Pattern 37; new `#mask-family`), `concepts/navsim-benchmark.md` (v1 and v2 rows; new `#residual-sign`; SUV residual sign corrected), `concepts/navhard-ood-evaluation.md`, `sources/suv.md` (residual sign fixed; Metis comparison; swap provenance), `sources/simwam.md` (backlinks), `index.md`
**Confidence**: **medium**. The mechanism ablation is clean, but the paper has two copied or mislabelled ablation tables, a probably cross-protocol headline comparison, and single runs.

**What it is**: the wiki's 77th paper. A Wan2.2-5B + ~1B MoT world-action model whose mask lets the future read the action but not the reverse, so video generation is only a training signal.

**Headline**: 89.5 EPDMS v2 (90.3 best-of-6), 32.2 navhard, 89.1 PDMS v1 (89.7 best-of-6), CityWalker MAOE 9.8° All, 147 ms at 2 steps on an RTX 4090.

**Finding 1 - the mask family.** SimWAM, Metis and SUV share a backbone and recipe and cover four visibility patterns. On navtest every mask is within 0.5; on navhard, future-reads-action gives +2.2 (Metis) and action-reads-future gives +4.1 (SUV). Metis's bidirectional mask is worst on navhard, which conflicts with SUV. Recorded hypothesis: the harm is the feedback loop. The missing experiment is named in `world-model-for-ad#mask-family`.

**Finding 2 - the residual's sign as a protocol fingerprint.** Two rows with known conventions flip sign as predicted: Human +4.1 pre-fix / −0.1 corrected; DriveFine +1.2 / −0.8. Metis's own row is at −0.6, among its table's otherwise pre-fix anchors, so its "+2.4 over VLA" is likely a cross-protocol comparison. Three exceptions are recorded, so this is a heuristic only.

**Finding 3 - navhard carries every mechanism claim.** The mask (+2.2), the video-prior scale (Wan2.1-1.3B −2.4) and SUV's access (+4.1) are all invisible on navtest.

**Corrections to earlier pages**:
- SUV's residual was written with the opposite sign convention (+0.5); corrected to −0.5.
- The Drive-JEPA TTC/Comf swap flagged on SUV is traced to Metis's Table 9.

**Limitations recorded**:
- Table 10's navhard triples duplicate Table 5's step rows, and Table 6 says "Wan2.2-14B" for the 5B model.
- The 8× speedup is really 2.9× (skipping video) × 3.3× (fewer steps).
- v1 "SOTA" relies on best-of-6.
- CityWalker is second overall to ABot-N0.
- navhard split size is 244/4,164 here vs. 450/5,462 in SUV.
- The "1,192-scenario navtrain subset" is unexplained.
- Real-robot results are qualitative only, and there are no video-generation metrics.

**Figure note**: 9 figures embedded; Fig. 3 (mask) has no image. The Fig. 7 caption says outdoor but the image is indoor. All 12 tables are reproduced (Tables 4/8 and 5/12 are merged on the source page).

**New methods for the gap list**: UniWorldVLA (89.4 v1), ImagiDrive, WorldRFT, World4Drive, ABot-N0, NavFoM, CityWalker, Fast-WAM.


## 2026-09-27 - Lint

**Structural checks**: 112 pages. No orphans; no broken wiki links; all frontmatter complete; every page in `index.md`; all embedded images resolve. `concepts/divergent-thinking-in-vlms.md` had inbound links only from frontmatter `related:` lists, so a body link was added from `rl-for-ad.md`.

**Stale claims fixed**:
- `navsim-benchmark.md`: the WAM-Diff rank list named CoWorld-VLA twice and predated SUV. Corrected to "tied seventh" in both places.
- `qwen-drive-1.0.md`: "the wiki's best world-action model" (SimWAM 91.5) contradicted DA-WAM 93.7. Scoped to "best video-prior WAM".
- `lwdrive.md`: "all eight highest non-BoN v1 entries train a scorer" is broken by Drive-HWM (no scorer, 93.8 / 93.3). Annotated.
- `world-model-for-ad.md` Open Questions: the frontier updated from DriveSuprim 93.5 to CLEAR/DA-WAM 93.7. The test-time imagination question marked partially answered (SUV/Metis, navhard). The open-loop vs. robustness question gains three same-backbone navtest-small / navhard-large effects. The WA-JEPA inference question now requires navhard.
- `README.md`: 74 → 77 papers; GRAVA / SUV / Metis rows added above the methodological block; SUV added under highest v2; the mixed-table count updated.

**Contradictions recorded, not resolved**:
- SUV (action reads future: +4.1 navhard) vs. Metis (bidirectional: worst on navhard). Hypothesis: feedback direction. See `world-model-for-ad#mask-family`.
- navhard split size: 244 / 4,164 (Metis) vs. 450 / 5,462 (SUV); SUV copies Metis rows.
- Residual-sign heuristic vs. WA-JEPA's protocol partition: four exceptions (DriveVLA-W0 86.1, DriveWorld-VLA 86.8, DiffusionDriveV2 87.5, DiffusionDrive 84.5).
- Metis's own row (corrected-looking) vs. its pre-fix baselines.

**Concept pages suggested** (by frequency across sources): latency and inference cost (49 source pages); flow-matching planners (29); single-run and seed-variance reporting (22); video-generation priors, i.e. Wan2.x backbones (13); attention masks in WAMs (12); reasoning faithfulness and interventions (9); teacher pseudo-labels (9).


## 2026-09-27 - New concept pages (from lint suggestions)

**Created**:
- `concepts/inference-latency.md`
- `concepts/evaluation-variance.md`
- `concepts/wam-attention-masks.md`
- `concepts/reasoning-faithfulness.md`
- `concepts/teacher-pseudo-labels.md`

**Not created, by design**:
- *Flow-matching planners*: already covered in depth by `diffusion-planner.md` (31 mentions); a cross-link to the step-cost section was added instead.
- *Wan2.x video priors*: covered by `foundation-backbones-for-ad.md#wan2.2-ti2v-5b`, which gains Metis and SUV rows and the navhard scale result.

**New syntheses surfaced while writing**:
- **WA-JEPA is a bidirectional mask with a stop-gradient on the future←action edge.** Metis's asymmetric mask deliberately *keeps* that gradient. The two papers imply opposite mechanisms for why coupling helps, and neither ran the stop-gradient ablation.
- **Hindsight annotation is the norm.** GRAVA, Qwen-Drive and Alpamayo-R1 all write the trace's decision with access to the recorded future. Consistency rewards (Alpamayo-R1, SpanVLA) therefore make traces reliable *verbalized actions*, not explanations.
- **DriveVLA-W0's latency is 74–240 ms by its own report and 690 ms as measured by SUV.** Treat cross-paper latency as incomparable.
- **navhard's scene-level standard error is plausibly ~1 point**, the scale of several recent navhard-only mechanism effects.
- **Step budgets are architecture-specific.** Metis and SUV are nearly converged at 2 steps; SimWAM collapses at 1 (68.9 PDMS).

**Backlinks added from**: `world-model-for-ad.md`, `navsim-benchmark.md`, `chain-of-thought-for-ad.md`, `perception-for-planning.md`, `diffusion-planner.md`, `foundation-backbones-for-ad.md`. `index.md` and `README.md` updated (38 concept pages).
