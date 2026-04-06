---
title: "Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2511.19221v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Jianhua Han <sup>1</sup>  Meng Tian <sup>1</sup> <sup>1</sup>  Jiangtong Zhu <sup>1</sup> <sup>1</sup>  Fan He <sup>1</sup>  Huixin Zhang <sup>1</sup>  Sitong Guo <sup>1</sup>  Dechang Zhu <sup>1</sup>  
Hao Tang <sup>1</sup>  Pei Xu <sup>1</sup>  Yuze Guo <sup>1</sup>  Minzhe Niu <sup>1</sup>  Haojie Zhu <sup>1</sup>  Qichao Dong <sup>1</sup>  Xuechao Yan <sup>1</sup>  
Siyuan Dong <sup>1</sup>  Lu Hou <sup>1</sup>  Qingqiu Huang <sup>1</sup>  Xiaosong Jia <sup>2</sup>  Hang Xu <sup>1</sup>  
<sup>1</sup> Yinwang Intelligent Technology Co. Ltd., <sup>2</sup> Fudan University Equal contribution.Corresponding author.

###### Abstract

Autonomous driving heavily relies on accurate and robust spatial perception. Many failures arise from inaccuracies and instability, especially in long-tail scenarios and complex interactions. However, current vision–language models are weak at spatial grounding and understanding, and VLA systems built on them therefore show limited perception and localization ability. To address these challenges, we introduce Percept-WAM, a perception-enhanced World-Awareness-Action Model that is the first to implicitly integrate 2D/3D scene understanding abilities within a single vision-language model (VLM). Instead of relying on QA-style spatial reasoning, Percept-WAM unifies 2D/3D perception tasks into World-PV and World-BEV tokens, which encode both spatial coordinates and confidence. We propose a grid-conditioned prediction mechanism for dense object perception, incorporating IoU-aware scoring and parallel autoregressive decoding, improving stability in long-tail, far-range, and small-object scenarios. Additionally, Percept-WAM leverages pretrained VLM parameters to retain general intelligence (e.g., logical reasoning) and can output perception results and trajectory control outputs directly. Experiments show that Percept-WAM matches or surpasses classical detectors and segmenters on downstream perception benchmarks, achieving 51.7/58.9 mAP on COCO 2D detection and nuScenes BEV 3D detection. When integrated with trajectory decoders, it further improves planning performance on nuScenes and NAVSIM, e.g., surpassing DiffusionDrive by 2.1 in PMDS on NAVSIM. Qualitative results further highlight its strong open-vocabulary and long-tail generalization.

## 1 Introduction

![[x1 6.png|Refer to caption]]

Figure 1: 1

Autonomous driving ultimately relies on precise spatial perception and the capability to reason about the environment to make appropriate control decisions [^3] [^4] [^5]. In practice, small geometric errors (e.g., detection bias, yaw drift, or BEV/occupancy mistakes) snowball into brittle decisions, especially under long-tail conditions (e.g., night, rain, and small or rare objects). Meanwhile, many recent VLA frameworks [^6] [^7] [^8] [^1] [^9] [^10] [^11] [^12] leverage VLM backbones to incorporate reasoning into driving control in complex scenarios. However, recent evaluations and benchmarks [^13] [^14] [^15] show that general-purpose VLMs still struggle in core spatial abilities (3D localization drift, temporal inconsistency and unreliable confidence), indicating that broad vision–language alignment does not guarantee geometric competence. This gap directly impacts closed-loop performance and stability on real-world routes and scenario-rich benchmarks [^16] [^17] [^18].

Many existing methods [^1] [^19] [^20] [^21] formulate spatial understanding as QA-style supervision (in Figure 1), e.g., asking “What is the distance to the moving object ahead?”. However, this provides only indirect localization signals, rarely yields persistent, localizable world states, and leads to duplicate detections with poorly calibrated confidence in crowded scenes [^22] [^23]. On the other hand, some methods [^2] [^24] [^25] [^26] abandon LLM-based architectures and adopt an encoder–diffusion-decoder pipeline to perform end-to-end (E2E) trajectory prediction directly. However, omitting explicit spatial task learning degrades E2E performance, and complex autonomous-driving scenarios also require the reasoning capacity of LLMs [^27] [^21] [^28]. These limitations motivate embedding explicit, persistent world states within a single VLM and jointly optimizing perception and trajectory to improve robustness—conceptually aligned with planning-oriented frameworks such as UniAD [^29], but instantiated within a VLM.

Therefore, we propose the Percept-WAM method, a World–Awareness–Action framework that embeds *world states* within a single VLM. Percept-WAM unifies 2D and 3D perception via two token families, World-PV (image-plane) and World-BEV (bird’s-eye view), each encoding metric coordinates together with calibrated confidence, providing localized, reusable evidence for downstream reasoning and planning. Specifically, Percept-WAM initializes from a pretrained VLM to preserve general capabilities (e.g., logical reasoning), and includes (i) a grid-conditioned prediction head that structures dense multi-object inference, (ii) an IoU-aware scoring objective that explicitly calibrates detection confidence, and (iii) parallel autoregressive decoding that maintains throughput without sacrificing stability. Furthermore, we introduce the World-Action tokens to align and fuse perceptual information, and a lightweight MLP decoder for accurate and efficient future-trajectory prediction. In summary, within a single backbone, the model can either output both perception results (e.g., 2D/3D bounding boxes) and trajectory, or only trajectory. We also equip Percept-WAM with efficient streaming inference technique to meet the demand of real-world applications.

In experiments, Percept-WAM matches or surpasses strong detector and segmenter baselines on downstream perception benchmarks, achieving 51.7 mAP on COCO [^30] 2D detection and 58.9 mAP on nuScenes [^18] BEV 3D detection, surpassing the 47.5/52.3 mAP of LMM-Det [^31] and PointPillars [^32], respectively. Planning performance on nuScenes and NAVSIM [^17] is further improved by equipping a query-based action decoder. It obtains 90.2 PMDS on NAVSIM, outperforming DiffusionDrive [^24] by 2.1, and exhibits a short frame latency of 707 ms via streaming inference. Qualitative evidence further shows robust open-vocabulary and long-tail generalization, underscoring enhanced perception-to-action capability within a unified World-Awareness-Action model.

We summarize our contributions as follows:

- Perception-enhanced World Tokens. Percept-WAM builds the first framework to seamlessly unify 2D/3D perception within a single VLM via *World-PV* and *World-BEV* tokens, which encode metric coordinates and calibrated confidence, yielding reusable, localizable world states for reasoning and control.
- Grid-conditioned Dense Perception. We introduce a novel grid-conditioned head, augmented with *IoU-aware* scoring and *parallel AR* decoding. This design significantly improves both accuracy and stability in long-tail, far-range, and small-objects perception.
- Perception-to-Action Paradigm. Beyond outperforming or matching SOTA detector/segmenter baselines on *nuImages* and *nuScenes*, Percept-WAM shows outstanding planning performance on *nuScenes* and *NAVSIM*, through the alignment of World-PV, World-BEV and World-Action tokens.

## 2 Related Work

![[x2 5.png|Refer to caption]]

Figure 2: The overall architecture of Percept-WAM. i) We use a pretrained VLM backbone to maintain general reasoning capability, ii) Percept-WAM unifies 2D and 3D perception via World-PV and World-BEV tokens. The learnable BEV-Level grid tokens implicitly model the mapping from PV features to BEV-space representations. iii) An Action Head is introduced to predict trajectories from world tokens via parallel decoding. An additional memory bank is introduced to support efficient streaming inference.

VLMs/VLAs and Spatial Grounding. Early foundation VLMs (e.g., CLIP [^33]) enable broad vision–language alignment and zero-shot transfer, and subsequent instruction-tuned architectures (Flamingo [^34], BLIP-2 [^35], LLaVA [^36]) improve few-shot adaptation with interleaved inputs. Yet recent studies report persistent gaps in metric geometry, relative spatial relations, and BEV/top-view reasoning [^15] [^37]. In parallel, Vision–Language–Action (VLA) models demonstrate that injecting action supervision into pretrained VLMs boosts embodied control (RT-2 [^38], OpenVLA [^39]), and driving-oriented VLM/VLA systems explore language-conditioned planning (DriveVLM [^40], DriveLM [^41]). These findings collectively motivate *encoding explicit geometry* rather than relying on QA-style supervision. Our Percept-WAM instantiates this by representing world states allowing the backbone to reason over spatially grounded evidence and to reuse tokens for downstream tasks.

BEV Perception and 3D Occupancy. BEV representations have become a de-facto interface for 3D understanding in AD: Lift–Splat–Shoot (LSS) [^42] lifts features to frustums and rasterizes them into BEV grids; BEVDepth [^43] improves depth quality with explicit supervision; and BEVSegFormer [^44] tailors deformable attention for BEV semantic segmentation. Beyond planar BEV, OccFormer [^45] pushes toward 3D semantic occupancy, enriching vertical structure and long-range context. For multi-sensor fidelity, BEVFusion [^46] disentangles camera–LiDAR fusion so the camera stream remains predictive under LiDAR dropouts. Percept-WAM packs spatial evidence as reconstructable World BEV tokens and parallel autoregressive decoding for 3D perception.

E2E AD Model and Closed-Loop Evaluation. Jointly optimizing perception and planning reduces error compounding. TransFuser [^47] integrates multi-modal attention for closed-loop control, while UniAD [^29] exemplifies planning-oriented full-stack learning. Surveys further codify challenges such as interpretability, world modeling, and causal confusion in E2E AD [^48]. Beyond open-loop L2 metrics, recent benchmarks move to scenario-driven closed-loop testing: Bench2Drive [^16] evaluates multi-ability behaviors and nuPlan [^49] establishes large-scale real-world planning protocols; NAVSIM [^17] enables data-driven pseudo-simulation at scale.

Reasoning VLM/VLA for Driving. Building on driving-centric VQA and language datasets, recent works focus on *reasoning* that spans perception, prediction, and planning [^50]. WOMD-Reasoning [^51] scales to millions of Q&As on interactive behaviors in the Waymo Open Motion Dataset. DRAMA [^52] localizes risk and provides natural-language captions for safety-critical cues. To explicitly structure multi-stage reasoning, DriveLM [^41] proposes Graph-VQA with a VLM-based agent. DriveCoT [^21] [^53] curates chain-of-thought traces in CARLA to enhance interpretability of E2E policies. Reason2Drive [^54] further targets interpretable, chain-based reasoning tasks tailored to AD. Percept-WAM *retains the general reasoning competence* of pretrained VLMs, while *strengthening spatial perception* via World-PV/World-BEV tokens.

## 3 Method

Overall Architecture of Percept-WAM: As shown in Figure 2, Percept-WAM contains: 1) the VLM backbone (i.e., InternVL2-8B [^55]) to maintain general reasoning capability, 2) the learnable BEV-Level grid tokens to implicitly model the mapping from PV features to BEV-space representations, and 3) an additional action expert head for efficient and accurate trajectory decoding. Note that these BEV-level grid tokens can be initialized from point cloud features produced by a pretrained LiDAR encoder [^32], if LiDAR modality is available.

Summary of the Tasks. As shown in Table 2, Percept-WAM accepts multi-view streaming video, LiDAR point (opt.), and textual queries as inputs. It supports PV Perception (2D detection, instance segmentation, semantic segmentation, mono-3D detection), BEV Perception (BEV 3D detection, BEV segmentation), and Trajectory Prediction.

### 3.1 World-PV: Perspective-View Perception

![[x3 4.png|Refer to caption]]

(a) Different Ways of Confidence-tuning Dataset Generation.

The perspective-view (PV) branch is a critical component of Percept-WAM for detecting objects across varying distances and scales. Distant vehicles and small or irregularly shaped objects are challenging to detect due to limited visual information. To overcome these difficulties, we support high-resolution input that preserves fine-grained details at large distances and efficient parallel AR decoding. This approach ensures high throughput while enabling reliable detection of objects under long-tail conditions.

As shown in Figure 2, to leverage the pre-trained VLM (i.e., InternVL [^55])’s understanding of image structure and semantics, the image inputs are first encoded by the VLM backbone of Percept-WAM. Then the obtained image features, denoted as World-PV tokens, are patchified into an $H{\times}W$ grid, and each grid location acts as a localized query for single-object perception. Inspired by UFO [^56], we construct the grid tokens by interpolating from the World-PV tokens, yielding fine-grained features tied to local image coordinates. Finally, each grid token predicts the bounding box or segmentation mask aligned with its coordinates, supervised by the corresponding ground truth.

#### 3.1.1 Task Formulation

2D Detection and Monocular 3D Detection. We retain the conventional language-based autoregressive (AR) decoding paradigm, serializing the output predictions into natural-language–like token sequences [^57]. Specifically for 2D detection in PV space, we formulate the output as

$cls$,<box> $x$,$y$,$w$,$h$ </box>,<conf> $s$ </conf>.

where $cls$, $s$ denote the object category and confidence score, $(x,y)$ and $(w,h)$ are the box center and size. For 3D detection, the output follows the same sequence format, while the <box> field is instead defined as $x,y,z,w,h,\ell,\theta,v_{x},v_{y}$, representing the 3D center $(x,y,z)$, box size $(w,h,\ell)$, yaw angle $\theta$ and horizontal and vertical velocities $(v_{x},v_{y})$. Following sequence-based detection methods [^58], continuous labels (e.g., coordinates and confidence) are normalized and discretized into integer bins, and supervised with cross-entropy loss [^58]. Since categories are provided as text, the detector naturally supports open-vocabulary detection [^59] [^60], which improves robustness in long-tail road scene understanding. Besides, Percept-WAM decodes predictions in parallel across grids, significantly enhancing inference efficiency without sacrificing perception accuracy.

2D Instance and Semantic Segmentation. Similar to UFO’s formulation [^56], we cast segmentation as feature retrieval task without adding new parameters: the model predicts $K{=}16$ <MASK> tokens, and we retrieve masks by dot-product similarity between World-PV tokens and the $K$ mask tokens. The generated mask is then interpolated to match the output resolution, with masks for all categories produced in a single forward pass.

High-Resolution Input. Following InternVL-style dynamic tiling [^61] [^55], we split high-resolution images into non-overlapping tiles, each encoded using shared ViT weights. The features of these tiles are then fused through global positional alignment, allowing us to preserve long-range detail while avoiding quadratic memory growth.

#### 3.1.2 IoU-based Confidence Prediction

Training–inference mismatch in MLLMs [^62] often yields duplicate boxes in perception tasks. Previous approaches, such as UFO [^56] derive box confidence from the softmax of class logits. However, large language models are systematically overconfident [^63]: softmax probabilities saturate even for ambiguous detections, yielding many false positives especially in duplicate scenes. Therefore, we add an *IoU-based confidence token* for each predicted box, conditioned on its label attributes (class/coordinates/size). This is aligned with prior quality-aware scoring [^64] [^65] and yields a more interpretable, localization-sensitive confidence.

- Confidence-tuning Dataset Generation. To support training with IoU-based confidence supervision, we build an auxiliary set with IoU score from the training set (in Figure 3(a)). We use the model from an intermediate training stage to inference on the training images, then the predicted boxes that match GT become samples paired with their IoU. We discretize IoU to $20$ bins for WAM’s token-learning scheme. Compared with random-offset synthesis from GT (which produces near-uniform IoU), the model-prediction distribution better reflects realistic confidence and reduces false detections. Refer to Section 4.3 for more details.
- Confidence Learning during Training. We mix GT data and the confidence-tuning data during training. As shown by the loss mask in Figure 3(b), for GT samples (IoU fixed to $1$), the model learns class and box without explicit IoU supervision to avoid collapse. For confidence-tuning samples, the model predicts IoU conditioned on the perturbed labels, and only the loss for confidence is applied.
- Confidence Computation during Inference. During inference, the final detection score is defined as the product of class confidence (the softmax of the class token) and the predicted IoU score, providing a more unified and interpretable reliability measure.

To summarize the PV perception losses, we adopt token-level cross-entropy on discretized labels for object detection, following Pix2Seq [^58], and a combination of cross-entropy, sigmoid focal loss [^66], and Dice loss [^67] for instance and semantic segmentation.

### 3.2 World-BEV: BEV-View Perception

![[x5 3.png|Refer to caption]]

Figure 4: Illustration of grid query tokens in dense prediction. Note that the grid tokens are interpolated from World-PV or World-BEV tokens to predict the matched bounding box.

3D spatial understanding is the cornerstone of reliable autonomous driving systems. To enhance the 3D perception capabilities of our Percept-WAM model, we explicitly integrate 3D object detection and semantic map segmentation tasks within the BEV representation space, enabling comprehensive scene understanding through multi-task learning. Specifically, we propose World-BEV tokens, which are a set of learnable query tokens that serve as the foundation for BEV perception. Similar to previous work [^46] [^68] [^69], we instantiate the BEV space as an $H\times W$ grid centered on the ego vehicle. Each token (i.e., grid cell) outputs a high-dimensional embedding, providing sufficient capacity to encode the spatial and semantic information, such as objects and map elements. These World-BEV tokens are then used to query the features of the World-PV tokens established in Section 3.1 via cross-attention, enabling the model to lift 2D evidence into a 3D BEV representation in a purely data-driven manner.

The design of World-BEV tokens has two major advantages: (i) Like World-PV tokens, World-BEV tokens can also be computed in a single forward pass in the prefilling stage, delivering high inference efficiency for trajectory prediction. (ii) World-BEV tokens can be easily adapted for seamless multi-modal sensor fusion. Under camera-only input, the word embeddings of these tokens are randomly initialized and optimized in the training stage. When other 3D sensor inputs, such as LiDAR, are available, we use an additional encoder to extract point cloud features and use them to initialize the World-BEV tokens. In particular, we first encode the point cloud with PointPillars [^32] and then downsample the features via PixelUnshuffle [^70] and an MLP layer. This strategy injects pretrained, metrically grounded 3D priors into the BEV representation, boosting geometric consistency and accuracy.

BEV 3D Object Detection. For the 3D object detection task, we represent each 3D bounding box as a plain-text sequence: cls, <box> x, y, z, w, h, l, $\theta$, $v_{x}$, $v_{y}$ </box>. The continuous value of each attribute is normalized and discretized into integers within $[0,1024)$. As shown in Figure 4, we uniformly sample grid queries from World-BEV tokens via bilinear interpolation. Then, by controlling the attention mask only to the relevant tokens (shown in Appendix A.1), the object proposals are predicted independently in a parallel AR decoding manner from the grid query token and World-BEV tokens.

BEV Map Segmentation. For the segmentation task, we reuse the PV segmentation formulation with 16 <mask> tokens (see Section 3.1.1 for more details). Similarly, we compute dot-product similarity scores between each World-BEV token and all <mask> tokens, and then interpolate the predicted mask to match the desired output resolution. It should be noted that different map categories may overlap (e.g., crosswalk is a subset of drivable area). Therefore, we cast map segmentation as independent binary segmentation tasks for each class. In order to train the BEV tasks, we use cross-entropy loss (CE) [^58] for BEV detection, and a combination of CE, sigmoid focal loss [^66], and Dice loss [^67] for BEV map segmentation.

### 3.3 From Perception to Action

![[x6 2.png|Refer to caption]]

Figure 5: Trajectory decoding. Four sets of point-level queries interact with different input modality information, and generate trajectory using MLP. 𝐐 ego \\mathbf{Q}\_{\\text{ego}}, pv \\mathbf{Q}\_{\\text{pv}}, and bev \\mathbf{Q}\_{\\text{bev}} are aligned with their corresponding modality tokens via attention masking, while full \\mathbf{Q}\_{\\text{full}} accesses all features to decode the final trajectory.

Table 1: Comparison of PV 2D and 3D perception performance on autonomous driving (AD) and general (Gen) datasets. Note that ‘Det’ and ‘Seg’ denote Detection and Segmentation tasks respectively. The model trained on both AD and the general dataset, shown in the last row of the table, is the final Percept-WAM model. Bold emphasizes top method; underline indicates the runner-up.

<table><tbody><tr><td rowspan="3">Method</td><td rowspan="3">VisualBackbone</td><td rowspan="3">LLM</td><td colspan="2">2D Det</td><td colspan="2">Mono3D Det</td><td colspan="2">2D Instance Seg</td><td colspan="3">2D Semantic Seg</td></tr><tr><td>nuImages</td><td>COCO</td><td colspan="2">nuScenes</td><td>nuImages</td><td>COCO</td><td>nuImages</td><td>ADE20K</td><td>COCOstuff</td></tr><tr><td>mAP</td><td>mAP</td><td>mAP</td><td>NDS</td><td>mAP</td><td>mAP</td><td>mIoU</td><td>mIoU</td><td>mIoU</td></tr><tr><td>Mask R-CNN <sup><a href="#fn:71">71</a></sup></td><td>RN101-FPN</td><td>–</td><td>–</td><td>39.8</td><td>–</td><td>–</td><td>–</td><td>37.1</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Pix2Seq v2 <sup><a href="#fn:72">72</a></sup></td><td>ViT-B</td><td>–</td><td>–</td><td>46.5</td><td>–</td><td>–</td><td>–</td><td>38.2</td><td>–</td><td>–</td><td>–</td></tr><tr><td>GiT <sup><a href="#fn:73">73</a></sup></td><td>ViT-B</td><td>–</td><td>–</td><td>46.7</td><td>–</td><td>–</td><td>–</td><td>31.9</td><td>–</td><td>47.8</td><td>–</td></tr><tr><td>DINO <sup><a href="#fn:74">74</a></sup></td><td>RN50</td><td>–</td><td>–</td><td>49.4</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Mask R-CNN <sup><a href="#fn:71">71</a></sup></td><td>RN50</td><td>–</td><td>47.8</td><td>–</td><td>–</td><td>–</td><td>38.6</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>FCOS3D <sup><a href="#fn:75">75</a></sup></td><td>RN101 w/FT</td><td>–</td><td>–</td><td>–</td><td>32.1</td><td>39.5</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Mask2Former <sup><a href="#fn:76">76</a></sup></td><td>Swin-B</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>52.4</td><td>–</td></tr><tr><td>DeepLab V2 <sup><a href="#fn:77">77</a></sup></td><td>VGG-16</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>47.7</td><td>33.2</td></tr><tr><td>Griffon-G-27B <sup><a href="#fn:78">78</a></sup></td><td>CLIP-ViT-L</td><td>Gemma2-27B</td><td>–</td><td>40.6</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Groma <sup><a href="#fn:79">79</a></sup></td><td>DINOv2</td><td>Vicuna-7B</td><td>–</td><td>43.6</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>VLM-FO1-3B <sup><a href="#fn:80">80</a></sup></td><td>DaViT-L</td><td>QwenVL2.5-3B</td><td>–</td><td>44.4</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>VisionLLM <sup><a href="#fn:22">22</a></sup></td><td>RN50</td><td>Alpaca-7B</td><td>–</td><td>44.8</td><td>–</td><td>–</td><td>–</td><td>25.2</td><td>–</td><td>–</td><td>–</td></tr><tr><td>LMM-Det <sup><a href="#fn:31">31</a></sup></td><td>OWLv2-L</td><td>Vicuna-7B</td><td>–</td><td>47.5</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>UFO-internvl2-8B <sup><a href="#fn:56">56</a></sup></td><td>InternViT</td><td>InternVL2-8B</td><td>13.7</td><td>48.9</td><td>–</td><td>–</td><td>13.1</td><td>42.6</td><td>8.9</td><td>54.5</td><td>30.2</td></tr><tr><td>Percept-WAM(2D AD)</td><td>InternViT</td><td>InternVL2-8B</td><td>46.7</td><td>–</td><td>–</td><td>–</td><td>36.8</td><td>–</td><td>64.7</td><td>–</td><td>–</td></tr><tr><td>Percept-WAM(AD)</td><td>InternViT</td><td>InternVL2-8B</td><td>49.9</td><td>–</td><td>32.6</td><td>38.0</td><td>41.7</td><td>–</td><td>63.9</td><td>–</td><td>–</td></tr><tr><td>Percept-WAM(AD+Gen)</td><td>InternViT</td><td>InternVL2-8B</td><td>49.6</td><td>51.7</td><td>33.0</td><td>38.6</td><td>41.7</td><td>45.9</td><td>62.8</td><td>54.3</td><td>50.3</td></tr></tbody></table>

By introducing World-PV and World-BEV tokens, our model gains strong 2D perception and 3D scene understanding capability. Building on this, we introduce World-Action tokens to enable future trajectory generation for autonomous driving. We adopted a query-based trajectory decoding approach, following the paradigm of imitation learning for training. To enhance the alignment between the World-Action Tokens and tokens of different modalities, we introduce four sets of point-level queries via parallel decoding under controlled attention mask for efficient training.

- Perception-Action Alignment. The BEV information (i.e., World-BEV tokens) captures accurate dynamic and static context, images (i.e., World-PV tokens) provide rich semantic details, and ego-state data supply vehicle kinematic information. Reliable future trajectory prediction requires features from all three perspectives. Therefore, as shown in Figure 5, we set up four sets of point-level queries: $\mathbf{Q}_{\mathbf{pv}}$, $\mathbf{Q}_{\text{bev}}$, $\mathbf{Q}_{\text{ego}}$ and $\mathbf{Q}_{\text{full}}$. The first three queries interact only with their corresponding modality features by controlling the attention mask, while $\mathbf{Q}_{\text{full}}$ has access to all features. This ensures that the action is fully aligned with different input features, avoiding over-reliance on any single modality.
- Parallel Trajectory Decoding. Each query set $\mathbf{Q}\in\mathbb{R}^{N\times C}$, where N represents the number of trajectory points and C is the dimensionality of the features, is randomly initialized. The query features are then encoded by Percept-WAM and decoded through an MLP to obtain the trajectory. During training, the four sets of queries are decoded in parallel to generate trajectories and supervised using Smooth-L1 loss [^81]. During inference, the trajectory decoded from $\mathbf{Q}_{\text{full}}$ is used as the final output.
- Streaming Inference. To further improve efficiency, we incorporate a streaming KV cache strategy into Percept-WAM. To mitigate distribution drift caused by training-inference paradigm mismatch, we adopt a longer-clip training scheme and a dual-recomputation KV cache mechanism. Details are provided in Appendix A.2.

Table 2: Summary of tasks and corresponding training datasets used in Percept-WAM training. Unless otherwise specified, we follow the official train-val-test dataset split.

<table><tbody><tr><td>Perception</td><td>Tasks</td><td>Datasets</td></tr><tr><td rowspan="5">PV</td><td>2D Det</td><td>nuImages <sup><a href="#fn:18">18</a></sup>, nuScenes <sup><a href="#fn:18">18</a></sup>, COCO <sup><a href="#fn:30">30</a></sup></td></tr><tr><td>Mono 3D Det</td><td>nuScenes, Waymo <sup><a href="#fn:82">82</a></sup></td></tr><tr><td>Instance Seg</td><td>nuImages, COCO</td></tr><tr><td>Semantic Seg</td><td><sup><a href="#fn:83">83</a></sup></td></tr><tr><td>Grounding</td><td><sup><a href="#fn:84">84</a></sup></td></tr><tr><td rowspan="2">BEV</td><td>3D Det</td><td rowspan="2"><sup><a href="#fn:18">18</a></sup></td></tr><tr><td>BEV Seg</td></tr><tr><td>Trajectory</td><td>Waypoint Pred</td><td>nuScenes, NAVSIM <sup><a href="#fn:17">17</a></sup></td></tr><tr><td>Driving QA</td><td>QA</td><td><sup><a href="#fn:41">41</a></sup></td></tr></tbody></table>

## 4 Experiments

### 4.1 Experimental Setup

Table 3: [^17]

<table><tbody><tr><td rowspan="2">Method</td><td colspan="4">nuScenes</td><td colspan="6">NAVSIM v1</td></tr><tr><td>L2-1s <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>L2-2s <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>L2-3s <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Avg.<math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>UniAD <sup><a href="#fn:29">29</a></sup></td><td>0.20</td><td>0.42</td><td>0.75</td><td>0.46</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>VAD-Base <sup><a href="#fn:93">93</a></sup> / VAD-v2 <sup><a href="#fn:94">94</a></sup></td><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:24">24</a></sup></td><td>0.27</td><td>0.54</td><td>0.90</td><td>0.57</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>DriveVLM <sup><a href="#fn:40">40</a></sup></td><td>0.18</td><td>0.34</td><td>0.68</td><td>0.4</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>BEV-Planner <sup><a href="#fn:95">95</a></sup></td><td>0.16</td><td>0.32</td><td>0.57</td><td>0.35</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>DRAMA <sup><a href="#fn:96">96</a></sup></td><td>–</td><td>–</td><td>–</td><td>–</td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:97">97</a></sup></td><td>–</td><td>–</td><td>–</td><td>–</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>Percept-WAM</td><td>0.17</td><td>0.35</td><td>0.63</td><td>0.38</td><td>98.7</td><td>97.8</td><td>93.2</td><td>92.8</td><td>84.4</td><td>88.6</td></tr><tr><td>Percept-WAM*</td><td>0.16</td><td>0.33</td><td>0.60</td><td>0.36 (+5.2%)</td><td>98.8</td><td>98.6</td><td>94.4</td><td>99.5</td><td>84.8</td><td>90.2 (+1.6)</td></tr></tbody></table>

We train the Percept-WAM from InternVL2-8B [^55] with various training datasets as shown in Table 2. The training process applies AdamW optimizer [^98] with the base LR $0.0002$ and weight decay $0.01$. We train the model in cosine decay learning rate schedule with linear warmup for $1000$ steps. Mixed precision [^99] and gradient checkpointing [^100] are enabled to save GPU memory. World-PV is discretized into a $10\times 10$ grid for both detection and segmentation, while World-BEV uses the $40\times 40$ grid for detection and a $10\times 10$ grid for segmentation to provide finer spatial granularity in BEV space. We adopt a two-stage curriculum that first consolidates spatial grounding for PV/BEV perception and then aligns the planner through E2E VLA fine-tuning. Further hyper-parameters and detailed data composition can be found in Appendix B.

### 4.2 Main Results

PV Perception Results. As shown in Table 1, Percept-WAM matches or surpasses specialist detectors and segmentors on both the *nuImages* [^18] and *nuScenes* [^18] PV tasks. For instance, it attains $49.9$ mAP in 2D detection and $41.7$ mAP in 2D instance segmentation compared to $47.8$ and $38.6$ mAP with Mask R-CNN [^71], and $33.0$ mAP in mono 3D detection versus $32.1$ mAP with FCOS3D [^75]. We observe a synergy between 2D and 3D PV detection, with 2D detection performance improving by $3.2$ mAP. This gain results from unified 2D and 3D modeling, as shown by joint training on AD datasets, where training across all PV tasks leads to consistent gains across benchmarks. Additionally, training on AD and general-purpose datasets achieves performance comparable to or better than specialist and multimodal large language models in both autonomous driving and general scenarios. Visualizations in Figure 7 demonstrate the excellent capability of Percept-WAM in detecting and segmenting multiple objects in complex scenarios.

Table 4: Results of BEV perception tasks on nuScenes val dataset, where Dri.→drivable area, Ped.→pedestrian crossing, Lane.→lane divider, Veh.→vehicle.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="2">Detection</td><td colspan="4">Segmentation (IoU)</td></tr><tr><td>mAP</td><td>NDS</td><td>Dri.</td><td>Ped.</td><td>Lane.</td><td>Veh.</td></tr><tr><td><sup><a href="#fn:69">69</a></sup></td><td>0.360</td><td>0.438</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td><sup><a href="#fn:68">68</a></sup></td><td>0.375</td><td>0.448</td><td>80.7</td><td>–</td><td>21.3</td><td>43.2</td></tr><tr><td><sup><a href="#fn:32">32</a></sup></td><td>0.523</td><td>0.613</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td><sup><a href="#fn:101">101</a></sup></td><td>0.526</td><td>0.630</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td><sup><a href="#fn:46">46</a></sup></td><td>0.685</td><td>0.714</td><td>85.5</td><td>60.5</td><td>67.7</td><td>–</td></tr><tr><td>Percept-WAM</td><td>0.589</td><td>0.645</td><td>87.0</td><td>70.9</td><td>62.7</td><td>60.2</td></tr></tbody></table>

BEV Perception Results. As shown in Table 4, without using sequential information, and with a relatively low image input resolution (i.e., $448\times 796$), Percept-WAM can outperform many specialist models on both detection and segmentation tasks. Specifically, for map segmentation, Percept-WAM can achieve superior performance with BEVFusion [^46] on drivable area and pedestrian crossing. Segmentation results on geographical train/val splits are summarized in Appendix C.1. For object detection, our method achieves an mAP of 0.589, which can outperform classical detectors such as PointPillars [^32] and SECOND [^101]. Qualitative results on nuScenes val set are shown in Figure 8. Please note that, rather than outperforming state-of-the-art methods on each sub-task, the main purpose of integrating BEV perception tasks is to strengthen the 3D spatial understanding capabilities of Percept-WAM.

E2E Trajectory Planning Results. Table 3 shows E2E driving performance of different methods. Specifically, our model achieves an average trajectory L2 error of 0.36m on nuScenes’ open-loop metrics and a score of 90.2 on NAVSIM’s closed-loop metrics, outperforming most existing BEV-based methods like UniAD [^29] and VLM-based methods like DriveVLM [^40]. Experimental results show that our two-stage training strategy further improves model performance. For NAVSIM, we draw inspiration from Hydra-MDP [^97], improving performance by scoring and selecting optimal trajectories from a static vocabulary. The visual results in Figure 9 further demonstrate the strong trajectory planning capability of the model, even in complex environments.

### 4.3 Ablations on Different Settings

Ablation on IoU-based Confidence Prediction. To validate the impact of different IoU score distributions in confidence-tuning dataset (refer to Section 3.1.2), we compare three construction schemes: (i) random-perturb: perturb ground-truth (GT) boxes and uniformly sample boxes across IoU levels; (ii) uniform model-pred: uniformly sample model predictions across IoU; and (iii) real model-pred: directly use model predictions with their realistic IoU distribution. As shown in Table 5, the proposed IoU-based confidence prediction, aligned with the realistic distribution of model predictions, significantly improves performance by $1.5$ AP and $2.3$ AP <sub>75</sub>, while ‘random-perturb’ and ‘uniform model-pred’ settings are inferior to the baseline.

Table 5: Ablation studies on training with different confidence-tuning datasets for IoU-based confidence prediction, evaluated on the 2D detection task of the nuImages val set.

| Confidence Computation | AP | AP <sub>50</sub> | AP <sub>75</sub> |
| --- | --- | --- | --- |
| Baseline (class score) | 48.1 | 70.9 | 51.4 |
| \+ IoU Conf. (random-perturb) | 46.9 | 70.0 | 50.7 |
| \+ IoU Conf. (uniform model-pred) | 46.2 | 69.1 | 49.3 |
| \+ IoU Conf. (real model-pred) | 49.6 | 70.4 | 53.7 |

![[x7 2.png|Refer to caption]]

(a) Class score (baseline)

To further demonstrate the impact of confidence-score training on model confidence, we visualize the relationship between predicted confidence (x-axis) and the IoU of the corresponding boxes with ground-truth (y-axis) in Figure 6. As shown in Figure 6(a), for class-score only setting, a large number of low-quality bounding boxes with high confidence scores (lower-right region) remain unfiltered in post-processing, which degrades precision. After incorporating IoU-based confidence prediction, the points in Figure 6(b) and Figure 6(c) move closer to the line $y=x$. The curve in Figure 6(c), trained on the model prediction dataset, aligns more closely with the diagonal, indicating improved predicted scores accuracy better suited for post-processing.

Ablation on BEV 3D Detection. We conduct ablation studies on the nuScenes validation set to assess the impact of each component related with BEV detection task, with all results listed in Table 6. Starting from a camera-only baseline, we initialize the World-BEV tokens with the feature extracted by a pretrained LiDAR encoder, which boosts the mAP by 8.2%. By adding data augmentations, such as LiDAR points and image data augmentation, the mAP is further improved by 8.1%. With an increase of the grid resolution (from $20\times 20$ to $40\times 40$), we observe an mAP improvement of 9.1%. Finally, we replace the AR coordinate decoding with MLP for parallel inference. This maintains a 50.4 mAP but accelerates inference speed by 16 $\times$.

Table 6: Ablations of BEV-perception related design choices on nuScenes val set.

| Method | mAP | NDS |
| --- | --- | --- |
| Baseline (Camera) | 25.0 | 25.7 |
| + Lidar Encoder | 33.2 | 32.2 |
| + Data Augmentation | 41.3 | 39.2 |
| + Number of Sampling Grids | 50.4 | 46.6 |
| + MLP Parallel (16× speedup) | 50.4 | 43.7 |

Table 7: Ablations of different decoding mechanisms and efficiency improvement from streaming inference on nuScenes val.

| Decoding Mechanisms | L2 (avg.) $\downarrow$ | Latency (ms) $\downarrow$ |
| --- | --- | --- |
| AR | 0.3970 | 2700 |
| + Streaming Infer. | 0.4058 | 2209 |
| AR(cluster) | 0.3919 | 1470 |
| Query-base | 0.3822 | 1174 |
| + Streaming Infer. | 0.3839 | 707 |

Ablation on E2E training. Table 7 presents the trajectory planning results of different decoding methods on the nuScenes validation set. ‘AR’ refers to directly converting trajectories into text and predicting them autoregressively. ‘AR (cluster)’ follows AutoVLA [^102]: trajectories are first segmented and clustered to generate new action tokens, which are then autoregressively predicted and decoded into trajectories. The results show that our query-based approach achieves the best trade-off between accuracy and inference speed. Our streaming inference strategy provides further efficiency gains, reducing the inference time by 18% and 40% for AR and query-based decoding approaches, respectively, with almost no impact on accuracy.

![[x10.png|Refer to caption]]

Figure 7: PV perception visualization. Percept-WAM demonstrates i) accurate and stable performance in crowded, long-range, and small-object scenarios for autonomous driving (see (a)–(d)); and ii) robust open-vocabulary perception in general-domain tasks (see (e) and (f)). Note that detection boxes of the same category are labeled with the same color.

![[x11 1.png|Refer to caption]]

Figure 8: BEV perception visualization. Left: BEV 3D object detection results projected onto surrounding images; Right: BEV map segmentation for the same scene sample.

![[x12.png|Refer to caption]]

Figure 9: Trajectory planning visualization. Left: Input surrounding images; Right: Ego vehicle trajectory planning in BEV perspective. Our model successfully navigates a construction zone, yielding to an oncoming vehicle while making the right turn.

## 5 Conclusion

This paper presents Percept-WAM, the first framework to unify 2D and 3D perception into one VLA model via *World-PV* and *World-BEV* tokens. It can achieve performances on par with specialized models across all sub-tasks. More importantly, by explicitly integrating these perception tasks into one unified model, the performance of trajectory planning can be further improved with high precision and low latency, demonstrating the effectiveness of our method. In the future, we will explore offline/online RL with rollout-based rewards that couple perception accuracy with trajectory prediction, aiming to enforce the overall consistency.

## References

Supplementary Material  

## Appendix A More Method Details

### A.1 The Attention Mask for Perception Tasks

To illustrate how Percep-WAM leverages World-PV and World-BEV tokens in the unified VLM backbone, Figure 10 visualizes the attention masks between input and output tokens for the PV- and BEV-detection tasks, respectively.

Specifically for the PV detection task, the attention mask is designed according to three principles: (i) World-PV tokens are fully visible to each other to better fuse PV features; (ii) for each grid-based prediction, the text tokens, grid tokens and output tokens follow the standard causal attention used in LLMs; (iii) to support grid-based parallel AR decoding, grid tokens and output tokens from different grid-based predictions are mutually masked. For BEV detection, two design choices apply: (i) similar to World-PV tokens, World-BEV tokens are fully mutually visible; (ii) to better capture the PV-to-BEV transformation and mitigate overfitting, each output token is constrained to attend only to World-BEV tokens and its corresponding grid token.

![[x13 1.png|Refer to caption]]

(a) PV detection.

### A.2 Streaming Inference

To meet the demands of real-world applications, we investigate the streaming inference capability of the Percept-WAM model for handling infinitely long conversational visual inputs. Inspired by streaming inference research in current mainstream VLMs [^103] [^104], we adopt a streaming strategy illustrated in Figure 11. Specifically, as shown in Figure 11(a), the trajectory prediction tokens at time step $T$ attend to the tokens of the two most recent frames (i.e., frame $T$ and frame $T\!-\!1$). Considering the high computational cost of processing visual tokens in the prefill phase, we reuse the KV cache from previous computations, with the specific strategy available for reference in Figure 11(b).

![[x15.png|Refer to caption]]

(a) Streaming Attention Map.

However, inconsistencies between training and inference paradigms inevitably lead to distribution drift: the cross-attention mechanism between frames theoretically enables historical tokens to implicitly encode historical information of infinite length. This differs significantly from standard video-clip-based training, which relies on fixed-length historical information. Thus, adopting a longer-clip training scheme is crucial for enhancing the model’s generalization capability on extended historical sequences.

To stabilize the computation process while gradually discarding historical visual tokens, we retain the attention sink [^105] (corresponding to the green grids in Figure 11(b)). However, this design causes deviations in positional embedding within the KV cache, arising from the discontinuity between the attention sink and visual frames. To address this issue, a dual-recomputation strategy is proposed with both local attention refinement and global cache recomputation: we recompute rotary positional embeddings (corresponding to the white numbers in Figure 11(b)) and adopt the specific token recomputation method [^106] to correct cross-attention results. Meanwhile, we mitigate error accumulation from increasing inference lengths by periodically caching ViT tokens and recomputing the complete KV cache, safeguarding long-sequence stability.

As shown in Table 7, we achieve 16% and 40% reduction in inference latency for the AR and the Query decoders, respectively, with an accuracy loss of less than 0.01.

![[x17.png|Refer to caption]]

Figure 12: Attention mask for our query-based trajectory decoding. 𝐐 ego \\mathbf{Q}\_{\\text{ego}}, pv \\mathbf{Q}\_{\\text{pv}}, and bev \\mathbf{Q}\_{\\text{bev}} are aligned with their corresponding modality tokens, while full \\mathbf{Q}\_{\\text{full}} accesses all features to decode the final trajectory.

### A.3 End-to-End Trajectory Prediction

Attention Mask for Query-based Trajectory Decoding. As described in Section 3.3, we introduce several sets of point-level queries and enforce modality-specific alignment by adjusting the attention mask. The attention mask is visualized in Figure 12: the first three query sets attend only to their corresponding modality tokens for trajectory decoding, while the final query set attends to all input tokens to generate the final trajectory.

Detail Settings of Percept-WAM as the Trajectory Selector. As mentioned in Section 4, training Percept-WAM solely to replicate ground-truth trajectories is insufficient for achieving strong closed-loop performance, as trajectory supervision often misaligns with real-world evaluation metrics [^107]. To mitigate this gap, we introduce a query-based trajectory scoring and selecting approach inspired by Hydra-MDP [^97] and GTRS [^108], evaluating it on the NAVSIM v1 benchmark.

As showed in Figure 13, instead of direct trajectory replication, we train the model to score a super-dense, pre-clustered trajectory vocabulary $\mathcal{V}_{XL}$. Half of $\mathcal{V}_{XL}$ is randomly dropped during training to improve robustness. At inference, a reduced vocabulary $\mathcal{V}_{L}$ is used. Each trajectory in $\mathcal{V}_{L}$ is first embedded through an MLP and then encoded as $\mathcal{V}^{\prime}_{L}$ via a stack of transformer layers:

$$
\mathcal{V}^{\prime}_{L}=\textit{Transformer}(Q,K,V=\textit{MLP}(\mathcal{V}_{L})).
$$

Percept-WAM further integrates contextual cues by querying features from the World-PV Tokens $T_{\text{WPV}}$, World-BEV Tokens $T_{\text{WBEV}}$ and Text Tokens $T_{\text{T}}$, generating World-Action Tokens $T_{\text{WA}}$ for trajectory scoring:

$$
T_{\text{WA}}=\textit{Percept-WAM}(Q=\mathcal{V}^{\prime}_{L},K,V=T_{\text{WPV}},T_{\text{WBEV}},T_{\text{T}}).
$$

These enriched features are passed through a set of prediction heads to compute trajectory scores. Using a binary cross-entropy objective, we distill rule-based driving priors into our model. At inference time, our model selects the trajectory with the highest composite score, reflecting optimal performance for the given scenario.

![[x18.png|Refer to caption]]

Figure 13: Query-based trajectory scoring and selecting. The embedded trajectory vocabularies are trained to be aligned with the World-PV, World-BEV, and Text Tokens, then decoded to generate the composite scores. The trajectory with the maximal score is selected as the planning result.

Table 8: Hyperparameter configurations for Percept-WAM across training stages. Note that within each stage, jointly trained tasks share the same base hyperparameters. All experiments are trained on 8 nodes.

<table><tbody><tr><td></td><td colspan="3">Stage 1</td><td>Stage 2</td></tr><tr><td>Training parameter</td><td>PV Perception</td><td>BEV Perception</td><td>Driving QA</td><td>Trajectory Prediction</td></tr><tr><td>Learning rate</td><td>0.0002</td><td>0.0002</td><td>0.0002</td><td>0.0002</td></tr><tr><td>Warmup iterations</td><td>1000</td><td>1000</td><td>1000</td><td>500</td></tr><tr><td>Training iterations</td><td>100000</td><td>100000</td><td>100000</td><td>3000</td></tr><tr><td>Batch size</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Image resolution</td><td><math><semantics><mrow><mn>1344</mn> <mo>×</mo> <mn>896</mn></mrow> <annotation>1344\times 896</annotation></semantics></math></td><td><math><semantics><mrow><mn>796</mn> <mo>×</mo> <mn>448</mn></mrow> <annotation>796\times 448</annotation></semantics></math></td><td><math><semantics><mrow><mn>796</mn> <mo>×</mo> <mn>448</mn></mrow> <annotation>796\times 448</annotation></semantics></math></td><td><math><semantics><mrow><mn>796</mn> <mo>×</mo> <mn>448</mn></mrow> <annotation>796\times 448</annotation></semantics></math></td></tr><tr><td>Grid number</td><td><math><semantics><mrow><mn>10</mn> <mo>×</mo> <mn>10</mn></mrow> <annotation>10\times 10</annotation></semantics></math></td><td><math><semantics><mrow><mn>40</mn> <mo>×</mo> <mn>40</mn></mrow> <annotation>40\times 40</annotation></semantics></math> for Det, <math><semantics><mrow><mn>10</mn> <mo>×</mo> <mn>10</mn></mrow> <annotation>10\times 10</annotation></semantics></math> for Seg</td><td>NA</td><td>NA</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>Schedule</td><td>Cosine Annealing</td><td>Cosine Annealing</td><td>Cosine Annealing</td><td>Cosine Annealing</td></tr></tbody></table>

Table 9: Comparison of referring expression comprehension (REC) performance, reported using P@0.5. VGM and MLLM refer to the vision generalist model and multimodal large language model, respectively.

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO <math><semantics><mo>+</mo> <annotation>+</annotation></semantics></math></td><td colspan="2">RefCOCOg</td><td></td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val</td><td>test</td><td>Avg</td></tr><tr><td>MDETR <sup><a href="#fn:109">109</a></sup></td><td rowspan="2">VGM</td><td>86.8</td><td>89.6</td><td>81.4</td><td>79.5</td><td>84.1</td><td>70.6</td><td>81.6</td><td>80.9</td><td>81.8</td></tr><tr><td>Grounding DINO T <sup><a href="#fn:110">110</a></sup></td><td>89.2</td><td>91.9</td><td>86.0</td><td>81.1</td><td>87.4</td><td>74.7</td><td>84.2</td><td>84.9</td><td>84.9</td></tr><tr><td>Shikra-13B <sup><a href="#fn:111">111</a></sup></td><td rowspan="4">MLLM</td><td>87.8</td><td>91.1</td><td>81.8</td><td>82.9</td><td>87.8</td><td>74.4</td><td>82.6</td><td>83.2</td><td>84.0</td></tr><tr><td>MiniGPT-v2-7B <sup><a href="#fn:112">112</a></sup></td><td>88.1</td><td>91.3</td><td>84.3</td><td>79.6</td><td>85.5</td><td>73.3</td><td>84.2</td><td>84.3</td><td>83.8</td></tr><tr><td>VistaLLM-7B <sup><a href="#fn:113">113</a></sup></td><td>88.1</td><td>91.5</td><td>83.0</td><td>82.9</td><td>89.8</td><td>74.8</td><td>83.6</td><td>84.4</td><td>84.8</td></tr><tr><td>Percept-WAM</td><td>89.9</td><td>90.8</td><td>89.3</td><td>85.4</td><td>88.2</td><td>81.7</td><td>86.5</td><td>87.0</td><td>87.4</td></tr></tbody></table>

Table 10: Comparison of referring expression segmentation (RES) performance, reported using cumulative IoU (cIoU). VGM and MLLM refer to the vision generalist model and multimodal large language model, respectively.

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO <math><semantics><mo>+</mo> <annotation>+</annotation></semantics></math></td><td colspan="2">RefCOCOg</td><td></td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val</td><td>test</td><td>Avg</td></tr><tr><td>GLEE-Pro <sup><a href="#fn:114">114</a></sup></td><td></td><td>80.0</td><td>–</td><td>–</td><td>69.6</td><td>–</td><td>–</td><td>72.9</td><td>–</td><td>74.2</td></tr><tr><td>UNINEXT-H <sup><a href="#fn:115">115</a></sup></td><td>VGM</td><td>82.2</td><td>83.4</td><td>81.3</td><td>72.5</td><td>76.4</td><td>66.2</td><td>74.7</td><td>76.4</td><td>76.6</td></tr><tr><td>LISA-7B <sup><a href="#fn:116">116</a></sup></td><td></td><td>74.1</td><td>76.5</td><td>71.1</td><td>62.4</td><td>67.4</td><td>56.5</td><td>66.4</td><td>68.5</td><td>67.9</td></tr><tr><td>VistaLLM-13B <sup><a href="#fn:113">113</a></sup></td><td></td><td>77.2</td><td>78.7</td><td>73.9</td><td>71.8</td><td>74.4</td><td>65.6</td><td>69.8</td><td>71.9</td><td>72.9</td></tr><tr><td>GLaMM-7B <sup><a href="#fn:117">117</a></sup></td><td></td><td>79.5</td><td>83.2</td><td>76.9</td><td>72.6</td><td>78.7</td><td>64.6</td><td>74.2</td><td>74.9</td><td>75.6</td></tr><tr><td>HiMTok-8B <sup><a href="#fn:118">118</a></sup></td><td></td><td>81.1</td><td>81.2</td><td>79.2</td><td>77.1</td><td>78.8</td><td>71.5</td><td>75.8</td><td>76.7</td><td>77.7</td></tr><tr><td>Percept-WAM</td><td>MLLM</td><td>86.5</td><td>87.4</td><td>86.6</td><td>79.9</td><td>83.6</td><td>75.2</td><td>81.3</td><td>81.9</td><td>82.8</td></tr></tbody></table>

## Appendix B More Experiment Settings

### B.1 Two-Stage Training Details.

Training Setting Details. To effectively optimize Percept-WAM for both perception and planning, we adopt a two-stage training scheme. The first stage focuses on enhancing the VLM’s overall 2D and 3D spatial perception, assisted by autonomous-driving general QA tasks, while the second stage trains end-to-end trajectory prediction on top of this perception-enhanced base model. The detailed training hyperparameters are summarized in Table 8.

Data Construction. As shown in Table 2 of the main paper, we employ task-specific training datasets for PV, BEV, and trajectory prediction. To preserve the model’s general capabilities, we further incorporate autonomous-driving QA data during training. For each task family, we specify the data mixture as follows: (i) for PV tasks, the sampling ratio among 2D Detection, Mono 3D Detection, Instance Segmentation, Semantic Segmentation, and Grounding is set to 1:1:2:1:1; (ii) for BEV tasks, the ratio between 3D Detection and BEV Segmentation is 1:1; (iii) when jointly training multiple main tasks (i.e., PV, BEV, trajectory prediction, and Driving QA), we sample each task with equal probability for simplicity.

Table 11: [^17]

<table><tbody><tr><td rowspan="2">Trajectory Planning Mechanism</td><td colspan="6">NAVSIM v1</td></tr><tr><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>AR-based trajectory generation</td><td>96.4</td><td>94.5</td><td>92.0</td><td>98.7</td><td>78.5</td><td>84.1</td></tr><tr><td>Query-based trajectory generation</td><td>96.5</td><td>90.3</td><td>91.0</td><td>99.7</td><td>75.6</td><td>80.4</td></tr><tr><td>Query-based trajectory scoring and selection</td><td>98.8</td><td>98.6</td><td>94.4</td><td>99.5</td><td>84.8</td><td>90.2</td></tr></tbody></table>

## Appendix C More Experiment Results

### C.1 Main Results

Comparison of Visual Grounding Performance. In this section, we evaluate the model’s ability to leverage fine-grained world features for complex visual grounding. Visual grounding is a critical task that associates textual descriptions with corresponding image regions or objects. This task can be further divided into referring expression comprehension (REC) and referring expression segmentation (RES), with output formats of bounding boxes or masks. We present comprehensive comparison results for both tasks in Table 9 and Table 10, respectively. As shown in Table 9, Percept-WAM achieves top-tier performance on the RefCOCO [^84], RefCOCO $+$ [^85], and RefCOCOg [^86] benchmarks among MLLMs, outperforming Grounding DINO [^74], a representative VGM, by an average of $2.5$ P@0.5. Table 10 demonstrates Percept-WAM’s exceptional pixel-level segmentation performance among VGMs and MLLMs, achieving an average cIoU of $82.8$. We demonstrate Percept-WAM’s visual grounding performance on the same images with different descriptions, as shown in Figure 14, highlighting its robust ability to distinguish specific objects with unique attributes from visually similar instances.

BEV Segmentation Results on Geographical Train/Val Splits. For the BEV segmentation task, we additionally report results on nuScenes with the geographical train/val split (following the split protocol in EAFT [^107]), as shown in Table 12. The results indicate that our Percept-WAM achieves performance comparable to EAFT, and that incorporating LiDAR inputs consistently provides stable performance gains.

Table 12: [^18]

<table><tbody><tr><td>Modality</td><td>Method</td><td>Dri.</td><td>Ped.</td><td>Lane.</td><td>Veh.</td></tr><tr><td rowspan="2">C</td><td><sup><a href="#fn:107">107</a></sup></td><td>58.06</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Percept-WAM</td><td>56.41</td><td>33.21</td><td>22.03</td><td>34.47</td></tr><tr><td>C + L</td><td>Percept-WAM</td><td>67.19</td><td>22.06</td><td>23.99</td><td>56.76</td></tr></tbody></table>

### C.2 Ablation Studies

Ablation of Query-based Trajectory Decoding Methods. We ablate different query–mask configurations for trajectory decoding, as summarized in Table 13. Compared to the “full” mode, which uses a single query set $\mathbf{Q}_{\mathbf{full}}$, parallel decoding with multiple query sets reduces the trajectory error by approximately 8%.

Ablation of Clustered-Action Design. The clustered-action method discretizes trajectories using the K-Disk clustering algorithm with a maximum vocabulary size of $2048$. A greedy clustering approach is applied based on a 0.05-meter distance threshold to group similar trajectories. To improve long-horizon stability, trajectories are segmented into $s$ -frame intervals, where $s\in\{1,2,3,6\}$. As shown in Table 14, a segment length of $2$ frames with a 0.05-meter threshold yields the best performance on nuScenes, achieving an L2\_avg of $0.3919$.

Table 13: Query mask ablation results on nuScenes trajectory prediction task. Lower is better. ‘Full’ means using a single query set which can attend to all the input tokens. ‘Parallel’ refers to our approach, where all query sets are decoded in parallel.

| Mask mode | L2-avg $\downarrow$ |
| --- | --- |
| Full | 0.4151 |
| Parallel | 0.3821 |

Table 14: Ablation of AR (cluster) configurations with different frame interval $s$ on nuScenes trajectory prediction task. Lower($\downarrow$) is better.

| Method | L2-1s $\downarrow$ | L2-2s $\downarrow$ | L2-3s $\downarrow$ | L2-avg $\downarrow$ |
| --- | --- | --- | --- | --- |
| AR | 0.160 | 0.356 | 0.674 | 0.3970 |
| AR(cluster) <sub>s=1</sub> | 0.434 | 0.744 | 1.129 | 0.7692 |
| AR(cluster) <sub>s=2</sub> | 0.181 | 0.356 | 0.638 | 0.3919 |
| AR(cluster) <sub>s=3</sub> | 0.196 | 0.390 | 0.692 | 0.4260 |
| AR(cluster) <sub>s=6</sub> | 0.236 | 0.494 | 0.892 | 0.5406 |

Comparison of E2E Performance. We ablate different trajectory planning strategies in the closed-loop setting on NAVSIM benchmark. Specifically, we compare three methods: (i) AR-based trajectory generation, (ii) query-based trajectory generation, (iii) query-based trajectory scoring and selection. The results are summarized in Table 11.

The comparison between AR-based and query-based methods in Table 7 appears inconsistent with the ablation results in Table 11. This discrepancy arises from the misalignment between open-loop and closed-loop metrics. While the query-based method reduces the L2 distance between planned trajectories and ground truth (from $1.1$ m to $0.8$ m on the NAVSIM navtrain validation split), improved trajectory replication does not always lead to better closed-loop performance, as evidenced by PDMS on the NAVSIM [^17] benchmark. Therefore, we adopt query-based trajectory scoring and selection which combines the imitation strength of the query-based approach with the closed-loop metrics, and achieves the best overall performance.

### C.3 Illustrations

Trajectory Prediction Results. As shown in Figure 15, our model demonstrates strong trajectory planning in various challenging scenarios. It effectively handles decisions such as yielding to vehicles and pedestrians, navigating hazards in adverse weather, and accounting for occluded objects in low-visibility conditions. These examples highlight the model’s robustness and adaptability, even in long-tail cases.

![[x19.png|Refer to caption]]

Figure 14: Illustration of Percept-WAM on the visual grounding task. Percept-WAM accurately localizes referred objects and exhibits robust understanding of diverse visual attributes.

![[x20.png|Refer to caption]]

Figure 15: More illustration of trajectory planning capabilities. The ego vehicle is shown as a green box, background vehicles as black boxes, pedestrians as purple boxes, and other detected objects as grey boxes. Ground truth trajectories are shown as blue dots, and planned trajectories as red dots. (a) Percept-WAM successfully navigates the ego vehicle turning left, yielding to pedestrians crossing the road. (b) Under rainy conditions, the model detects a jaywalking pedestrian and safely avoids a potential collision. (c) While turning right at night, the model accounts for a cyclist obstructed in the front camera view, ensuring enough space for safe passage. Overall, our model demonstrates strong planning performance, even in challenging and long-tail scenarios.

## Appendix D Limitations

Limitations & Future Work. We plan to explore more efficient and higher-accuracy architectures for multi-task joint training (perception & trajectory prediction), such as Mixture-of-Experts (MoE), where different tasks will be routed to specialized experts. In addition, for both perception and end-to-end trajectory prediction, we aim to employ a unified reinforcement learning framework with a multi-objective reward design to jointly optimize these tasks.

[^1]: Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262, 2024.

[^2]: Yinan Zheng, Ruiming Liang, Kexin Zheng, Jinliang Zheng, Liyuan Mao, Jianxiong Li, Weihao Gu, Rui Ai, Shengbo Eben Li, Xianyuan Zhan, et al. Diffusion-based planning for autonomous driving with flexible guidance. arXiv preprint arXiv:2501.15564, 2025.

[^3]: Rui Fan, Sicen Guo, and Mohammud Junaid Bocus. Autonomous driving perception. Cham, Switzerland: Springer, 2023.

[^4]: Kexin Tian, Jingrui Mao, Yunlong Zhang, Jiwan Jiang, Yang Zhou, and Zhengzhong Tu. Nuscenes-spatialqa: A spatial understanding and reasoning benchmark for vision-language models in autonomous driving. arXiv preprint arXiv:2504.03164, 2025.

[^5]: Xianda Guo, Ruijun Zhang, Yiqun Duan, Yuhang He, Chenming Zhang, Shuai Liu, and Long Chen. Drivemllm: A benchmark for spatial understanding with multimodal large language models in autonomous driving. arXiv e-prints, pages arXiv–2411, 2024.

[^6]: Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, et al. Alpamayo-r1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025.

[^7]: Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755, 2025.

[^8]: Yi Xu, Yuxin Hu, Zaiwei Zhang, Gregory P Meyer, Siva Karthik Mustikovela, Siddhartha Srinivasa, Eric M Wolff, and Xin Huang. Vlm-ad: End-to-end autonomous driving through vision-language model supervision. arXiv preprint arXiv:2412.14446, 2024.

[^9]: Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855, 2025.

[^10]: Yujin Wang, Quanfeng Liu, Zhengxin Jiang, Tianyi Wang, Junfeng Jiao, Hongqing Chu, Bingzhao Gao, and Hong Chen. Rad: Retrieval-augmented decision-making of meta-actions with vision-language models in autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3838–3848, 2025.

[^11]: Mingjie Pan, Jiyao Zhang, Tianshu Wu, Yinghao Zhao, Wenlong Gao, and Hao Dong. Omnimanip: Towards general robotic manipulation via object-centric interaction primitives as spatial constraints. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17359–17369, 2025.

[^12]: Max Argus, Jelena Bratulic, Houman Masnavi, Maxim Velikanov, Nick Heppert, Abhinav Valada, and Thomas Brox. cvla: Towards efficient camera-space vlas. arXiv preprint arXiv:2507.02190, 2025.

[^13]: Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025.

[^14]: Erik Daxberger, Nina Wenzel, David Griffiths, Haiming Gang, Justin Lazarow, Gefen Kohavi, Kai Kang, Marcin Eichner, Yinfei Yang, Afshin Dehghan, et al. Mm-spatial: Exploring 3d spatial understanding in multimodal llms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7395–7408, 2025.

[^15]: An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024.

[^16]: Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan. Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. Advances in Neural Information Processing Systems, 37:819–844, 2024.

[^17]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2024.

[^18]: Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.

[^19]: Shuang Zeng, Xinyuan Chang, Mengwei Xie, Xinran Liu, Yifan Bai, Zheng Pan, Mu Xu, and Xing Wei. Futuresightdrive: Thinking visually with spatio-temporal cot for autonomous driving. arXiv preprint arXiv:2505.17685, 2025.

[^20]: Haohan Chi, Huan-ang Gao, Ziming Liu, Jianing Liu, Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu, Zeda Wang, Wenyi Li, et al. Impromptu vla: Open weights and open data for driving vision-language-action models. arXiv preprint arXiv:2505.23757, 2025.

[^21]: Tianqi Wang, Enze Xie, Ruihang Chu, Zhenguo Li, and Ping Luo. Drivecot: Integrating chain-of-thought reasoning with end-to-end driving. arXiv preprint arXiv:2403.16996, 2024.

[^22]: Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. Advances in Neural Information Processing Systems, 36:61501–61513, 2023.

[^23]: Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: an embodied multimodal language model. In Proceedings of the 40th International Conference on Machine Learning, pages 8469–8488, 2023.

[^24]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12037–12047, 2025.

[^25]: Rui Zhao, Yuze Fan, Ziguo Chen, Fei Gao, and Zhenhai Gao. Diffe2e: Rethinking end-to-end driving with a hybrid action diffusion and supervised policy. arXiv preprint arXiv:2505.19516, 2025.

[^26]: Shu Liu, Wenlin Chen, Weihao Li, Zheng Wang, Lijin Yang, Jianing Huang, Yipin Zhang, Zhongzhan Huang, Ze Cheng, and Hao Yang. Bridgedrive: Diffusion bridge policy for closed-loop trajectory planning in autonomous driving. arXiv preprint arXiv:2509.23589, 2025.

[^27]: Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. arXiv preprint arXiv:2310.01415, 2023.

[^28]: Qiming Zhang, Meixin Zhu, and Hao Frank Yang. Think-driver: From driving-scene understanding to decision-making with vision language models. In European Conference on Computer Vision Workshop, 2024.

[^29]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17853–17862, 2023.

[^30]: Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[^31]: Jincheng Li, Chunyu Xie, Ji Ao, Dawei Leng, and Yuhui Yin. Lmm-det: Make large multimodal models excel in object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 308–318, 2025.

[^32]: Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12697–12705, 2019.

[^33]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[^34]: Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

[^35]: Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

[^36]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

[^37]: Ilias Stogiannidis, Steven McDonagh, and Sotirios A Tsaftaris. Mind the gap: Benchmarking spatial reasoning in vision-language models. arXiv preprint arXiv:2503.19707, 2025.

[^38]: Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023. URL https://arxiv. org/abs/2307.15818, 2024.

[^39]: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[^40]: Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024.

[^41]: Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In European conference on computer vision, pages 256–274. Springer, 2024.

[^42]: Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In European conference on computer vision, pages 194–210. Springer, 2020.

[^43]: Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, and Zeming Li. Bevdepth: Acquisition of reliable depth for multi-view 3d object detection. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 1477–1485, 2023.

[^44]: Lang Peng, Zhirong Chen, Zhangjie Fu, Pengpeng Liang, and Erkang Cheng. Bevsegformer: Bird’s eye view semantic segmentation from arbitrary camera rigs. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5935–5943, 2023.

[^45]: Yunpeng Zhang, Zheng Zhu, and Dalong Du. Occformer: Dual-path transformer for vision-based 3d semantic occupancy prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9433–9443, 2023.

[^46]: Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela Rus, and Song Han. Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation. arXiv preprint arXiv:2205.13542, 2022.

[^47]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence, 45(11):12878–12895, 2022.

[^48]: Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[^49]: Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810, 2021.

[^50]: Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4542–4550, 2024.

[^51]: Yiheng Li, Cunxin Fan, Chongjian Ge, Zhihao Zhao, Chenran Li, Chenfeng Xu, Huaxiu Yao, Masayoshi Tomizuka, Bolei Zhou, Chen Tang, et al. Womd-reasoning: A large-scale dataset for interaction reasoning in driving. arXiv preprint arXiv:2407.04281, 2024.

[^52]: Srikanth Malla, Chiho Choi, Isht Dwivedi, Joon Hee Choi, and Jiachen Li. Drama: Joint risk localization and captioning in driving. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1043–1052, 2023.

[^53]: Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In Conference on robot learning, pages 1–16. PMLR, 2017.

[^54]: Ming Nie, Renyuan Peng, Chunwei Wang, Xinyue Cai, Jianhua Han, Hang Xu, and Li Zhang. Reason2drive: Towards interpretable and chain-based reasoning for autonomous driving. In European Conference on Computer Vision, pages 292–308. Springer, 2024.

[^55]: Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

[^56]: Hao Tang, Chenwei Xie, Haiyang Wang, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Ufo: A unified approach to fine-grained visual perception via open-ended language interface. arXiv preprint arXiv:2503.01342, 2025.

[^57]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[^58]: Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. arXiv preprint arXiv:2109.10852, 2021.

[^59]: Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10965–10975, 2022.

[^60]: Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In European conference on computer vision, pages 728–755. Springer, 2022.

[^61]: Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024.

[^62]: Qing Jiang, Junan Huo, Xingyu Chen, Yuda Xiong, Zhaoyang Zeng, Yihao Chen, Tianhe Ren, Junzhi Yu, and Lei Zhang. Detect anything via next point prediction. arXiv preprint arXiv:2510.12798, 2025.

[^63]: Mozhi Zhang, Mianqiu Huang, Rundong Shi, Linsen Guo, Chong Peng, Peng Yan, Yaqian Zhou, and Xipeng Qiu. Calibrating the confidence of large language models by eliciting fidelity. arXiv preprint arXiv:2404.02655, 2024.

[^64]: Borui Jiang, Ruixuan Luo, Jiayuan Mao, Tete Xiao, and Yuning Jiang. Acquisition of localization confidence for accurate object detection. In Proceedings of the European conference on computer vision (ECCV), pages 784–799, 2018.

[^65]: Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. Advances in neural information processing systems, 33:21002–21012, 2020.

[^66]: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

[^67]: Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016.

[^68]: Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[^69]: Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790, 2021.

[^70]: Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1874–1883, 2016.

[^71]: Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.

[^72]: Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J Fleet, and Geoffrey E Hinton. A unified sequence interface for vision tasks. Advances in Neural Information Processing Systems, 35:31333–31346, 2022.

[^73]: Haiyang Wang, Hao Tang, Li Jiang, Shaoshuai Shi, Muhammad Ferjad Naeem, Hongsheng Li, Bernt Schiele, and Liwei Wang. Git: Towards generalist vision transformer through universal language interface. In European Conference on Computer Vision, pages 55–73. Springer, 2024.

[^74]: Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022.

[^75]: Tai Wang, Xinge Zhu, Jiangmiao Pang, and Dahua Lin. Fcos3d: Fully convolutional one-stage monocular 3d object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 913–922, 2021.

[^76]: Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022.

[^77]: Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1209–1218, 2018.

[^78]: Yufei Zhan, Hongyin Zhao, Yousong Zhu, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon-g: Bridging vision-language and vision-centric tasks via large multimodal models. arXiv preprint arXiv:2410.16163, 2024.

[^79]: Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. In European Conference on Computer Vision, pages 417–435. Springer, 2024.

[^80]: Peng Liu, Haozhan Shen, Chunxin Fang, Zhicheng Sun, Jiajia Liao, and Tiancheng Zhao. Vlm-fo1: Bridging the gap between high-level reasoning and fine-grained perception in vlms. arXiv preprint arXiv:2509.25916, 2025.

[^81]: Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 1440–1448, 2015.

[^82]: Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020.

[^83]: Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127(3):302–321, 2019.

[^84]: Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.

[^85]: Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In European conference on computer vision, pages 69–85. Springer, 2016.

[^86]: Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016.

[^87]: Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, and Oleg Sinavski. Lingoqa: Visual question answering for autonomous driving. arXiv preprint arXiv:2312.14115, 2023.

[^88]: Yanze Li, Wenhua Zhang, Kai Chen, Yanxin Liu, Pengxiang Li, Ruiyuan Gao, Lanqing Hong, Meng Tian, Xinhai Zhao, Zhenguo Li, et al. Automated evaluation of large vision-language models on self-driving corner cases. arXiv preprint arXiv:2404.10595, 2024.

[^89]: Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, and Chaowei Xiao. Dolphins: Multimodal language model for driving. In European Conference on Computer Vision, pages 403–420. Springer, 2024.

[^90]: Yuhang Lu, Yichen Yao, Jiadong Tu, Jiangnan Shao, Yuexin Ma, and Xinge Zhu. Can lvlms obtain a driver’s license? a benchmark towards reliable agi for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 5838–5846, 2025.

[^91]: Xu Cao, Tong Zhou, Yunsheng Ma, Wenqian Ye, Can Cui, Kun Tang, Zhipeng Cao, Kaizhao Liang, Ziran Wang, James M Rehg, et al. Maplm: A real-world large-scale vision-language benchmark for map and traffic scene understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21819–21830, 2024.

[^92]: Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 2024.

[^93]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8350, 2023.

[^94]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243, 2024.

[^95]: Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14864–14873, 2024.

[^96]: Chengran Yuan, Zhanqi Zhang, Jiawei Sun, Shuo Sun, Zefan Huang, Christina Dao Wen Lee, Dongen Li, Yuhang Han, Anthony Wong, Keng Peng Tee, et al. Drama: An efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601, 2024.

[^97]: Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978, 2024.

[^98]: Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[^99]: Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. arXiv preprint arXiv:1710.03740, 2017.

[^100]: Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.

[^101]: Yan Yan, Yuxing Mao, and Bo Li. Second: Sparsely embedded convolutional detection. Sensors, 18(10):3337, 2018.

[^102]: Zewei Zhou, Tianhui Cai, Seth Z Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757, 2025.

[^103]: Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. arXiv preprint arXiv:2510.09608, 2025.

[^104]: Zhenyu Ning, Guangda Liu, Qihao Jin, Wenchao Ding, Minyi Guo, and Jieru Zhao. Livevlm: Efficient online video understanding via streaming-oriented kv cache and retrieval. arXiv preprint arXiv:2505.15269, 2025.

[^105]: Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

[^106]: Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. Cacheblend: Fast large language model serving for rag with cached knowledge fusion. In Proceedings of the Twentieth European Conference on Computer Systems, pages 94–109, 2025.

[^107]: Christian Witte, Jens Behley, Cyrill Stachniss, and Marvin Raaijmakers. Epipolar attention field transformers for bird’s eye view semantic segmentation. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 8660–8669. IEEE, 2025.

[^108]: Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Joshua Chen, Nadine Chang, Maying Shen, Zuxuan Wu, Shiyi Lan, and Jose M Alvarez. Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664, 2025.

[^109]: Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1780–1790, 2021.

[^110]: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.

[^111]: Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.

[^112]: Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023.

[^113]: Shraman Pramanick, Guangxing Han, Rui Hou, Sayan Nag, Ser-Nam Lim, Nicolas Ballas, Qifan Wang, Rama Chellappa, and Amjad Almahairi. Jack of all tasks master of many: Designing general-purpose coarse-to-fine vision-language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14076–14088, 2024.

[^114]: Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, and Song Bai. General object foundation model for images and videos at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3783–3795, 2024.

[^115]: Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15325–15336, 2023.

[^116]: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024.

[^117]: Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024.

[^118]: Tao Wang, Changxu Cheng, Lingfeng Wang, Senda Chen, and Wuyue Zhao. Himtok: Learning hierarchical mask tokens for image segmentation with large multimodal model. arXiv preprint arXiv:2503.13026, 2025.