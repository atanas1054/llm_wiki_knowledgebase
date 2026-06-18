---
title: "Understanding R1-Zero-Like Training: A Critical Perspective"
source: "https://arxiv.org/html/2503.20783v2"
author:
published:
created: 2026-06-18
description:
tags:
  - "clippings"
---
Zichen Liu <sup>* <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math> 1,2</sup>, Changyu Chen <sup>*1,3</sup>, Wenjun Li <sup>*3</sup>, Penghui Qi <sup>*1,2</sup>,  
Tianyu Pang <sup>1</sup>, Chao Du <sup>1</sup>, Wee Sun Lee <sup>2</sup>, Min Lin <sup>1</sup>  
<sup>1</sup> Sea AI Lab  
<sup>2</sup> National University of Singapore  
<sup>3</sup> Singapore Management University <sup>∗</sup> Core Contributors.<sup>†</sup> Project Lead.

###### Abstract

DeepSeek-R1-Zero has shown that reinforcement learning (RL) at scale can directly enhance the reasoning capabilities of LLMs without supervised fine-tuning. In this work, we critically examine R1-Zero-like training by analyzing its two core components: base models and RL. We investigate a wide range of base models, including DeepSeek-V3-Base, to understand how pretraining characteristics influence RL performance. Our analysis reveals that DeepSeek-V3-Base already exhibit “Aha moment”, while Qwen2.5 base models demonstrate strong reasoning capabilities even without prompt templates, suggesting potential pretraining biases. Additionally, we identify an optimization bias in Group Relative Policy Optimization (GRPO), which artificially increases response length (especially for incorrect outputs) during training. To address this, we introduce Dr. GRPO, an unbiased optimization method that improves token efficiency while maintaining reasoning performance. Leveraging these insights, we present a minimalist R1-Zero recipe that achieves $43.3\%$ accuracy on AIME 2024 with a 7B base model, establishing a new state-of-the-art.

[https://github.com/sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) <sup>1</sup>

![[x1 33.png|Refer to caption]]

Figure 1: Left: Dr. GRPO introduces simple yet significant modifications to address the biases in GRPO 27, by removing the length and std normalization terms. Right: Our unbiased optimizer effectively prevents the model from generating progressively longer incorrect responses, thereby enhancing token efficiency.

![[x2 32.png|Refer to caption]]

Figure 2: Model performance comparison. Oat-Zero-7B is RL-tuned with our minimalist recipe described in Sec. 1 (third paragraph). Please see App. B for more results.

## 1 Introduction

DeepSeek-R1-Zero [^8] revolutionizes the pipeline of large language model (LLM) post-training by introducing the R1-Zero-like training paradigm: directly applying RL to base LLMs without relying on supervised fine-tuning (SFT) as a preliminary step. This new paradigm is appealing due to its simplicity and the demonstrated RL scaling phenomenon: the model reasoning capabilities improve along with a continual increase in model’s response length. This phenomenon is also accompanied by the “Aha moment”, at which the model learns emergent skills such as self-reflections.

In this paper, we aim to understand R1-Zero-like training by studying two essential components: *base models* and *RL*. In the first part, we investigate various attributes of base models, with the focus on the Qwen2.5 model family [^33] [^34], which has been used in recent attempts to reproduce R1-Zero [^23] [^36] [^21] [^12], as well as DeepSeek-V3-Base [^19], from which the real R1-Zero model was RL-tuned. In the second part, we identify the bias in optimization of GRPO [^27], which may lead to progressively longer incorrect responses. To this end, we propose a simple modification to eliminate the bias, i.e., to get GRPO Done Right (Dr. GRPO), which leads to better token efficiency (highlighted in Fig. 1).

Our analysis on base models and RL suggests a minimalist recipe for R1-Zero-like training: we RL-tune Qwen2.5-Math-7B using the (unbiased) Dr. GRPO algorithm on MATH [^10] level 3-5 questions with the Qwen-Math template, and achieve state-of-the-art performance (Fig. 2) with only $27$ hours compute on $8\times$ A100 GPUs. We hope our findings presented in this paper, models released, and the codebase open-sourced could benefit future research in the field. As an overview, we summarize the takeaways of this paper below:

<svg height="9544.5" id="S1.p4.pic1" overflow="visible" version="1.1" viewBox="0 0 600 9544.5" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,9544.5) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0" style="--ltx-fill-color:#000000;"><path d="M 0 5.91 L 0 9526.45 C 0 9529.71 2.64 9532.36 5.91 9532.36 L 594.09 9532.36 C 597.36 9532.36 600 9529.71 600 9526.45 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F0F0FF" fill-opacity="1.0" style="--ltx-fill-color:#F0F0FF;"><path d="M 1.97 5.91 L 1.97 9526.45 C 1.97 9528.63 3.73 9530.39 5.91 9530.39 L 594.09 9530.39 C 596.27 9530.39 598.03 9528.63 598.03 9526.45 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 15 9522.36)"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)"><g fill="#FFFFFF" fill-opacity="1.0" style="--ltx-fill-color:#FFFFFF;"><path d="M 0 2.95 L 0 19.19 C 0 20.82 1.32 22.14 2.95 22.14 L 158.39 22.14 C 160.02 22.14 161.34 20.82 161.34 19.19 L 161.34 2.95 C 161.34 1.32 160.02 0 158.39 0 L 2.95 0 C 1.32 0 0 1.32 0 2.95 Z" style="stroke:none"></path></g><g fill="#000000" fill-opacity="1.0" style="--ltx-fill-color:#000000;"><path d="M 0 2.95 L 0 19.19 C 0 20.82 1.32 22.14 2.95 22.14 L 158.39 22.14 C 160.02 22.14 161.34 20.82 161.34 19.19 L 161.34 2.95 C 161.34 1.32 160.02 0 158.39 0 L 2.95 0 C 1.32 0 0 1.32 0 2.95 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 11.81 7.61)"><foreignObject color="#000000" height="12.3" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :10.09em;--fo_height:0.69em;--fo_depth :0.19em;" transform="matrix(1 0 0 -1 0 9.61)" width="139.64"><span style="--ltx-fg-color:#FFFFFF;">Overview of takeaways</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 9506.39)"><foreignObject color="#000000" height="9508.09" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :40.23em;--fo_height:0.75em;--fo_depth :686.4em;" transform="matrix(1 0 0 -1 0 10.38)" width="556.69"><span style="width:40.23em;"><span id="S1.I1"><span id="S1.I1.i1" style="list-style-type:none;">• <span id="S1.I1.i1.p1">(Sec.&nbsp;2.1) Template is crucial to make base models answer questions instead of completing sentences. In addition, all base models already possess math-solving capability prior to RL.</span></span> <span id="S1.I1.i2" style="list-style-type:none;">• <span id="S1.I1.i2.p1">(Sec.&nbsp;2.2) Intriguingly, Qwen-2.5 base models get an immediate <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\sim 60\%"><semantics><mrow><mo>∼</mo> <mrow><mn>60</mn> <mo>%</mo></mrow></mrow> <annotation encoding="application/x-tex">\sim 60\%</annotation></semantics></math> improvement by not using template, making us hypothesize that they may pretrain on concatenated question-answer texts when cooking the models.</span></span> <span id="S1.I1.i3" style="list-style-type:none;">• <span id="S1.I1.i3.p1">(Sec.&nbsp;2.3) Nearly all base models already exhibit the “Aha moment”, including DeepSeek-V3-Base.</span></span> <span id="S1.I1.i4" style="list-style-type:none;">• <span id="S1.I1.i4.p1">(Sec.&nbsp;3.1, Sec.&nbsp;3.2) Dr.&nbsp;GRPO effectively fixes GRPO’s bias in optimization, achieving better token efficiency.</span></span> <span id="S1.I1.i5" style="list-style-type:none;">• <span id="S1.I1.i5.p1">(Sec.&nbsp;3.3) Model-template mismatch can destroy reasoning capabilities before RL reconstructs it.</span></span> <span id="S1.I1.i6" style="list-style-type:none;">• <span id="S1.I1.i6.p1">(Sec.&nbsp;3.4) Math pretraining on Llama-3.2-3B improves its RL ceiling.</span></span></span></span></foreignObject></g></g></svg>

## 2 Analysis on Base Models

In this section, we scrutinize a wide range of base models, including the Qwen-2.5 family [^33] [^34], Llama-3.1 [^7] and DeepSeek series [^19] [^27] [^8], asking them $500$ questions sampled from the MATH [^10] training set and analyzing their responses.

### 2.1 R1-Zero Trainability: Templates Construct Exploratory Base Policies

Since training from a base model is a fundamental setting of the R1-Zero-like paradigm, we first investigate whether widely used open-source base models, which are typically trained for sentence completion (i.e., $p_{\theta}({\mathbf{x}})$), can have their question-answering capabilities effectively elicited through appropriate templates, thereby functioning as a question-answering base policy $\pi_{\theta}(\cdot|{\mathbf{q}})$. In addition to the R1 template (Template 1) in [^8], we consider the Qwen-Math template (Template 2) used by [^36], as well as No template (Template 3):

<svg height="887.78" id="S2.SS1.p2.pic1" overflow="visible" version="1.1" viewBox="0 0 600 887.78" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,887.78) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0" style="--ltx-fill-color:#000000;"><path d="M 0 5.32 L 0 882.45 C 0 885.39 2.38 887.78 5.32 887.78 L 594.68 887.78 C 597.62 887.78 600 885.39 600 882.45 L 600 5.32 C 600 2.38 597.62 0 594.68 0 L 5.32 0 C 2.38 0 0 2.38 0 5.32 Z" style="stroke:none"></path></g><g fill="#F1F7FC" fill-opacity="1.0" style="--ltx-fill-color:#F1F7FC;"><path d="M 1.38 5.32 L 1.38 882.45 C 1.38 884.63 3.15 886.39 5.32 886.39 L 594.68 886.39 C 596.85 886.39 598.62 884.63 598.62 882.45 L 598.62 5.32 C 598.62 3.15 596.85 1.38 594.68 1.38 L 5.32 1.38 C 3.15 1.38 1.38 3.15 1.38 5.32 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 6.92 339.01)"><foreignObject color="#000000" height="871.17" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :42.36em;--fo_height:39.06em;--fo_depth :23.9em;" transform="matrix(1 0 0 -1 0 540.47)" width="586.16"><span style="width:42.36em;"><span id="Thmtemplate1"><h6>Template 1 (R1 template).</h6><span id="Thmtemplate1.p1">A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> think <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> /think <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> and answer is enclosed within <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> answer <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> /answer <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> tags, respectively, i.e., <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> think <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> reasoning process here <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> /think <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> answer <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> answer here <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> /answer <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math>.\nUser: <span style="--ltx-fg-color:#FF0000;">{question}</span> \nAssistant: <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> think <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math></span></span><span id="Thmtemplate2"><h6>Template 2 (Qwen-Math template).</h6><span id="Thmtemplate2.p1"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> |im_start| <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> system\nPlease reason step by step, and put your final answer within \\boxed{}.<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> |im_end| <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> \n <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> |im_start | <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> user\n <span style="--ltx-fg-color:#FF0000;">{question}<br><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo mathcolor="#000000" style="--ltx-fg-color:#000000;">&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math></span> |im_end| <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> \n <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&lt;"><semantics><mo>&lt;</mo> <annotation encoding="application/x-tex">&lt;</annotation></semantics></math> |im_start| <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="&gt;"><semantics><mo>&gt;</mo> <annotation encoding="application/x-tex">&gt;</annotation></semantics></math> assistant\n</span></span><span id="Thmtemplate3"><h6>Template 3 (No template).</h6><span id="Thmtemplate3.p1"><span style="--ltx-fg-color:#FF0000;">{question}</span></span></span></span></foreignObject></g></g></svg>

Experimental settings. We include Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-7B, Llama-3.1-8B, DeepSeek-Math-7B and DeepSeek-V3-Base-685B for experiments. For each model, we first apply No template to get the model responses, then let GPT-4o-mini to judge whether the model responses are in an answering format (regardless of quality) or in a sentence-completion pattern. We record the percentage of responses that tend to answer the question as the metric. We then apply both R1 template and Qwen-Math template to obtain model responses, and determine the most suitable template for each model based on the metric. Finally, we evaluate the pass@8 accuracy of each model with the corresponding template to assess whether the base policies can explore rewarding trajectories for RL improvement.

![[x3 30.png|Refer to caption]]

Figure 3: Model attributes across three aspects. Question-Answering Ability: the extent to which a pretrained language model provides a direct answer to a question rather than continuing or expanding upon it; Exploration Ability: pass@8 measures how well base models explore; Self-Reflection: counts are obtained through cross-validation between keyword-based detection and LLM-based detection, as detailed in Appendix D.

Results. The left plot of Fig. 3 shows how well base models (with or without templates) answer the provided questions. We observe that Llama and DeepSeek models all improve the answering ability by employing the proper template (R1 template). However, Qwen2.5 models work best (with $100\%$ answering rate) when no template is used. This intriguing property motivates further investigation which will be discussed in Sec. 2.2. Meanwhile, the lowest answering rate with no template suggests that DeepSeek-V3-Base is a nearly pure base model. This observation motivates us to explore whether a pure base model like DeepSeek-V3-Base demonstrates the Aha moment (Sec. 2.3). The middle plot of Fig. 3 shows the pass@8 accuracy of different base models (with template) at different sampling temperatures. This metric can serve as an indicator of base policy’s exploration ability. For example, if a base policy cannot even sample a single trajectory that leads to the correct final answer, it is impossible for RL to improve the policy because there is no reward signal. Our results demonstrate that all tested models are exploratory (thus ready for RL), with Qwen2.5 models performing the best (even surpassing DeekSeek-V3-Base). This might partially explain that most R1-Zero projects [^36] [^12] are based on Qwen2.5 models.

### 2.2 Qwen-2.5 Models Unlock the Best Performance When Discarding Template

We next dig into the intriguing observation (c.f. Fig. 3(Left)) that all Qwen2.5 base models readily serve as chat models even without any template. We take a step further to evaluate the reasoning ability of Qwen2.5-Math models on five standard benchmarks: AIME 2024 [^17], AMC [^17], MATH500 [^10], Minerva Math [^16], and OlympiadBench [^9]. Following common practice, we use greedy decoding and limit the sampling budget to 3000 tokens.

| Base model + Template | AIME24 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Math-1.5B |  |  |  |  |  |  |
| (4-shot prompting) | 0.0 | 20.0 | 50.4 | 12.1 | 15.9 | 19.7 |
| R1 template | 0.0 | 9.6 | 21.2 | 6.6 | 2.2 | 7.9 |
| Qwen template | 20.0 | 32.5 | 33.0 | 12.5 | 22.8 | 24.2 |
| No template | 16.7 | 43.4 | 61.8 | 15.1 | 28.4 | 33.1 |
| Qwen2.5-Math-7B |  |  |  |  |  |  |
| (4-shot prompting) | 3.3 | 22.5 | 61.6 | 10.7 | 20.9 | 23.8 |
| R1 template | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 |
| Qwen template | 16.7 | 38.6 | 50.6 | 9.9 | 16.6 | 26.5 |
| No template | 0.2 | 45.8 | 69.0 | 21.3 | 34.7 | 38.2 |

Table 1: Qwen2.5-Math models might be pretrained on concatenated question-answer text, resulting in peak performance when no template is applied.

As shown in Table 1, not using any template can drastically boost the average performance, resulting in an improvement of about $60\%$ compared to the traditional 4-shot prompting. Since Qwen2.5-Math [^34] uses chat model’s data (question-answer pairs) during the pretraining stage, we hypothesize that they might pretrain on the concatenated text to maximize $\log p_{\theta}({\mathbf{q}};{\mathbf{o}})$ directly. If our hypothesis turns out true, we shall be more careful about using Qwen2.5 models to reproduce DeepSeek-R1-Zero, since the base models are already SFT-like without templates.

### 2.3 Aha Moment Already Appears in Base Models Including DeepSeek-V3-Base

One of the most inspiring results of DeepSeek-R1-Zero is the emergence of self-reflection behaviors, a.k.a., Aha moment, through pure RL training. A few prior studies [^21] [^35] have suggested that there may not be Aha moment in open-source R1 replications because the base models they use already exhibit self-reflection keywords. However, they have not tested DeepSeek-V3-Base, on which the real R1-Zero model was RL-tuned. We complete this missing piece by hosting DeepSeek-V3-Base-685B ourselves and investigating its responses to the $500$ MATH questions with the R1 template. From the right plot of Fig. 3, we can observe that DeepSeek-V3-Base also generates a decent amount of self-reflections, further validating the claims of [^21]. We also show examples in App. E (Fig. 13) where DeepSeek-V3-Base generates keywords such as “Aha” and “wait”.

An additional important question is whether self-reflection behaviors are associated with improved model performance after RL training. To investigate this, we host DeepSeek-R1-Zero and analyze its responses to the same questions from the MATH dataset. Although self-reflection behaviors occur more frequently in R1-Zero, we observe that these behaviors are not positively correlated with higher accuracy. Detailed analysis can be found in App. F.

## 3 Analysis on Reinforcement Learning

Language model generation can be formulated as a token-level Markov Decision Process (MDP) ${\mathcal{M}}=({\mathcal{S}},{\mathcal{A}},r,p_{{\mathcal{Q}}})$. At each generation step $t$, the state $s_{t}\in{\mathcal{S}}$ is the concatenation of the input question and the output response generated so far: $s_{t}={\mathbf{q}};{\mathbf{o}}_{<t}=[q_{1},\dots,q_{M},o_{1},\dots,o_{t-1}]$. The policy $\pi_{\theta}(\cdot|s_{t})$ will select the next token $o_{t}$ from the vocabulary ${\mathcal{A}}$, resulting in a deterministic transition to the next state $s_{t+1}=s_{t};[o_{t}]$. The generation process starts from sampling an initial state $s_{1}={\mathbf{q}}\sim p_{{\mathcal{Q}}}$ from a set of questions, and stops when the autoregressive policy generates the \[eos\] token or exhausts the budget.

Typically, we maximize the entropy-regularized objective [^25]:

$$
\mathcal{J}(\pi_{\theta})=\underset{{{\mathbf{q}}\sim p_{\mathcal{Q}}}}{\mathbb{E}}\left[\underset{{\mathbf{o}}\sim\pi_{\theta}(\cdot|{\mathbf{q}})}{\mathbb{E}}[R({\mathbf{q}},{\mathbf{o}})]-\beta{\mathbb{D}}_{KL}[\pi_{\theta}(\cdot|{\mathbf{q}}))||\pi_{\text{ref}}(\cdot|{\mathbf{q}})]\right],
$$

where $R({\mathbf{q}},{\mathbf{o}})=\sum_{t=1}^{|{\mathbf{o}}|}r(s_{t},o_{t})$ is the return [^31] of the trajectory ${\mathbf{q}};{\mathbf{o}}$, and $\pi_{\text{ref}}$ is a reference policy. The KL regularization term is usually adopted ($\beta>0$) for reinforcement learning from human feedback [^5], where $r$ is a reward model learned from data collected by $\pi_{\text{ref}}$. In this case, regularization helps prevent $\pi_{\theta}$ from deviating too far from the distribution where the reward model is accurate [^13] [^30]. However, RL-tuning reasoning models typically employs rule-based verifiers as $r$ [^15], eliminating the concerns of distributional shift. This allows us to remove the KL term, which not only saves the memory and computation required by $\pi_{\text{ref}}$ during training, but also potentially leads to better performance for R1-Zero-like training [^12]. We will assume $\beta=0$ throughout this paper.

Policy optimization algorithms. To optimize $\pi_{\theta}$ with the above objective (Eq. 1 with $\beta=0$), Proximal Policy Optimization (PPO) [^26] maximizes the following surrogate objective:

$$
\begin{split}{\mathcal{J}}_{PPO}(\pi_{\theta})&=\mathbb{E}_{{\mathbf{q}}\sim p_{{\mathcal{Q}}},{\mathbf{o}}\sim\pi_{\theta_{\text{old}}}(\cdot|{\mathbf{q}})}\\
&\sum_{t=1}^{|{\mathbf{o}}|}\left\{\min\left[\frac{\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})}{\pi_{\theta_{\text{old}}}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})}\hat{A}_{t},\text{clip}(\frac{\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})}{\pi_{\theta_{\text{old}}}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})},1-\epsilon,1+\epsilon)\hat{A}_{t}\right]\right\},\end{split}
$$

where $\pi_{\theta_{\text{old}}}$ is the policy before the update, $\epsilon$ is the clipping hyperparameter, and $\hat{A}_{t}$ is an estimator of the advantage function of the $t$ -th token. A standard way to estimate $\hat{A}_{t}$ is to compute the Generalized Advantage Estimation (GAE) [^24] with a learned value model $V_{\phi}$. However, in the context of LLM RL-tuning, learning the value model is computationally expensive, so methods that estimate $\hat{A}_{t}$ without $V_{\phi}$ are practically preferred. For example, [^27] proposed GRPO, which first samples a group of responses $\{{\mathbf{o}}_{1},\dots,{\mathbf{o}}_{G}\}$ per question and computes their returns $\mathbf{R}=\{R_{1},\dots,R_{G}\}$, then sets the advantage of all tokens from ${\mathbf{o}}_{i}$ as $\hat{A}_{t}=\frac{R_{i}-\operatorname{mean}(\mathbf{R})}{\operatorname{std}(\mathbf{R})}$.

### 3.1 GRPO Leads to Biased Optimization

In Deepseek-R1-Zero [^8], a notable trend is the consistent increase in response length throughout the training process. This is frequently interpreted as an indication of the development of advanced reasoning abilities such as self-reflection. Recent studies [^23] [^36] [^12] have replicated this phenomenon using various algorithms and implementations. However, we argue that the observed increase in response length may also be attributed to a bias inherent in the GRPO [^27] objective function:

$$
\begin{split}\mathcal{J}_{GRPO}&(\pi_{\theta})=\mathbb{E}_{{\mathbf{q}}\sim p_{{\mathcal{Q}}},\{{\mathbf{o}}_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot|{\mathbf{q}})}\\
&\frac{1}{G}\sum_{i=1}^{G}{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\frac{1}{|{\mathbf{o}}_{i}|}}\sum_{t=1}^{|{\mathbf{o}}_{i}|}\left\{\min\left[\frac{\pi_{\theta}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})}\hat{A}_{i,t},\text{clip}\left(\frac{\pi_{\theta}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})},1-\epsilon,1+\epsilon\right)\hat{A}_{i,t}\right]\right\},\end{split}
$$

where

$$
\hat{A}_{i,t}=\frac{R({\mathbf{q}},{\mathbf{o}}_{i})-\operatorname{mean}({\{R({\mathbf{q}},{\mathbf{o}}_{1}),\dots,R({\mathbf{q}},{\mathbf{o}}_{G})\}})}{{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\operatorname{std}({\{R({\mathbf{q}},{\mathbf{o}}_{1}),\dots,R({\mathbf{q}},{\mathbf{o}}_{G})\}})}},
$$

with the return $R({\mathbf{q}},{\mathbf{o}}_{i})$ typically only including the outcome verifiable reward in LLM reasoning (the analysis also applies to process reward cases).

Compared to the objective function in Eq. 2, GRPO introduces two biases (see also Fig. 4):

- Response-level length bias: This arises from dividing by $|{\mathbf{o}}_{i}|$. For positive advantages ($\hat{A}_{i,t}>0$, indicating a correct response), this bias results in greater gradient updates for shorter responses, leading the policy to favor brevity in correct answers. Conversely, for negative advantages ($\hat{A}_{i,t}<0$, indicating an incorrect response), longer responses are penalized less due to their larger $|{\mathbf{o}}_{i}|$, causing the policy to prefer lengthier responses among incorrect ones.
- Question-level difficulty bias: This is caused by dividing the centered outcome reward by $\operatorname{std}(\{R({\mathbf{q}},{\mathbf{o}}_{1}),\dots,R({\mathbf{q}},{\mathbf{o}}_{G})\})$. Questions with lower standard deviations (e.g., those that are too easy or too hard, with the outcome rewards being almost all 1 or 0) are given higher weights during policy updates. While advantage normalization is a common trick in RL [^3], it is typically computed across an entire batch. In contrast, question-level normalization results in varying weights in the objective for different questions, leading to a difficulty bias in optimization.

![[x4 27.png|Refer to caption]]

Figure 4: Illustration of the biases in GRPO. Note that the effective advantage of GRPO a i, t a\_{i,t} is equivalent to a reweighted version of the unbiased advantage A ~ = R ( 𝐪 𝐨 ) − mean ⁡ 𝐑 \\tilde{A}\_{i,t}=R({\\mathbf{q}},{\\mathbf{o}}\_{i})-\\operatorname{mean}(\\mathbf{R}). The terms std \\operatorname{std}(\\mathbf{R}) and | |{\\mathbf{o}}\_{i}| could bias the optimization by assigning different weights to different questions and responses, as denoted by the sizes of the blue circles and the lengths of the orange arrows. Upward arrows indicate positive advantages, and vice versa.

Length Bias Also Exists in Open-Source PPO Implementations. We also examined several popular open-source implementations of vanilla PPO algorithms for LLM post-training. To our surprise, all of these implementations normalize the loss by response length (see LABEL:lst:ppo\_impl and Table 2), which misaligns with the PPO objective as defined in Eq. 2. This formulation-implementation misalignment was present even before the publication of GRPO. We speculate that the misalignment might originate from the pretraining stage [^29], where all tokens are packed into a fixed-length context and normalizing the loss by the context length (i.e., computing loss.mean(-1)) improves the numerical stability. However, in the RL-tuning stage, typical implementations [^32] normalize the loss by the response length, which is not a constant, introducing an unintended length bias.

Listing 1: Comparison between a typical open-source PPO loss implementation that is biased (red) and our implementation (green). MAX\_TOKENS is a global constant during the entire training (unless budget curriculum is enabled), which specifies the maximum number of generation tokens. Other constants also work with differences in gradient norm.

[⬇](data:text/plain;base64,ZGVmIG1hc2tlZF9tZWFuKHRlbnNvciwgbWFzaywgZGltKToKLSAgICByZXR1cm4gKHRlbnNvciAqIG1hc2spLnN1bShheGlzPWRpbSkgLyBtYXNrLnN1bShheGlzPWRpbSkKKyAgICByZXR1cm4gKHRlbnNvciAqIG1hc2spLnN1bShheGlzPS0xKSAvIE1BWF9UT0tFTlMKCnBwb19sb3NzID0gLi4uICAgICAgICMgY29tcHV0ZSBwZXItdG9rZW4gcHBvIGxvc3MKcmVzcG9uc2VfbWFzayA9IC4uLiAgIyBwZXItdG9rZW4gcmVzcG9uc2UgbWFzawojIHBlci1yZXNwb25zZSBsZW5ndGggbm9ybWFsaXphdGlvbiAoZS5nLiwgT3BlblJMSEYpCmxvc3NfdmFyaWFudDEgPSBtYXNrZWRfbWVhbihwcG9fbG9zcywgcmVzcG9uc2VfbWFzaywgZGltPS0xKS5tZWFuKCkKIyBPUiBwZXItYmF0Y2ggbGVuZ3RoIG5vcm1hbGl6YXRpb24gKGUuZy4sIHRybCwgdmVybCkKbG9zc192YXJpYW50MiA9IG1hc2tlZF9tZWFuKHBwb19sb3NzLCByZXNwb25zZV9tYXNrLCBkaW09Tm9uZSkubWVhbigp)

def masked\_mean(tensor, mask, dim):

\- return (tensor \* mask).sum(axis=dim) / mask.sum(axis=dim)

\+ return (tensor \* mask).sum(axis=-1) / MAX\_TOKENS

ppo\_loss =... # compute per-token ppo loss

response\_mask =... # per-token response mask

\# per-response length normalization (e.g., OpenRLHF)

loss\_variant1 = masked\_mean(ppo\_loss, response\_mask, dim=-1).mean()

\# OR per-batch length normalization (e.g., trl, verl)

loss\_variant2 = masked\_mean(ppo\_loss, response\_mask, dim=None).mean()

| Repository | Code Link | Unbiased? |
| --- | --- | --- |
| trl [^32] | [PPO Loss](https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/ppo_trainer.py#L573C1-L574C1) |  |
| OpenRLHF [^11] | [PPO Loss](https://github.com/OpenRLHF/OpenRLHF/blob/15d31511d7f63c410bdbea8be34854aafc90c0ac/openrlhf/models/loss.py#L76) |  |
| verl [^28] | [PPO Loss](https://github.com/volcengine/verl/blob/c6dc8b73cf011aa75b8c6a47b0322f50aed800ad/verl/trainer/ppo/core_algos.py#L301) |  |
| SimpleRL-Zero [^36] | [PPO Loss](https://github.com/hkust-nlp/simpleRL-reason/blob/41c9a893ea17dc4b5399dc2e5a14a53d81b373f6/train/openrlhf/models/loss.py#L48) |  |
| Open-Reasoner-Zero [^12] | [PPO Loss](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/e008f6d95f0b9a0e992f6b8bac912515b50a4634/orz/ppo/actors.py#L130) |  |

Table 2: Many open-sourced PPO implementations contain length bias.

### 3.2 Dr. GRPO: Group Relative Policy Optimization Done Right

To avoid the aforementioned optimization bias in GRPO, we propose to simply remove the ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\frac{1}{|{\mathbf{o}}_{i}|}}$ and ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\operatorname{std}({\{R({\mathbf{q}},{\mathbf{o}}_{1}),\dots,R({\mathbf{q}},{\mathbf{o}}_{G})\}})}$ normalization terms. Meanwhile, to faithfully implement the unbiased optimization objective, we could replace the mask.sum(axis=dim) with a constant value (e.g., generation budget) in the masked\_mean function in LABEL:lst:ppo\_impl, as highlighted by the line in green. Notably, these simple modifications recover the PPO objective in Eq. 2, with the advantage estimated by Monte Carlo return with an unbiased baseline [^31]. We give detailed derivations in App. A. We refer to our new optimization algorithm as Dr. GRPO. We next experimentally validate its effectiveness.

Experimental settings. We implement our algorithm using Oat [^20], a modular, research-friendly and efficient LLM RL framework. We adopt the Qwen2.5-1.5B base model and the R1 template (Template 1) for online RL-tuning. We implement the verification-based reward function using Math-Verify <sup>2</sup>, with the following minimalistic rule:

$$
R({\mathbf{q}},{\mathbf{o}})=\begin{cases}1&\text{if ${\mathbf{o}}$ contains the correct final answer to ${\mathbf{q}}$}\\
0&\text{otherwise}\end{cases}
$$

We run RL on questions sampled from the MATH [^10] training dataset, and compare the vanilla GRPO with the proposed Dr. GRPO. We evaluate the online model on five benchmarks: AIME2024, AMC, MATH500, Minerva Math and OlympiadBench. More experimental details including hyperparameters can be found in App. G.

![[x5 25.png|Refer to caption]]

Figure 5: Comparison of Dr. GRPO and GRPO in terms of training dynamics (Top) and evaluation results (Bottom).

Results. We report various metrics in Fig. 5 to demonstrate that Dr. GRPO can effectively mitigate the optimization bias and lead to better token efficiency. In particular, we first note that both GRPO and Dr. GRPO exhibit similar trend to DeepSeek-R1-Zero [^8], namely their response length increases along with training reward (Plots 1 & 2). However, we observe that GRPO tends to continually generate longer responses even when the reward improvement slows down (Plot 2). Although such a phenomenon is often referred to as the “emergence” of long-CoT through RL [^36] [^12], we argue that it is also confounded by the response-level length bias (Sec. 3.1) during optimization <sup>3</sup>. In contrast, by computing the unbiased policy gradients, Dr. GRPO prevents the response length from growing wildly during training (Plot 2). Moreover, on evaluation benchmarks, the length of incorrect responses is substantially reduced by Dr. GRPO compared to the baseline (Plot 4), suggesting that an unbiased optimizer also mitigates overthinking [^4].

![[x6 21.png|Refer to caption]]

Figure 6: The average benchmark accuracy of different {template, question set} combinations during RL training.

### 3.3 A Duet of Template and Question Set Coverage in RL dynamics

Recall that the Qwen2.5-Math base models can readily answer questions with high accuracy without any prompt template (Sec. 2.2). Based on this intriguing observation, we are interested in how different templates affect the RL training. Furthermore, given the general belief that larger question set coverage leads to better performance [^22] [^12], we also study the interaction between different templates and different levels of question coverage.

Experimental settings. Starting from the Qwen2.5-Math-1.5B base model, we apply R1 template, Qwen-Math template and No template respectively to run RL using Dr. GRPO. All experiments are repeated for different question sets that are detailed in Table 3.

| Question set | # | Description |
| --- | --- | --- |
| ORZ | 57k | Combining AIME, Numina-Math, Tulu3 MATH; diverse and large amount |
| MATH | 12k | High-school math competition questions |
| GSM | 8k | Simpler grade-school math questions |
| ASDiv | 2k | Basic algebra ($+-\times\div)$ questions |

Table 3: Different question sets that have different levels of difficulty and coverage.

Results. Fig. 6 shows the RL curves of different runs, from which we can make several interesting observations: 1) Templates determine the performance of the initial policies, but RL can improve all policies to a comparable performance of $\sim 40\%$ (given a proper question set); 2) When using the R1 template, question sets have a significant impact on the dynamics of RL, with too narrow coverage leading to lower plateau performance. However, when using the Qwen-Math template, the best final performance is attained by RL on GSM-8K, demonstrating that training on much simpler (and o.o.d.) questions can largely improve (nearly double) the test accuracy on harder questions. From these observations, we draw the following insights:

- The Qwen2.5-Math-1.5B base model already possesses strong math-solving capabilities (see the starting point in the right plot of Fig. 6). Applying templates in fact destroys the capability before RL reconstructs it. This implies that we should be more conservative in claiming the huge gains brought about by pure RL.
- When there is a large mismatch between base models and templates (e.g., R1 template mismatches Qwen2.5-Math-1.5B), the policy improvement mainly comes from RL-tuning, thus requiring question set to have good coverage (left plot of Fig. 6). Otherwise, even a small and completely o.o.d. question set could induce the reasoning ability equally well, by reinforcing useful reasoning behaviors instead of infusing new knowledge.

### 3.4 Domain-Specific Pretraining Improves RL Ceiling

Recent successful R1-Zero-like replications of math reasoners mostly employ Qwen2.5 base models as the initial policies [^36] [^6] [^12], which are already strong math solvers and exhibit self-reflection patterns (Sec. 2.2 and 2.3). In this section we hope to explore the other side: can R1-Zero-like training succeed on originally weak (in terms of math reasoning) base models? We answer this question affirmatively, with the observation that math pretraining would improve the ceiling of RL.

![[x7 17.png|Refer to caption]]

Figure 7: Left: The average benchmark performance curves of different base models. Right: The comparison between Dr. GRPO and GRPO with respect to reasoning accuracy (solid lines) and model response length (dashed lines).

Experimental settings. We adopt the Llama-3.2-3B base model as our starting point, and use the unbiased Dr. GRPO algorithm for RL-tuning with the R1 template. We hypothesize that domain-specific pretraining would help RL, hence we adopt the Llama-3.2-3B-FineMath <sup>4</sup>, which is continual pretrained on the FineMath dataset [^2]. Moreover, as we hypothesize that Qwen2.5 models are likely to be pretrained on concatenated question-response texts (Sec. 2.2), we similarly prepare a concatenated dataset from NuminaMath-1.5 [^18], and continual pretrain Llama-3.2-3B-FineMath for 2 epochs with learning rate 1e-5. We refer to the concatanated continual pretrained model as Llama-3.2-3B-NuminaQA.

Results. We present the RL curves of different base models in the left plot of Fig. 7. We observe that RL can even improve the vanilla Llama base model, but the gain is minimal. After continual pretraining (and concatenated continual pretraining) to embed math domain knowledge, Llama models can show much stronger RL performance, validating our hypothesis. We also revisit the GRPO’s optimization bias with the Llama base model. The right plot of Fig. 7 compares the model performance and response length trained with GRPO and Dr. GRPO. We can clearly see that GRPO can produce the “double-increase” phenomenon, potentially leading to a misperception that long-CoT can also emerge on Llama models after math pretraining. Unfortunately, the increase of length might be due to the optimization bias (Sec. 3.1), which can be effectively mitigated by the proposed Dr. GRPO (Sec. 3.2 & right plot of Fig. 7).

## 4 Closing Remarks

We have taken a critical perspective to examine base models used for R1-Zero-like training, as well as algorithms used for RL. Through the analysis, we demystified how pretraining biases influence RL outcomes and how optimization choices, like GRPO, can unintentionally shape model behavior. With the proposed Dr. GRPO, we offer a simple fix that improves token efficiency while preserving reasoning performance. Our results show that scaling RL can be both effective and efficient—sometimes, less really is more.

## References

## Appendix A Policy Gradient Derivations

In the context of RL for LLM post-training, we typically maximize the value of

$$
\mathcal{J}(\pi_{\theta})=\underset{{{\mathbf{q}}\sim p_{\mathcal{Q}}}}{\mathbb{E}}\left[\underset{{\mathbf{o}}\sim\pi_{\theta}(\cdot|{\mathbf{q}})}{\mathbb{E}}[R({\mathbf{q}},{\mathbf{o}})]\right],
$$

where $R({\mathbf{q}},{\mathbf{o}})=\sum_{t=1}^{|{\mathbf{o}}|}r({\mathbf{q}},{\mathbf{o}}_{\leq t})$ is the return [^31] of the trajectory ${\mathbf{q}};{\mathbf{o}}$, and $r({\mathbf{q}},{\mathbf{o}}_{\leq t})$ represents the token-level reward for $t$ -th token in response ${\mathbf{o}}$.

The Monte Carlo policy gradient [^31] of Eq. 4 is

$$
\begin{split}\nabla_{\theta}\mathcal{J}(\pi_{\theta})&=\underset{{{\mathbf{q}}\sim p_{\mathcal{Q}}}}{\mathbb{E}}\left[\underset{{\mathbf{o}}\sim\pi_{\theta}(\cdot|{\mathbf{q}})}{\mathbb{E}}[\nabla_{\theta}\log\pi_{\theta}({\mathbf{o}}|{\mathbf{q}})R({\mathbf{q}},{\mathbf{o}})]\right]\\
&=\underset{{{\mathbf{q}}\sim p_{\mathcal{Q}}}}{\mathbb{E}}\left[\underset{{\mathbf{o}}\sim\pi_{\theta}(\cdot|{\mathbf{q}})}{\mathbb{E}}[\nabla_{\theta}\sum_{t=1}^{|{\mathbf{o}}|}\log\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})R({\mathbf{q}},{\mathbf{o}})]\right]\\
&=\underset{{{\mathbf{q}}\sim p_{\mathcal{Q}}}}{\mathbb{E}}\left[\underset{{\mathbf{o}}\sim\pi_{\theta}(\cdot|{\mathbf{q}})}{\mathbb{E}}[\sum_{t=1}^{|{\mathbf{o}}|}\nabla_{\theta}\log\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})\sum_{t^{\prime}=t}^{|{\mathbf{o}}|}r({\mathbf{q}},{\mathbf{o}}_{\leq t^{\prime}})]\right]\\
&=\underset{{{\mathbf{q}}\sim p_{\mathcal{Q}}}}{\mathbb{E}}\left[\underset{{\mathbf{o}}\sim\pi_{\theta}(\cdot|{\mathbf{q}})}{\mathbb{E}}\left[\sum_{t=1}^{|{\mathbf{o}}|}\nabla_{\theta}\log\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})\left(\sum_{t^{\prime}=t}^{|{\mathbf{o}}|}r({\mathbf{q}},{\mathbf{o}}_{\leq t^{\prime}})-B({\mathbf{q}},{\mathbf{o}}_{<t})\right)\right]\right],\end{split}
$$

where $B({\mathbf{q}},{\mathbf{o}}_{<t})$ is a variance reduction term, which is invariant with respect to $o_{t}$ so that

$$
\displaystyle\underset{o_{t}\sim\pi_{\theta}(\cdot|{\mathbf{q}},{\mathbf{o}}_{<t})}{\mathbb{E}}[\nabla_{\theta}\log\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})B({\mathbf{q}},{\mathbf{o}}_{<t})]
$$
 
$$
\displaystyle=\underset{o_{t}\sim\pi_{\theta}(\cdot|{\mathbf{q}},{\mathbf{o}}_{<t})}{\mathbb{E}}[\nabla_{\theta}\log\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})]B({\mathbf{q}},{\mathbf{o}}_{<t})
$$
 
$$
\displaystyle=[\sum_{o_{t}}\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})\nabla_{\theta}\log\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})]B({\mathbf{q}},{\mathbf{o}}_{<t})
$$
 
$$
\displaystyle=[\sum_{o_{t}}\nabla_{\theta}\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})]B({\mathbf{q}},{\mathbf{o}}_{<t})
$$
 
$$
\displaystyle=[\nabla_{\theta}\sum_{o_{t}}\pi_{\theta}(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})]B({\mathbf{q}},{\mathbf{o}}_{<t})
$$
 
$$
\displaystyle=[\nabla_{\theta}1]B({\mathbf{q}},{\mathbf{o}}_{z<t})=0.
$$

Typically, we set $B({\mathbf{q}},{\mathbf{o}}_{<t})=\underset{{\mathbf{o}}_{\geq t}\sim\pi_{\theta}(\cdot|{\mathbf{q}},{\mathbf{o}}_{<t})}{\mathbb{E}}[\sum_{t^{\prime}=t}^{|{\mathbf{o}}|}r({\mathbf{q}},{\mathbf{o}}_{\leq t^{\prime}})]$, which is the expected cumulative reward in the future (also known as the value of the current state), and denote $A(o_{t}|{\mathbf{q}},{\mathbf{o}}_{<t})=\sum_{t^{\prime}=t}^{|{\mathbf{o}}|}r({\mathbf{q}},{\mathbf{o}}_{\leq t^{\prime}})-B({\mathbf{q}},{\mathbf{o}}_{<t})$ as the advantage. In the case of outcome reward, $\sum_{t^{\prime}=t}^{|{\mathbf{o}}|}r({\mathbf{q}},{\mathbf{o}}_{\leq t^{\prime}})=\sum_{t=1}^{|{\mathbf{o}}|}r({\mathbf{q}},{\mathbf{o}}_{\leq t})=R({\mathbf{q}},{\mathbf{o}})$.

By setting $B({\mathbf{q}},{\mathbf{o}}_{<t})=\operatorname{mean}({\{R({\mathbf{q}},{\mathbf{o}}_{1}),\dots,R({\mathbf{q}},{\mathbf{o}}_{G})\}})$, the policy gradient of Eq. 5 becomes

$$
\begin{split}\nabla_{\theta}\mathcal{J}(\pi_{\theta})&=\underset{{{\mathbf{q}}\sim p_{\mathcal{Q}}}}{\mathbb{E}}\left[\underset{\{{\mathbf{o}}_{i}\}_{i=1}^{G}\sim\pi_{\theta}(\cdot|{\mathbf{q}})}{\mathbb{E}}[\frac{1}{G}\sum_{i=1}^{G}\sum_{t=1}^{|{\mathbf{o}}|}\nabla_{\theta}\log\pi_{\theta}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})\tilde{A}_{i,t}]\right],\end{split}
$$

where

$$
\tilde{A}_{i,t}=\frac{R({\mathbf{q}},{\mathbf{o}}_{i})-\operatorname{mean}({\{R({\mathbf{q}},{\mathbf{o}}_{1}),\dots,R({\mathbf{q}},{\mathbf{o}}_{G})\}})}{{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\xcancel{\operatorname{std}({\{R({\mathbf{q}},{\mathbf{o}}_{1}),\dots,R({\mathbf{q}},{\mathbf{o}}_{G})\}})}}}.
$$
  

We adopt the PPO [^26] objective to compute Eq. 6:

$$
\begin{split}\mathcal{J}(\pi_{\theta})&=\mathbb{E}{[{\mathbf{q}}\sim p_{\mathcal{Q}},\{{\mathbf{o}}_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot|{\mathbf{q}})]}\\
&\frac{1}{G}\sum_{i=1}^{G}{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\xcancel{\frac{1}{|{\mathbf{o}}_{i}|}}}\sum_{t=1}^{|{\mathbf{o}}_{i}|}\left\{\min\left[\frac{\pi_{\theta}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})}\tilde{A}_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|{\mathbf{q}},{\mathbf{o}}_{i,<t})},1-\epsilon,1+\epsilon\right)\tilde{A}_{i}\right]\right\},\end{split}
$$

from which we conclude that both $\operatorname{std}$ and $|{\mathbf{o}}|$ should not appear in the RL objective.

Unbiasedness of $\tilde{A}_{i,t}$. We note that $\tilde{A}_{i,t}$ computed above is equivalent to that of REINFORCE Leave-One-Out (RLOO) [^1] [^14] up to a scaling factor, which can be subsumed into the learning rate without affecting the RL dynamics. Specifically,

$$
\begin{split}{\color[rgb]{0.6,0,0.4}\definecolor[named]{pgfstrokecolor}{rgb}{0.6,0,0.4}\frac{G}{G-1}}\cdot\tilde{A}_{i,t}&={\color[rgb]{0.6,0,0.4}\definecolor[named]{pgfstrokecolor}{rgb}{0.6,0,0.4}\frac{G}{G-1}}R({\mathbf{q}},{\mathbf{o}}_{i})-{\color[rgb]{0.6,0,0.4}\definecolor[named]{pgfstrokecolor}{rgb}{0.6,0,0.4}\frac{G}{G-1}}\frac{1}{G}\sum_{j=1}^{G}R({\mathbf{q}},{\mathbf{o}}_{j})\\
&={\color[rgb]{0.6,0,0.4}\definecolor[named]{pgfstrokecolor}{rgb}{0.6,0,0.4}\frac{G}{G-1}}R({\mathbf{q}},{\mathbf{o}}_{i})-\frac{1}{G-1}\sum_{j=1,j\neq i}^{G}R({\mathbf{q}},{\mathbf{o}}_{j})-\frac{1}{G-1}R({\mathbf{q}},{\mathbf{o}}_{i})\\
&=\hat{A}^{\text{RLOO}}_{i,t}.\end{split}
$$

## Appendix B Detailed Benchmark Results

We show the detailed benchmark results for three scales (1.5B, 3B and 7B) in Table 4. We also include the instruct models at the same scale and R1-Distill models for comparison. Note that since we employ the Qwen2.5-Math base models, which have a context length of 4k, we thus limit the generation budget at 3k for all baselines compared. For models that are trained for a longer context (OpenReasoner-Zero end R1-Distill-Qwen), we also report their performance at 8k generation budget.

| Base model + Method | AIME24 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Math-1.5B | 20.0 | 32.5 | 33.0 | 12.5 | 22.8 | 24.2 |
| Qwen2.5-Math-1.5B\* | 16.7 | 43.4 | 61.8 | 15.1 | 28.4 | 33.1 |
| Oat-Zero-1.5B | 20.0 | 53.0 | 74.2 | 25.7 | 37.6 | 42.1 |
| R1-Distill-Qwen-1.5B @ 3k | 2.5 | 21.7 | 52.2 | 16.3 | 17.3 | 22.0 |
| R1-Distill-Qwen-1.5B @ 8k | 20.0 | 49.4 | 77.4 | 25.0 | 35.8 | 41.5 |
| Qwen2.5-Math-1.5B-Instruct | 10.0 | 48.2 | 74.2 | 26.5 | 40.2 | 39.8 |
| Llama-3.2-3B | 0.0 | 2.4 | 6.4 | 6.3 | 1.3 | 3.3 |
| \+ RL w. Dr. GRPO | 3.3 | 7.2 | 10.0 | 11.0 | 2.2 | 6.8 |
| Llama-3.2-3B-FineMath | 0.0 | 3.6 | 18.4 | 5.9 | 2.2 | 6.0 |
| \+ RL w. Dr. GRPO | 3.3 | 10.8 | 38.0 | 12.9 | 9.0 | 14.8 |
| Llama-3.2-3B-NuminaQA | 0.0 | 0.0 | 0.6 | 0.0 | 0.1 | 0.14 |
| \+ RL w. Dr. GRPO (Oat-Zero-3B) | 6.7 | 18.1 | 50.0 | 14.3 | 14.7 | 20.7 |
| Llama-3.2-3B-Instruct | 6.7 | 15.7 | 38.8 | 11.8 | 12.6 | 17.1 |
| Qwen2.5-Math-7B | 16.7 | 38.6 | 50.6 | 9.9 | 16.6 | 26.5 |
| Qwen2.5-Math-7B\* | 0.2 | 45.8 | 69.0 | 21.3 | 34.7 | 38.2 |
| SimpleRL-Zero-7B | 26.7 | 60.2 | 78.2 | 27.6 | 40.3 | 46.6 |
| PRIME-Zero-7B | 16.7 | 62.7 | 83.8 | 36.0 | 40.9 | 48.0 |
| OpenReasoner-Zero-7B @ 3k | 13.3 | 47.0 | 79.2 | 31.6 | 44.0 | 43.0 |
| OpenReasoner-Zero-7B @ 8k | 13.3 | 54.2 | 82.4 | 31.6 | 47.9 | 45.9 |
| Oat-Zero-7B | 43.3 | 62.7 | 80.0 | 30.1 | 41.0 | 51.4 |
| R1-Distill-Qwen-7B @ 3k | 10.0 | 26.2 | 60.1 | 23.0 | 23.1 | 28.5 |
| R1-Distill-Qwen-7B @ 8k | 33.3 | 68.4 | 88.1 | 35.9 | 47.7 | 54.7 |
| Qwen2.5-Math-7B-Instruct | 16.7 | 53.0 | 83.6 | 29.8 | 42.7 | 45.1 |

Table 4: A comparison on benchmark scores. Ours models are RL-tuned by our minimalist recipe (Sec. 1). \* means we employ the best template (no template) to generate answers, such that the test scores are highest and can faithfully reflect the capabilities of the base models.

## Appendix C Extended Empirical Results

In this section we present two extended empirical results for (1) the ablation of different bias terms in GRPO and (2) statistical significance of Dr. GRPO’s results. We RL-tune the Qwen2.5-1.5B base model on a mixture of 3K diverse math questions drawn from ASDiv, MATH, and AIME (pre-2023).

![[x8 12.png|Refer to caption]]

Figure 8: Ablation results on the two bias terms in GRPO.

Fig. 8 shows the training and evaluation curves for the following variants: Dr. GRPO, GRPO w/o length normalization, GRPO w/o standard deviation (std) normalization and Vanilla GRPO. From the middle subplot, we observe that both Dr. GRPO and the variant without length normalization generate shorter responses compared to the other two. This confirms that the length bias term has a more significant influence on response length–consistent with our expectations.

In terms of performance, Dr. GRPO and the other ablated variants consistently outperform vanilla GRPO in both training rewards and evaluation accuracy. This indicates that removing bias terms (either length or std) improves policy learning, validating our motivation for Dr. GRPO.

![[x9 6.png|Refer to caption]]

Figure 9: Evaluation results of 3 independent RL runs. The mean curves are drawn in solid lines and the standard deviation is plotted in the shaded areas.

Fig. 9 compares GRPO and Dr. GRPO across three independent runs. We observe that Dr. GRPO consistently demonstrates statistically significant improvements–both in token efficiency and final accuracy–across different random seeds.

## Appendix D Keyword-based Detection and LLM-Based Identification of Self-Reflection Behaviors

We construct a pool of carefully selected keywords and phrases that signal self-reflection behaviors in the LLM’s responses. However, LLM-generated responses often contain hallucinations and off-topic content, leading to the presence of simple, ambiguous keywords that do not necessarily indicate genuine self-reflection. For instance, terms like “wait” and “try again” frequently result in false positive detections. To reduce false positives, we maintain a small, highly selective keyword pool consisting of terms that are strongly indicative of self-reflection. In our experiment, the keyword pool is limited to: recheck, rethink, reassess, reevaluate, re-evaluate, reevaluation, re-examine, reexamine, reconsider, reanalyze, double-check, check again, think again, verify again, and go over the steps.

![[x10 7.png|Refer to caption]]

Figure 10: Count of keyword occurrences out of 40,000 responses (500 questions × \\times 8 responses per question 10 temperatures). y is in log scale.

We present the occurrences of various keywords in the responses generated by different models in Figure 10. Interestingly, different model families emphasize different keywords. For instance, phrases such as “check again”, “double-check”, “re-evaluate”, “re-examine”, “recheck”, “reconsider”, and “verify again” appear most frequently in the Qwen2.5 family. In contrast, “re-evaluate”, “re-examine”, and “verify again” do not appear in the responses of the DeepSeek family, while Llama models frequently use the phrase “think again.” We hypothesize that this phenomenon results from differences in the pretraining data, particularly in relation to reasoning and mathematics.

Although we meticulously select the keyword pool, it may still be insufficient to identify some implicit behaviors of self-reflection that do not contain a specific keyword. Additionally, it can lead to false positives, as illustrated in Case (a) of Figure 11. To address these limitations and more accurately assess the self-reflection capability of base models, we leverage stronger LLMs (GPT-4o-mini in our experiments) to analyze the responses and determine whether they exhibit explicit self-reflection (e.g., keywords like ”recheck” and ”reevaluate”) or implicit self-reflection (e.g., more sophisticated patterns that cannot be easily captured through keyword matching). This approach helps distinguish true self-reflection behaviors from superficial or incidental use of related terms.

![[x11 4.png|Refer to caption]]

Figure 11: Case (a): a false positive in keyword-based detection. Case (b): a false positive in LLM-based detection.

While LLM-based detection effectively filters out false positives from keyword-based detection and identifies implicit self-reflection behaviors, it can still misclassify responses, particularly when they are lengthy and complex. For instance, Case (b) in Figure 11 shows a false positive in LLM-based detection, where the response is categorized as self-reflection by the LLM but does not actually exhibit self-reflection. This type of error can be filtered out by keyword-based detection. To enhance robustness, we integrate keyword-based and LLM-based detection through cross-validation. The combined detection results, along with the individual results from keyword-based and LLM-based methods, are presented in Figure 12.

![[x12 3.png|Refer to caption]]

Figure 12: Comparison of keyword-based detection, LLM-based detection, and cross detection. Self-reflections are counted at the question level across 500 questions, where a question is marked as having self-reflection if at least one of its eight responses exhibits self-reflection.

## Appendix E Examples of Aha Moment in DeepSeek-V3-Base

Fig. 13 shows two examples to demonstrate that the DeepSeek-V3-Base model already exhibits the so-called “aha moment” even before the RL-tuning.

![[x13 6.png|Refer to caption]]

Figure 13: Cases showing that DeepSeek-V3-Base already exhibits “Aha moment” even before RL tunning.

## Appendix F Comparison Between DeepSeek-V3-Base and DeepSeek-R1-Zero

![[x14 2.png|Refer to caption]]

Figure 14: Breakdown of response categories across difficulty levels in the MATH dataset for DeepSeek-V3-Base and DeepSeek-R1-Zero.

We analyze DeepSeek-V3-Base and DeepSeek-R1-Zero to understand changes in model behavior during R1-Zero training. In Fig. 14, we present the breakdown of response categories across difficulty levels for 500 MATH questions evaluated on both models. The results indicate that most incorrect responses are corrected after RL training, demonstrating substantial performance gains from R1-Zero training. Meanwhile, we find an increase in unformatted responses, which aligns with the observation in [^21].

In Table 5, we report the average response lengths across categories. Note that truncated responses would fall into any of the other three categories if a larger context size were used; thus, we exclude them from the table. The results show a substantial increase in response lengths across all categories, including correct responses, consistent with the results in the Fig. 3 of [^8]. However, the average length of incorrect responses is notably longer than that of correct responses. We hypothesize this is because more challenging questions generally require longer responses due to increased reasoning complexity, and incorrect responses are more likely to originate from harder questions, resulting in a longer average length.

![[x15 2.png|Refer to caption]]

Figure 15: Accuracy difference between responses with and without self-reflection for each question (responses sampled from DeepSeek-R1-Zero).

Self-reflection does not necessarily imply higher accuracy. To investigate whether self-reflection behaviors are associated with model performance during the inference (acknowledging that self-reflection may improve exploration during training—a potential positive effect outside this section’s scope), we analyze questions that elicit at least one response with self-reflection from DeepSeek-R1-Zero across eight trials. For each question, we sample 100 responses and divide them into two groups: those with self-reflection and those without. We then compute the accuracy difference between these two groups for each question. As shown in Fig. 15, the results indicate that nearly half responses with self-reflection do not achieve higher accuracy than those without self-reflection, suggesting that self-reflection does not necessarily imply higher inference-stage accuracy for DeepSeek-R1-Zero.

## Appendix G Detailed Experimental Settings

All our experiments are performed on 8 $\times$ A100 GPUs and finished in about one day. We enable the actor-learner collocation supported by Oat [^20] to optimize the training efficiency. We show the experimental configurations in Table 6.

<table><tbody><tr><th>Parameter</th><td>Value</td></tr><tr><th colspan="2">Actor</th></tr><tr><th>Maximum response length</th><td><math><semantics><mn>3000</mn> <annotation>3000</annotation></semantics></math> tokens</td></tr><tr><th>Sampling temperature</th><td>1.0</td></tr><tr><th>(top P, top k)</th><td>(1.0, -1)</td></tr><tr><th>Number of responses per question</th><td>8</td></tr><tr><th colspan="2">Learner</th></tr><tr><th>Optimizer</th><td>AdamW</td></tr><tr><th>Adam parameters (<math><semantics><mrow><msub><mi>β</mi> <mn>1</mn></msub><mo>,</mo><msub><mi>β</mi> <mn>2</mn></msub></mrow> <annotation>\beta_{1},\beta_{2}</annotation></semantics></math>)</th><td>(0.9, 0.95)</td></tr><tr><th>Weight decay</th><td>0.0</td></tr><tr><th>Gradient norm clipping</th><td>1.0</td></tr><tr><th>Learning rate scheduler</th><td>Constant</td></tr><tr><th>Learning rate</th><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>6</mn></mrow></msup></mrow> <annotation>1\times 10^{-6}</annotation></semantics></math></td></tr><tr><th>Inner proximal update epoch</th><td>1</td></tr><tr><th>KL loss coefficient</th><td>0.0</td></tr><tr><th>KL penalty coefficient</th><td>0.0</td></tr><tr><th>Policy clipping parameter</th><td>0.2</td></tr></tbody></table>

Table 6: Hyperparameter configurations used in all experiments.

[^1]: Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. *arXiv preprint arXiv:2402.14740*, 2024.

[^2]: Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, et al. Smollm2: When smol goes big–data-centric training of a small language model. *arXiv preprint arXiv:2502.02737*, 2025.

[^3]: Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphaël Marinier, Leonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al. What matters for on-policy deep actor-critic methods? a large-scale study. In *International conference on learning representations*, 2021.

[^4]: Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. *arXiv preprint arXiv:2412.21187*, 2024.

[^5]: Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. *Advances in neural information processing systems*, 30, 2017.

[^6]: Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. *arXiv preprint arXiv:2502.01456*, 2025.

[^7]: Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

[^8]: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

[^9]: Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. *arXiv preprint arXiv:2402.14008*, 2024.

[^10]: Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021.

[^11]: Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. *arXiv preprint arXiv:2405.11143*, 2024.

[^12]: Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang. Open-reasoner-zero: An open source approach to scaling reinforcement learning on the base model. [https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), 2025.

[^13]: Natasha Jaques, Asma Ghandeharioun, Judy Hanwen Shen, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, and Rosalind Picard. Way off-policy batch deep reinforcement learning of implicit human preferences in dialog. *arXiv preprint arXiv:1907.00456*, 2019.

[^14]: Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free!, 2019.

[^15]: Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T $\backslash$ ” ulu 3: Pushing frontiers in open language model post-training. *arXiv preprint arXiv:2411.15124*, 2024.

[^16]: Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. *Advances in Neural Information Processing Systems*, 35:3843–3857, 2022.

[^17]: Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. *Hugging Face repository*, 13:9, 2024a.

[^18]: Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath, 2024b.

[^19]: Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.

[^20]: Zichen Liu, Changyu Chen, Chao Du, Wee Sun Lee, and Min Lin. Oat: A research-friendly framework for llm online alignment. [https://github.com/sail-sg/oat](https://github.com/sail-sg/oat), 2025a.

[^21]: Zichen Liu, Changyu Chen, Wenjun Li, Tianyu Pang, Chao Du, and Min Lin. There may not be aha moment in r1-zero-like training — a pilot study. [https://oatllm.notion.site/oat-zero](https://oatllm.notion.site/oat-zero), 2025b. Notion Blog.

[^22]: Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. [https://github.com/agentica-project/deepscaler](https://github.com/agentica-project/deepscaler), 2025.

[^23]: Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero. https://github.com/Jiayi-Pan/TinyZero, 2025. Accessed: 2025-01-24.

[^24]: John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*, 2015.

[^25]: John Schulman, Xi Chen, and Pieter Abbeel. Equivalence between policy gradients and soft q-learning. *arXiv preprint arXiv:1704.06440*, 2017a.

[^26]: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017b.

[^27]: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

[^28]: Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv:2409.19256*, 2024.

[^29]: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*, 2019.

[^30]: Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. *Advances in neural information processing systems*, 33:3008–3021, 2020.

[^31]: Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*. The MIT Press, second edition, 2018.

[^32]: Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. Trl: Transformer reinforcement learning. [https://github.com/huggingface/trl](https://github.com/huggingface/trl), 2020.

[^33]: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*, 2024a.

[^34]: An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. *arXiv preprint arXiv:2409.12122*, 2024b.

[^35]: Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. *arXiv preprint arXiv:2502.03373*, 2025.

[^36]: Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, and Junxian He. 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient. [https://hkust-nlp.notion.site/simplerl-reason](https://hkust-nlp.notion.site/simplerl-reason), 2025. Notion Blog.