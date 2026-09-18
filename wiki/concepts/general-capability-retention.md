---
title: General-Capability Retention After Driving Adaptation
type: concept
sources: [raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md]
related: [concepts/vlm-domain-adaptation.md, concepts/foundation-backbones-for-ad.md, concepts/mixture-of-experts.md, concepts/dual-system-vla.md, sources/qwen-drive-1.0.md, sources/automot.md, sources/unidrivevla.md, sources/alpamayo-r1.md, sources/percept-wam.md, sources/onedrive.md]
created: 2026-09-18
updated: 2026-09-18
confidence: medium
---

## What This Page Tracks

Every driving VLA in this wiki starts from a general vision-language model and trains it on driving data. **What happens to the general model is almost never measured.** This page collects the measurements that exist, the two arguments for why retention matters, and the one protocol that has been run across many models at once.

It is a sibling of [[concepts/vlm-domain-adaptation.md]], which tracks *how* adaptation is done. This page tracks *what it costs*.

## Two Arguments for Caring

**1. Open-set robustness.** The standard argument, made by [[sources/automot.md]], [[sources/unidrivevla.md]] and [[sources/qwen-drive-1.0.md]] alike: no finite driving dataset covers deployment, and the pretrained model's world knowledge is what remains when the scene is unlike anything in `navtrain`. This argument is nearly untestable — no ingested paper has shown a case where retained general knowledge demonstrably rescued a driving decision.

**2. Cockpit-driving integration.** [[sources/qwen-drive-1.0.md]] introduces a second argument that *is* testable:

> "Production vehicles are moving towards cockpit-driving integration, in which the intelligent cockpit and the driving system share a single compute platform rather than two separate domain controllers... A model that trades general capability for driving performance forfeits this benefit, because the cockpit functions would then require a separate model and additional compute."

This changes the target. Under argument 1, "acceptable degradation" is a judgement call. Under argument 2 the bar is **parity with the base model on ordinary multimodal tasks**, because the alternative is shipping a second model — and a model that has lost its instruction-following interface fails the bar even if the knowledge is still in the weights.

## The Measurements

### Qwen-Drive-1.0: 15 benchmarks, 13 models, one protocol {#the-table}

[[sources/qwen-drive-1.0.md]]'s Table 3 is the only cross-method measurement in the wiki. Every model is re-evaluated by the authors under one near-deterministic decoding setting (top-$k$ 1, top-$p$ 0.001, temperature 0.01), with no per-model prompt adaptation, and an unparsable response scored as zero.

Group (a) is knowledge/reasoning/recognition (MMBench, MMStar, MMMU, MMMU-Pro ×2, CharXiv, OCRBench, RealWorldQA, SimpleVQA, CountQA); group (b) is spatial understanding and grounding (EmbSpatial, ERQA, RefSpatial, Omni3D, ODinW13).

| Model | Params | Specialization | (a) avg | (b) avg | 15-setting avg |
|---|---:|---|---:|---:|---:|
| **Qwen3.5-4B** | 4 B | none (base model) | **67.40** | 52.99 | **62.60** |
| **Qwen-Drive-1.0-SFT** | 4 B + 1.1 B | driving: perception + VQA + planning | **66.41** | **53.96** | **62.26** |
| Gemma4-12B | 12 B | none | 62.77 | 23.03 | 49.52 |
| Cosmos-Reason2-32B | 32 B | physical AI | 60.93 | 48.49 | 56.78 |
| Cosmos3-nano | – | omnimodal | 55.98 | 37.45 | – |
| Cosmos-Reason2-8B | 8 B | physical AI | 55.13 | 49.14 | – |
| InternVL3.5-8B-Instruct | 8 B | none | 54.84 | 23.24 | – |
| Cosmos-Reason1-7B | 7 B | physical AI | 52.77 | 22.48 | – |
| LLaVA-OV2-8B | 8 B | none | 51.49 | 24.14 | – |
| **UniDriveVLA-8B** | 8 B | driving | **48.73** | **21.59** | – |
| Cosmos-Reason2-2B | 2 B | physical AI | 46.87 | 40.42 | – |
| **MiMo-Embodied-7B** | 7 B | embodied + driving | **26.53** | **17.39** | – |
| **Alpamayo-1.5-10B** | 10 B | driving (chain-of-causation) | **14.65** | **9.62** | – |

**The ordering by specialization is close to monotone.** General models cluster at 51–63 on group (a). The physical-AI family sits 47–61. The three purely driving-or-embodiment models sit at 48.73, 26.53 and 14.65. The one model that adapted for driving *and* kept its general mixture sits at 66.41, within a point of the strongest base model in the table.

### The retention recipe that produced 66.41

Four choices, none of them novel individually, all four together:

1. **The pretrained architecture is unchanged** — no added special tokens, no vocabulary expansion, no MoT split, no replaced FFNs. View and frame tags use ordinary vocabulary.
2. **26.0 % of the Stage-2 mixture is general-purpose vision-language data** (31.0 % after repetition), against 64.3 % driving VL and 9.7 % perception.
3. **The driving supervision is heterogeneous and format-diverse** — 24 datasets, multi-view, temporal, video, MCQ and open-ended, grounding and captioning — rather than one task template.
4. **Everything below the action head is trained only once.** Stages 3 and 4 freeze the vision encoder and VLM entirely, so RL cannot damage what Stage 2 preserved.

### The two prior single-model measurements

**[[sources/automot.md]]** compared a frozen understanding expert against the same expert fine-tuned on driving data:

| Benchmark | Frozen | AD fine-tuned | Δ |
|---|---:|---:|---|
| ScienceQA | 88.60 | 87.80 | −0.8 |
| FigureQA | 97.60 | 91.20 | −6.4 |
| TallyQA | 81.40 | 52.40 | **−35 %** |
| InfographicVQA | 89.30 | 50.20 | **−44 %** |
| VizWiz | 75.60 | 50.20 | **−34 %** |

and concluded that fine-tuning should be restricted to action-level components.

**[[sources/unidrivevla.md]]** showed that MoT reduces but does not remove the damage — MMStar 63.0 → 43.3, RealWorldQA 69.0 → 49.9, VLMsAreBlind 61.9 → 26.6 — and its 48.73 in the table above is an independent re-measurement of the same model by a different group, consistent in direction.

### What the three together say {#reconciliation}

AutoMoT's conclusion — *do not fine-tune the VLM* — was drawn from one experiment in which the driving data was, by its own description, a driving-only mixture. Qwen-Drive fine-tunes the vision encoder **and** the full VLM, under perception losses that backpropagate through the entire language model, and loses 0.99 points.

**The variable that separates them is not whether the backbone is frozen. It is what is in the mixture.** That is a cheaper fix than architecture — no MoT, no frozen expert, no async KV cache — and it is the reading this page currently holds:

> Catastrophic forgetting in driving adaptation is a **data-mixture** failure before it is an architectural one. Roughly a quarter to a third general-purpose data appears to be enough to hold parity, at 4 B, with an unchanged architecture.

It is one data point, from the lab that owns the base model, and nobody has run the mixture-ratio sweep that would turn it into a rule. **That sweep is the single cheapest high-value experiment this page can name**: hold Qwen-Drive's recipe fixed and vary the general fraction across 0 %, 10 %, 26 %, 50 %.

## The Confound: Knowledge vs. Interface {#format-confound}

Alpamayo-1.5-10B scores **7.51 on MMBench and 3.20 on OCRBench**. Those are not capability scores for a 10 B model built on a strong physical-AI backbone — they are parse failures. MiMo-Embodied-7B returns unparsable output on three of ten group-(a) benchmarks and its LingoQA answer in Appendix C repeats a self-correction for ~4,000 characters.

So the table measures two things at once:

- **retained knowledge** — what the weights still encode, and
- **retained interface** — whether the model will still answer an arbitrary prompt in an arbitrary requested format.

They cannot be separated from these numbers. A per-model prompt-tuning pass would separate them and no one has run it.

**But the conflation is not obviously a flaw.** Under the cockpit argument the interface *is* the product: a cockpit model that cannot follow "answer with a single letter" is unusable regardless of what it knows. The page's position: **treat these numbers as deployable general capability, and treat any claim about knowledge in the weights as unmeasured.**

Qwen-Drive's own text draws the same distinction and lands on the same side:

> "Several other specialized models produce invalid responses on some benchmarks, showing that strong domain specialization does not necessarily preserve the general instruction-following interface required across diverse tasks."

## The Second Confound: Who Is Judging {#judge-confound}

Qwen-Drive is a Qwen3.5-4B derivative; the LLM judge is Qwen3.5-Plus; the data filter is Qwen3.5-Flash; the CoC training traces were written by Qwen3.7-Plus. The general-VLM benchmarks in group (a)/(b) are mostly multiple-choice or exact-match and are largely insulated from this, which is why this page treats the retention table as more trustworthy than the same paper's driving-VQA table. But the base-model row is the lab's own model, and "our adaptation preserves our base model" is a claim with an interested party on both sides.

## Open Questions

1. **What is the minimum general-data fraction?** Unmeasured. 26 % works at 4 B; 0 % clearly fails; nothing between has been run.
2. **Does retention survive RL?** Qwen-Drive freezes the VLM during Stages 3–4, so its RL cannot forget anything — and the paper reports no post-RL VQA evaluation at all. Every wiki method that runs GRPO *through* the VLM ([[sources/autovla.md]], [[sources/recogdrive.md]], [[sources/adathinkdrive.md]], [[sources/nord.md]], and a dozen others) has an unmeasured exposure here.
3. **Does retained general capability ever help driving?** Argument 1 predicts it should show up in long-tail or OOD scenarios. The closest available evidence is indirect and points the other way: Qwen-Drive's own Table 8 prices its entire Stage-2 knowledge adaptation at **+0.08 RFS** for planning. Retention may be worth having for the cockpit and worth nothing for the plan.
4. **Is the base model a fair reference?** Qwen-Drive is compared against the model it was built from. UniDriveVLA was compared against Qwen3-VL-8B by its own authors and re-measured here by others; the two disagree in level but not in direction. Nobody has run this protocol on a method's *own* base outside the authoring lab.
5. **Does anyone else report it?** Of the 74 papers in this wiki, three measure post-adaptation general capability. That is the fact this page most wants to change.
