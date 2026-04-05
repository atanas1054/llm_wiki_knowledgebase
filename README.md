# LLM-Powered AD Research Wiki

A structured knowledge base for end-to-end autonomous driving research, built by ingesting academic papers with an LLM assistant (Claude). Each paper is read, discussed, and distilled into cross-linked wiki pages covering source summaries and reusable concept notes.

## Structure

```
raw/
  papers/      # Source papers (markdown, converted from arXiv HTML)
  assets/      # Figures extracted from papers
wiki/
  sources/     # One page per paper — summary, figures, tables, limitations
  concepts/    # Cross-paper concept notes (updated as new papers are ingested)
  index.md     # Master catalog of all wiki pages
  log.md       # Append-only ingest log
CLAUDE.md      # Workflow instructions for the LLM assistant
```

## Papers Ingested

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| [ReCogDrive](wiki/sources/recogdrive.md) | 2025 | VLM + DiT diffusion planner + GRPO RL; 89.6 PDMS on NAVSIM-v1 |
| [WAM-Flow](wiki/sources/wam-flow.md) | 2025 | Discrete flow matching (CTMC) + GRPO; 90.3 PDMS with 1 camera |
| [UniUGP](wiki/sources/uniugp.md) | 2025 | Unified VLA + world model; MoT architecture; SOTA nuScenes FID/FVD |
| [Senna-2](wiki/sources/senna2.md) | 2025 | Dual-system VLM+E2E consistency alignment; 3DGS HRL; 86.6 EPDMS |
| [ReflectDrive](wiki/sources/reflectdrive.md) | 2025 | Masked discrete diffusion + gradient-free reflective inference |
| [Reasoning-VLA](wiki/sources/reasoning-vla.md) | 2025 | Learnable action queries (1-step parallel); unified 8-dataset corpus |

## Concept Pages

| Concept | Description |
|---------|-------------|
| [Diffusion-Based Trajectory Planner](wiki/concepts/diffusion-planner.md) | Continuous diffusion, DFM, masked diffusion, FM, and learnable-query paradigms compared |
| [Discrete Flow Matching](wiki/concepts/discrete-flow-matching.md) | CTMC-based DFM theory; WAM-Flow vs. masked diffusion distinction |
| [RL for Autonomous Driving](wiki/concepts/rl-for-ad.md) | GRPO variants: simulator-based, GT-based, hierarchical (3DGS) |
| [VLM Domain Adaptation](wiki/concepts/vlm-domain-adaptation.md) | Data curation, CoT integration, multi-dataset generalization strategies |
| [NAVSIM Benchmark](wiki/concepts/navsim-benchmark.md) | PDMS/EPDMS metrics; current SOTA table |
| [World Models for AD](wiki/concepts/world-model-for-ad.md) | Video generation as causal learning; FID/FVD metrics |
| [Dual-System VLA](wiki/concepts/dual-system-vla.md) | VLM decisions + E2E trajectory; consistency alignment |
| [Inference-Time Safety](wiki/concepts/inference-time-safety.md) | Gradient-free safety correction; inpainting-as-repair pattern |

## Workflow

The ingest workflow is defined in `CLAUDE.md`. Each paper goes through:
1. Read source + figures
2. Discuss key takeaways and limitations
3. Create/update `wiki/sources/<paper>.md` with embedded figures and full tables
4. Create/update relevant `wiki/concepts/` pages
5. Update `wiki/index.md` and append to `wiki/log.md`
