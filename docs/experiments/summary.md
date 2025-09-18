---
title: [Category] - [Specific Topic/ID, e.g., Run 001: Baseline HPO]
date: 2025-09-18
tags: [e.g., fov-90, real-data, ray-attention]
status: [e.g., in-progress, complete]
links: [e.g., [[understanding/ray_attention.md]], ClearML Task ID: 12345]
---

# [Category] - [Topic/ID]

## Summary
[1-2 paragraph overview: What was done? Key findings? Why it matters. E.g., "Tested ray-attention on 90° FOV real airport data; reduced depth MAE by 8% but increased yaw error."]

## Context/Background
[Link to prior work. E.g., "Builds on baseline from run_000; motivated by domain shift observations in understanding/mono_depth.md."]

## Details
### Setup/Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 0.001 | Adjusted from paper for real noise |
| num_heads | 4 | Ray-attention hyper |
| Dataset | Airport train split | 80% of sequences |

[Include code snippets if relevant, e.g.,]
```python
# Key config from ClearML
ray_params = {'num_heads': 4, 'num_layers': 1}
``` 

## Process/Methodology
[Bullet steps. E.g.,

Preprocessed with FOV=90° adjustment (F_W=0.5).
Ran ClearML pipeline: prep → train (100 epochs) → eval.
Monitored metrics: depth MAE, pose error <0.5m/<5°.
]

## Results
[Tables for quantitative data.]

## Analysis
[Bullets for insights, what went wrong. E.g.,

High variance in open spaces: Ray-attention smoothed local noise but over-generalized distant walls.
Overfitting signal: Train/val gap widened after epoch 50—consider more dropout.
]

## Improvements/Next Steps
[Bullets with rationale. E.g.,

Increase d_max to 20m for airport scales; test in next run.
Ablate without semantic branch: Hypothesis—will hurt but clarify attention's role.
Link to experiments/run_002.md for HPO on num_layers.
]

## References/Artifacts

[Paper section: F3Loc observation model]
ClearML Artifacts: Model checkpoint, Ray plots
Related: [[data/preprocessing.md]]

[End with open questions: "Does FOV change require ray re-sampling? To investigate."]

### Approach to Documentation and Note Writing
Approach documentation as an **iterative, reflective habit** integrated into your workflow—not a post-hoc chore. This builds understanding, tracks thinking, and proves progress (e.g., for stakeholders). Key principles:
- **Timeliness**: Write during/after each session (e.g., 10-15 min post-experiment). Use voice-to-text or quick bullets if time-constrained.
- **Audience Dualism**: Write for "future you" (detailed, personal examples) and "others" (clear, standalone summaries). Use simple language, avoid jargon without explanation.
- **Depth vs. Brevity**: Aim for "just enough"—summaries for skimmers, details for divers. Quantify where possible (e.g., "MAE dropped 12% because...").
- **Reflection Loop**: Always include "analysis" and "improvements" sections to turn notes into actionable insights. Revisit weekly (e.g., via ClearML dashboard) to link/update.
- **Versioning**: Commit to Git after each note (e.g., `git commit -m "Doc: Analyzed run_001 failures"`). Use branches for drafts.

#### In Practice: Step-by-Step Workflow
1. **Daily Setup (5 min)**: Open Obsidian/GitHub; review open questions from last note. Tag with current focus (e.g., #real-adaptation).
2. **During Work**: Jot quick bullets in a "scratch.md" (e.g., "Ray-attention: heads=8 caused OOM—why?"). Log live in ClearML (e.g., report scalars with notes).
3. **Post-Session (15-30 min)**: Fill template in the relevant category file. Add personal examples (e.g., "To grok mono net: Fed in toy image of a room; saw probs peak at wall depths because...").
4. **Weekly Review (30 min)**: Scan experiments/ folder; update summary.md with trends (e.g., table of all runs). Link to understanding/ for "aha" moments.
5. **Milestone Dumps**: At phase ends (e.g., 3-week HPO), create a "synthesis.md" aggregating categories (e.g., "Key learnings: Uncertainty modeling mitigated 15% of real noise").
6. **Sharing/Proof**: Push to GitHub; share README links in meetings. For visuals, screenshot ClearML dashboards into notes.