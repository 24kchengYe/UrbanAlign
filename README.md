# UrbanAlign: Few-Shot Urban Perception via Semantic Feature Distillation

> Synthesizing high-fidelity, interpretable urban perception data from minimal crowdsourced annotations.

**Paper**: *UrbanAlign: Few-Shot Urban Perception via Semantic Feature Distillation and Hybrid Visual Relationship Mapping* (ECCV 2026)

---

## Overview

UrbanAlign is a training-free framework that elicits domain-specific urban perception capabilities from Vision-Language Models (VLMs) through a three-stage post-processing pipeline. Given only a small set of crowdsourced pairwise comparisons (e.g., "Which scene looks safer?"), UrbanAlign synthesizes perception scores across six perceptual dimensions — **safety, beauty, liveliness, wealth, boringness, and depressingness** — with interpretable, dimension-level explanations.

### Key Contributions

1. **Semantic Dimension Extraction** — Automatically discovers 5-8 universal evaluation dimensions (e.g., Facade Quality, Vegetation Coverage) from a handful of consensus samples, replacing hand-crafted rules with transferable, interpretable dimensions.

2. **Multi-Agent Feature Distillation** — An Observer-Debater-Judge deliberation pipeline that mitigates VLM confirmation bias through progressive reasoning and self-consistency verification.

3. **Hybrid Embedding Space & LWRR Alignment** — Fuses CLIP visual features with LLM semantic scores, then calibrates synthetic judgments against human ground truth via Locally Weighted Ridge Regression.

### Results

| Metric | Value |
|--------|-------|
| Average accuracy (6 categories) | 61.3% |
| Best category (wealthy) | 72.3% |
| Improvement over zero-shot VLM | +17.9 pp |
| LWRR calibration gain | +7.0 pp |
| Cost reduction vs. full annotation | 97% ($167K -> $4.5K) |

---

## Project Structure

```
.
├── config.py                              # Global configuration (API, paths, hyperparams)
├── requirements.txt                       # Python dependencies
├── .env.example                           # Environment variable template
│
├── abc_preprocess0_extract_clip_features.py   # [Optional] Extract CLIP embeddings
├── abc_preprocess1_human_choices_dataset.py   # Preprocess Place Pulse 2.0 annotations
├── abc_preprocess2_human_choices_check.py     # Validate filtered annotations
├── abc_preprocess3_human_choice_trueskill.py  # Compute TrueSkill ratings
│
├── abc_stage1_semantic_extractor.py       # Stage 1: Semantic dimension extraction
├── abc_stage2_multi_mode_synthesis.py     # Stage 2: Multi-agent feature distillation
├── abc_stage3_hybrid_vrm.py              # Stage 3: Hybrid VRM + LWRR alignment
├── abc_stage4_comprehensive_evaluation.py # Stage 4: Evaluation against human labels
├── abc_stage5_sensitivity_analysis.py     # Stage 5: Hyperparameter sensitivity
├── abc_stage6_e2e_dimension_optimization.py # Stage 6: End-to-end dimension search
├── abc_stage7_traditional_baselines.py    # Stage 7: Baseline comparisons
├── abc_stage8_results_summary.py          # Stage 8: Results aggregation for paper
│
├── abc_stage2_auto_all_modes.py           # Helper: run all Stage 2 modes sequentially
├── abc_specs_transfer_experiment.py       # Cross-dataset transfer to SPECS
├── generate_paper_figures.py              # Generate figures for paper
│
├── ECCV_2026_Paper_Template/              # LaTeX paper source
│   ├── main.tex                           # Main paper
│   ├── supplementary.tex                  # Supplementary material
│   ├── main.bib                           # References
│   └── fig_*.pdf                          # Paper figures
│
└── archive/                               # Legacy code and paper drafts
    ├── legacy_1.0/                        # UrbanAlign 1.0 (rule-based) scripts
    ├── paper_drafts/                      # Early paper versions
    └── temp_scripts/                      # One-off analysis scripts
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For CLIP feature extraction (optional — pre-extracted features can be reused):
```bash
pip install torch transformers
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `URBANALIGN_API_KEY` — API key for the VLM service
- `PLACE_PULSE_DIR` — Path to the [Place Pulse 2.0](http://pulse.media.mit.edu/) dataset

### 3. Prepare data

Download the Place Pulse 2.0 dataset and set `PLACE_PULSE_DIR` to its root. Expected structure:

```
<PLACE_PULSE_DIR>/
├── final_data_reliable_agg_N3.csv    # Aggregated pairwise annotations
├── final_data_reliable_raw_N3.csv    # Raw individual annotations
└── final_photo_dataset/              # Street-view images
```

---

## Usage

### Preprocessing (run once)

```bash
python abc_preprocess0_extract_clip_features.py   # Extract CLIP features (optional)
python abc_preprocess1_human_choices_dataset.py    # Filter & analyze annotations
python abc_preprocess3_human_choice_trueskill.py   # Compute TrueSkill ratings
```

### Main pipeline

```bash
python abc_stage1_semantic_extractor.py        # Extract semantic dimensions
python abc_stage2_multi_mode_synthesis.py       # Score image pairs (Mode 4 = multi-agent)
python abc_stage3_hybrid_vrm.py                # LWRR calibration
python abc_stage4_comprehensive_evaluation.py  # Evaluate accuracy & kappa
```

### Analysis & ablations

```bash
python abc_stage5_sensitivity_analysis.py          # Hyperparameter sensitivity
python abc_stage6_e2e_dimension_optimization.py    # Dimension set optimization
python abc_stage7_traditional_baselines.py         # Run baseline methods
python abc_stage8_results_summary.py               # Generate result tables
```

All outputs are saved to `urbanalign_outputs/` (auto-created, gitignored).

### Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STAGE2_MODE` | 4 | 1: single-direct, 2: pairwise-direct, 3: single-multiagent, **4: pairwise-multiagent** |
| `N_POOL_MULTIPLIER` | 0.01 | Sampling ratio from human annotation pool |
| `LABELED_SET_RATIO` | 0.05 | Reference set ratio for LWRR calibration |
| `ALPHA_HYBRID` | 0.3 | Hybrid weight (30% CLIP + 70% semantic) |
| `SELECTION_RATIO` | 0.15 | Post-alignment retention ratio |

---

## Method

<p align="center">
  <img src="ECCV_2026_Paper_Template/fig_framework.pdf" width="90%">
</p>

**Stage 1 — Semantic Dimension Extraction**: Using TrueSkill-stratified consensus samples, the VLM identifies universal evaluation dimensions (e.g., *Facade Quality*, *Infrastructure Condition*) with operational definitions.

**Stage 2 — Multi-Agent Feature Distillation**: For each image pair, three VLM agents collaborate:
- **Observer**: Describes visual evidence without judgment
- **Debater**: Argues opposing perspectives
- **Judge**: Synthesizes a final dimension-level score (1-10)

**Stage 3 — Hybrid VRM + LWRR**: Constructs a hybrid embedding (CLIP visual + semantic scores), then aligns synthetic judgments to human ground truth via Locally Weighted Ridge Regression in this joint space.

---

## Citation

```bibtex
@inproceedings{zhang2026urbanalign,
  title     = {UrbanAlign: Few-Shot Urban Perception via Semantic Feature
               Distillation and Hybrid Visual Relationship Mapping},
  author    = {Zhang, Yecheng and others},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

This project is for academic and research purposes only.
