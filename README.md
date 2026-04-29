# UrbanAlign

> Post-hoc Semantic Calibration for VLM-Human Preference Alignment

UrbanAlign is a **training-free framework** that synthesizes interpretable urban perception data from minimal crowdsourced annotations. It elicits domain-specific perception capabilities from Vision-Language Models (VLMs) through a three-stage post-processing pipeline.

Given a small set of pairwise comparisons (e.g., *"Which scene looks safer?"*), UrbanAlign produces dimension-level perception scores across six categories: **safety, beauty, liveliness, wealth, boringness, and depressingness**.

<p align="center">
  <img src="assets/framework.pdf" width="90%">
</p>

---

## Method

### Stage 1 — Semantic Dimension Extraction

Using TrueSkill-stratified consensus samples, the VLM discovers 5-8 universal evaluation dimensions (e.g., *Facade Quality*, *Vegetation Coverage*, *Infrastructure Condition*) with operational definitions — replacing hand-crafted rules with transferable, interpretable dimensions.

### Stage 2 — Multi-Agent Feature Distillation

For each image pair, three VLM agents collaborate through deliberation:

- **Observer** — Describes visual evidence without premature judgment
- **Debater** — Argues opposing perspectives to reduce confirmation bias
- **Judge** — Synthesizes a final dimension-level score (1-10)

### Stage 3 — Hybrid VRM + LWRR Alignment

Constructs a hybrid embedding space (CLIP visual features + LLM semantic scores), then calibrates synthetic judgments against human ground truth via Locally Weighted Ridge Regression.

---

## Results

| Metric | Value |
|--------|-------|
| Average accuracy (6 categories) | 61.3% |
| Best single category (wealthy) | 72.3% |
| Gain over zero-shot VLM baseline | +17.9 pp |
| LWRR calibration gain | +7.0 pp |
| Cost reduction vs. full human annotation | 97% |

---

## Project Structure

```
UrbanAlign/
├── urbanalign/                          # Core package
│   ├── config.py                        # Global configuration (API, paths, hyperparams)
│   ├── preprocessing/
│   │   ├── extract_clip_features.py     # [Optional] CLIP embedding extraction
│   │   ├── prepare_dataset.py           # Filter & analyze Place Pulse annotations
│   │   ├── validate_data.py             # Data quality checks
│   │   └── compute_trueskill.py         # TrueSkill rating computation
│   ├── pipeline/
│   │   ├── stage1_semantic_extractor.py # Semantic dimension discovery
│   │   ├── stage2_multi_agent_synthesis.py  # Multi-agent VLM scoring
│   │   └── stage3_hybrid_vrm.py         # LWRR calibration & alignment
│   ├── evaluation/
│   │   ├── evaluate.py                  # Accuracy & kappa evaluation
│   │   ├── sensitivity_analysis.py      # Hyperparameter sensitivity
│   │   ├── dimension_optimization.py    # End-to-end dimension search
│   │   └── results_summary.py           # Result table generation
│   └── baselines/
│       └── traditional_baselines.py     # Siamese / Segmentation / Zero-shot baselines
├── scripts/
│   ├── run_all_modes.py                 # Run all Stage 2 modes sequentially
│   ├── specs_transfer_experiment.py     # Cross-dataset transfer to SPECS
│   └── generate_figures.py              # Generate paper figures
├── assets/                              # Framework diagrams
├── requirements.txt
├── .env.example
└── .gitignore
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

```bash
cp .env.example .env
```

Fill in your `.env`:
- `URBANALIGN_API_KEY` — API key for the VLM service
- `PLACE_PULSE_DIR` — Path to the [Place Pulse 2.0](http://pulse.media.mit.edu/) dataset

Expected data layout:
```
<PLACE_PULSE_DIR>/
├── final_data_reliable_agg_N3.csv
├── final_data_reliable_raw_N3.csv
└── final_photo_dataset/
```

---

## Usage

### Preprocessing (run once)

```bash
python -m urbanalign.preprocessing.extract_clip_features    # Optional
python -m urbanalign.preprocessing.prepare_dataset
python -m urbanalign.preprocessing.compute_trueskill
```

### Core pipeline

```bash
python -m urbanalign.pipeline.stage1_semantic_extractor
python -m urbanalign.pipeline.stage2_multi_agent_synthesis
python -m urbanalign.pipeline.stage3_hybrid_vrm
```

### Evaluation

```bash
python -m urbanalign.evaluation.evaluate
python -m urbanalign.evaluation.sensitivity_analysis
python -m urbanalign.baselines.traditional_baselines
python -m urbanalign.evaluation.results_summary
```

All outputs are saved to `urbanalign_outputs/` (auto-created, gitignored).

### Key configuration

Edit `urbanalign/config.py` or set via environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STAGE2_MODE` | 4 | 1: single-direct, 2: pairwise-direct, 3: single-multiagent, **4: pairwise-multiagent** |
| `N_POOL_MULTIPLIER` | 0.01 | Sampling ratio from human annotation pool |
| `LABELED_SET_RATIO` | 0.05 | Reference set ratio for LWRR calibration |
| `ALPHA_HYBRID` | 0.3 | Hybrid weight (30% CLIP + 70% semantic) |
| `SELECTION_RATIO` | 0.15 | Post-alignment retention ratio |

---

## Citation

```bibtex
@article{zhang2026urbanalign,
  title   = {UrbanAlign: Post-hoc Semantic Calibration for
             VLM-Human Preference Alignment},
  author  = {Zhang, Yecheng and others},
  year    = {2026}
}
```

## License

[MIT License](LICENSE)
