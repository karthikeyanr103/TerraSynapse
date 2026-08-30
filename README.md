# TerraSynapse

Cross-modal satellite image retrieval for the ISRO BAH 2026 challenge.

TerraSynapse retrieves semantically similar Earth-observation patches across SAR, optical, and multispectral imagery. The project focuses on learning a shared embedding space so a query from one modality can retrieve matching or related scenes from another modality.

## Project Highlights

- Built an end-to-end cross-modal retrieval workflow for SAR, optical, and multispectral satellite data.
- Trained and evaluated same-modal and cross-modal retrieval directions.
- Compared EfficientNetV2 and hybrid attention-based encoder variants.
- Reported retrieval quality using F1@K, Recall@K, mAP@K, and per-query retrieval time.
- Prepared final ISRO submission report, presentation deck, architecture diagrams, and result visualizations.

## Problem Statement

Satellite datasets often contain multiple sensor modalities, but users may need to search using only one available modality. TerraSynapse addresses this by aligning representations across:

- SAR imagery
- Optical imagery
- Multispectral imagery

The goal is to support retrieval such as SAR to optical, optical to SAR, optical to multispectral, and multispectral to optical.

## Repository Structure

```text
TerraSynapse/
+-- assets/
|   +-- architecture/        # Model architecture and workflow diagrams
|   +-- figures/             # Inference examples and supporting visuals
+-- certificates/            # Add ISRO/BAH participation certificates here
+-- data/                    # Lightweight preprocessing code
+-- Notebooks/
|   +-- final/               # Final training notebooks
|   +-- retrieval/           # Same-modal and cross-modal retrieval notebooks
|   +-- training/            # Experiment and training notebooks
+-- Results/                 # Metrics, retrieval samples, and result plots
+-- submission/              # Final report, presentation, and problem statement files
+-- requirements.txt
+-- README.md
```

## Key Artifacts

- Final presentation: `submission/Final-ISRO.pdf`
- Final report: `submission/EuroData_ISRO_Final_Report_Completed.pdf`
- Metrics: `Results/retrieval_metrics_at_k.csv`
- Retrieval samples: `Results/random_retrieval_samples.csv`
- Architecture visuals: `assets/architecture/`
- Inference examples: `assets/figures/`

## Methods

The project uses deep visual encoders and projection heads to map different satellite modalities into a comparable feature space. Retrieval is performed by encoding queries and gallery images, then ranking candidates using embedding similarity.

Evaluation covers both same-modal retrieval and cross-modal retrieval so the results can show whether modality alignment is working beyond ordinary visual similarity.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open the notebooks under `Notebooks/final/` or `Notebooks/training/` to inspect training and evaluation experiments.

## Notes

Large datasets, model checkpoints, Kaggle credentials, and local environments are intentionally excluded from version control. Add certificates to `certificates/` before sharing the final project folder or GitHub repository.
