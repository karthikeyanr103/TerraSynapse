# TerraSynapse

Cross-modal satellite image retrieval project for the ISRO BAH 2026 challenge.

TerraSynapse explores retrieval across SAR, optical, and multispectral satellite imagery by learning a shared embedding space. The goal is to query in one modality and retrieve semantically relevant scenes from another modality.

## Highlights

- Same-modal and cross-modal satellite image retrieval experiments
- SAR, optical, and multispectral image workflows
- Hybrid attention-based architecture exploration
- Evaluation with F1@K, Recall@K, mAP@K, and query latency
- Final presentation, report, architecture diagrams, results, and certificates

## Clean Repo Structure

```text
TerraSynapse/
+-- notebooks/       # Experiment, training, and retrieval notebooks
+-- architecture/    # Model architecture and workflow diagrams
+-- results/         # Metrics, retrieval examples, plots, and figures
+-- ppt/             # Final presentation, report, and submission documents
+-- certificates/    # ISRO/BAH participation certificates
+-- README.md
```

## Key Files

- `ppt/Final-ISRO.pdf`
- `ppt/Final-ISRO.pptx`
- `ppt/EuroData_ISRO_Final_Report_Completed.pdf`
- `results/retrieval_metrics_at_k.csv`
- `results/random_retrieval_samples.csv`
- `architecture/Architecture-scaled-dot-product-att.drawio`

## Notes

The public repository is intentionally focused on showcase material only. Source-code helpers, prompt drafts, local archives, generated Graphify output, packages, and temporary assistant files are kept out of Git.

