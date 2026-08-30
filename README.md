# TerraSynapse

Cross-modal satellite image retrieval for the ISRO BAH 2026 challenge.

TerraSynapse learns a shared embedding space for SAR, optical, and multispectral imagery so a query from one modality can retrieve semantically similar scenes from another.

## Architecture

- Modality-specific image encoders project satellite inputs into a common representation space.
- Attention and pooling blocks refine cross-modal features before retrieval.
- Evaluation covers same-modal and cross-modal retrieval using F1@K, Recall@K, mAP@K, and latency.

**System workflow**

<img src="architecture/system-workflow.png" alt="TerraSynapse system workflow" width="900">

**Model blocks**

<img src="architecture/Architecture-Blocks.drawio.png" alt="TerraSynapse model architecture blocks" width="900">

**Image encoder**

<img src="architecture/image-encoder.drawio.png" alt="TerraSynapse image encoder architecture" width="900">

## Results

**Cross-modal retrieval example**

<img src="results/figures/inference-example.png" alt="Optical to SAR cross-modal retrieval example" width="900">

**Optical to SAR retrieval sample**

<img src="results/optical-to-sar-random-query-860.png" alt="Optical to SAR random retrieval sample" width="900">

**Training history**

<img src="results/training-history-optical-sar.png" alt="Optical to SAR training history" width="900">

Result tables:

- [Optical to SAR metrics](results/retrieval-metrics-optical-sar.csv)
- [Optical to multispectral metrics](results/retrieval-metrics-optical-multispectral.csv)

## Submission

- [Final presentation PDF](submission/final-isro-presentation.pdf)
- [Final presentation PPTX](submission/final-isro-presentation.pptx)
- [Final report PDF](submission/final-report.pdf)
- [Problem statement](submission/problem-statement-cross-modal-satellite-retrieval.pdf)
- [Dataset reference](submission/dataset-reference.pdf)

## Certificate

<!-- <img src="certificates/isro-bah-2026-certificate.png" alt="ISRO BAH 2026 participation certificate" width="900">
-->
<table>
  <tr>
    <td>
      <p align="center"><b>Karthikeyan</b></p>
      <img src="certificates/isro-bah-2026-certificate.png" width="400" alt="Certificate 1">
    </td>
    <td>
      <p align="center"><b>Rebecca</b></p>
      <img src="certificates/isro-bah-2026-certificate.png" width="400" alt="Certificate 2">
    </td>
    <td>
      <p align="center"><b>Bharath</b></p>
      <img src="certificates/isro-bah-2026-certificate.png" width="400" alt="Certificate 2">
    </td>
      <td>
      <p align="center"><b>Musharraf</b></p>
      <img src="certificates/isro-bah-2026-certificate.png" width="400" alt="Certificate 2">
    </td>
  </tr>
</table>

[Certificate PDF](certificates/isro-bah-2026-certificate.pdf)

## Repository Structure

```text
TerraSynapse/
|-- architecture/   # Diagrams and model workflow visuals
|-- certificates/   # Participation certificate
|-- notebooks/      # Experiment notebooks retained for reproducibility
|-- results/        # Metrics, samples, and evaluation figures
|-- submission/     # Final report, slides, and challenge references
|-- archive/        # Local drafts and source-code archive
`-- README.md
```
## Team

Built by Team EuroData for ISRO BAH 2026.

| Member | Role |
|---|---|
| [Karthikeyan Ramadoss](https://github.com/karthikeyanr103) | Model architecture, retrieval pipeline, evaluation |
| [Rebecca John](https://github.com/member-2-username) | Data preprocessing and experiments |
| [Bharath Ilayaperumal](https://github.com/member-3-username) | Training workflows and analysis |
| [Musharraf Hamdan](https://github.com/Hamdan-Musharraf) | Report, presentation, and validation |
