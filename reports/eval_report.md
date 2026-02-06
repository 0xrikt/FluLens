# FluLens Evaluation Report

Date: 2026-02-06

## Dataset
- Source: IDR idr0128 (Georgi et al. Influenza A virus screen)
- Labeling: infection intensity proxy
  - Low infection (label=0): bottom 20% `numberOfInfectedNuclei`
  - High infection (label=1): top 20% `numberOfInfectedNuclei`
- Samples used (available from current download cache):
  - Train: 38
  - Val: 10
  - Test: 30

## Model
- Backbone: ResNeSt-50 (timm `resnest50d`)
- Attention: CBAM-style channel + spatial attention
- Input: 224x224, RGB composed from nuclei channel (R) + virus channel (G,B)
- Training: pretrained ResNeSt-50 with frozen backbone, head+attention fine-tuned

## Metrics (Test Split)
- Accuracy: 0.5333
- Sensitivity (Recall for high infection): 0.5625
- Specificity (Recall for low infection): 0.5000

Confusion matrix:
- TP: 9
- TN: 7
- FP: 7
- FN: 7

## Notes & Limitations
- The dataset is small due to download constraints; results are **not** statistically stable.
- Labels are infection-intensity proxies, not strict infected vs. uninfected.
- These metrics are a baseline to validate the pipeline and web demo; they should improve with more data and BYOL pretraining.
