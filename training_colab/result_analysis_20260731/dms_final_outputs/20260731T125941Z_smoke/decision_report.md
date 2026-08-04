# LightGBM DMS Decision Report

- Bundle status: **EXPERIMENTAL_SMOKE**
- Selected feature count: **65**
- Selected threshold (OOF only): **0.221000**
- OOF constraints met: **False**
- Test Precision / Recall / F1: **N/A / N/A / N/A**
- Test FPR / PR-AUC: **N/A / N/A**

## Decision

The offline gate failed. Keep the bundle in telemetry-only mode, inspect per-dataset/LODO errors, and do not represent it as production-ready.

## Scientific caveat

Labels are video-level weak labels. Window metrics therefore measure agreement with inherited labels, not frame-level clinical drowsiness ground truth.

## Experiment comparison

```text
stage  feature_count  macro_pr_auc  worst_dataset_pr_auc  precision   recall       f1  constraints_met  seed
    A             65      0.692798              0.472933   0.569290 0.895651 0.696117            False   NaN
    A             40      0.682867              0.449216   0.570955 0.883293 0.693582            False   NaN
    A             20      0.619269              0.371039   0.574824 0.902413 0.702296            False   NaN
    A              7      0.579886              0.360970   0.523723 0.969103 0.679974            False   NaN
    C             65      0.692798              0.472933   0.569290 0.895651 0.696117            False  42.0
```
