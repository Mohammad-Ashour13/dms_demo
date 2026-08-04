# LightGBM DMS Decision Report

- Bundle status: **EXPERIMENTAL**
- Selected feature count: **65**
- Selected threshold (OOF only): **0.372000**
- OOF constraints met: **False**
- Test Precision / Recall / F1: **0.587 / 0.698 / 0.638**
- Test FPR / PR-AUC: **0.457 / 0.691**

## Decision

The offline gate failed. Keep the bundle in telemetry-only mode, inspect per-dataset/LODO errors, and do not represent it as production-ready.

## Scientific caveat

Labels are video-level weak labels. Window metrics therefore measure agreement with inherited labels, not frame-level clinical drowsiness ground truth.

## Experiment comparison

```text
stage  feature_count  macro_pr_auc  worst_dataset_pr_auc  precision   recall       f1  constraints_met  seed
    A             40      0.642781              0.593097   0.498078 0.999886 0.664931            False   NaN
    A             65      0.635094              0.572795   0.537756 0.892146 0.671035            False   NaN
    A             20      0.621348              0.531822   0.497972 1.000000 0.664862            False   NaN
    A              7      0.604666              0.539776   0.516812 0.950797 0.669638            False   NaN
    B             40      0.663198              0.602351        NaN      NaN      NaN            False   NaN
    B             65      0.685909              0.657553        NaN      NaN      NaN            False   NaN
    C             65      0.685904              0.657553   0.548481 0.875262 0.674370            False  42.0
    C             65      0.715975              0.647338   0.546432 0.936354 0.690125            False  43.0
    C             65      0.649629              0.454704   0.556426 0.869105 0.678473            False  44.0
```
