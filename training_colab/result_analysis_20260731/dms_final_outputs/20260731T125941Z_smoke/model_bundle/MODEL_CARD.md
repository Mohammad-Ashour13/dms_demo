# DMS Drowsiness Model Card

- Status: **EXPERIMENTAL_SMOKE**
- Model: LightGBM binary classifier
- Features: 65 runtime-computable V3 window features
- Window: 2.0s, 30 resampled points, update 0.5s
- Calibration: Platt scaling fitted only on OOF training predictions
- Threshold: 0.22100000, selected only from OOF training predictions
- Intended use: one evidence source inside the documented Fusion FSM; never a standalone safety decision.
- Limitations: video-level weak labels, domain shift, camera/landmark sensitivity. Review per-dataset and LODO metrics.
