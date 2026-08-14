# RISEF YOLO11s seat-belt classifier

- Source: <https://huggingface.co/RISEF/yolov11s-seatbelt>
- Revision: `0187cdb`
- Downloaded artifact: `weights/best.onnx`
- SHA-256: `bb945fdd1750dc4fdd98b4683a605ac6a1c9c7f4417324faf76b0c1b4f319df2`
- License declared by the source: AGPL-3.0.

The local `model.ncnn.param` and `model.ncnn.bin` were generated from the
source ONNX artifact with PNNX on the export workstation. Their hashes and the
required RGB 224x224 preprocessing contract are recorded in
`model_manifest.json`.

The source model card reports a highly imbalanced training set and windshield
views. This integration therefore begins in telemetry-only shadow mode. Do not
enable Fusion, incidents, or alerts until it passes evaluation on the actual
in-cabin camera footage.
