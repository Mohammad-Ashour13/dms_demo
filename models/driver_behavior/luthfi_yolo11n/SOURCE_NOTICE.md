# YOLO11n driver-behavior weight

- Source: https://github.com/luthfajr/driver-inattention-detection
- Verified source commit: `7cb52d367a936e8803d61b960d830291fa8f880e`
- Original path: `models/best_yolo11n.pt`
- Local SHA-256: `37ab5aeee2c4f655fcbe733195e53cff8c4ea03ab36accd064d1bee4b22c0c9d`
- Intended local classes: `phone`, `cigarette`, `drink_or_food`.

The upstream README states that training data uses a front-facing driver camera and
warns that side-view performance degrades. The upstream repository did not contain an
explicit project license when this artifact was downloaded. Keep this integration in
research/controlled-demo mode until software, model and dataset licensing are reviewed.

The DMS runtime does not reuse the upstream alert or state-machine implementation. It
uses only the trained weight and maps temporally stabilized detections to the local
versioned `EvidenceEvent` contract.
