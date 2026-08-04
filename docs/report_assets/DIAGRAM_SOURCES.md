# Editable Mermaid sources

هذه المصادر النصية تقابل الرسومات الثابتة المستخدمة في نسخة Word. يمكن تعديلها أو
إعادة تصديرها باستخدام Mermaid CLI.

## 1. System architecture

```mermaid
flowchart LR
  Camera --> Capture --> MediaPipe --> Calibration
  Calibration --> Windows --> LightGBM --> Platt --> Fusion
  Calibration --> Events --> Fusion
  YOLO --> EvidenceBus --> Fusion
  Fusion --> AlarmController --> AudioHMI[Audio + HMI]
  Fusion --> Recorder --> Outbox
  Fusion --> Telemetry
```

## 2. Data pipeline

```mermaid
flowchart LR
  Videos[Video-level weak labels] --> Frames[18 frame channels]
  Frames --> Clean[Cleaning and physical clipping]
  Clean --> Window[2 s windows / 0.5 s stride]
  Window --> F390[390 columns]
  F390 --> Select[65 selected features]
  Select --> Train[Grouped training]
```

## 3. Temporal resampling

```mermaid
flowchart TD
  F15[15 FPS: about 30 frames] --> Time[Timestamp-based 2 s interval]
  F30[30 FPS: about 60 frames] --> Time
  F60[60 FPS: about 120 frames] --> Time
  Time --> R[Resample to 30 temporal points]
```

## 4. Frame versus temporal decision

```mermaid
flowchart LR
  Frame[One closed-eye frame] --> Blink[Normal blink?]
  Frame --> Closure[Dangerous closure?]
  Frame --> Gaze[Downward gaze?]
  Blink --> Temporal[Duration + context + quality]
  Closure --> Temporal
  Gaze --> Temporal
  Temporal --> Decision[Explainable decision]
```

## 5. Training protocol

```mermaid
flowchart LR
  Train --> A[Stage A: K=7/20/40/65]
  A --> B[Stage B: Optuna]
  B --> C[Stage C: 3 seeds + OOF]
  C --> Cal[OOF Platt calibration and threshold]
  Cal --> Final[Final fit on train]
  Final --> Test[One-time fixed test]
```

## 6. Model comparison

```mermaid
flowchart TD
  Raw[Raw temporal sequence] --> LSTM[LSTM experiment]
  Raw --> GRU[GRU experiment]
  V3[V3 engineered windows] --> LGBM[LightGBM deployed experiment]
  LSTM --> Reject[Not adopted: generalization/FPR]
  GRU --> Reject
  LGBM --> Fusion[Adopted as supporting evidence in Fusion]
```

## 7. Runtime queues

```mermaid
flowchart LR
  Capture --> Latest[Bounded latest-frame AI queue] --> AI[MediaPipe + Events + Model]
  Capture --> RecorderQ[Bounded recorder queue] --> Video
  AI --> Fusion --> TelemetryQ[Bounded telemetry queue] --> JSONL
```

## 8. Fusion evidence

```mermaid
flowchart LR
  Risk[Calibrated model risk] --> FSM[Fusion FSM]
  Eye[Eye events and PERCLOS] --> FSM
  Mouth[Yawn and head nod] --> FSM
  Objects[YOLO evidence] --> FSM
  Quality[Signal quality and drift] --> FSM
  FSM --> State
  FSM --> Violations
  FSM --> Reasons[Reason codes]
```

## 9. Driver states

```mermaid
stateDiagram-v2
  [*] --> UNKNOWN
  UNKNOWN --> NORMAL: calibration + trusted signal
  NORMAL --> FATIGUE_WARNING: persistent supported risk
  FATIGUE_WARNING --> DROWSY: physiological evidence
  DROWSY --> CRITICAL: strong bilateral closure >= 1.5 s
  CRITICAL --> FATIGUE_WARNING: trusted open-eye recovery
  FATIGUE_WARNING --> NORMAL: exit threshold + persistence
  NORMAL --> UNKNOWN: sustained quality loss
```

## 10. Alarm controller

```mermaid
stateDiagram-v2
  [*] --> IDLE
  IDLE --> ONSET: DROWSY or CRITICAL begins
  ONSET --> REMINDER: state persists
  REMINDER --> ESCALATION: recurrence/long duration
  ONSET --> ACKNOWLEDGED: trusted eye open 0.5 s
  REMINDER --> ACKNOWLEDGED
  ESCALATION --> ACKNOWLEDGED
  ACKNOWLEDGED --> COOLDOWN
  COOLDOWN --> ONSET: rearmed dangerous episode
  COOLDOWN --> IDLE: episode ends
```

## 11. Rolling recorder

```mermaid
flowchart LR
  Ring[Compressed circular buffer] --> Pre[10 s pre-alert]
  Trigger --> Incident[Active incident]
  Pre --> Incident
  Incident --> Post[10 s post-alert]
  Post --> Package[video.mp4 + incident.json + telemetry.jsonl]
```

## 12. Offline outbox

```mermaid
flowchart LR
  Camera --> Local[Local detection and warning]
  Local --> Outbox[Filesystem Outbox]
  Outbox -. connection returns .-> Adapter[Future CompanyUploadAdapter]
  Adapter --> API[Future Backend API]
  API --> Dashboard[Future Dashboard]
```

## 13. Embedded deployment

```mermaid
flowchart LR
  USB[USB Camera] --> Pi[Raspberry Pi 5 CPU-only]
  Pi --> Speaker
  Pi --> Display
  Pi --> Storage
  Storage -. future .-> Server
```

## 14. Confusion matrix

```mermaid
quadrantChart
  x-axis Predicted Awake --> Predicted Drowsy
  y-axis Actual Awake --> Actual Drowsy
  quadrant-1 TP 6651
  quadrant-2 FN 956
  quadrant-3 TN 1948
  quadrant-4 FP 6236
```

## 15. Decision timeline

```mermaid
sequenceDiagram
  participant C as Camera
  participant P as Perception
  participant E as Event Engine
  participant M as LightGBM
  participant F as Fusion
  participant A as Alarm
  participant R as Recorder
  C->>P: timestamped frame
  P->>E: calibrated eye/mouth/pose signals
  P->>M: 2 s V3 feature window
  E->>F: events + quality
  M->>F: calibrated probability
  F->>A: state transition
  F->>R: incident trigger + reason codes
```
