# Runtime V2.1 replay regression

Source session: `raspberry-live-df16b6e3`  
Final replay: `raspberry-live-210110d6`  
Video duration: about 304.7 seconds

## Before/after

| Metric | Before | Runtime V2.1 |
|---|---:|---:|
| UNKNOWN after calibration | 19.95% | 3.71% |
| State transitions | 58 | 20 |
| CRITICAL → UNKNOWN transitions | 8 | 0 |
| First CRITICAL | 175.27s | 175.33s |
| FATIGUE entries before first closure block | 2 | 0 |
| NORMAL duration | 93.40s | 196.00s |
| DROWSY duration | 42.96s | 1.53s |

The two yawns remain violations and trigger recording without changing the
driver state by themselves. The first yawn-only incident in the final replay
has `highest_state=NORMAL` and `violations=[YAWNING]`.

The final replay emitted no CRITICAL before the visually reviewed closure near
175 seconds and no CRITICAL/UNKNOWN oscillation. Post-critical PERCLOS memory is
reported as `FATIGUE_WARNING` with `POST_CRITICAL_FATIGUE`.

## Verification

- Runtime tests: 30 passed.
- Golden samples: 12 passed.
- Maximum raw prediction error: `0.0`.
- Maximum calibrated prediction error: `1.1102230246251565e-16`.
- Required tolerance: `1e-6`.

This is a deterministic replay regression, not an accuracy acceptance report.
Blink recall and episode precision/recall still require the annotated pilot.
