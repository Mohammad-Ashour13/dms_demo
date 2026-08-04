from __future__ import annotations

import threading

import cv2


class VisualHMI:
    """Main-thread display; overlays never touch recorder/evaluation frames.

    OpenCV HighGUI uses Qt on the supported Linux targets.  Qt GUI calls are
    not thread-safe and may terminate the whole process with SIGSEGV when
    ``imshow``/``waitKey`` run in a worker while MediaPipe or PyTorch are also
    executing native code.  ``push`` is therefore intentionally synchronous
    and must be called by the application's main loop.
    """

    def __init__(self, config, telemetry, *, enabled=True):
        self.config = config
        self.telemetry = telemetry
        self.enabled = bool(config.enabled and config.show_camera and enabled)
        self.failed = False
        self.window_created = False
        self.owner_thread_id = threading.get_ident()

    def push(
        self,
        frame,
        decision,
        alarm_status,
        behavior_snapshot=None,
        face_signal=None,
    ) -> None:
        if not self.enabled or self.failed:
            return
        if threading.get_ident() != self.owner_thread_id:
            self.failed = True
            self.telemetry.emit(
                "HMI",
                "display_failed",
                {"error": "OpenCV HighGUI must run on the HMI owner/main thread"},
                level="ERROR",
            )
            return
        try:
            cv2.imshow(
                self.config.window_name,
                self._draw(
                    frame.copy(),
                    decision,
                    alarm_status,
                    behavior_snapshot,
                    face_signal,
                ),
            )
            self.window_created = True
            cv2.waitKey(1)
        except cv2.error as exc:
            self.failed = True
            self.telemetry.emit(
                "HMI", "display_failed", {"error": repr(exc)}, level="ERROR"
            )

    def _draw(
        self,
        frame,
        decision,
        alarm_status,
        behavior_snapshot=None,
        face_signal=None,
    ):
        state = decision.driver_state.value
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 34), (20, 20, 20), -1)
        cv2.putText(frame, f"SHADOW - LOCAL AUDIO DEMO | {state}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if state == "DROWSY":
            cv2.rectangle(frame, (0, height - 64), (width, height), (0, 170, 255), -1)
            cv2.putText(frame, "DROWSINESS DETECTED - TAKE A SAFE BREAK", (12, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        elif state == "CRITICAL":
            phase = int(decision.monotonic_sec * self.config.critical_flash_hz * 2) % 2
            color = (0, 0, 255) if phase == 0 else (15, 15, 90)
            cv2.rectangle(frame, (0, height - 72), (width, height), color, -1)
            cv2.putText(frame, "CRITICAL - OPEN EYES / STOP SAFELY", (12, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        audio = "PLAYING" if alarm_status.audio_playing else alarm_status.backend_health
        if alarm_status.acknowledged:
            audio = "ACKNOWLEDGED"
        cv2.putText(frame, f"AUDIO: {audio}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        driver_too_far = bool(
            getattr(face_signal, "driver_distance_status", "") == "TOO_FAR"
            or "DRIVER_TOO_FAR" in getattr(decision, "current_reason_codes", [])
        )
        if driver_too_far:
            cv2.rectangle(frame, (0, height - 58), (width, height), (0, 120, 220), -1)
            cv2.putText(
                frame,
                "MOVE CLOSER / ADJUST CAMERA - EYES TOO SMALL",
                (12, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        if behavior_snapshot is not None:
            active = ", ".join(behavior_snapshot.active_behaviors) or "NONE"
            cv2.putText(
                frame,
                f"OBJECT DETECTOR: {behavior_snapshot.health} | ACTIVE: {active}",
                (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (80, 255, 255) if behavior_snapshot.active_behaviors else (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            # Detections are produced asynchronously. Draw only a recent result;
            # recorder/evaluation always receive the untouched raw frame.
            if decision.monotonic_sec - behavior_snapshot.monotonic_sec <= 0.75:
                for detection in behavior_snapshot.detections:
                    x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 210, 255), 2)
                    cv2.putText(
                        frame,
                        f"{detection.label} {detection.confidence:.2f}",
                        (x1, max(96, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (40, 210, 255),
                        1,
                        cv2.LINE_AA,
                    )
        return frame

    def close(self) -> None:
        if not self.window_created:
            return
        try:
            cv2.destroyWindow(self.config.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        finally:
            self.window_created = False
