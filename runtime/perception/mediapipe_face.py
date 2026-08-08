"""MediaPipe face signals with the same units expected by V3 windows."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from dms_final_system.shared.contracts import FaceSignal, FramePacket


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_HOR = (362, 263)
LEFT_EYE_TOPS = [384, 385, 386, 387, 388]
LEFT_EYE_BOTTOMS = [373, 374, 380, 381, 382]
RIGHT_EYE_HOR = (33, 133)
RIGHT_EYE_TOPS = [157, 158, 159, 160, 161]
RIGHT_EYE_BOTTOMS = [154, 153, 163, 144, 145]
HEAD_POINTS = [1, 152, 33, 263, 61, 291]
FACE_MODEL = np.asarray(
    [(0, 0, 0), (0, -63.6, -12.5), (-43.3, 32.7, -26),
     (43.3, 32.7, -26), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)],
    dtype=np.float64,
)


def _distance(a, b, width: int, height: int) -> float:
    return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)


def _ear(landmarks, indices, width, height) -> float:
    if indices == LEFT_EYE:
        horizontal_pair, tops, bottoms = LEFT_EYE_HOR, LEFT_EYE_TOPS, LEFT_EYE_BOTTOMS
    elif indices == RIGHT_EYE:
        horizontal_pair, tops, bottoms = RIGHT_EYE_HOR, RIGHT_EYE_TOPS, RIGHT_EYE_BOTTOMS
    else:
        a, b, c, d, e, f = (landmarks[i] for i in indices)
        horizontal = _distance(a, d, width, height)
        return (_distance(b, f, width, height) + _distance(c, e, width, height)) / max(2 * horizontal, 1e-9)
    horizontal = _distance(landmarks[horizontal_pair[0]], landmarks[horizontal_pair[1]], width, height)
    verticals = sorted(_distance(landmarks[top], landmarks[bottom], width, height) for top in tops for bottom in bottoms)
    return sum(verticals[:2]) / max(2 * horizontal, 1e-9)


def _event_ear(landmarks, indices, width, height) -> float:
    """Conventional matched-pair EAR used only by the physiological event path."""
    a, b, c, d, e, f = (landmarks[i] for i in indices)
    horizontal = _distance(a, d, width, height)
    return (
        _distance(b, f, width, height) + _distance(c, e, width, height)
    ) / max(2 * horizontal, 1e-9)


def _mar(landmarks, width, height) -> float:
    p = [landmarks[i] for i in MOUTH]
    horizontal = _distance(p[0], p[4], width, height)
    vertical = _distance(p[1], p[7], width, height) + _distance(p[2], p[6], width, height) + _distance(p[3], p[5], width, height)
    return vertical / max(3 * horizontal, 1e-9)


def _normalize_angle(angle: float) -> float:
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return float(angle)


def _head_pose(landmarks, width, height) -> tuple[float, float, float]:
    image_points = np.asarray([(landmarks[i].x * width, landmarks[i].y * height) for i in HEAD_POINTS], dtype=np.float64)
    camera = np.asarray([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(FACE_MODEL, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    matrix, _ = cv2.Rodrigues(rotation)
    sy = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(-matrix[2, 0], sy)
        roll = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        pitch = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = math.atan2(-matrix[2, 0], sy)
        roll = 0.0
    return _normalize_angle(math.degrees(pitch)), float(math.degrees(yaw)), _normalize_angle(math.degrees(roll))


def _iris_offset(landmarks, corners, iris, tops, bottoms) -> tuple[float, float]:
    left, right = landmarks[corners[0]], landmarks[corners[1]]
    cx, cy = (left.x + right.x) / 2, (min(landmarks[i].y for i in tops) + max(landmarks[i].y for i in bottoms)) / 2
    ix = float(np.mean([landmarks[i].x for i in iris]))
    iy = float(np.mean([landmarks[i].y for i in iris]))
    width = max(abs(right.x - left.x), 1e-6)
    height = max(landmarks[i].y for i in bottoms) - min(landmarks[i].y for i in tops)
    if height < 1e-6:
        height = width
    return (ix - cx) / (width / 2), (iy - cy) / (height / 2)


def _gaze(landmarks) -> tuple[float, float, str]:
    if len(landmarks) < 478:
        return 0.0, 0.0, "UNKNOWN"
    left = _iris_offset(landmarks, (362, 263), LEFT_IRIS, [384, 385, 386, 387, 388], [373, 374, 380, 381, 382])
    right = _iris_offset(landmarks, (33, 133), RIGHT_IRIS, [157, 158, 159, 160, 161], [154, 153, 163, 144, 145])
    x, y = float(np.clip((left[0] + right[0]) / 2, -1, 1)), float(np.clip((left[1] + right[1]) / 2, -1, 1))
    if abs(x) < 0.15 and abs(y) < 0.15:
        zone = "CENTER"
    elif abs(x) >= abs(y):
        zone = "LEFT" if x < 0 else "RIGHT"
    else:
        zone = "UP" if y < 0 else "DOWN"
    return x, y, zone


def _eye_pixel_geometry(landmarks, width: int, height: int) -> tuple[float, float, float]:
    """Return left/right eye width and eye-centre distance in image pixels."""
    left_width = _distance(
        landmarks[LEFT_EYE_HOR[0]], landmarks[LEFT_EYE_HOR[1]], width, height
    )
    right_width = _distance(
        landmarks[RIGHT_EYE_HOR[0]], landmarks[RIGHT_EYE_HOR[1]], width, height
    )
    left_center = (
        (landmarks[LEFT_EYE_HOR[0]].x + landmarks[LEFT_EYE_HOR[1]].x) * width / 2.0,
        (landmarks[LEFT_EYE_HOR[0]].y + landmarks[LEFT_EYE_HOR[1]].y) * height / 2.0,
    )
    right_center = (
        (landmarks[RIGHT_EYE_HOR[0]].x + landmarks[RIGHT_EYE_HOR[1]].x) * width / 2.0,
        (landmarks[RIGHT_EYE_HOR[0]].y + landmarks[RIGHT_EYE_HOR[1]].y) * height / 2.0,
    )
    interocular = math.hypot(
        left_center[0] - right_center[0], left_center[1] - right_center[1]
    )
    return float(left_width), float(right_width), float(interocular)


class MediaPipeFacePerception:
    def __init__(
        self,
        model_path: Path,
        min_detection=0.5,
        min_tracking=0.5,
        min_eye_signal_quality=0.45,
        max_eye_pose_deg=25.0,
        max_eye_asymmetry_ratio=0.45,
        min_face_width_ratio=0.20,
        min_interocular_distance_px=50.0,
        min_eye_width_px=24.0,
        process_width=256,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe face landmarker not found: {model_path}")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("Install requirements-raspberry.txt to use MediaPipe") from exc
        self.mp = mp
        self.process_width = int(process_width)
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.previous: tuple[float, float, float, float, float, float] | None = None
        self.last_timestamp_ms = -1
        self.latest_face_bbox_xyxy: tuple[float, float, float, float] | None = None
        self.latest_face_bbox_frame_id: int | None = None
        self.latest_face_bbox_monotonic_sec: float | None = None
        self.min_eye_signal_quality = float(min_eye_signal_quality)
        self.max_eye_pose_deg = float(max_eye_pose_deg)
        self.max_eye_asymmetry_ratio = float(max_eye_asymmetry_ratio)
        self.min_face_width_ratio = float(min_face_width_ratio)
        self.min_interocular_distance_px = float(min_interocular_distance_px)
        self.min_eye_width_px = float(min_eye_width_px)

    def process(self, packet: FramePacket) -> FaceSignal:
        height, width = packet.frame.shape[:2]
        processing_frame = packet.frame
        if self.process_width > 0 and width > self.process_width:
            process_width = max(1, int(self.process_width))
            process_height = max(1, int(round(height * (process_width / width))))
            processing_frame = cv2.resize(
                packet.frame,
                (process_width, process_height),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(self.last_timestamp_ms + 1, int(packet.monotonic_sec * 1000))
        self.last_timestamp_ms = timestamp_ms
        result = self.detector.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            self.previous = None
            self.latest_face_bbox_xyxy = None
            self.latest_face_bbox_frame_id = packet.frame_id
            self.latest_face_bbox_monotonic_sec = packet.monotonic_sec
            return FaceSignal(
                packet.frame_id,
                packet.utc_timestamp,
                packet.monotonic_sec,
                False,
                0.0,
                eye_resolution_valid=False,
                driver_distance_status="NO_FACE",
            )
        landmarks = result.face_landmarks[0]
        xs, ys = np.asarray([p.x for p in landmarks]), np.asarray([p.y for p in landmarks])
        face_w, face_h = float(xs.max() - xs.min()), float(ys.max() - ys.min())
        face_width_px, face_height_px = face_w * width, face_h * height
        bbox_x1, bbox_y1 = float(np.clip(xs.min(), 0.0, 1.0)), float(np.clip(ys.min(), 0.0, 1.0))
        bbox_x2, bbox_y2 = float(np.clip(xs.max(), 0.0, 1.0)), float(np.clip(ys.max(), 0.0, 1.0))
        self.latest_face_bbox_xyxy = (
            bbox_x1 * width,
            bbox_y1 * height,
            bbox_x2 * width,
            bbox_y2 * height,
        )
        self.latest_face_bbox_frame_id = packet.frame_id
        self.latest_face_bbox_monotonic_sec = packet.monotonic_sec
        center_x, center_y = float((xs.max() + xs.min()) / 2), float((ys.max() + ys.min()) / 2)
        border_margin = min(xs.min(), ys.min(), 1 - xs.max(), 1 - ys.max())
        size_score = float(np.clip(min(face_w / 0.20, face_h / 0.20), 0, 1))
        border_score = float(np.clip(border_margin / 0.03, 0, 1))
        quality = 0.7 * size_score + 0.3 * border_score
        left_ear, right_ear = _ear(landmarks, LEFT_EYE, width, height), _ear(landmarks, RIGHT_EYE, width, height)
        ear = (left_ear + right_ear) / 2
        event_left_ear = _event_ear(landmarks, LEFT_EYE, width, height)
        event_right_ear = _event_ear(landmarks, RIGHT_EYE, width, height)
        event_ear = (event_left_ear + event_right_ear) / 2
        left_eye_width_px, right_eye_width_px, interocular_px = _eye_pixel_geometry(
            landmarks, width, height
        )
        eye_resolution_valid = bool(
            face_w >= self.min_face_width_ratio
            and interocular_px >= self.min_interocular_distance_px
            and min(left_eye_width_px, right_eye_width_px) >= self.min_eye_width_px
        )
        mar = _mar(landmarks, width, height)
        pitch, yaw, roll = _head_pose(landmarks, width, height)
        eye_asymmetry = abs(event_left_ear - event_right_ear) / max(event_ear, 1e-6)
        pose_ratio = max(abs(pitch), abs(yaw), abs(roll)) / max(self.max_eye_pose_deg, 1e-6)
        pose_quality = float(np.clip(1.0 - pose_ratio, 0.0, 1.0))
        # Eye asymmetry is diagnostic rather than a hard geometry failure. During
        # a normal blink one eyelid commonly leads the other by one or two frames;
        # rejecting that frame entirely made valid blinks look like signal loss.
        eye_quality = float(quality * (0.5 + 0.5 * pose_quality))
        binocular_consistent = bool(eye_asymmetry <= self.max_eye_asymmetry_ratio)
        eye_valid = bool(
            np.isfinite(event_left_ear)
            and np.isfinite(event_right_ear)
            and event_left_ear > 0
            and event_right_ear > 0
            and max(abs(pitch), abs(yaw), abs(roll)) <= self.max_eye_pose_deg
            and eye_quality >= self.min_eye_signal_quality
            and eye_resolution_valid
        )
        gaze_x, gaze_y, gaze_zone = _gaze(landmarks)
        velocity_x = velocity_y = head_motion = 0.0
        if self.previous is not None:
            px, py, ppitch, pyaw, proll, previous_time = self.previous
            dt = max(packet.monotonic_sec - previous_time, 1e-3)
            velocity_x = (center_x - px) / max(face_w, 1e-6) / dt
            velocity_y = (center_y - py) / max(face_h, 1e-6) / dt
            head_motion = math.sqrt((pitch - ppitch) ** 2 + (yaw - pyaw) ** 2 + (roll - proll) ** 2) / dt
        self.previous = (center_x, center_y, pitch, yaw, roll, packet.monotonic_sec)
        return FaceSignal(
            packet.frame_id, packet.utc_timestamp, packet.monotonic_sec, True, quality,
            left_ear=float(np.clip(left_ear, 0, 1)), right_ear=float(np.clip(right_ear, 0, 1)),
            ear=float(np.clip(ear, 0, 1)), mar=float(np.clip(mar, 0, 2)),
            event_left_ear=float(np.clip(event_left_ear, 0, 1)),
            event_right_ear=float(np.clip(event_right_ear, 0, 1)),
            event_ear=float(np.clip(event_ear, 0, 1)),
            eye_signal_valid=eye_valid, eye_signal_quality=eye_quality,
            face_width_px=float(face_width_px),
            face_height_px=float(face_height_px),
            face_width_ratio=float(face_w),
            interocular_distance_px=float(interocular_px),
            left_eye_width_px=float(left_eye_width_px),
            right_eye_width_px=float(right_eye_width_px),
            eye_resolution_valid=eye_resolution_valid,
            driver_distance_status="OK" if eye_resolution_valid else "TOO_FAR",
            binocular_consistent=binocular_consistent,
            pitch=float(np.clip(pitch, -90, 90)), yaw=float(np.clip(yaw, -90, 90)), roll=float(np.clip(roll, -90, 90)),
            gaze_x=gaze_x, gaze_y=gaze_y, gaze_zone=gaze_zone,
            face_velocity_x=float(np.clip(velocity_x, -15, 15)), face_velocity_y=float(np.clip(velocity_y, -15, 15)),
            head_motion=float(np.clip(head_motion, 0, 20_000)),
        )

    def close(self) -> None:
        self.detector.close()
