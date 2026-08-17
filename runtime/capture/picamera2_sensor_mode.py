from __future__ import annotations

from typing import Any


def select_full_fov_sensor_mode(
    sensor_modes: list[dict[str, Any]],
    requested_size: tuple[int, int],
    requested_fps: float,
) -> dict[str, Any]:
    """Choose a full-FOV sensor mode that satisfies the output and frame rate."""
    valid_modes: list[dict[str, Any]] = []
    for mode in sensor_modes:
        size = mode.get("size")
        crop_limits = mode.get("crop_limits")
        bit_depth = mode.get("bit_depth")
        fps = mode.get("fps")
        if (
            isinstance(size, (tuple, list))
            and len(size) == 2
            and isinstance(crop_limits, (tuple, list))
            and len(crop_limits) == 4
            and bit_depth is not None
            and fps is not None
        ):
            valid_modes.append(mode)

    if not valid_modes:
        raise RuntimeError("Picamera2 did not report any usable sensor modes")

    requested_width, requested_height = requested_size
    large_enough = [
        mode
        for mode in valid_modes
        if int(mode["size"][0]) >= requested_width
        and int(mode["size"][1]) >= requested_height
    ]
    candidates = large_enough or valid_modes

    widest_crop_area = max(
        int(mode["crop_limits"][2]) * int(mode["crop_limits"][3])
        for mode in candidates
    )
    full_fov_modes = [
        mode
        for mode in candidates
        if int(mode["crop_limits"][2]) * int(mode["crop_limits"][3])
        == widest_crop_area
    ]
    frame_rate_modes = [
        mode
        for mode in full_fov_modes
        if float(mode["fps"]) + 1e-6 >= requested_fps
    ]
    if not frame_rate_modes:
        maximum_fps = max(float(mode["fps"]) for mode in full_fov_modes)
        raise RuntimeError(
            "Full-FOV Picamera2 sensor modes do not support the requested "
            f"{requested_fps:g} FPS (maximum {maximum_fps:g} FPS); lower camera.fps"
        )

    # With equal FOV, prefer the smallest sufficient mode to reduce bandwidth.
    return min(
        frame_rate_modes,
        key=lambda mode: int(mode["size"][0]) * int(mode["size"][1]),
    )
