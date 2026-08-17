#!/usr/bin/env python3
"""Keep a transistor-switched buzzer on until the process is stopped."""

from __future__ import annotations

import argparse
import signal
import threading


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep a Raspberry Pi GPIO buzzer continuously on"
    )
    parser.add_argument(
        "--pin",
        type=int,
        default=18,
        help="BCM GPIO number (default: 18, physical header pin 12)",
    )
    parser.add_argument(
        "--buzzer-type",
        choices=("active", "passive"),
        default="active",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=2000.0,
        help="Tone frequency for a passive buzzer",
    )
    args = parser.parse_args()

    if not 0 <= args.pin <= 27:
        parser.error("--pin must be a BCM GPIO number between 0 and 27")
    if args.frequency <= 0:
        parser.error("--frequency must be positive")

    try:
        from gpiozero import DigitalOutputDevice, PWMOutputDevice
    except ImportError as exc:
        raise SystemExit(
            "gpiozero is required; install it with: "
            "sudo apt install -y python3-gpiozero python3-lgpio"
        ) from exc

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.buzzer_type == "active":
        device = DigitalOutputDevice(args.pin, active_high=True, initial_value=False)
    else:
        device = PWMOutputDevice(
            args.pin,
            active_high=True,
            initial_value=0.0,
            frequency=args.frequency,
        )

    try:
        if args.buzzer_type == "active":
            device.on()
        else:
            device.value = 0.5
        print(
            f"Buzzer continuously ON: BCM GPIO{args.pin}, {args.buzzer_type}. "
            "Press Ctrl+C to stop.",
            flush=True,
        )
        stop_event.wait()
    finally:
        device.off()
        device.close()
        print("Buzzer OFF", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
