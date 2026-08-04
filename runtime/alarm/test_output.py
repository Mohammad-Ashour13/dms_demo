from __future__ import annotations

import argparse
import threading
import time

from dms_final_system.runtime.alarm.outputs import LinuxAudioOutput
from dms_final_system.runtime.alarm.patterns import ALARM_PATTERNS
from dms_final_system.shared.contracts import AlarmCommand, DriverState


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly test one local DMS alarm sound")
    parser.add_argument("--pattern", choices=sorted(ALARM_PATTERNS), default="DROWSY")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="default")
    parser.add_argument("--gain", type=float, default=0.40)
    args = parser.parse_args()
    if not 0 <= args.gain <= 1:
        raise SystemExit("--gain must be between 0 and 1")
    output = LinuxAudioOutput(args.backend, args.device, args.gain)
    ok, backend = output.probe()
    if not ok:
        raise SystemExit(backend)
    state = DriverState.CRITICAL if args.pattern == "CRITICAL" else DriverState.DROWSY
    command = AlarmCommand(
        "manual-sound-test", "manual-sound-test", time.monotonic(), state,
        args.pattern, "ONSET", 1, ["MANUAL_SOUND_TEST"], None,
    )
    print(f"Playing {args.pattern} through {backend} at gain {args.gain:.2f}")
    try:
        print(output.play(command, threading.Event()))
    finally:
        output.close()


if __name__ == "__main__":
    main()
