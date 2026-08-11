from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from dms_final_system.shared.contracts import IncidentRecord


class IncidentSink(ABC):
    @abstractmethod
    def publish(self, incident: IncidentRecord) -> None:
        """Accept a finalized local incident without blocking the AI loop."""


class FilesystemOutboxSink(IncidentSink):
    def publish(self, incident: IncidentRecord) -> None:
        path = incident.video_path.parent / "incident.json"
        payload = {}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(incident.to_dict())
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)


class CompanyUploadAdapter(IncidentSink):
    """Integration contract for the other team; deliberately no network implementation."""

    def publish(self, incident: IncidentRecord) -> None:
        raise NotImplementedError(
            "Implement authenticated upload here. Upload video.mp4, incident.json and telemetry.jsonl; "
            "update upload_status only after the company API acknowledges all files."
        )
