from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )


def read_windows(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    if frame.empty:
        raise ValueError(f"Window CSV is empty: {path}")
    return frame


def csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)
