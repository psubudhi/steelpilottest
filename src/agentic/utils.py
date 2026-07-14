from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def infer_stand_from_fault_label(label: str) -> str:
    label = str(label).lower()
    match = re.search(r"_(\d)$", label)
    if match:
        return f"stand_{match.group(1)}"
    return "mill_level"


def risk_level_from_score(score: float) -> str:
    if score >= 76:
        return "critical"
    if score >= 56:
        return "high"
    if score >= 31:
        return "medium"
    return "low"
