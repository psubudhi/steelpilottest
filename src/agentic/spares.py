from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_default_spares(path: Path = Path("data/synthetic/spare_parts.csv")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        rows = [
            ["BRG_ST1", "Bearing Set Stand 1", "stand_1", 1, 0, 45000],
            ["BRG_ST2", "Bearing Set Stand 2", "stand_2", 0, 7, 45000],
            ["BRG_ST3", "Bearing Set Stand 3", "stand_3", 1, 0, 45000],
            ["BRG_ST4", "Bearing Set Stand 4", "stand_4", 1, 0, 45000],
            ["BRG_ST5", "Bearing Set Stand 5", "stand_5", 0, 10, 45000],
            ["MTR_ST1", "Motor Drive Module Stand 1", "stand_1", 0, 14, 120000],
            ["MTR_ST2", "Motor Drive Module Stand 2", "stand_2", 1, 0, 120000],
            ["MTR_ST3", "Motor Drive Module Stand 3", "stand_3", 1, 0, 120000],
            ["MTR_ST4", "Motor Drive Module Stand 4", "stand_4", 0, 14, 120000],
            ["MTR_ST5", "Motor Drive Module Stand 5", "stand_5", 1, 0, 120000],
            ["ROLL_ST1", "Work Roll Stand 1", "stand_1", 2, 0, 90000],
            ["ROLL_ST2", "Work Roll Stand 2", "stand_2", 2, 0, 90000],
            ["ROLL_ST3", "Work Roll Stand 3", "stand_3", 1, 0, 90000],
            ["ROLL_ST4", "Work Roll Stand 4", "stand_4", 0, 5, 90000],
            ["ROLL_ST5", "Work Roll Stand 5", "stand_5", 1, 0, 90000],
        ]
        pd.DataFrame(rows, columns=["part_id", "part_name", "equipment_id", "stock_available", "lead_time_days", "cost_inr"]).to_csv(path, index=False)
    return path


def get_spares_for_fault(predicted_fault: str, predicted_stand: str, path: Path = Path("data/synthetic/spare_parts.csv")) -> list[dict[str, Any]]:
    ensure_default_spares(path)
    df = pd.read_csv(path)
    fault = predicted_fault.lower()
    stand = predicted_stand.lower()
    if stand == "mill_level":
        return df.head(5).to_dict(orient="records")
    if "bearing" in fault:
        q = df[(df["equipment_id"] == stand) & df["part_name"].str.contains("Bearing", case=False, na=False)]
    elif "electric" in fault:
        q = df[(df["equipment_id"] == stand) & df["part_name"].str.contains("Motor|Drive", case=False, na=False)]
    elif "workroll" in fault or "work_roll" in fault:
        q = df[(df["equipment_id"] == stand) & df["part_name"].str.contains("Work Roll", case=False, na=False)]
    else:
        q = df[df["equipment_id"] == stand]
    return q.to_dict(orient="records")
