from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def _z(row: pd.Series, signal: str) -> float:
    return _num(row, f"{signal}_z_recent_30", 0.0)


def _stand_no(stand: str | None) -> int | None:
    m = re.search(r"(\d+)$", str(stand or ""))
    return int(m.group(1)) if m else None


def analyze_cascading_impact(row: pd.Series, ml_result: dict[str, Any]) -> dict[str, Any]:
    predicted_stand = ml_result.get("predicted_stand", "mill_level")
    s = _stand_no(predicted_stand)
    if s is None:
        affected = []
        for i in range(1, 6):
            stand_score = max(abs(_z(row, f"force_{i}")), abs(_z(row, f"torque_{i}")), abs(_z(row, f"motor_power_{i}")))
            if stand_score >= 1.5:
                affected.append({"stand": f"stand_{i}", "score": round(float(stand_score), 2)})
        risk_score = min(1.0, len(affected) / 5.0 + float(ml_result.get("anomaly_probability", 0)) * 0.35)
        return {
            "primary_stand": "mill_level",
            "cascading_risk_score": round(risk_score, 3),
            "cascading_risk": "high" if risk_score >= 0.65 else "medium" if risk_score >= 0.35 else "low",
            "upstream_effect": "Entry/early-stand load schedule may be affected by reduction/gap setup.",
            "downstream_effect": "Exit-side thickness, tension, and load sharing may become unstable across multiple stands.",
            "affected_neighbors": affected,
            "recommended_checks": [
                "Verify pass schedule and gap/reduction settings across all stands.",
                "Review inter-stand tension and exit thickness stability.",
                "Check whether stand-level symptoms are secondary to a mill-level process setup issue.",
            ],
        }

    neighbors = []
    if s > 1:
        neighbors.append((s - 1, "upstream"))
    if s < 5:
        neighbors.append((s + 1, "downstream"))

    affected = []
    for n, direction in neighbors:
        signals = {
            f"torque_{n}": _z(row, f"torque_{n}"),
            f"motor_power_{n}": _z(row, f"motor_power_{n}"),
            f"force_{n}": _z(row, f"force_{n}"),
        }
        tension_idx = n if 0 <= n <= 5 else None
        if tension_idx is not None:
            signals[f"tension_{tension_idx}"] = _z(row, f"tension_{tension_idx}")
        max_abs = max(abs(v) for v in signals.values()) if signals else 0
        if max_abs >= 1.2:
            affected.append({
                "stand": f"stand_{n}",
                "direction": direction,
                "max_abs_z": round(float(max_abs), 2),
                "signals": {k: round(float(v), 2) for k, v in signals.items()},
            })

    primary_score = max(
        abs(_z(row, f"torque_{s}")),
        abs(_z(row, f"motor_power_{s}")),
        abs(_z(row, f"force_{s}")),
        abs(_z(row, f"tension_{s}")) if 0 <= s <= 5 else 0,
    )
    neighbor_score = max([a["max_abs_z"] for a in affected], default=0.0)
    risk_score = min(1.0, primary_score / 6.0 + neighbor_score / 8.0 + 0.15 * len(affected))

    return {
        "primary_stand": f"stand_{s}",
        "cascading_risk_score": round(float(risk_score), 3),
        "cascading_risk": "high" if risk_score >= 0.65 else "medium" if risk_score >= 0.35 else "low",
        "upstream_effect": f"Stand {s-1} may be influencing or responding to Stand {s} load/tension changes." if s > 1 else "No upstream stand before Stand 1.",
        "downstream_effect": f"Stand {s+1} may inherit tension/load disturbance from Stand {s}." if s < 5 else "No downstream stand after Stand 5.",
        "affected_neighbors": affected,
        "recommended_checks": [
            f"Compare Stand {s} torque, force, motor power, and tension against adjacent stands.",
            "Check whether the predicted component fault is primary or a symptom of upstream/downstream strip instability.",
            "Monitor downstream motor load and exit tension before resuming normal operating speed.",
        ],
    }
