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


def _z(row: pd.Series, signal: str, window: int = 30) -> float:
    return _num(row, f"{signal}_z_recent_{window}", 0.0)


def _abs_z(row: pd.Series, signal: str, window: int = 30) -> float:
    return abs(_z(row, signal, window))


def _severity_from_score(score: float) -> str:
    if score >= 0.80:
        return "critical"
    if score >= 0.55:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def _stand_number(stand: str | None) -> int | None:
    if not stand:
        return None
    m = re.search(r"(\d+)$", str(stand))
    return int(m.group(1)) if m else None


def _rule(rule_id: str, title: str, severity_score: float, matched: bool, explanation: str, evidence: list[str], recommendation: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "title": title,
        "matched": bool(matched),
        "severity": _severity_from_score(float(severity_score)) if matched else "info",
        "severity_score": round(float(severity_score), 3),
        "explanation": explanation,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def apply_physical_constraint_rules(row: pd.Series, ml_result: dict[str, Any]) -> list[dict[str, Any]]:
    
    rules: list[dict[str, Any]] = []
    predicted_stand = ml_result.get("predicted_stand", "mill_level")
    candidate_stands = [int(predicted_stand[-1])] if re.match(r"stand_\d", str(predicted_stand)) else [1, 2, 3, 4, 5]

    best = (0, 0.0, [])
    for i in candidate_stands:
        torque_z = _z(row, f"torque_{i}")
        power_z = _z(row, f"motor_power_{i}")
        reduction_z = _z(row, f"reduction_{i}")
        score = min(1.0, (max(0, torque_z) + max(0, power_z)) / 6.0) * (1.0 if abs(reduction_z) <= 1.5 else 0.65)
        if score > best[1]:
            best = (i, score, [f"torque_{i} z={torque_z:.2f}", f"motor_power_{i} z={power_z:.2f}", f"reduction_{i} z={reduction_z:.2f}"])
    i, score, ev = best
    rules.append(_rule(
        "PHY_MECH_LOAD_001",
        "Mechanical load increase check",
        score,
        score >= 0.35,
        "Torque and motor power rising together under mostly stable reduction indicates load-side stress such as bearing load, friction, misalignment, or lubrication degradation.",
        ev,
        "Inspect mechanical load path first: bearing lubrication, coupling alignment, roll friction, and abnormal loading before treating it as an isolated electrical issue.",
    ))

    # R2: motor power rises without torque rise => likely electrical/drive efficiency issue.
    best = (0, 0.0, [])
    for i in candidate_stands:
        power_z = _z(row, f"motor_power_{i}")
        torque_z = _z(row, f"torque_{i}")
        score = min(1.0, max(0, power_z) / 4.0) * (1.0 if abs(torque_z) <= 1.2 else 0.5)
        if score > best[1]:
            best = (i, score, [f"motor_power_{i} z={power_z:.2f}", f"torque_{i} z={torque_z:.2f}"])
    i, score, ev = best
    rules.append(_rule(
        "PHY_ELEC_DRIVE_002",
        "Electrical drive efficiency check",
        score,
        score >= 0.35,
        "Motor power deviation without proportional torque deviation is more consistent with drive, motor, cooling, supply, or efficiency issues.",
        ev,
        "Check drive alarms, current imbalance, motor cooling, and supply quality; then rule out hidden mechanical drag.",
    ))

    best = (0, 0.0, [])
    for i in candidate_stands:
        force_z = _z(row, f"force_{i}")
        torque_z = _z(row, f"torque_{i}")
        mileage = _num(row, f"mileage_norm_{i}")
        score = min(1.0, (max(0, force_z) + max(0, torque_z)) / 7.0 + 0.25 * mileage)
        if score > best[1]:
            best = (i, score, [f"force_{i} z={force_z:.2f}", f"torque_{i} z={torque_z:.2f}", f"mileage_norm_{i}={mileage:.2f}"])
    i, score, ev = best
    rules.append(_rule(
        "PHY_WORKROLL_003",
        "Work roll friction / wear check",
        score,
        score >= 0.45,
        "High force and torque with meaningful work-roll mileage can indicate roll surface wear, emulsion/lubrication degradation, or friction increase.",
        ev,
        "Check emulsion concentration, roll cooling, roll surface condition, and roll-change threshold.",
    ))

    red_hits = []
    gap_hits = []
    for i in range(1, 6):
        rz = _z(row, f"reduction_{i}")
        gz = _z(row, f"gap_{i}")
        if abs(rz) >= 2.0:
            red_hits.append(f"reduction_{i} z={rz:.2f}")
        if abs(gz) >= 2.0:
            gap_hits.append(f"gap_{i} z={gz:.2f}")
    score = min(1.0, (len(red_hits) + len(gap_hits)) / 5.0)
    rules.append(_rule(
        "PHY_REDUCTION_004",
        "Reduction/gap scheme consistency check",
        score,
        score >= 0.35 or "reduction" in str(ml_result.get("predicted_fault", "")),
        "Coordinated deviation in gap or reduction across stands suggests process setup or reduction-schedule abnormality rather than a single isolated component fault.",
        red_hits[:5] + gap_hits[:5],
        "Verify pass schedule, roll gap setup, thickness target, AGC settings, and inter-stand tension coordination.",
    ))


    stand_no = _stand_number(predicted_stand)
    tension_evidence = []
    if stand_no is not None:
        for t in [stand_no - 1, stand_no, stand_no + 1]:
            if 0 <= t <= 5:
                tz = _z(row, f"tension_{t}")
                if abs(tz) >= 1.5:
                    tension_evidence.append(f"tension_{t} z={tz:.2f}")
    else:
        for t in range(0, 6):
            tz = _z(row, f"tension_{t}")
            if abs(tz) >= 2.0:
                tension_evidence.append(f"tension_{t} z={tz:.2f}")
    score = min(1.0, len(tension_evidence) / 3.0)
    rules.append(_rule(
        "PHY_CASCADE_005",
        "Adjacent tension cascade check",
        score,
        score >= 0.30,
        "A local stand abnormality can propagate through inter-stand strip tension and disturb upstream/downstream load sharing.",
        tension_evidence,
        "Review adjacent stand trends before final isolation; monitor tension stability and downstream motor-load changes.",
    ))

    return sorted(rules, key=lambda r: (not r["matched"], -r["severity_score"]))
