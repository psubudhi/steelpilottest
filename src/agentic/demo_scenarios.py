from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEMO_SCENARIOS: dict[str, dict[str, Any]] = {
    "normal_operation": {
        "label": "Normal operation / healthy stream",
        "description": "Clears major abnormal z-scores to show a stable shift-start condition.",
        "fault_type": "normal",
    },
    "bearing_mechanical_overload": {
        "label": "Bearing-like mechanical overload",
        "description": "Torque and motor power rise together while reduction remains mostly stable.",
        "fault_type": "bearing",
    },
    "electric_motor_efficiency": {
        "label": "Electric motor / drive efficiency anomaly",
        "description": "Motor power rises strongly without a proportional torque increase.",
        "fault_type": "electric",
    },
    "workroll_friction": {
        "label": "Work-roll friction / lubrication issue",
        "description": "Force and torque rise with high work-roll mileage, indicating friction or wear.",
        "fault_type": "workroll",
    },
    "reduction_scheme_anomaly": {
        "label": "Reduction scheme / gap setup anomaly",
        "description": "Gap and reduction pattern shifts, creating a mill-level process anomaly.",
        "fault_type": "reduction",
    },
    "cascading_instability": {
        "label": "Cascading instability across neighbouring stands",
        "description": "A primary stand disturbance also moves adjacent tension/load signals.",
        "fault_type": "cascade",
    },
}


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _safe_set(row: pd.Series, col: str, value: float) -> None:
    if col in row.index and np.isfinite(value):
        row[col] = float(value)


def _bump(row: pd.Series, col: str, pct: float, severity: float) -> None:
    if col in row.index:
        base = _num(row[col])
        if base == 0:
            base = 1.0
        row[col] = base * (1.0 + pct * severity)


def _drop(row: pd.Series, col: str, pct: float, severity: float) -> None:
    if col in row.index:
        base = _num(row[col])
        row[col] = base * max(0.0, 1.0 - pct * severity)


def _set_z(row: pd.Series, signal: str, z: float) -> None:
    for w in (10, 30, 60):
        z_col = f"{signal}_z_recent_{w}"
        if z_col in row.index:
            row[z_col] = float(z)


def _recompute_z_from_mean_std(row: pd.Series, signal: str) -> None:
    if signal not in row.index:
        return
    current = _num(row[signal])
    for w in (10, 30, 60):
        mean_col = f"{signal}_roll_mean_{w}"
        std_col = f"{signal}_roll_std_{w}"
        z_col = f"{signal}_z_recent_{w}"
        if mean_col in row.index and std_col in row.index and z_col in row.index:
            std = abs(_num(row[std_col]))
            if std < 1e-9:
                continue
            row[z_col] = float((current - _num(row[mean_col])) / (std + 1e-9))


def _recompute_ratios_for_stand(row: pd.Series, stand: int) -> None:
    s = int(stand)
    torque = f"torque_{s}"
    force = f"force_{s}"
    power = f"motor_power_{s}"
    reduction = f"reduction_{s}"
    speed = f"roll_speed_{s}"

    if torque in row.index and force in row.index and _num(row[force]) != 0:
        row[f"torque_to_force_ratio_{s}"] = _num(row[torque]) / (_num(row[force]) + 1e-9)
    if power in row.index and torque in row.index and _num(row[torque]) != 0:
        row[f"power_to_torque_ratio_{s}"] = _num(row[power]) / (_num(row[torque]) + 1e-9)
    if force in row.index and reduction in row.index and _num(row[reduction]) != 0:
        row[f"force_per_reduction_{s}"] = _num(row[force]) / (_num(row[reduction]) + 1e-9)
    if power in row.index and torque in row.index:
        row[f"motor_load_index_{s}"] = _num(row[power]) * _num(row[torque])
    if speed in row.index and power in row.index and _num(row[power]) != 0:
        row[f"speed_power_ratio_{s}"] = _num(row[speed]) / (_num(row[power]) + 1e-9)


def _recompute_adjacent_features(row: pd.Series) -> None:
    for base in ["torque", "force", "motor_power", "roll_speed", "gap", "reduction"]:
        for i in range(2, 6):
            c1, c0 = f"{base}_{i}", f"{base}_{i-1}"
            diff, ratio = f"{base}_diff_{i}_{i-1}", f"{base}_ratio_{i}_{i-1}"
            if c1 in row.index and c0 in row.index:
                if diff in row.index:
                    row[diff] = _num(row[c1]) - _num(row[c0])
                if ratio in row.index and abs(_num(row[c0])) > 1e-9:
                    row[ratio] = _num(row[c1]) / (_num(row[c0]) + 1e-9)


def _scenario_meta(scenario: str, stand: int, severity: float, evidence: list[str]) -> dict[str, Any]:
    cfg = DEMO_SCENARIOS.get(scenario, DEMO_SCENARIOS["normal_operation"])
    sev = float(np.clip(severity, 0.5, 2.0))
    s = int(stand)

    if scenario == "normal_operation":
        return {
            "scenario": scenario,
            "label": cfg["label"],
            "description": cfg["description"],
            "forced_fault": "normal_operation",
            "forced_stand": f"stand_{s}",
            "asset_name": f"TCM Stand {s}",
            "forced_anomaly_probability": 0.04,
            "forced_fault_confidence": 0.02,
            "forced_risk_score": 18.0,
            "forced_rul_band": "monitor",
            "forced_is_alert": False,
            "evidence": evidence,
            "secondary_faults": [],
        }

    risk = float(np.clip(54 + 22 * sev, 0, 98))
    prob = float(np.clip(0.62 + 0.18 * sev, 0, 0.99))
    conf = float(np.clip(0.58 + 0.18 * sev, 0, 0.98))
    rul = "immediate" if risk >= 76 else "within_1_shift"

    if scenario == "bearing_mechanical_overload":
        fault, forced_stand = f"anomaly_bearing_{s}", f"stand_{s}"
    elif scenario == "electric_motor_efficiency":
        fault, forced_stand = f"anomaly_electric_{s}", f"stand_{s}"
    elif scenario == "workroll_friction":
        fault, forced_stand = f"anomaly_workroll_{s}", f"stand_{s}"
    elif scenario == "reduction_scheme_anomaly":
        fault, forced_stand = "anomaly_reduction", "mill_level"
        risk = float(np.clip(62 + 20 * sev, 0, 99))
        prob = float(np.clip(0.70 + 0.15 * sev, 0, 0.995))
        conf = float(np.clip(0.68 + 0.14 * sev, 0, 0.99))
        rul = "immediate" if risk >= 76 else "within_1_shift"
    elif scenario == "cascading_instability":
        fault, forced_stand = f"anomaly_bearing_{s}", f"stand_{s}"
        risk = float(np.clip(60 + 20 * sev, 0, 99))
        prob = float(np.clip(0.68 + 0.15 * sev, 0, 0.995))
        conf = float(np.clip(0.62 + 0.15 * sev, 0, 0.98))
        rul = "immediate" if risk >= 76 else "within_1_shift"
    else:
        fault, forced_stand = f"anomaly_bearing_{s}", f"stand_{s}"

    return {
        "scenario": scenario,
        "label": cfg["label"],
        "description": cfg["description"],
        "forced_fault": fault,
        "forced_stand": forced_stand,
        "asset_name": "TCM Mill Level" if forced_stand == "mill_level" else f"TCM Stand {s}",
        "forced_anomaly_probability": prob,
        "forced_fault_confidence": conf,
        "forced_risk_score": risk,
        "forced_rul_band": rul,
        "forced_is_alert": True,
        "evidence": evidence,
        "secondary_faults": [],
    }


def apply_demo_scenario(row: pd.Series, scenario: str, stand: int = 3, severity: float = 1.0) -> tuple[pd.Series, dict[str, Any]]:
    
    r = row.copy()
    scenario = scenario if scenario in DEMO_SCENARIOS else "normal_operation"
    s = int(stand)
    sev = float(np.clip(severity, 0.5, 2.0))
    evidence: list[str] = []

    if scenario == "normal_operation":
        for sig in ["torque", "force", "motor_power", "gap", "reduction", "roll_speed"]:
            _set_z(r, f"{sig}_{s}", 0.15)
        for t in [max(0, s - 1), s, min(5, s)]:
            _set_z(r, f"tension_{t}", 0.10)
        evidence = [f"stand_{s} major z-scores normalized for demo healthy condition"]
        return r, _scenario_meta(scenario, s, sev, evidence)

    if scenario == "bearing_mechanical_overload":
        _bump(r, f"torque_{s}", 0.30, sev)
        _bump(r, f"motor_power_{s}", 0.22, sev)
        _bump(r, f"force_{s}", 0.08, sev)
        _set_z(r, f"torque_{s}", 2.8 + 0.45 * sev)
        _set_z(r, f"motor_power_{s}", 2.4 + 0.45 * sev)
        _set_z(r, f"force_{s}", 1.4 + 0.35 * sev)
        _set_z(r, f"reduction_{s}", 0.35)
        evidence = [
            f"torque_{s}: recent z-score {2.8 + 0.45 * sev:.2f}",
            f"motor_power_{s}: recent z-score {2.4 + 0.45 * sev:.2f}",
            f"reduction_{s}: stable recent z-score 0.35",
        ]

    elif scenario == "electric_motor_efficiency":
        _bump(r, f"motor_power_{s}", 0.35, sev)
        _bump(r, f"torque_{s}", 0.03, sev)
        _set_z(r, f"motor_power_{s}", 3.0 + 0.45 * sev)
        _set_z(r, f"torque_{s}", 0.55)
        _set_z(r, f"force_{s}", 0.35)
        _set_z(r, f"reduction_{s}", 0.20)
        evidence = [
            f"motor_power_{s}: recent z-score {3.0 + 0.45 * sev:.2f}",
            f"torque_{s}: stable recent z-score 0.55",
            f"force_{s}: stable recent z-score 0.35",
        ]

    elif scenario == "workroll_friction":
        _bump(r, f"force_{s}", 0.30, sev)
        _bump(r, f"torque_{s}", 0.22, sev)
        _bump(r, f"motor_power_{s}", 0.12, sev)
        if f"mileage_norm_{s}" in r.index:
            r[f"mileage_norm_{s}"] = min(1.0, max(_num(r[f"mileage_norm_{s}"]), 0.75 + 0.08 * sev))
        _set_z(r, f"force_{s}", 2.8 + 0.45 * sev)
        _set_z(r, f"torque_{s}", 2.3 + 0.40 * sev)
        _set_z(r, f"motor_power_{s}", 1.5 + 0.25 * sev)
        _set_z(r, f"reduction_{s}", 0.25)
        evidence = [
            f"force_{s}: recent z-score {2.8 + 0.45 * sev:.2f}",
            f"torque_{s}: recent z-score {2.3 + 0.40 * sev:.2f}",
            f"mileage_norm_{s}: high work-roll usage",
        ]

    elif scenario == "reduction_scheme_anomaly":
        for i in range(1, 6):
            if i in {s, max(1, s - 1), min(5, s + 1)}:
                _bump(r, f"gap_{i}", 0.10, sev)
                _drop(r, f"reduction_{i}", 0.12, sev)
                _bump(r, f"force_{i}", 0.18, sev)
                _bump(r, f"torque_{i}", 0.14, sev)
                _set_z(r, f"gap_{i}", 2.0 + 0.30 * sev)
                _set_z(r, f"reduction_{i}", -(2.2 + 0.35 * sev))
                _set_z(r, f"force_{i}", 2.2 + 0.25 * sev)
                _set_z(r, f"torque_{i}", 2.1 + 0.25 * sev)
        evidence = [
            f"gap_{s}: recent z-score {2.0 + 0.30 * sev:.2f}",
            f"reduction_{s}: recent z-score {-1*(2.2 + 0.35 * sev):.2f}",
            f"force_{s}: recent z-score {2.2 + 0.25 * sev:.2f}",
        ]

    elif scenario == "cascading_instability":
        _bump(r, f"torque_{s}", 0.25, sev)
        _bump(r, f"motor_power_{s}", 0.18, sev)
        _set_z(r, f"torque_{s}", 2.7 + 0.35 * sev)
        _set_z(r, f"motor_power_{s}", 2.3 + 0.30 * sev)
        _set_z(r, f"force_{s}", 1.7 + 0.25 * sev)
        if s > 1:
            _bump(r, f"tension_{s-1}", 0.20, sev)
            _set_z(r, f"tension_{s-1}", 2.1 + 0.35 * sev)
        _bump(r, f"tension_{s}", 0.22, sev)
        _set_z(r, f"tension_{s}", 2.4 + 0.35 * sev)
        if s < 5:
            _bump(r, f"motor_power_{s+1}", 0.10, sev)
            _bump(r, f"force_{s+1}", 0.08, sev)
            _set_z(r, f"motor_power_{s+1}", 1.8 + 0.25 * sev)
            _set_z(r, f"force_{s+1}", 1.6 + 0.20 * sev)
        evidence = [
            f"torque_{s}: recent z-score {2.7 + 0.35 * sev:.2f}",
            f"tension_{s}: recent z-score {2.4 + 0.35 * sev:.2f}",
            f"adjacent stand load/tension disturbance applied",
        ]

    for i in range(1, 6):
        _recompute_ratios_for_stand(r, i)
    _recompute_adjacent_features(r)

    meta = _scenario_meta(scenario, s, sev, evidence)
    meta["stand"] = s
    meta["severity"] = sev
    return r, meta
