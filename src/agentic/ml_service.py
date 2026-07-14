from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import settings
from .utils import infer_stand_from_fault_label, risk_level_from_score


class MLArtifactError(RuntimeError):
    pass


class MLModelService:

    def __init__(self, modelling_root: Path | None = None):
        self.root = modelling_root or settings.modelling_root
        self.model_dir = self.root / "models"
        self.output_dir = self.root / "outputs"
        self.processed_dir = self.root / "data" / "processed"
        self._loaded = False
        self.demo_override_enabled = False
        self.demo_override_index: int | None = None
        self.demo_override_row: pd.Series | None = None
        self.demo_override_meta: dict[str, Any] = {}

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise MLArtifactError(
                f"Missing artifact: {path}. Run the ML notebook first or set TCM_MODELLING_ROOT correctly."
            )
        return path

    def load(self) -> "MLModelService":
        with open(self._require(self.model_dir / "model_metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.feature_cols = self.metadata["feature_cols"]
        self.anomaly_cols = self.metadata["anomaly_cols"]
        self.anomaly_threshold = float(
            self.metadata.get("anomaly_threshold_recall_biased", self.metadata.get("anomaly_threshold_f1", 0.5))
        )
        self.obs_per_shift = int(self.metadata.get("obs_per_shift_assumption", 200))
        rb = self.metadata.get("rul_band_names", {})
        try:
            self.rul_band_names = {int(k): v for k, v in rb.items()}
        except Exception:
            self.rul_band_names = rb
        if not self.rul_band_names:
            self.rul_band_names = {0: "monitor", 1: "within_7_shifts", 2: "within_3_shifts", 3: "within_1_shift", 4: "immediate"}

        self.anomaly_model = joblib.load(self._require(self.model_dir / "anomaly_classifier.joblib"))
        self.fault_model = joblib.load(self._require(self.model_dir / "fault_multilabel_classifier.joblib"))
        self.rul_clf = joblib.load(self._require(self.model_dir / "rul_urgency_classifier.joblib"))
        self.rul_reg = joblib.load(self._require(self.model_dir / "proxy_rul_regressor.joblib"))

        self.case_pre = None
        self.case_nn = None
        self.case_meta = pd.DataFrame()
        if (self.model_dir / "case_memory_preprocessor.joblib").exists() and (self.model_dir / "case_memory_nearest_neighbors.joblib").exists():
            self.case_pre = joblib.load(self.model_dir / "case_memory_preprocessor.joblib")
            self.case_nn = joblib.load(self.model_dir / "case_memory_nearest_neighbors.joblib")
        if (self.output_dir / "historical_case_memory.csv").exists():
            self.case_meta = pd.read_csv(self.output_dir / "historical_case_memory.csv")

        feat_path = self._require(self.processed_dir / "tcm_features_dataset3.parquet")
        self.features_df = pd.read_parquet(feat_path)
        self.features_df = self.features_df.loc[:, ~self.features_df.columns.duplicated()]
        self.features_df = self.features_df.reset_index(drop=True)

        self.priority_board = self._safe_read_csv(self.output_dir / "maintenance_priority_board.csv")
        self.health_predictions = self._safe_read_csv(self.output_dir / "stand_health_predictions.csv")
        self.feature_importance = self._safe_read_csv(self.output_dir / "anomaly_feature_importance.csv")
        self.drift_psi = self._safe_read_csv(self.output_dir / "drift_psi_train_vs_dataset5.csv")
        self.metrics_summary = self._safe_json(self.output_dir / "metrics_summary.json")
        self._loaded = True
        return self

    @staticmethod
    def _safe_read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    @staticmethod
    def _safe_json(path: Path) -> dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def set_demo_override(self, row_index: int, row: pd.Series | dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        self.ensure_loaded()
        idx = self._get_row_index(row_index=row_index, strategy="latest")
        if isinstance(row, dict):
            row = pd.Series(row)
        base = self.features_df.loc[idx].copy()
        for col, val in row.items():
            if col in base.index:
                base[col] = val
        self.demo_override_enabled = True
        self.demo_override_index = int(idx)
        self.demo_override_row = base
        self.demo_override_meta = dict(meta or {})

    def clear_demo_override(self) -> None:
        self.demo_override_enabled = False
        self.demo_override_index = None
        self.demo_override_row = None
        self.demo_override_meta = {}

    def get_demo_override(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.demo_override_enabled),
            "row_index": self.demo_override_index,
            "meta": self.demo_override_meta,
        }

    def _row_with_demo_override(self, idx: int) -> pd.Series:
        if (
            self.demo_override_enabled
            and self.demo_override_index is not None
            and int(idx) == int(self.demo_override_index)
            and self.demo_override_row is not None
        ):
            return self.demo_override_row.copy()
        return self.features_df.loc[idx].copy()

    def _row_frame_with_demo_override(self, idx: int) -> pd.DataFrame:
        row = self._row_with_demo_override(idx)
        data = {col: row.get(col, np.nan) for col in self.feature_cols}
        return pd.DataFrame([data], index=[idx])

    def _apply_demo_result_overlay(self, result: dict[str, Any]) -> dict[str, Any]:
        if not (self.demo_override_enabled and self.demo_override_index is not None):
            return result
        if int(result.get("row_index", -1)) != int(self.demo_override_index):
            return result
        meta = self.demo_override_meta or {}
        if not meta:
            return result

        forced_fault = meta.get("forced_fault")
        forced_stand = meta.get("forced_stand")
        forced_asset = meta.get("asset_name")
        risk_score = float(meta.get("forced_risk_score", result.get("risk_score", 0)))
        anom_prob = float(meta.get("forced_anomaly_probability", result.get("anomaly_probability", 0)))
        fault_conf = float(meta.get("forced_fault_confidence", result.get("fault_confidence", 0)))
        is_alert = bool(meta.get("forced_is_alert", anom_prob >= self.anomaly_threshold))

        result.update({
            "demo_mode": True,
            "demo_scenario": meta.get("scenario"),
            "demo_scenario_label": meta.get("label"),
            "demo_scenario_description": meta.get("description"),
            "is_alert": is_alert,
            "anomaly_probability": round(anom_prob, 4),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level_from_score(risk_score),
            "health_index": round(100 - risk_score, 2),
            "predicted_rul_band": meta.get("forced_rul_band", result.get("predicted_rul_band")),
            "fault_confidence": round(fault_conf, 4),
            "scenario_note": "Demo scenario override applied. The base TCM row is unchanged; only the in-memory demo context was perturbed.",
        })
        if forced_fault:
            result["predicted_fault"] = forced_fault
        if forced_stand:
            result["predicted_stand"] = forced_stand
            result["asset_name"] = forced_asset or ("TCM Mill Level" if forced_stand == "mill_level" else f"TCM Stand {str(forced_stand)[-1]}")
        if meta.get("evidence"):
            result["evidence"] = list(meta["evidence"]) + [e for e in result.get("evidence", []) if e not in meta["evidence"]][:3]
        if meta.get("secondary_faults"):
            result["secondary_faults"] = meta["secondary_faults"]
        return result

    @staticmethod
    def _multioutput_probs(prob_list: list[np.ndarray]) -> np.ndarray:
        cols = []
        for p in prob_list:
            if p.shape[1] == 1:
                cols.append(np.zeros(p.shape[0]))
            else:
                cols.append(p[:, 1])
        return np.vstack(cols).T

    def _get_row_index(self, row_index: int | None = None, strategy: str = "latest") -> int:
        self.ensure_loaded()
        if row_index is not None:
            if 0 <= int(row_index) < len(self.features_df):
                return int(row_index)
            matches = self.features_df.index[self.features_df.get("timestamp_index", pd.Series(dtype=int)) == int(row_index)].tolist()
            if matches:
                return int(matches[0])
            raise ValueError(f"row_index/timestamp_index {row_index} not found in processed feature data.")

        if strategy == "highest_risk" and not self.health_predictions.empty and "risk_score" in self.health_predictions.columns:
            ts = int(self.health_predictions.sort_values("risk_score", ascending=False).iloc[0]["timestamp_index"])
            matches = self.features_df.index[self.features_df["timestamp_index"] == ts].tolist()
            if matches:
                return int(matches[0])

        return int(self.features_df.index[-1])

    def find_row_from_query(self, query: str) -> int | None:
        m = re.search(r"(?:row|index|timestamp|alarm|alm)\s*[:#-]?\s*(\d+)", query.lower())
        return int(m.group(1)) if m else None

    def row_by_index(self, row_index: int | None = None, strategy: str = "latest") -> pd.Series:
        idx = self._get_row_index(row_index=row_index, strategy=strategy)
        return self._row_with_demo_override(idx)

    def evidence_for_row(self, row: pd.Series, top_n: int = 8) -> list[str]:
        z_cols = [c for c in self.feature_cols if c.endswith("z_recent_30") and c in row.index]
        evidence = []
        for zc in z_cols:
            val = float(row.get(zc, 0))
            if np.isfinite(val):
                signal = zc.replace("_z_recent_30", "")
                evidence.append((signal, val, abs(val)))
        evidence = sorted(evidence, key=lambda x: x[2], reverse=True)[:top_n]
        return [f"{signal}: recent z-score {z:.2f}" for signal, z, _ in evidence]

    def compute_trend_risk(self, row: pd.Series) -> float:
        z_cols = [c for c in self.feature_cols if c.endswith("z_recent_30") and c in row.index]
        if not z_cols:
            return 0.0
        vals = np.abs(row[z_cols].astype(float).values)
        return float(np.clip(np.nanpercentile(vals, 90) / 5.0, 0, 1))

    def similar_cases(self, row_X: pd.DataFrame, top_k: int = 5) -> list[dict[str, Any]]:
        if self.case_pre is None or self.case_nn is None or self.case_meta.empty:
            return []
        try:
            row_scaled = self.case_pre.transform(row_X[self.feature_cols])
            dist, ind = self.case_nn.kneighbors(row_scaled, n_neighbors=min(top_k, len(self.case_meta)))
            rows = []
            for d, i in zip(dist[0], ind[0]):
                meta = self.case_meta.iloc[int(i)].to_dict()
                meta["similarity"] = round(float(1 - d), 4)
                rows.append(meta)
            return rows
        except Exception as exc:
            return [{"error": f"similar case lookup failed: {exc}"}]

    def predict_condition(self, row_index: int | None = None, strategy: str = "latest") -> dict[str, Any]:
        self.ensure_loaded()
        idx = self._get_row_index(row_index=row_index, strategy=strategy)
        row = self._row_with_demo_override(idx)
        row_X = self._row_frame_with_demo_override(idx)

        if hasattr(self.anomaly_model, "predict_proba"):
            anom_prob = float(self.anomaly_model.predict_proba(row_X)[:, 1][0])
        else:
            raw = self.anomaly_model.decision_function(row_X)
            anom_prob = float(1 / (1 + np.exp(-raw[0])))
        is_alert = bool(anom_prob >= self.anomaly_threshold)

        if hasattr(self.fault_model, "named_steps") and "model" in self.fault_model.named_steps:
            transformed = self.fault_model.named_steps["imputer"].transform(row_X)
            prob_list = self.fault_model.named_steps["model"].predict_proba(transformed)
        else:
            prob_list = self.fault_model.predict_proba(row_X)
        fault_probs = self._multioutput_probs(prob_list)[0]
        top_faults = sorted(zip(self.anomaly_cols, fault_probs), key=lambda x: x[1], reverse=True)[:3]
        top_fault, top_fault_conf = top_faults[0]

        rul_id = int(self.rul_clf.predict(row_X)[0])
        rul_band = self.rul_band_names.get(rul_id, str(rul_id))
        rul_obs = float(self.rul_reg.predict(row_X)[0])

        trend_risk = self.compute_trend_risk(row)
        risk_score = float(np.clip(100 * (
            0.35 * anom_prob
            + 0.25 * (rul_id / 4.0)
            + 0.15 * float(top_fault_conf)
            + 0.15 * trend_risk
            + 0.10 * 0.80
        ), 0, 100))

        predicted_stand = infer_stand_from_fault_label(top_fault)
        result = {
            "alarm_id": f"ALM-{int(row.get('timestamp_index', idx)):06d}",
            "timestamp_index": int(row.get("timestamp_index", idx)),
            "row_index": int(idx),
            "is_alert": is_alert,
            "anomaly_probability": round(anom_prob, 4),
            "anomaly_threshold": round(self.anomaly_threshold, 4),
            "predicted_fault": top_fault,
            "predicted_stand": predicted_stand,
            "asset_name": "TCM Mill Level" if predicted_stand == "mill_level" else f"TCM Stand {predicted_stand[-1]}",
            "fault_confidence": round(float(top_fault_conf), 4),
            "secondary_faults": [
                {"fault": f, "confidence": round(float(p), 4), "stand": infer_stand_from_fault_label(f)}
                for f, p in top_faults[1:]
            ],
            "predicted_rul_band": rul_band,
            "proxy_rul_observations": round(rul_obs, 2),
            "proxy_rul_shifts": round(rul_obs / max(1, self.obs_per_shift), 2),
            "trend_risk": round(trend_risk, 4),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level_from_score(risk_score),
            "health_index": round(100 - risk_score, 2),
            "evidence": self.evidence_for_row(row, top_n=8),
            "similar_historical_cases": self.similar_cases(row_X, top_k=5),
            "true_is_anomaly": int(row.get("is_anomaly", -1)) if "is_anomaly" in row.index else None,
            "true_dominant_fault": str(row.get("dominant_fault", "unknown")),
            "true_rul_band": str(row.get("rul_urgency_band", "unknown")),
            "disclaimer": "Proxy RUL is an urgency estimate derived from anomaly progression, not true run-to-failure life.",
        }
        result = self._apply_demo_result_overlay(result)
        return result

    def _stand_signal_z(self, row: pd.Series, stand: int) -> dict[str, float]:
        signals = ["torque", "force", "motor_power", "roll_speed", "gap", "reduction"]
        out = {}
        for sig in signals:
            z_col = f"{sig}_{stand}_z_recent_30"
            if z_col in row.index:
                out[f"{sig}_{stand}"] = float(row.get(z_col, 0.0))
        return out

    def plant_topology(self, row_index: int | None = None, strategy: str = "latest") -> list[dict[str, Any]]:
        self.ensure_loaded()
        idx = self._get_row_index(row_index=row_index, strategy=strategy)
        row = self._row_with_demo_override(idx)
        ml = self.predict_condition(row_index=idx)
        cards = []
        for s in range(1, 6):
            zmap = self._stand_signal_z(row, s)
            top_signal, top_z = ("none", 0.0)
            if zmap:
                top_signal, top_z = max(zmap.items(), key=lambda kv: abs(kv[1]))
            mileage = float(row.get(f"mileage_norm_{s}", 0.0))
            base_signal_risk = min(1.0, abs(top_z) / 5.0)
            predicted_here = ml.get("predicted_stand") == f"stand_{s}"
            anomaly_boost = float(ml.get("anomaly_probability", 0.0)) if predicted_here else 0.0
            stand_risk = float(np.clip(100 * (0.55 * base_signal_risk + 0.25 * mileage + 0.20 * anomaly_boost), 0, 100))
            cards.append({
                "stand": f"stand_{s}",
                "asset": f"TCM Stand {s}",
                "timestamp_index": int(row.get("timestamp_index", idx)),
                "health_index": round(100 - stand_risk, 1),
                "risk_score": round(stand_risk, 1),
                "risk_level": risk_level_from_score(stand_risk),
                "active_alarm": bool(stand_risk >= 56 or predicted_here and ml.get("is_alert")),
                "predicted_fault": ml.get("predicted_fault") if predicted_here else "no dominant stand fault",
                "anomaly_probability": ml.get("anomaly_probability") if predicted_here else None,
                "proxy_rul_band": ml.get("predicted_rul_band") if predicted_here else "monitor",
                "top_abnormal_signal": top_signal,
                "top_signal_z": round(float(top_z), 2),
                "work_roll_mileage_norm": round(mileage, 2),
            })
        return cards

    def telemetry_window(self, row_index: int | None = None, stand: int = 3, window: int = 250) -> pd.DataFrame:
        self.ensure_loaded()
        idx = self._get_row_index(row_index=row_index, strategy="latest")
        start = max(0, idx - int(window) + 1)
        cols = ["timestamp_index"]
        cols += [c for c in [
            f"torque_{stand}", f"force_{stand}", f"motor_power_{stand}", f"roll_speed_{stand}",
            f"gap_{stand}", f"reduction_{stand}", f"work_roll_mileage_{stand}",
            f"torque_to_force_ratio_{stand}", f"power_to_torque_ratio_{stand}", f"motor_load_index_{stand}",
        ] if c in self.features_df.columns]
        df = self.features_df.loc[start:idx, cols].copy()
        if (
            self.demo_override_enabled
            and self.demo_override_index is not None
            and int(self.demo_override_index) == int(idx)
            and self.demo_override_row is not None
        ):
            for col in cols:
                if col in self.demo_override_row.index and idx in df.index:
                    df.loc[idx, col] = self.demo_override_row[col]
        return df.reset_index(drop=True)

    def active_alarms(self, top_n: int = 20) -> pd.DataFrame:
        self.ensure_loaded()
        if self.health_predictions.empty:
            res = self.predict_condition(strategy="highest_risk")
            return pd.DataFrame([res])
        df = self.health_predictions.copy()
        if "risk_score" in df.columns:
            df = df.sort_values(["risk_score", "anomaly_probability"], ascending=False)
        df = df.head(top_n).copy()
        if "timestamp_index" in df.columns:
            df["alarm_id"] = df["timestamp_index"].apply(lambda x: f"ALM-{int(x):06d}")
        if "abnormal_stand" in df.columns:
            df["asset"] = df["abnormal_stand"].apply(lambda s: "TCM Mill Level" if str(s) == "mill_level" else f"TCM Stand {str(s)[-1]}")
        if self.demo_override_enabled and self.demo_override_index is not None:
            demo = self.predict_condition(row_index=int(self.demo_override_index))
            demo_row = {
                "alarm_id": demo.get("alarm_id"),
                "timestamp_index": demo.get("timestamp_index"),
                "asset": demo.get("asset_name"),
                "abnormal_stand": demo.get("predicted_stand"),
                "top_fault": demo.get("predicted_fault"),
                "latest_fault": demo.get("predicted_fault"),
                "risk_level": demo.get("risk_level"),
                "risk_score": demo.get("risk_score"),
                "anomaly_probability": demo.get("anomaly_probability"),
                "predicted_rul_band": demo.get("predicted_rul_band"),
                "latest_rul_band": demo.get("predicted_rul_band"),
                "health_index": demo.get("health_index"),
                "demo_scenario": demo.get("demo_scenario_label"),
            }
            df = pd.concat([pd.DataFrame([demo_row]), df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp_index"], keep="first")
        return df.reset_index(drop=True)


    def alarm_events(self, top_n: int = 25, min_risk: str = "high", gap: int = 8) -> pd.DataFrame:
        self.ensure_loaded()
        df = self.active_alarms(top_n=max(200, top_n * 20)).copy()
        if df.empty:
            return df
        risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_rank = risk_order.get(str(min_risk).lower(), 2)
        if "risk_level" in df.columns:
            df = df[df["risk_level"].astype(str).str.lower().map(risk_order).fillna(0) >= min_rank]
        if df.empty:
            return df
        if "timestamp_index" not in df.columns:
            return df.head(top_n)
        if "abnormal_stand" not in df.columns and "predicted_stand" in df.columns:
            df["abnormal_stand"] = df["predicted_stand"]
        if "top_fault" not in df.columns:
            df["top_fault"] = df.get("latest_fault", "unknown")
        df = df.sort_values(["abnormal_stand", "top_fault", "timestamp_index"]).reset_index(drop=True)
        events = []
        for (_, _), g in df.groupby([df.get("abnormal_stand", "unknown"), df.get("top_fault", "unknown")], dropna=False):
            g = g.sort_values("timestamp_index")
            current = []
            last_ts = None
            for _, row in g.iterrows():
                ts = int(row["timestamp_index"])
                if not current or (last_ts is not None and ts - last_ts <= gap):
                    current.append(row)
                else:
                    events.append(self._summarize_alarm_event(current))
                    current = [row]
                last_ts = ts
            if current:
                events.append(self._summarize_alarm_event(current))
        out = pd.DataFrame(events)
        if out.empty:
            return out
        out = out.sort_values(["max_risk_score", "max_anomaly_probability", "event_count"], ascending=False).head(top_n)
        out.insert(0, "priority_rank", range(1, len(out) + 1))
        return out.reset_index(drop=True)

    def _summarize_alarm_event(self, rows: list[pd.Series]) -> dict[str, Any]:
        g = pd.DataFrame(rows)
        idxmax = g["risk_score"].astype(float).idxmax() if "risk_score" in g.columns else g.index[-1]
        peak = g.loc[idxmax]
        ts_peak = int(peak.get("timestamp_index", idxmax))
        stand = str(peak.get("abnormal_stand", peak.get("predicted_stand", "mill_level")))
        fault = str(peak.get("top_fault", peak.get("latest_fault", peak.get("predicted_fault", "unknown"))))
        asset = str(peak.get("asset", "TCM Mill Level" if stand == "mill_level" else f"TCM Stand {stand[-1]}"))
        row = self.row_by_index(ts_peak)
        evidence = self.evidence_for_row(row, top_n=3)
        top_signal = evidence[0].split(":")[0] if evidence else "not available"
        return {
            "alarm_id": f"ALM-{ts_peak:06d}",
            "event_start": int(g["timestamp_index"].min()),
            "event_end": int(g["timestamp_index"].max()),
            "timestamp_index": ts_peak,
            "asset": asset,
            "abnormal_stand": stand,
            "fault": fault,
            "risk_level": str(peak.get("risk_level", risk_level_from_score(float(peak.get("risk_score", 0))))),
            "max_risk_score": round(float(g.get("risk_score", pd.Series([0])).astype(float).max()), 2),
            "mean_risk_score": round(float(g.get("risk_score", pd.Series([0])).astype(float).mean()), 2),
            "max_anomaly_probability": round(float(g.get("anomaly_probability", pd.Series([0])).astype(float).max()), 4),
            "rul_band": str(peak.get("predicted_rul_band", peak.get("latest_rul_band", "unknown"))),
            "event_count": int(len(g)),
            "top_evidence_signal": top_signal,
            "recommended_action": self.default_action_for_fault(fault, str(peak.get("risk_level", ""))),
            "status": "open",
            "evidence": evidence,
        }

    @staticmethod
    def default_action_for_fault(fault: str, risk_level: str = "") -> str:
        f = str(fault).lower()
        risk = str(risk_level).lower()
        prefix = "Inspect immediately" if risk == "critical" else "Schedule focused inspection"
        if "bearing" in f:
            return f"{prefix}: verify bearing lubrication, housing temperature, vibration/noise, and coupling alignment."
        if "electric" in f:
            return f"{prefix}: check drive alarms, current imbalance, cooling, and power-to-torque behaviour."
        if "workroll" in f:
            return f"{prefix}: check emulsion/lubrication, roll surface condition, cooling nozzles, and roll mileage."
        if "reduction" in f:
            return f"{prefix}: verify reduction schedule, roll gap calibration, force distribution, and inter-stand tension."
        return f"{prefix}: review telemetry evidence, physical rules, SOP guidance, and adjacent stand impact."

    def maintenance_queue(self, top_n: int = 8) -> pd.DataFrame:
        events = self.alarm_events(top_n=top_n, min_risk="medium")
        if events.empty:
            return events
        cols = [
            "priority_rank", "alarm_id", "asset", "fault", "risk_level", "max_risk_score",
            "rul_band", "event_count", "top_evidence_signal", "recommended_action", "status", "timestamp_index",
        ]
        return events[[c for c in cols if c in events.columns]].copy()

    def telemetry_window_normalized(self, row_index: int | None = None, stand: int = 3, window: int = 250, mode: str = "zscore") -> pd.DataFrame:
        """Return telemetry window normalized for charting mixed-scale signals."""
        df = self.telemetry_window(row_index=row_index, stand=stand, window=window)
        if df.empty:
            return df
        out = pd.DataFrame({"timestamp_index": df["timestamp_index"]})
        signal_cols = [c for c in df.columns if c != "timestamp_index"]
        for c in signal_cols:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            if mode == "pct_change":
                base = s.dropna().iloc[0] if not s.dropna().empty else 0.0
                out[c] = 100 * (s - base) / (abs(base) + 1e-9)
            elif mode == "minmax":
                out[c] = (s - s.min()) / (s.max() - s.min() + 1e-9)
            else:
                out[c] = (s - s.mean()) / (s.std() + 1e-9)
        return out

    def telemetry_groups(self, row_index: int | None = None, stand: int = 3, window: int = 250) -> dict[str, pd.DataFrame]:
        df = self.telemetry_window(row_index=row_index, stand=stand, window=window)
        groups = {
            "Mechanical load": [f"torque_{stand}", f"force_{stand}"],
            "Electrical load": [f"motor_power_{stand}", f"power_to_torque_ratio_{stand}"],
            "Process setup": [f"gap_{stand}", f"reduction_{stand}"],
            "Wear / derived load": [f"work_roll_mileage_{stand}", f"torque_to_force_ratio_{stand}", f"motor_load_index_{stand}"],
        }
        out = {}
        for name, cols in groups.items():
            keep = ["timestamp_index"] + [c for c in cols if c in df.columns]
            if len(keep) > 1:
                out[name] = df[keep].copy()
        return out

    def priority_table(self, top_n: int = 10) -> list[dict[str, Any]]:
        self.ensure_loaded()
        if self.priority_board.empty:
            return []
        return self.priority_board.head(top_n).to_dict(orient="records")

    def drift_summary(self, top_n: int = 10) -> list[dict[str, Any]]:
        self.ensure_loaded()
        if self.drift_psi.empty:
            return []
        return self.drift_psi.head(top_n).to_dict(orient="records")

    def metrics(self) -> dict[str, Any]:
        self.ensure_loaded()
        return self.metrics_summary


ml_service = MLModelService()
