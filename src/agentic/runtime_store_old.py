from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_loads(text: str | None, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _shorten(value: Any, max_len: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = _json_dumps(value)
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


class SteelPilotRuntimeStore:
    

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.sqlite_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS logbook_events (
                    log_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    alarm_id TEXT,
                    row_index INTEGER,
                    asset TEXT,
                    fault TEXT,
                    risk_level TEXT,
                    risk_score REAL,
                    anomaly_probability REAL,
                    rul_band TEXT,
                    status TEXT,
                    recommended_action TEXT,
                    query TEXT,
                    final_answer TEXT,
                    decision_trace_json TEXT,
                    physical_rules_json TEXT,
                    cascading_impact_json TEXT,
                    rag_sources_json TEXT,
                    raw_event_json TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    feedback_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    alarm_id TEXT,
                    row_index INTEGER,
                    asset TEXT,
                    predicted_fault TEXT,
                    risk_level TEXT,
                    anomaly_probability REAL,
                    actual_fault TEXT,
                    action_taken TEXT,
                    outcome TEXT,
                    status TEXT,
                    engineer_name TEXT,
                    notes TEXT,
                    source TEXT,
                    raw_feedback_json TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_health_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    selected_test_ids TEXT,
                    average_score REAL,
                    passed INTEGER,
                    failed INTEGER,
                    result_json TEXT
                )
                """
            )
            con.commit()

    def save_logbook_event(self, event: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        log_id = event.get("log_id") or f"LOG-{uuid.uuid4().hex[:10].upper()}"
        decision_trace = event.get("decision_trace", [])
        physical_rules = event.get("physical_rules", [])
        cascading = event.get("cascading_impact", {})
        rag_sources = event.get("rag_sources", event.get("rag_context", []))
        row_index = event.get("row_index")
        try:
            row_index = None if row_index in [None, ""] else int(row_index)
        except Exception:
            row_index = None
        record = {
            "log_id": log_id,
            "created_at": event.get("created_at") or now,
            "updated_at": now,
            "alarm_id": event.get("alarm_id"),
            "row_index": row_index,
            "asset": event.get("asset") or event.get("asset_name"),
            "fault": event.get("fault") or event.get("predicted_fault"),
            "risk_level": event.get("risk_level"),
            "risk_score": event.get("risk_score"),
            "anomaly_probability": event.get("anomaly_probability"),
            "rul_band": event.get("rul_band") or event.get("predicted_rul_band"),
            "status": event.get("status") or "open",
            "recommended_action": event.get("recommended_action") or event.get("action") or "Review Steel Pilot recommendation.",
            "query": event.get("query"),
            "final_answer": event.get("final_answer"),
            "decision_trace_json": _json_dumps(decision_trace),
            "physical_rules_json": _json_dumps(physical_rules),
            "cascading_impact_json": _json_dumps(cascading),
            "rag_sources_json": _json_dumps(rag_sources),
            "raw_event_json": _json_dumps(event),
        }
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO logbook_events (
                    log_id, created_at, updated_at, alarm_id, row_index, asset, fault,
                    risk_level, risk_score, anomaly_probability, rul_band, status,
                    recommended_action, query, final_answer, decision_trace_json,
                    physical_rules_json, cascading_impact_json, rag_sources_json, raw_event_json
                ) VALUES (
                    :log_id, :created_at, :updated_at, :alarm_id, :row_index, :asset, :fault,
                    :risk_level, :risk_score, :anomaly_probability, :rul_band, :status,
                    :recommended_action, :query, :final_answer, :decision_trace_json,
                    :physical_rules_json, :cascading_impact_json, :rag_sources_json, :raw_event_json
                )
                """,
                record,
            )
            con.commit()
        return {**record, **event, "log_id": log_id, "created_at": record["created_at"], "updated_at": now}

    def update_log_status(self, log_id: str, status: str) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE logbook_events SET status = ?, updated_at = ? WHERE log_id = ?",
                (status, utc_now(), log_id),
            )
            con.commit()

    def load_logbook_events(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM logbook_events ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_logbook_event(self, log_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM logbook_events WHERE log_id = ?", (log_id,)).fetchone()
        if not row:
            return None
        rec = dict(row)
        rec["decision_trace"] = _json_loads(rec.get("decision_trace_json"), [])
        rec["physical_rules"] = _json_loads(rec.get("physical_rules_json"), [])
        rec["cascading_impact"] = _json_loads(rec.get("cascading_impact_json"), {})
        rec["rag_sources"] = _json_loads(rec.get("rag_sources_json"), [])
        rec["raw_event"] = _json_loads(rec.get("raw_event_json"), {})
        return rec

    def logbook_display_df(self, limit: int = 200) -> pd.DataFrame:
        rows = self.load_logbook_events(limit=limit)
        display_rows: list[dict[str, Any]] = []
        for r in rows:
            trace = _json_loads(r.get("decision_trace_json"), []) or []
            rules = _json_loads(r.get("physical_rules_json"), []) or []
            rag_sources = _json_loads(r.get("rag_sources_json"), []) or []
            matched_rules = 0
            if isinstance(rules, list):
                matched_rules = sum(1 for x in rules if isinstance(x, dict) and x.get("matched"))
            sources = []
            if isinstance(rag_sources, list):
                for src in rag_sources:
                    if isinstance(src, dict):
                        sources.append(str(src.get("source", "")))
                    else:
                        sources.append(str(src))
            display_rows.append(
                {
                    "log_id": str(r.get("log_id") or ""),
                    "created_at": str(r.get("created_at") or ""),
                    "updated_at": str(r.get("updated_at") or ""),
                    "alarm_id": str(r.get("alarm_id") or ""),
                    "row_index": r.get("row_index"),
                    "asset": str(r.get("asset") or ""),
                    "fault": str(r.get("fault") or ""),
                    "risk_level": str(r.get("risk_level") or ""),
                    "risk_score": r.get("risk_score"),
                    "status": str(r.get("status") or ""),
                    "short_recommendation": _shorten(r.get("recommended_action"), 140),
                    "trace_steps": len(trace) if isinstance(trace, list) else 0,
                    "rules_triggered": matched_rules,
                    "sop_sources": len([s for s in sources if s]),
                }
            )
        df = pd.DataFrame(display_rows)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
        return df

    def summarize_logbook(self) -> dict[str, Any]:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM logbook_events").fetchone()[0]
            open_count = con.execute("SELECT COUNT(*) FROM logbook_events WHERE LOWER(COALESCE(status,'')) IN ('open','acknowledged','assigned','work_order_created')").fetchone()[0]
            critical = con.execute("SELECT COUNT(*) FROM logbook_events WHERE LOWER(COALESCE(risk_level,'')) = 'critical'").fetchone()[0]
            latest = con.execute("SELECT MAX(created_at) FROM logbook_events").fetchone()[0]
        return {"total_events": int(total), "open": int(open_count), "critical": int(critical), "latest_created_at": latest}

    def save_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        feedback_id = feedback.get("feedback_id") or f"FDB-{uuid.uuid4().hex[:10].upper()}"
        row_index = feedback.get("row_index")
        try:
            row_index = None if row_index in [None, ""] else int(row_index)
        except Exception:
            row_index = None
        record = {
            "feedback_id": feedback_id,
            "created_at": feedback.get("created_at") or now,
            "updated_at": now,
            "alarm_id": feedback.get("alarm_id"),
            "row_index": row_index,
            "asset": feedback.get("asset") or feedback.get("asset_name"),
            "predicted_fault": feedback.get("predicted_fault"),
            "risk_level": feedback.get("risk_level"),
            "anomaly_probability": feedback.get("anomaly_probability"),
            "actual_fault": feedback.get("actual_fault"),
            "action_taken": feedback.get("action_taken"),
            "outcome": feedback.get("outcome"),
            "status": feedback.get("status") or "feedback_recorded",
            "engineer_name": feedback.get("engineer_name"),
            "notes": feedback.get("notes"),
            "source": feedback.get("source") or "streamlit",
            "raw_feedback_json": _json_dumps(feedback),
        }
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO feedback_events (
                    feedback_id, created_at, updated_at, alarm_id, row_index, asset,
                    predicted_fault, risk_level, anomaly_probability, actual_fault,
                    action_taken, outcome, status, engineer_name, notes, source, raw_feedback_json
                ) VALUES (
                    :feedback_id, :created_at, :updated_at, :alarm_id, :row_index, :asset,
                    :predicted_fault, :risk_level, :anomaly_probability, :actual_fault,
                    :action_taken, :outcome, :status, :engineer_name, :notes, :source, :raw_feedback_json
                )
                """,
                record,
            )
            con.commit()
        return {**record, **feedback, "feedback_id": feedback_id, "created_at": record["created_at"], "updated_at": now}

    def feedback_display_df(self, limit: int = 200) -> pd.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM feedback_events ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        data = []
        for r in rows:
            d = dict(r)
            data.append(
                {
                    "feedback_id": str(d.get("feedback_id") or ""),
                    "created_at": str(d.get("created_at") or ""),
                    "updated_at": str(d.get("updated_at") or ""),
                    "alarm_id": str(d.get("alarm_id") or ""),
                    "row_index": d.get("row_index"),
                    "asset": str(d.get("asset") or ""),
                    "predicted_fault": str(d.get("predicted_fault") or ""),
                    "actual_fault": str(d.get("actual_fault") or ""),
                    "status": str(d.get("status") or ""),
                    "action_taken": _shorten(d.get("action_taken"), 140),
                    "outcome": _shorten(d.get("outcome"), 140),
                }
            )
        df = pd.DataFrame(data)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
        return df

    def save_health_run(self, selected_ids: list[str], result_df: pd.DataFrame) -> dict[str, Any]:
        now = utc_now()
        run_id = f"EVAL-{uuid.uuid4().hex[:10].upper()}"
        avg = float(result_df["score"].mean()) if not result_df.empty and "score" in result_df else 0.0
        passed = int((result_df.get("score", pd.Series(dtype=float)) >= 70).sum()) if not result_df.empty else 0
        failed = int(len(result_df) - passed)
        payload = result_df.to_dict(orient="records") if not result_df.empty else []
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO agent_health_runs (run_id, created_at, selected_test_ids, average_score, passed, failed, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, now, _json_dumps(selected_ids), avg, passed, failed, _json_dumps(payload)),
            )
            con.commit()
        return {"run_id": run_id, "created_at": now, "average_score": avg, "passed": passed, "failed": failed, "results": payload}

    def health_runs_df(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT run_id, created_at, selected_test_ids, average_score, passed, failed FROM agent_health_runs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        data = [dict(r) for r in rows]
        df = pd.DataFrame(data)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
        return df


runtime_store = SteelPilotRuntimeStore()
