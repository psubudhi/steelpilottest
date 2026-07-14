from __future__ import annotations

from typing import Any


def build_decision_trace(state: dict[str, Any]) -> list[dict[str, Any]]:
    ml = state.get("ml_result", {}) or {}
    rules = state.get("physical_rules", []) or []
    cascade = state.get("cascading_impact", {}) or {}
    rag = state.get("rag_context", []) or []
    plan = state.get("maintenance_plan", {}) or {}
    root = state.get("root_cause", {}) or {}
    spares = state.get("spares", []) or []

    trace: list[dict[str, Any]] = []
    trace.append({
        "step": 1,
        "agent": "Query Router",
        "decision": "Classified the request and selected the sensor + RAG maintenance workflow.",
        "evidence": state.get("route", {}),
    })
    trace.append({
        "step": 2,
        "agent": "ML Sensor Intelligence Agent",
        "decision": f"Predicted {ml.get('predicted_fault', 'unknown')} on {ml.get('predicted_stand', 'unknown')} with risk {ml.get('risk_level', 'unknown')}.",
        "evidence": {
            "anomaly_probability": ml.get("anomaly_probability"),
            "fault_confidence": ml.get("fault_confidence"),
            "proxy_rul_band": ml.get("predicted_rul_band"),
            "top_signals": ml.get("evidence", [])[:5],
        },
    })
    matched_rules = [r for r in rules if r.get("matched")]
    trace.append({
        "step": 3,
        "agent": "Physical Constraint Rule Engine",
        "decision": f"Validated the ML output with {len(matched_rules)} matched rolling-mill engineering rule(s).",
        "evidence": [{"rule": r.get("title"), "severity": r.get("severity"), "evidence": r.get("evidence", [])[:3]} for r in matched_rules[:4]],
    })
    trace.append({
        "step": 4,
        "agent": "Cascading Impact Analyst",
        "decision": f"Estimated cascading risk as {cascade.get('cascading_risk', 'unknown')}.",
        "evidence": cascade,
    })
    trace.append({
        "step": 5,
        "agent": "Knowledge Retrieval Agent",
        "decision": f"Retrieved {len(rag)} SOP/manual chunks from FAISS.",
        "evidence": [{"source": d.get("source"), "score": d.get("score")} for d in rag[:5]],
    })
    trace.append({
        "step": 6,
        "agent": "Root Cause Agent",
        "decision": "Combined ML evidence, rules, similar cases, and SOP context into probable root causes.",
        "evidence": root,
    })
    trace.append({
        "step": 7,
        "agent": "Maintenance Planner Agent",
        "decision": "Generated immediate, short-term, planned, spare, monitoring, and safety actions.",
        "evidence": {"plan": plan, "spares": spares},
    })
    trace.append({
        "step": 8,
        "agent": "Report Agent",
        "decision": "Produced a traceable maintenance answer and saved the session to the digital logbook.",
        "evidence": {"audit": state.get("audit_event", {})},
    })
    return trace
