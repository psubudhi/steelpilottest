from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

from src.agentic.audit import digital_logbook
from src.agentic.cascading import analyze_cascading_impact
from src.agentic.demo_scenarios import DEMO_SCENARIOS, apply_demo_scenario
from src.agentic.eval_harness import GOLD_TESTS, run_gold_tests
from src.agentic.graph import answer_maintenance_query
from src.agentic.memory import feedback_memory
from src.agentic.ml_service import ml_service
from src.agentic.rag import rag
from src.agentic.rules import apply_physical_constraint_rules
from src.agentic.runtime_store import runtime_store

APP_NAME = "Steel Pilot V2"

st.set_page_config(page_title=f"{APP_NAME} — Maintenance Investigation Workspace", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    /* ------------------------------------------------------------------
       Steel Pilot V2 theme-safe styling
       Streamlit dark mode changes the default text color, but custom HTML
       cards do not automatically receive good foreground/background pairs.
       These variables use Streamlit theme variables when available so the
       same cards remain readable in both light and dark themes.
    ------------------------------------------------------------------ */
    :root {
        --steel-pilot-card-bg: var(--secondary-background-color, #ffffff);
        --steel-pilot-page-bg: var(--background-color, #ffffff);
        --steel-pilot-text: var(--text-color, #0f172a);
        --steel-pilot-muted: var(--text-color, #475569);
        --steel-pilot-border: rgba(148, 163, 184, 0.42);
        --steel-pilot-shadow: rgba(15, 23, 42, 0.10);
        --steel-pilot-soft-blue: rgba(99, 102, 241, 0.10);
        --steel-pilot-soft-red: rgba(239, 68, 68, 0.10);
        --steel-pilot-soft-green: rgba(34, 197, 94, 0.10);
    }

    /* Universal text helpers */
    .main-title {
        color: var(--steel-pilot-text) !important;
        font-size: 2.0rem;
        font-weight: 850;
        margin-bottom: 0.1rem;
    }
    .subtitle, .small-muted {
        color: var(--steel-pilot-muted) !important;
        opacity: 0.78;
    }
    .subtitle {margin-bottom: 1rem;}
    .small-muted {font-size: 0.88rem;}

    /* Theme-safe cards used across Plant Command Center, RCA, Copilot, and Evidence sections */
    .card,
    .stand-card,
    .alarm-box,
    .ok-box,
    .evidence-step {
        background: var(--steel-pilot-card-bg) !important;
        color: var(--steel-pilot-text) !important;
        border: 1px solid var(--steel-pilot-border) !important;
        border-radius: 16px;
        box-shadow: 0 1px 8px var(--steel-pilot-shadow);
    }
    .card *,
    .stand-card *,
    .alarm-box *,
    .ok-box *,
    .evidence-step * {
        color: inherit !important;
    }
    .card .small-muted,
    .stand-card .small-muted,
    .alarm-box .small-muted,
    .ok-box .small-muted,
    .evidence-step .small-muted {
        color: var(--steel-pilot-muted) !important;
        opacity: 0.78;
    }

    .card {
        padding: 14px;
        margin-bottom: 10px;
    }
    .stand-card {
        padding: 14px;
        min-height: 225px;
    }
    .stand-title {
        color: var(--steel-pilot-text) !important;
        font-weight: 800;
        font-size: 1.02rem;
        margin-bottom: 0.3rem;
    }

    /* Risk side bars keep strong colors but retain theme-safe text/background */
    .risk-low {border-left: 8px solid #22c55e !important;}
    .risk-medium {border-left: 8px solid #f59e0b !important;}
    .risk-high {border-left: 8px solid #ef4444 !important;}
    .risk-critical {border-left: 8px solid #7f1d1d !important;}

    /* Context cards */
    .alarm-box {
        border-color: rgba(239, 68, 68, 0.60) !important;
        border-left: 8px solid #ef4444 !important;
        padding: 12px;
        border-radius: 12px;
        background: linear-gradient(0deg, var(--steel-pilot-soft-red), var(--steel-pilot-soft-red)), var(--steel-pilot-card-bg) !important;
    }
    .ok-box {
        border-color: rgba(34, 197, 94, 0.60) !important;
        border-left: 8px solid #22c55e !important;
        padding: 12px;
        border-radius: 12px;
        background: linear-gradient(0deg, var(--steel-pilot-soft-green), var(--steel-pilot-soft-green)), var(--steel-pilot-card-bg) !important;
    }

    /* Decision evidence and rule cards */
    .evidence-step {
        border-left: 4px solid #6366f1 !important;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 10px;
        background: linear-gradient(0deg, var(--steel-pilot-soft-blue), var(--steel-pilot-soft-blue)), var(--steel-pilot-card-bg) !important;
    }
    .rule-triggered {border-left: 6px solid #ef4444 !important;}
    .rule-muted {
        border-left: 6px solid #94a3b8 !important;
        opacity: 1 !important;
    }

    /* Make generated HTML tables/code snippets inside expanders readable as well */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: var(--steel-pilot-text);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="main-title">⚡ {APP_NAME} — Maintenance Investigation Workspace for Tandem Cold Mills</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Detect alarm → investigate RCA → explain evidence → recommend action → record feedback → monitor model health.</div>',
    unsafe_allow_html=True,
)

try:
    ml_service.ensure_loaded()
except Exception as exc:
    st.error(f"ML artifacts not ready: {exc}")
    st.info("Run the TCM ML/RUL notebook first and set TCM_MODELLING_ROOT in .env.")
    st.stop()

# ----------------------------- session state -----------------------------
for key, default in {
    "active_alarm_row": None,
    "active_alarm_id": None,
    "last_agent_result": None,
    "messages": [],
    "answer_mode": "Concise",
    "selected_alarm_event": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------- helper funcs -----------------------------
def risk_badge(level: str | None) -> str:
    level = str(level or "low").lower()
    return {"low": "🟢 Low", "medium": "🟠 Medium", "high": "🔴 High", "critical": "🛑 Critical"}.get(level, level)


def bind_alarm(row_index: int, alarm_id: str | None = None) -> None:
    st.session_state.active_alarm_row = int(row_index)
    st.session_state.active_alarm_id = alarm_id or f"ALM-{int(row_index):06d}"


def get_active_result() -> dict[str, Any] | None:
    if st.session_state.active_alarm_row is None:
        return None
    return ml_service.predict_condition(row_index=int(st.session_state.active_alarm_row))


def render_active_context() -> None:
    res = get_active_result()
    if not res:
        st.info("No active alarm bound. Select an alarm from **Alarm Investigation & RCA** or bind a row from telemetry replay.")
        return
    box_cls = "alarm-box" if res.get("is_alert") else "ok-box"
    demo = ""
    if res.get("demo_mode"):
        demo = f"<br><b>Demo scenario:</b> {res.get('demo_scenario_label')}"
    st.markdown(
        f"""
        <div class="{box_cls}">
        <b>Active Context:</b> {res.get('alarm_id')} · {res.get('asset_name')} · {risk_badge(res.get('risk_level'))}<br>
        <b>Fault:</b> {res.get('predicted_fault')} · <b>Anomaly Prob:</b> {res.get('anomaly_probability')} · <b>Proxy RUL:</b> {res.get('predicted_rul_band')}{demo}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(ml: dict[str, Any]) -> None:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Risk Level", risk_badge(ml.get("risk_level")))
    m2.metric("Risk Score", ml.get("risk_score"))
    m3.metric("Anomaly Prob.", ml.get("anomaly_probability"))
    m4.metric("Health Index", ml.get("health_index"))
    m5.metric("Proxy RUL", ml.get("predicted_rul_band"))


def render_rules(rules: list[dict[str, Any]]) -> None:
    if not rules:
        st.info("No physical-rule output available.")
        return
    for r in rules:
        matched = bool(r.get("matched"))
        css = "rule-triggered" if matched else "rule-muted"
        icon = "✅ Triggered" if matched else "⚪ Not triggered"
        st.markdown(
            f"""
            <div class="card {css}">
            <b>{r.get('title', r.get('rule_id', 'Rule'))}</b> — {icon}<br>
            <b>Severity:</b> {r.get('severity')} · <b>Score:</b> {r.get('severity_score')}<br>
            <span class="small-muted">{r.get('explanation', '')}</span><br>
            <b>Recommendation:</b> {r.get('recommendation', '')}
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Evidence for {r.get('rule_id', r.get('title', 'rule'))}"):
            st.json(r.get("evidence", {}))


def render_cascade(cascade: dict[str, Any]) -> None:
    if not cascade:
        st.info("No cascading impact output available.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Primary Stand", cascade.get("primary_stand", "unknown"))
    c2.metric("Cascading Risk", risk_badge(cascade.get("cascading_risk")))
    c3.metric("Risk Score", cascade.get("cascading_risk_score", 0))
    st.markdown("**Affected Neighbours / Checks**")
    neighbours = cascade.get("affected_neighbors") or []
    checks = cascade.get("recommended_checks") or []
    if neighbours:
        for n in neighbours:
            st.write(f"- {n}")
    if checks:
        st.markdown("**Recommended adjacent-stand checks**")
        for x in checks:
            st.write(f"- {x}")
    with st.expander("Full cascading impact JSON"):
        st.json(cascade)


def render_sop_sources(docs: list[dict[str, Any]]) -> None:
    if not docs:
        st.info("No SOP source retrieved. Run `python scripts/create_demo_docs.py` and `python scripts/ingest_faiss.py`, or use keyword fallback docs.")
        return
    for i, d in enumerate(docs, start=1):
        src = d.get("source", "unknown")
        mode = d.get("retrieval_mode", "faiss")
        score = d.get("score", "")
        with st.expander(f"SOP Source {i}: {src} · mode={mode} · score={score}"):
            st.write(d.get("content", ""))


def render_decision_evidence(result: dict[str, Any] | None) -> None:
    if not result:
        st.info("Run RCA or chat first to generate decision evidence.")
        return
    trace = result.get("decision_trace") or []
    if not trace:
        st.info("No decision trace available for this result.")
        return
    for step in trace:
        st.markdown(
            f"""
            <div class="evidence-step">
            <b>Step {step.get('step')}: {step.get('agent')}</b><br>
            {step.get('decision')}
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Evidence details — Step {step.get('step')}"):
            st.json(step.get("evidence", {}))


def render_line_chart(df: pd.DataFrame, y_cols: list[str], title: str, active_ts: int | None = None) -> None:
    y_cols = [c for c in y_cols if c in df.columns]
    if df.empty or not y_cols:
        st.info(f"No data available for {title}.")
        return
    st.markdown(f"**{title}**")
    if alt is not None:
        long = df[["timestamp_index"] + y_cols].melt("timestamp_index", var_name="signal", value_name="value")
        chart = alt.Chart(long).mark_line().encode(x="timestamp_index:Q", y="value:Q", color="signal:N", tooltip=["timestamp_index", "signal", "value"]).properties(height=280)
        if active_ts is not None:
            rule = alt.Chart(pd.DataFrame({"timestamp_index": [active_ts]})).mark_rule(strokeDash=[6, 4]).encode(x="timestamp_index:Q")
            chart = chart + rule
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(df.set_index("timestamp_index")[y_cols])
        if active_ts is not None:
            st.caption(f"Active alarm timestamp: {active_ts}")


def fault_label_to_title(fault: str) -> str:
    return str(fault or "unknown").replace("anomaly_", "").replace("_", " ").title()


# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.header("Steel Pilot V2 Navigation")
    page = st.radio(
        "Page",
        [
            "Plant Command Center",
            "Alarm Investigation & RCA",
            "Live Telemetry Replay",
            "Steel Pilot Maintenance Copilot",
            "Operations Logbook & Feedback",
            "Model Health & Evaluation",
        ],
    )
    st.divider()
    st.subheader("Demo Mode / Scenario Simulator")
    scenario_keys = list(DEMO_SCENARIOS.keys())
    scenario = st.selectbox("Scenario", scenario_keys, format_func=lambda k: DEMO_SCENARIOS[k]["label"])
    stand = st.selectbox("Affected stand", [1, 2, 3, 4, 5], index=2)
    severity = st.slider("Severity", 0.5, 2.0, 1.0, 0.1)
    max_idx = len(ml_service.features_df) - 1
    default_base = int(st.session_state.active_alarm_row) if st.session_state.active_alarm_row is not None else min(max_idx, 18792)
    base_idx = st.number_input("Base row/timestamp", min_value=0, max_value=max_idx, value=min(default_base, max_idx), step=1)
    c_apply, c_clear = st.columns(2)
    if c_apply.button("Apply", use_container_width=True):
        base_row = ml_service.row_by_index(int(base_idx))
        demo_row, meta = apply_demo_scenario(base_row, scenario=scenario, stand=int(stand), severity=float(severity))
        ml_service.set_demo_override(int(base_idx), demo_row, meta=meta)
        bind_alarm(int(base_idx), f"ALM-{int(base_idx):06d}")
        st.success("Scenario applied and bound as active alarm.")
        st.rerun()
    if c_clear.button("Clear", use_container_width=True):
        ml_service.clear_demo_override()
        st.success("Demo override cleared.")
        st.rerun()
    demo_state = ml_service.get_demo_override()
    if demo_state["enabled"]:
        st.info(f"Active demo: {demo_state['meta'].get('label')} @ row {demo_state['row_index']}")
    st.divider()
    st.subheader("Active Alarm Context")
    if st.session_state.active_alarm_row is not None:
        st.success(f"{st.session_state.active_alarm_id} / row {st.session_state.active_alarm_row}")
        if st.button("Clear active alarm"):
            st.session_state.active_alarm_row = None
            st.session_state.active_alarm_id = None
            st.rerun()
    else:
        st.info("None bound")
    thread_id = st.text_input("LangGraph Thread ID", value="steel-pilot-v2-demo")


# ----------------------------- pages -----------------------------
if page == "Plant Command Center":
    st.subheader("Plant Command Center")
    st.caption("Plant-level view of the 5-stand tandem cold mill plus top maintenance priorities. The priority board is integrated here as an operational queue, not a separate static page.")

    top = ml_service.predict_condition(strategy="highest_risk")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Highest Risk Alarm", top.get("alarm_id"))
    k2.metric("Highest Risk", f"{top.get('risk_score')} · {top.get('risk_level')}")
    k3.metric("Fault", fault_label_to_title(top.get("predicted_fault")))
    k4.metric("Logbook Events", digital_logbook.summarize().get("total_events", 0))

    st.markdown("### Live Plant Topology")
    strategy = st.selectbox("Topology row", ["highest_risk", "latest"], index=0)
    row_index = int(st.session_state.active_alarm_row) if st.session_state.active_alarm_row is not None else None
    topology = ml_service.plant_topology(row_index=row_index, strategy=strategy)
    st.markdown("**Entry Coil → Stand 1 → Stand 2 → Stand 3 → Stand 4 → Stand 5 → Exit Coil**")
    cols = st.columns(5)
    for i, card in enumerate(topology):
        level = str(card.get("risk_level", "low")).lower()
        with cols[i]:
            st.markdown(
                f"""
                <div class="stand-card risk-{level}">
                <div class="stand-title">{card['asset']}</div>
                <b>{risk_badge(level)}</b><br>
                Health Index: <b>{card['health_index']}</b><br>
                Risk Score: <b>{card['risk_score']}</b><br>
                Top Signal: <b>{card['top_abnormal_signal']}</b><br>
                z-score: <b>{card['top_signal_z']}</b><br>
                RUL: <b>{card['proxy_rul_band']}</b><br>
                Fault: <small>{card['predicted_fault']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Bind {card['asset']}", key=f"bind_topology_{i}"):
                bind_alarm(int(card["timestamp_index"]), f"ALM-{int(card['timestamp_index']):06d}")
                st.rerun()

    st.markdown("### Top Maintenance Priorities")
    queue = ml_service.maintenance_queue(top_n=8)
    if queue.empty:
        st.info("No priority events available. Re-run the ML export cells.")
    else:
        st.dataframe(queue, use_container_width=True, hide_index=True)
        selected = st.selectbox("Investigate priority item", queue.index.tolist(), format_func=lambda i: f"#{queue.loc[i].get('priority_rank')} {queue.loc[i].get('alarm_id')} · {queue.loc[i].get('asset')} · {queue.loc[i].get('risk_level')}")
        b1, b2, b3 = st.columns(3)
        row_idx = int(queue.loc[selected, "timestamp_index"])
        alarm_id = str(queue.loc[selected, "alarm_id"])
        if b1.button("Investigate selected alarm"):
            bind_alarm(row_idx, alarm_id)
            st.session_state.selected_alarm_event = queue.loc[selected].to_dict()
            st.success(f"Bound {alarm_id}. Open Alarm Investigation & RCA.")
        if b2.button("Add priority to logbook"):
            ev = digital_logbook.save_event(queue.loc[selected].to_dict())
            st.success(f"Saved {ev['log_id']}")
        if b3.button("Run quick RCA"):
            bind_alarm(row_idx, alarm_id)
            with st.spinner("Running Steel Pilot RCA..."):
                st.session_state.last_agent_result = answer_maintenance_query("Diagnose the selected priority alarm and recommend the next action.", thread_id=thread_id, row_index=row_idx, answer_mode="concise")
            st.success("RCA complete. Open Alarm Investigation or Copilot for details.")

elif page == "Alarm Investigation & RCA":
    st.subheader("Alarm Investigation & RCA")
    st.caption("Active alarms are grouped high-risk abnormal telemetry windows detected from the TCM replay stream. In production, this queue would come from live SCADA/historian events.")

    events = ml_service.alarm_events(top_n=25, min_risk="medium")
    if events.empty:
        st.warning("No grouped alarm events found. Try applying a demo scenario or re-run ML export cells.")
    else:
        view_cols = ["priority_rank", "alarm_id", "event_start", "event_end", "asset", "fault", "risk_level", "max_risk_score", "max_anomaly_probability", "rul_band", "event_count", "top_evidence_signal", "recommended_action"]
        st.dataframe(events[[c for c in view_cols if c in events.columns]], use_container_width=True, hide_index=True)
        selected = st.selectbox("Select alarm event", events.index.tolist(), format_func=lambda i: f"{events.loc[i].get('alarm_id')} · {events.loc[i].get('asset')} · {events.loc[i].get('risk_level')} · {fault_label_to_title(events.loc[i].get('fault'))}")
        event = events.loc[selected].to_dict()
        row_idx = int(event["timestamp_index"])
        alarm_id = str(event["alarm_id"])
        if st.button("Bind selected alarm as active context"):
            bind_alarm(row_idx, alarm_id)
            st.session_state.selected_alarm_event = event
            st.rerun()

    ml = get_active_result()
    if not ml:
        st.info("Bind an alarm event to run RCA.")
    else:
        st.markdown("### Selected Alarm Summary")
        render_active_context()
        metric_row(ml)
        st.markdown("### Telemetry Evidence")
        evidence = ml.get("evidence", [])
        if evidence:
            for e in evidence[:8]:
                st.write(f"- {e}")
        row = ml_service.row_by_index(int(ml["row_index"]))
        rules = apply_physical_constraint_rules(row, ml)
        cascade = analyze_cascading_impact(row, ml)
        st.markdown("### Physical Constraint Rules")
        render_rules(rules)
        st.markdown("### Cascading Impact")
        render_cascade(cascade)
        st.markdown("### SOP Evidence")
        query = f"{ml.get('predicted_fault')} {ml.get('predicted_stand')} {' '.join(ml.get('evidence', []))}"
        docs = rag.retrieve(query, k=4)
        render_sop_sources(docs)
        st.markdown("### RCA Report")
        c1, c2, c3 = st.columns(3)
        if c1.button("Generate Concise RCA"):
            with st.spinner("Running concise RCA..."):
                st.session_state.last_agent_result = answer_maintenance_query("Give a concise RCA and next action for the active alarm.", thread_id=thread_id, row_index=int(ml["row_index"]), answer_mode="concise")
            st.rerun()
        if c2.button("Generate Detailed RCA"):
            with st.spinner("Running detailed RCA..."):
                st.session_state.last_agent_result = answer_maintenance_query("Generate a detailed RCA report with SOP evidence and cascading impact.", thread_id=thread_id, row_index=int(ml["row_index"]), answer_mode="detailed")
            st.rerun()
        if c3.button("Save Investigation to Logbook"):
            result = st.session_state.last_agent_result or {}
            ev = digital_logbook.save_event({
                "query": "Manual RCA save from Alarm Investigation page",
                "row_index": ml.get("row_index"),
                "alarm_id": ml.get("alarm_id"),
                "asset": ml.get("asset_name"),
                "fault": ml.get("predicted_fault"),
                "risk_level": ml.get("risk_level"),
                "risk_score": ml.get("risk_score"),
                "anomaly_probability": ml.get("anomaly_probability"),
                "rul_band": ml.get("predicted_rul_band"),
                "recommended_action": ml_service.default_action_for_fault(ml.get("predicted_fault", ""), ml.get("risk_level", "")),
                "decision_trace": result.get("decision_trace", []),
                "physical_rules": rules,
                "cascading_impact": cascade,
                "rag_sources": docs,
                "status": "open",
            })
            st.success(f"Saved {ev['log_id']}")
        if st.session_state.last_agent_result:
            st.markdown(st.session_state.last_agent_result.get("final_answer", ""))
            with st.expander("Decision Evidence — Why this recommendation?"):
                render_decision_evidence(st.session_state.last_agent_result)
            with st.expander("SOP citations used by agent"):
                render_sop_sources(st.session_state.last_agent_result.get("rag_context", []))

elif page == "Live Telemetry Replay":
    st.subheader("Live Telemetry Replay")
    st.caption("Chronological TCM telemetry replay with normalized and grouped views. Default chart uses z-score normalization so force does not hide torque, power, and ratios.")
    max_idx = len(ml_service.features_df) - 1
    default_idx = int(st.session_state.active_alarm_row) if st.session_state.active_alarm_row is not None else max_idx
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stand_sel = st.selectbox("Stand", [1, 2, 3, 4, 5], index=2)
    with c2:
        row_idx = st.slider("Replay timestamp", 0, max_idx, min(default_idx, max_idx), step=1)
    with c3:
        window = st.slider("Window", 50, 1000, 250, step=50)
    with c4:
        chart_mode = st.selectbox("Chart mode", ["Normalized z-score", "Raw grouped", "Individual signal", "Percent change", "Min-max"])
    if st.button("Bind this timestamp as active alarm context"):
        bind_alarm(int(row_idx), f"ALM-{int(row_idx):06d}")
        st.success(f"Bound ALM-{int(row_idx):06d}")
    ml = ml_service.predict_condition(row_index=int(row_idx))
    metric_row(ml)
    if chart_mode == "Raw grouped":
        groups = ml_service.telemetry_groups(row_index=int(row_idx), stand=int(stand_sel), window=int(window))
        for name, df_g in groups.items():
            render_line_chart(df_g, [c for c in df_g.columns if c != "timestamp_index"], name, active_ts=int(row_idx))
    elif chart_mode == "Individual signal":
        raw = ml_service.telemetry_window(row_index=int(row_idx), stand=int(stand_sel), window=int(window))
        signals = [c for c in raw.columns if c != "timestamp_index"]
        signal = st.selectbox("Signal", signals)
        render_line_chart(raw, [signal], f"Raw signal: {signal}", active_ts=int(row_idx))
    else:
        mode = "zscore" if chart_mode == "Normalized z-score" else "pct_change" if chart_mode == "Percent change" else "minmax"
        norm = ml_service.telemetry_window_normalized(row_index=int(row_idx), stand=int(stand_sel), window=int(window), mode=mode)
        default_cols = [c for c in norm.columns if c != "timestamp_index"]
        selected_cols = st.multiselect("Signals to compare", default_cols, default=default_cols[:6])
        render_line_chart(norm, selected_cols, f"{chart_mode} comparison", active_ts=int(row_idx))
    with st.expander("Raw telemetry table"):
        raw = ml_service.telemetry_window(row_index=int(row_idx), stand=int(stand_sel), window=int(window))
        st.dataframe(raw.tail(50), use_container_width=True, hide_index=True)

elif page == "Steel Pilot Maintenance Copilot":
    st.subheader("Steel Pilot Maintenance Copilot")
    render_active_context()
    st.caption("Use concise mode for exact operational answers and detailed mode for RCA/report-style responses. Out-of-domain queries are handled by a guardrail.")
    st.session_state.answer_mode = st.radio("Answer Mode", ["Concise", "Detailed"], horizontal=True, index=0 if st.session_state.answer_mode == "Concise" else 1)
    override = st.number_input("Optional row/timestamp override", value=-1, step=1)
    selected_row = int(override) if override >= 0 else st.session_state.active_alarm_row
    prompt_examples = [
        "Diagnose the active alarm and recommend next action.",
        "Is this more likely mechanical or electrical?",
        "What telemetry evidence supports this diagnosis?",
        "Which SOP supports the recommended action?",
        "Can this issue cascade to nearby stands?",
        "Generate a shift handover summary.",
    ]
    with st.expander("Suggested maintenance questions"):
        for p in prompt_examples:
            st.code(p)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_query = st.chat_input("Ask Steel Pilot about alarms, telemetry, RCA, SOPs, drift, priority, or feedback...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Running Steel Pilot agents..."):
                result = answer_maintenance_query(user_query, thread_id=thread_id, row_index=selected_row, answer_mode=st.session_state.answer_mode.lower())
                st.session_state.last_agent_result = result
                answer = result.get("final_answer", "No answer generated.")
                st.markdown(answer)
                if result.get("guardrail"):
                    st.warning("Guardrail handled this query as out-of-domain.")
                with st.expander("Decision Evidence"):
                    render_decision_evidence(result)
                with st.expander("SOP Sources"):
                    render_sop_sources(result.get("rag_context", []))
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif page == "Operations Logbook & Feedback":
    st.subheader("Operations Logbook & Feedback")
    st.caption("Persistent SQLite-backed audit trail and engineer feedback memory. Complex JSON is stored safely; the display table is flattened to avoid PyArrow errors.")
    summary = digital_logbook.summarize()
    a, b, c, d = st.columns(4)
    a.metric("Total Events", summary.get("total_events", 0))
    b.metric("Open Events", summary.get("open", 0))
    c.metric("Critical Events", summary.get("critical", 0))
    d.metric("SQLite DB", "Persistent")
    tab1, tab2 = st.tabs(["Logbook", "Feedback"])
    with tab1:
        df = digital_logbook.to_dataframe(limit=300)
        if df.empty:
            st.info("No logbook events yet. Run an RCA/chat query or save an investigation.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            selected_id = st.selectbox("Open log detail", df["log_id"].tolist())
            detail = digital_logbook.get_event(selected_id)
            if detail:
                c1, c2, c3 = st.columns(3)
                c1.metric("Alarm", detail.get("alarm_id"))
                c2.metric("Risk", risk_badge(detail.get("risk_level")))
                c3.metric("Status", detail.get("status"))
                new_status = st.selectbox("Update status", ["open", "acknowledged", "assigned", "work_order_created", "resolved", "false_alarm"], index=0)
                if st.button("Update log status"):
                    digital_logbook.update_status(selected_id, new_status)
                    st.success("Status updated.")
                    st.rerun()
                with st.expander("Full recommendation / answer"):
                    st.write(detail.get("final_answer") or detail.get("recommended_action") or "")
                with st.expander("Decision Trace"):
                    st.json(detail.get("decision_trace", []))
                with st.expander("Physical Rules"):
                    st.json(detail.get("physical_rules", []))
                with st.expander("Cascading Impact"):
                    st.json(detail.get("cascading_impact", {}))
                with st.expander("SOP Sources"):
                    st.json(detail.get("rag_sources", []))
    with tab2:
        ml = get_active_result()
        st.markdown("### Auto-filled feedback from active alarm")
        if not ml:
            st.info("Bind an active alarm first to auto-populate predicted fault and timestamp. You can still save manual feedback.")
            ml = {}
        with st.form("feedback_form"):
            c1, c2, c3 = st.columns(3)
            alarm_id = c1.text_input("Alarm ID", value=str(ml.get("alarm_id", "")), disabled=bool(ml))
            row_index_val = c2.text_input("Timestamp / row", value=str(ml.get("row_index", "")), disabled=bool(ml))
            asset = c3.text_input("Asset", value=str(ml.get("asset_name", "")), disabled=bool(ml))
            predicted_fault = st.text_input("Predicted fault", value=str(ml.get("predicted_fault", "")), disabled=bool(ml))
            risk_level = st.text_input("Risk level", value=str(ml.get("risk_level", "")), disabled=bool(ml))
            actual_fault = st.text_input("Actual root cause found by engineer")
            action_taken = st.text_area("Action taken")
            outcome = st.text_area("Outcome / observed result")
            status = st.selectbox("Resolution status", ["feedback_recorded", "resolved", "false_alarm", "monitoring", "needs_followup"])
            engineer_name = st.text_input("Engineer name / shift", value="demo_engineer")
            notes = st.text_area("Additional notes")
            submitted = st.form_submit_button("Save persistent feedback")
            if submitted:
                try:
                    row_idx_int = int(row_index_val) if str(row_index_val).strip() else None
                except Exception:
                    row_idx_int = None
                event = feedback_memory.save_feedback({
                    "alarm_id": alarm_id or ml.get("alarm_id"),
                    "row_index": row_idx_int,
                    "asset": asset or ml.get("asset_name"),
                    "predicted_fault": predicted_fault or ml.get("predicted_fault"),
                    "risk_level": risk_level or ml.get("risk_level"),
                    "anomaly_probability": ml.get("anomaly_probability"),
                    "actual_fault": actual_fault,
                    "action_taken": action_taken,
                    "outcome": outcome,
                    "status": status,
                    "engineer_name": engineer_name,
                    "notes": notes,
                    "source": "operations_feedback_form",
                })
                digital_logbook.save_event({
                    "alarm_id": alarm_id or ml.get("alarm_id"),
                    "row_index": row_idx_int,
                    "asset": asset or ml.get("asset_name"),
                    "fault": predicted_fault or ml.get("predicted_fault"),
                    "risk_level": risk_level or ml.get("risk_level"),
                    "anomaly_probability": ml.get("anomaly_probability"),
                    "status": status,
                    "recommended_action": action_taken,
                    "query": "Engineer feedback recorded",
                    "final_answer": f"Actual root cause: {actual_fault}\nAction: {action_taken}\nOutcome: {outcome}",
                })
                st.success(f"Feedback saved persistently: {event['feedback_id']}")
        fdf = feedback_memory.to_dataframe(limit=300)
        st.markdown("### Feedback history")
        if fdf.empty:
            st.info("No feedback saved yet.")
        else:
            st.dataframe(fdf, use_container_width=True, hide_index=True)

elif page == "Model Health & Evaluation":
    st.subheader("Model Health & Evaluation")
    st.caption("Graphical model monitoring, drift interpretation, and periodic agent health checks.")
    metrics = ml_service.metrics()
    sup = metrics.get("supervised_anomaly", {}) or {}
    rec = metrics.get("recall_biased_anomaly", {}) or {}
    rul = metrics.get("proxy_rul_regression", {}) or {}
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("F1", round(float(sup.get("f1", 0)), 3))
    m2.metric("Recall", round(float(sup.get("recall", 0)), 3))
    m3.metric("Precision", round(float(sup.get("precision", 0)), 3))
    m4.metric("ROC-AUC", round(float(sup.get("roc_auc", 0)), 3))
    m5.metric("PR-AUC", round(float(sup.get("pr_auc", 0)), 3))
    m6.metric("RUL MAE shifts", round(float(rul.get("mae_shifts", 0)), 3))
    if rec:
        st.info(f"Safety-first threshold mode: recall={rec.get('recall'):.3f}, precision={rec.get('precision'):.3f}, threshold={rec.get('threshold'):.2f}.")

    st.markdown("### Drift Monitoring")
    drift = ml_service.drift_psi.copy()
    if drift.empty:
        st.warning("No drift PSI file found.")
    else:
        top_drift = drift.head(15).copy()
        psi_col = "psi_train_vs_drift" if "psi_train_vs_drift" in top_drift.columns else top_drift.columns[1]
        high_count = int((drift[psi_col] > 0.25).sum())
        d1, d2, d3 = st.columns(3)
        d1.metric("Overall Drift", "High" if high_count else "Low/Moderate")
        d2.metric("High-drift Features", high_count)
        d3.metric("Retraining Signal", "Review required" if high_count else "Monitor")
        chart_df = top_drift[["feature", psi_col]].set_index("feature")
        st.bar_chart(chart_df)
        st.caption("PSI > 0.25 usually indicates high drift. In this dataset, force/tension/motor-power rolling features often drift strongly between dataset 3 and dataset 5.")

    st.markdown("### Feature Importance")
    imp = ml_service.feature_importance.copy()
    if imp.empty:
        st.warning("No feature importance file found.")
    else:
        chart = imp.head(20)[["feature", "importance"]].set_index("feature")
        st.bar_chart(chart)
        st.caption("Important features show what the anomaly model relies on: power-to-torque balance, force imbalance, speed-power relationship, reduction/gap variation, and tension instability.")

    st.markdown("### Agent Health Check")
    st.caption("Periodic evaluation prompts check whether the agent includes ML context, risk, evidence, SOP citations, actions, physical rules, cascading impact, and guardrails.")
    select_all = st.checkbox("Select all tests", value=True)
    selected_ids: list[str] = []
    cols = st.columns(2)
    for i, test in enumerate(GOLD_TESTS):
        with cols[i % 2]:
            checked = st.checkbox(f"{test['id']} — {test['category']}: {test['query']}", value=select_all, key=f"eval_{test['id']}")
            if checked:
                selected_ids.append(test["id"])
    if st.button("Run selected health checks"):
        if not selected_ids:
            st.warning("Select at least one test.")
        else:
            with st.spinner("Running selected agent health checks..."):
                df = run_gold_tests(answer_maintenance_query, selected_ids=selected_ids, row_index=st.session_state.active_alarm_row)
                run = runtime_store.save_health_run(selected_ids, df)
            st.success(f"Saved health-check run {run['run_id']}")
            st.dataframe(df, use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Average Score", round(float(df["score"].mean()), 1))
            c2.metric("Passed", int(df["pass"].sum()))
            c3.metric("Failed", int((~df["pass"]).sum()))
    history = runtime_store.health_runs_df(limit=20)
    if not history.empty:
        st.markdown("### Previous health-check runs")
        st.dataframe(history, use_container_width=True, hide_index=True)
