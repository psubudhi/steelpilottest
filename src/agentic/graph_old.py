from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .audit import digital_logbook
from .cascading import analyze_cascading_impact
from .decision_trace import build_decision_trace
from .llm import invoke_json, invoke_text
from .memory import feedback_memory
from .ml_service import ml_service
from .prompts import COUNCIL_SYSTEM, PLANNER_SYSTEM, REPORT_SYSTEM, ROOT_CAUSE_SYSTEM, ROUTER_SYSTEM
from .rag import rag
from .rules import apply_physical_constraint_rules
from .spares import get_spares_for_fault


DOMAIN_TERMS = {
    "steel", "mill", "rolling", "roll", "stand", "tcm", "steelpilot", "steel-pilot", "scada", "telemetry",
    "anomaly", "alarm", "fault", "bearing", "motor", "electric", "workroll", "work-roll",
    "reduction", "gap", "torque", "force", "power", "tension", "speed", "rul", "risk",
    "health", "maintenance", "sop", "rca", "root", "cause", "drift", "logbook", "feedback",
    "priority", "cascade", "cascading", "inspect", "lubrication", "shift", "handover",
}


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    row_index: int | None
    answer_mode: str
    route: dict[str, Any]
    guardrail: dict[str, Any]
    ml_result: dict[str, Any]
    physical_rules: list[dict[str, Any]]
    cascading_impact: dict[str, Any]
    rag_context: list[dict[str, Any]]
    priority_board: list[dict[str, Any]]
    drift_summary: list[dict[str, Any]]
    spares: list[dict[str, Any]]
    root_cause: dict[str, Any]
    maintenance_plan: dict[str, Any]
    council: str
    decision_trace: list[dict[str, Any]]
    final_answer: str
    feedback_saved: dict[str, Any]
    audit_event: dict[str, Any]
    errors: list[str]


def _is_probably_in_domain(query: str) -> bool:
    q = re.sub(r"[^a-zA-Z0-9_ -]", " ", query.lower())
    terms = set(q.replace("_", " ").split())
    if terms & DOMAIN_TERMS:
        return True
    if re.search(r"\b(row|timestamp|alarm|alm)[-:# ]*\d+\b", q):
        return True
    return False


def _fallback_route(query: str, supplied_row: int | None = None) -> dict[str, Any]:
    lower = query.lower()
    intent = "diagnosis"
    if any(k in lower for k in ["actual issue", "feedback", "action taken", "resolved", "correct fault"]):
        intent = "feedback"
    elif any(k in lower for k in ["cascade", "adjacent", "upstream", "downstream"]):
        intent = "cascade_check"
    elif any(k in lower for k in ["priority", "first", "maintain first"]):
        intent = "risk_priority"
    elif any(k in lower for k in ["sop", "manual", "procedure", "citation", "source"]):
        intent = "general_sop_question"
    elif any(k in lower for k in ["drift", "model health", "retrain"]):
        intent = "drift_check"
    elif any(k in lower for k in ["mechanical", "electrical", "electric"]):
        intent = "mechanical_vs_electrical"
    elif any(k in lower for k in ["handover", "report", "summary"]):
        intent = "report_generation"
    elif any(k in lower for k in ["what is", "meaning", "explain"]):
        intent = "general_sop_question" if _is_probably_in_domain(query) else "out_of_domain"
    if not _is_probably_in_domain(query):
        intent = "out_of_domain"
    return {
        "intent": intent,
        "stand": None,
        "needs_sensor": intent not in {"out_of_domain"},
        "needs_rag": intent not in {"out_of_domain", "feedback"},
        "needs_priority": intent == "risk_priority",
        "needs_report": intent == "report_generation",
        "row_index": supplied_row or ml_service.find_row_from_query(query),
        "domain_status": "out_of_domain" if intent == "out_of_domain" else "in_domain",
    }


def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    supplied_row = state.get("row_index")
    lower = query.lower()
    if not _is_probably_in_domain(query):
        route = _fallback_route(query, supplied_row)
        return {**state, "route": route, "row_index": route.get("row_index")}
    if any(k in lower for k in ["actual issue", "feedback", "diagnosis was", "correct fault", "action taken", "resolved"]):
        route = _fallback_route(query, supplied_row)
    else:
        route = invoke_json(ROUTER_SYSTEM, query, temperature=0.0)
        if "intent" not in route or route.get("fallback"):
            route = _fallback_route(query, supplied_row)
        route["row_index"] = supplied_row or route.get("row_index") or ml_service.find_row_from_query(query)
        route["domain_status"] = route.get("domain_status") or "in_domain"
        if route.get("intent") == "out_of_domain":
            route["needs_sensor"] = False
            route["needs_rag"] = False
    return {**state, "route": route, "row_index": route.get("row_index")}


def route_after_router(state: AgentState) -> Literal["feedback", "guardrail", "sensor"]:
    intent = state.get("route", {}).get("intent")
    if intent == "feedback":
        return "feedback"
    if intent == "out_of_domain":
        return "guardrail"
    return "sensor"


def guardrail_node(state: AgentState) -> AgentState:
    answer = (
        "I’m designed for Steel Pilot steel plant maintenance workflows: telemetry, alarms, RCA, physical rules, "
        "SOP guidance, risk, proxy RUL, drift, feedback, and dashboard usage. Please ask a question related "
        "to the active alarm, plant health, maintenance action, model output, or steel rolling-mill process."
    )
    guardrail = {"domain_status": "out_of_domain", "action": "refused_gracefully"}
    return {**state, "guardrail": guardrail, "final_answer": answer}


def feedback_node(state: AgentState) -> AgentState:
    ml = {}
    if state.get("row_index") is not None:
        try:
            ml = ml_service.predict_condition(row_index=state.get("row_index"))
        except Exception:
            ml = {}
    feedback = feedback_memory.save_feedback({
        "query": state["query"],
        "row_index": state.get("row_index") or ml.get("row_index"),
        "alarm_id": ml.get("alarm_id"),
        "asset": ml.get("asset_name"),
        "predicted_fault": ml.get("predicted_fault"),
        "risk_level": ml.get("risk_level"),
        "anomaly_probability": ml.get("anomaly_probability"),
        "actual_fault": state["query"],
        "notes": "Engineer feedback captured from chat query. Review/structure in Operations Logbook & Feedback.",
        "source": "agent_chat",
    })
    audit = digital_logbook.save_event({
        "query": state["query"],
        "row_index": state.get("row_index") or ml.get("row_index"),
        "alarm_id": ml.get("alarm_id"),
        "asset": ml.get("asset_name"),
        "fault": ml.get("predicted_fault"),
        "risk_level": ml.get("risk_level"),
        "anomaly_probability": ml.get("anomaly_probability"),
        "status": "feedback_recorded",
        "recommended_action": "Review feedback and add it to validated case memory if confirmed.",
    })
    answer = (
        f"Feedback saved persistently in SQLite. Feedback ID: {feedback['feedback_id']}. "
        f"Logbook event: {audit['log_id']}. It will still be visible after reopening the dashboard."
    )
    return {**state, "feedback_saved": feedback, "audit_event": audit, "final_answer": answer}


def sensor_node(state: AgentState) -> AgentState:
    query = state["query"]
    row_index = state.get("row_index")
    strategy = "highest_risk" if any(k in query.lower() for k in ["worst", "highest risk", "critical", "maintained first", "maintain first"]) else "latest"
    result = ml_service.predict_condition(row_index=row_index, strategy=strategy)
    priority = ml_service.maintenance_queue(top_n=8).to_dict(orient="records")
    drift = ml_service.drift_summary(top_n=8)
    return {**state, "ml_result": result, "priority_board": priority, "drift_summary": drift, "row_index": result.get("row_index")}


def physical_rules_node(state: AgentState) -> AgentState:
    row = ml_service.row_by_index(state.get("row_index"), strategy="latest")
    rules = apply_physical_constraint_rules(row, state.get("ml_result", {}))
    cascade = analyze_cascading_impact(row, state.get("ml_result", {}))
    return {**state, "physical_rules": rules, "cascading_impact": cascade}


def rag_node(state: AgentState) -> AgentState:
    ml = state.get("ml_result", {})
    matched_rules = [r for r in state.get("physical_rules", []) if r.get("matched")]
    retrieval_query = "\n".join([
        state["query"],
        f"Predicted fault: {ml.get('predicted_fault')}",
        f"Predicted stand: {ml.get('predicted_stand')}",
        f"Evidence: {'; '.join(ml.get('evidence', []))}",
        f"Physical rules: {'; '.join([r.get('title', '') for r in matched_rules])}",
        f"Cascading impact: {state.get('cascading_impact', {}).get('cascading_risk')}",
    ])
    docs = rag.retrieve(retrieval_query, k=5)
    return {**state, "rag_context": docs}


def root_cause_node(state: AgentState) -> AgentState:
    user = json.dumps({
        "query": state["query"],
        "answer_mode": state.get("answer_mode", "concise"),
        "ml_result": state.get("ml_result", {}),
        "physical_rules": state.get("physical_rules", []),
        "cascading_impact": state.get("cascading_impact", {}),
        "rag_context": state.get("rag_context", []),
        "similar_cases": state.get("ml_result", {}).get("similar_historical_cases", []),
    }, indent=2)
    rc = invoke_json(ROOT_CAUSE_SYSTEM, user, temperature=0.1)
    if rc.get("fallback"):
        rc = {
            "probable_root_causes": [state.get("ml_result", {}).get("predicted_fault", "unknown fault")],
            "reasoning": "Fallback root cause based on ML predicted fault and telemetry evidence.",
            "uncertainty_notes": "LLM unavailable; review rules and SOP evidence manually.",
        }
    return {**state, "root_cause": rc}


def planner_node(state: AgentState) -> AgentState:
    ml = state.get("ml_result", {})
    spares = get_spares_for_fault(ml.get("predicted_fault", ""), ml.get("predicted_stand", "mill_level"))
    user = json.dumps({
        "query": state["query"],
        "answer_mode": state.get("answer_mode", "concise"),
        "ml_result": ml,
        "physical_rules": state.get("physical_rules", []),
        "cascading_impact": state.get("cascading_impact", {}),
        "root_cause": state.get("root_cause", {}),
        "rag_context": state.get("rag_context", []),
        "priority_board": state.get("priority_board", []),
        "spares": spares,
    }, indent=2)
    plan = invoke_json(PLANNER_SYSTEM, user, temperature=0.1)
    if plan.get("fallback"):
        fault = ml.get("predicted_fault", "the predicted fault")
        plan = {
            "immediate_actions": [ml_service.default_action_for_fault(fault, ml.get("risk_level", ""))],
            "short_term_actions": ["Monitor the top abnormal telemetry signals and adjacent stand behaviour."],
            "planned_actions": ["Review event in shift handover and validate with maintenance engineer feedback."],
            "spare_strategy": ["Check relevant spares listed for this fault category."],
            "monitoring_plan": ["Track anomaly probability, risk score, and z-score evidence until condition normalizes."],
            "safety_notes": ["Follow plant SOP and authorized engineer approval for production changes or shutdown decisions."],
        }
    return {**state, "spares": spares, "maintenance_plan": plan}


def council_node(state: AgentState) -> AgentState:
    if state.get("answer_mode") == "concise":
        return {**state, "council": "Concise mode: council summary suppressed; see Decision Evidence for detailed agent outputs."}
    user = json.dumps({
        "ml_result": state.get("ml_result", {}),
        "physical_rules": state.get("physical_rules", []),
        "cascading_impact": state.get("cascading_impact", {}),
        "root_cause": state.get("root_cause", {}),
        "maintenance_plan": state.get("maintenance_plan", {}),
        "rag_context": state.get("rag_context", []),
    }, indent=2)
    council = invoke_text(COUNCIL_SYSTEM, user, temperature=0.2)
    return {**state, "council": council}


def trace_node(state: AgentState) -> AgentState:
    return {**state, "decision_trace": build_decision_trace(state)}


def _deterministic_report(state: AgentState) -> str:
    ml = state.get("ml_result", {})
    plan = state.get("maintenance_plan", {}) or {}
    rules = [r for r in state.get("physical_rules", []) if r.get("matched")]
    rag_docs = state.get("rag_context", [])
    mode = state.get("answer_mode", "concise")
    action = "Review Steel Pilot recommendation."
    if isinstance(plan, dict):
        ia = plan.get("immediate_actions", [])
        if isinstance(ia, list) and ia:
            action = str(ia[0])
    if mode == "concise":
        return (
            f"- **Diagnosis:** {ml.get('predicted_fault')} on {ml.get('asset_name')} ({ml.get('risk_level')} risk, score {ml.get('risk_score')}).\n"
            f"- **Why:** anomaly probability {ml.get('anomaly_probability')}; top evidence: {'; '.join(ml.get('evidence', [])[:3])}.\n"
            f"- **Rule check:** {rules[0].get('title') if rules else 'No strong physical rule triggered; inspect telemetry evidence.'}\n"
            f"- **Next action:** {action}\n"
            f"- **SOP evidence:** {', '.join([d.get('source','') for d in rag_docs[:3]]) or 'No SOP source retrieved.'}"
        )
    return json.dumps({
        "diagnosis": ml,
        "matched_rules": rules,
        "cascading_impact": state.get("cascading_impact", {}),
        "maintenance_plan": plan,
        "sop_sources": [d.get("source") for d in rag_docs],
        "proxy_rul_note": ml.get("disclaimer"),
    }, indent=2)


def report_node(state: AgentState) -> AgentState:
    user = json.dumps({
        "answer_mode": state.get("answer_mode", "concise"),
        "user_query": state["query"],
        "route": state.get("route", {}),
        "ml_result": state.get("ml_result", {}),
        "physical_rules": state.get("physical_rules", []),
        "cascading_impact": state.get("cascading_impact", {}),
        "root_cause": state.get("root_cause", {}),
        "maintenance_plan": state.get("maintenance_plan", {}),
        "multi_agent_council": state.get("council", ""),
        "decision_trace": state.get("decision_trace", []),
        "priority_board": state.get("priority_board", []),
        "drift_summary": state.get("drift_summary", []),
        "spares": state.get("spares", []),
        "rag_context": state.get("rag_context", []),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2)
    final = invoke_text(REPORT_SYSTEM, user, temperature=0.15)
    if final.startswith("Steel Pilot could not call"):
        final = _deterministic_report(state)
    return {**state, "final_answer": final}


def audit_node(state: AgentState) -> AgentState:
    ml = state.get("ml_result", {})
    plan = state.get("maintenance_plan", {}) or {}
    immediate = plan.get("immediate_actions", []) if isinstance(plan, dict) else []
    first_action = immediate[0] if isinstance(immediate, list) and immediate else "Review Steel Pilot recommendation."
    audit = digital_logbook.save_event({
        "query": state.get("query"),
        "final_answer": state.get("final_answer"),
        "row_index": state.get("row_index"),
        "alarm_id": ml.get("alarm_id"),
        "asset": ml.get("asset_name") or ml.get("predicted_stand"),
        "fault": ml.get("predicted_fault"),
        "risk_level": ml.get("risk_level"),
        "risk_score": ml.get("risk_score"),
        "anomaly_probability": ml.get("anomaly_probability"),
        "rul_band": ml.get("predicted_rul_band"),
        "recommended_action": first_action,
        "status": "open" if ml.get("is_alert") else "reviewed",
        "decision_trace": state.get("decision_trace", []),
        "physical_rules": state.get("physical_rules", []),
        "cascading_impact": state.get("cascading_impact", {}),
        "rag_sources": state.get("rag_context", []),
    })
    trace = state.get("decision_trace", [])
    if trace:
        trace[-1]["evidence"] = {"audit": {"log_id": audit.get("log_id"), "status": audit.get("status")}}
    return {**state, "audit_event": audit, "decision_trace": trace}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("sensor", sensor_node)
    graph.add_node("physical_rules", physical_rules_node)
    graph.add_node("rag", rag_node)
    graph.add_node("root_cause", root_cause_node)
    graph.add_node("planner", planner_node)
    graph.add_node("council", council_node)
    graph.add_node("trace", trace_node)
    graph.add_node("report", report_node)
    graph.add_node("audit", audit_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {"feedback": "feedback", "guardrail": "guardrail", "sensor": "sensor"})
    graph.add_edge("guardrail", END)
    graph.add_edge("feedback", END)
    graph.add_edge("sensor", "physical_rules")
    graph.add_edge("physical_rules", "rag")
    graph.add_edge("rag", "root_cause")
    graph.add_edge("root_cause", "planner")
    graph.add_edge("planner", "council")
    graph.add_edge("council", "trace")
    graph.add_edge("trace", "report")
    graph.add_edge("report", "audit")
    graph.add_edge("audit", END)
    return graph.compile(checkpointer=MemorySaver())


app_graph = build_graph()


def answer_maintenance_query(
    query: str,
    thread_id: str = "steel-pilot-demo",
    row_index: int | None = None,
    answer_mode: str = "concise",
) -> dict[str, Any]:
    mode = "detailed" if str(answer_mode).lower().startswith("detail") else "concise"
    inputs: AgentState = {"query": query, "thread_id": thread_id, "row_index": row_index, "answer_mode": mode, "errors": []}
    return app_graph.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
