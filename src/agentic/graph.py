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
    "operator", "procedure", "procedures", "troubleshoot", "troubleshooting", "gearbox",
    "hydraulic", "emulsion", "coolant", "quality", "strip", "gauge", "thickness",
    "vibration", "cooling", "drive", "agc", "work", "rolls", "millstand", "chock",
}

FOLLOW_UP_TERMS = {
    "why", "how", "what", "which", "when", "where", "who", "should", "could", "can",
    "next", "then", "else", "more", "detail", "details", "explain", "compare",
    "recommend", "recommendation", "action", "actions", "inspect", "check", "monitor",
    "sop", "procedure", "steps", "risk", "evidence",
}

OUT_OF_DOMAIN_HINTS = {
    "poem", "joke", "weather", "recipe", "movie", "song", "travel", "holiday", "email",
    "resume", "interview", "code", "python", "javascript", "football", "cricket", "stock",
}

SMALLTALK_PATTERNS = (
    r"^\s*(hi|hello|hey|thanks|thank you|good morning|good afternoon|good evening)[!. ]*$",
    r"\bhow are you\b",
    r"\bwho are you\b",
    r"\bwhat('?s| is) up\b",
    r"\btell me a joke\b",
    r"\bwrite (me )?(a )?poem\b",
)


def _textify_action(value: Any) -> str:
    
    if value is None:
        return "Review Steel Pilot recommendation."
    if isinstance(value, dict):
        for key in ("action", "recommended_action", "description", "task", "title", "summary", "text"):
            if value.get(key):
                return str(value.get(key)).strip()
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, (list, tuple)):
        return "; ".join(_textify_action(v) for v in value) or "Review Steel Pilot recommendation."
    return str(value).strip() or "Review Steel Pilot recommendation."


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    row_index: int | None
    answer_mode: str
    surface: str
    conversation_context: str
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


def _normalized_terms(*parts: str | None) -> set[str]:
    merged = " ".join(part for part in parts if part)
    q = re.sub(r"[^a-zA-Z0-9_ -]", " ", merged.lower())
    return set(q.replace("_", " ").split())


def _is_smalltalk_or_meta(query: str) -> bool:
    lower = query.lower().strip()
    return any(re.search(pattern, lower) for pattern in SMALLTALK_PATTERNS)


def _is_probably_in_domain(query: str, conversation_context: str | None = None, has_active_context: bool = False) -> bool:
    if _is_smalltalk_or_meta(query):
        return False
    terms = _normalized_terms(query, conversation_context)
    if terms & DOMAIN_TERMS:
        return True
    q = re.sub(r"[^a-zA-Z0-9_ -]", " ", query.lower())
    if re.search(r"\b(row|timestamp|alarm|alm)[-:# ]*\d+\b", q):
        return True
    if has_active_context and not (terms & OUT_OF_DOMAIN_HINTS):
        if terms & FOLLOW_UP_TERMS:
            return True
    return False


def _fallback_route(query: str, supplied_row: int | None = None, conversation_context: str | None = None) -> dict[str, Any]:
    lower = query.lower()
    has_active_context = supplied_row is not None
    combined = "\n".join(part for part in [conversation_context, query] if part).lower()
    intent = "diagnosis"
    if any(k in combined for k in ["actual issue", "feedback", "action taken", "resolved", "correct fault"]):
        intent = "feedback"
    elif any(k in combined for k in ["cascade", "adjacent", "upstream", "downstream"]):
        intent = "cascade_check"
    elif any(k in combined for k in ["priority", "first", "maintain first"]):
        intent = "risk_priority"
    elif any(k in combined for k in ["sop", "manual", "procedure", "citation", "source", "work instruction", "checklist"]):
        intent = "general_sop_question"
    elif any(k in combined for k in ["drift", "model health", "retrain"]):
        intent = "drift_check"
    elif any(k in combined for k in ["mechanical", "electrical", "electric"]):
        intent = "mechanical_vs_electrical"
    elif any(k in combined for k in ["handover", "report", "summary"]):
        intent = "report_generation"
    elif any(k in lower for k in ["what is", "meaning", "explain"]):
        intent = "general_sop_question" if _is_probably_in_domain(query, conversation_context, has_active_context) else "out_of_domain"
    elif has_active_context and any(k in lower for k in ["why", "how", "what next", "next step", "what should", "should we", "can we", "could this"]):
        intent = "diagnosis"
    if not _is_probably_in_domain(query, conversation_context, has_active_context):
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
    conversation_context = state.get("conversation_context")
    lower = query.lower()
    if not _is_probably_in_domain(query, conversation_context, supplied_row is not None):
        route = _fallback_route(query, supplied_row, conversation_context)
        return {**state, "route": route, "row_index": route.get("row_index")}
    if any(k in lower for k in ["actual issue", "feedback", "diagnosis was", "correct fault", "action taken", "resolved"]):
        route = _fallback_route(query, supplied_row, conversation_context)
    else:
        router_prompt = query if not conversation_context else f"Conversation context:\n{conversation_context}\n\nCurrent user query:\n{query}"
        route = invoke_json(ROUTER_SYSTEM, router_prompt, temperature=0.0)
        if "intent" not in route or route.get("fallback"):
            route = _fallback_route(query, supplied_row, conversation_context)
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
    if state.get("surface") == "copilot":
        answer = (
            "I can help with the rolling-mill alarm, telemetry, RCA reasoning, maintenance actions, and relevant SOP or checklist guidance. "
            "I can’t help with general chat or unrelated topics in this copilot."
        )
    else:
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


def _stringify_list(value: Any, fallback: list[str]) -> list[str]:
    items: list[str] = []
    if isinstance(value, str):
        for chunk in re.split(r"[\n;]+", value):
            text = chunk.strip(" -\t")
            if text:
                items.append(text)
    elif isinstance(value, dict):
        text = _textify_action(value)
        if text:
            items.append(text)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            text = _textify_action(item)
            if text:
                items.append(text)
    if not items:
        return fallback
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(item)
    return deduped


def _normalize_root_cause(rc: dict[str, Any], ml: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(rc.get("raw_text") or "").strip()
    fallback_cause = ml.get("predicted_fault", "unknown fault")
    causes = _stringify_list(rc.get("probable_root_causes"), [fallback_cause])
    reasoning = str(rc.get("reasoning") or "").strip() or raw_text or "Root cause derived from predicted fault, telemetry evidence, and matched rule checks."
    uncertainty = str(rc.get("uncertainty_notes") or "").strip() or "Validate against SOP steps, local inspection findings, and engineer confirmation."
    return {
        "probable_root_causes": causes,
        "reasoning": reasoning,
        "uncertainty_notes": uncertainty,
    }


def _default_plan(ml: dict[str, Any]) -> dict[str, Any]:
    fault = ml.get("predicted_fault", "the predicted fault")
    return {
        "immediate_actions": [ml_service.default_action_for_fault(fault, ml.get("risk_level", ""))],
        "short_term_actions": ["Monitor the top abnormal telemetry signals and adjacent stand behaviour."],
        "planned_actions": ["Review event in shift handover and validate with maintenance engineer feedback."],
        "spare_strategy": ["Check relevant spares listed for this fault category."],
        "monitoring_plan": ["Track anomaly probability, risk score, and z-score evidence until condition normalizes."],
        "safety_notes": ["Follow plant SOP and authorized engineer approval for production changes or shutdown decisions."],
    }


def _normalize_plan(plan: dict[str, Any], ml: dict[str, Any]) -> dict[str, Any]:
    default_plan = _default_plan(ml)
    raw_text = str(plan.get("raw_text") or "").strip()
    normalized: dict[str, Any] = {}
    for key, fallback in default_plan.items():
        normalized[key] = _stringify_list(plan.get(key), fallback)
    if raw_text:
        raw_lines = _stringify_list(raw_text, [])
        if raw_lines:
            for key in ("immediate_actions", "short_term_actions", "planned_actions"):
                if normalized[key] == default_plan[key]:
                    normalized[key] = raw_lines[:3]
                    break
    return normalized


def _format_list(title: str, items: list[str]) -> str:
    body = "\n".join(f"- {item}" for item in items if item)
    return f"**{title}**\n{body or '- Not available.'}"


def _format_rule(rule: dict[str, Any]) -> str:
    evidence = rule.get("evidence", [])
    evidence_text = "; ".join(str(item) for item in evidence[:3]) if isinstance(evidence, list) else str(evidence)
    title = rule.get("title") or rule.get("rule_id") or "Rule"
    severity = rule.get("severity") or "unknown"
    explanation = rule.get("explanation") or rule.get("recommendation") or "No explanation available."
    if evidence_text:
        return f"- {title} ({severity}): {explanation} Evidence: {evidence_text}"
    return f"- {title} ({severity}): {explanation}"


def _deterministic_detailed_report(state: AgentState) -> str:
    ml = state.get("ml_result", {})
    plan = state.get("maintenance_plan", {}) or {}
    rules = [r for r in state.get("physical_rules", []) if r.get("matched")]
    cascade = state.get("cascading_impact", {}) or {}
    rag_docs = state.get("rag_context", [])
    root = state.get("root_cause", {}) or {}
    diagnosis = f"{ml.get('predicted_fault', 'Unknown fault')} on {ml.get('asset_name', ml.get('predicted_stand', 'unknown asset'))}"
    evidence = _stringify_list(ml.get("evidence", []), ["Review telemetry evidence for the active alarm window."])[:5]
    cascade_checks = _stringify_list(cascade.get("recommended_checks"), ["Check upstream/downstream load sharing and tension response."])
    sop_lines = [f"{doc.get('source', 'unknown source')} ({doc.get('retrieval_mode', 'rag')})" for doc in rag_docs[:5]]
    sections = [
        f"## Diagnosis\n{diagnosis} with {ml.get('risk_level', 'unknown')} risk (score {ml.get('risk_score', 'n/a')}).",
        (
            "## Why It Matters\n"
            f"Anomaly probability is {ml.get('anomaly_probability', 'n/a')}. "
            f"Proxy RUL band is {ml.get('predicted_rul_band', 'unknown')}, which should be treated as urgency guidance rather than true run-to-failure life."
        ),
        _format_list("Evidence", evidence),
        "**Physical Rules**\n" + ("\n".join(_format_rule(rule) for rule in rules) if rules else "- No matched rule strongly triggered; inspect telemetry and SOP evidence together."),
        (
            "## Cascading Impact\n"
            f"Primary stand: {cascade.get('primary_stand', 'unknown')}. "
            f"Cascading risk: {cascade.get('cascading_risk', 'unknown')} "
            f"(score {cascade.get('cascading_risk_score', 'n/a')}).\n"
            + "\n".join(f"- {item}" for item in cascade_checks)
        ),
        _format_list("SOP Grounding", sop_lines or ["No SOP source retrieved."]),
        _format_list("Recommended Actions", _stringify_list(plan.get("immediate_actions"), ["Review Steel Pilot recommendation."]))
        + "\n"
        + _format_list("Short-Term Actions", _stringify_list(plan.get("short_term_actions"), ["Increase monitoring and validate the diagnosis."]))
        + "\n"
        + _format_list("Planned Actions", _stringify_list(plan.get("planned_actions"), ["Record findings in shift handover and schedule validation."])),
        (
            "## Root Cause Notes\n"
            f"- Probable causes: {', '.join(_stringify_list(root.get('probable_root_causes'), [ml.get('predicted_fault', 'unknown fault')]))}\n"
            f"- Reasoning: {root.get('reasoning', 'Review telemetry, rules, and SOP evidence together.')}\n"
            f"- Uncertainty: {root.get('uncertainty_notes', 'Validate with inspection and engineer confirmation.')}"
        ),
    ]
    return "\n\n".join(sections)


def _deterministic_copilot_reply(state: AgentState) -> str:
    ml = state.get("ml_result", {})
    plan = state.get("maintenance_plan", {}) or {}
    rules = [r for r in state.get("physical_rules", []) if r.get("matched")]
    rag_docs = state.get("rag_context", [])
    cascade = state.get("cascading_impact", {}) or {}
    mode = state.get("answer_mode", "concise")
    action = _textify_action((plan.get("immediate_actions") or ["Review Steel Pilot recommendation."])[0])
    diagnosis = f"This looks most consistent with {ml.get('predicted_fault', 'the current fault pattern')} on {ml.get('asset_name', ml.get('predicted_stand', 'the active asset'))}."
    evidence = "; ".join(_stringify_list(ml.get("evidence", []), ["review the active telemetry window"])[:3])
    rule_text = rules[0].get("title") if rules else "No single physical rule dominated, so telemetry and SOP checks should be reviewed together."
    sop = rag_docs[0].get("source") if rag_docs else "No SOP source retrieved yet."
    if mode == "concise":
        return (
            f"{diagnosis} Risk is {ml.get('risk_level', 'unknown')} with anomaly probability {ml.get('anomaly_probability', 'n/a')}.\n\n"
            f"The main evidence is {evidence}. The strongest rule check is: {rule_text}.\n\n"
            f"My next step would be: {action} SOP reference: {sop}"
        )
    return (
        f"{diagnosis} I’d treat it as a {ml.get('risk_level', 'unknown')} risk issue, not just a generic RCA template.\n\n"
        f"What is driving that view: {evidence}. The physical-rule check that matters most here is {rule_text}. "
        f"Cascading risk is {cascade.get('cascading_risk', 'unknown')}, so adjacent-stand behavior should stay in view.\n\n"
        f"What I’d do next: {action} The most relevant SOP or checklist right now is {sop}."
    )


def rag_node(state: AgentState) -> AgentState:
    ml = state.get("ml_result", {})
    matched_rules = [r for r in state.get("physical_rules", []) if r.get("matched")]
    retrieval_query = "\n".join([
        f"Conversation context: {state.get('conversation_context', '')}",
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
        "conversation_context": state.get("conversation_context", ""),
        "answer_mode": state.get("answer_mode", "concise"),
        "ml_result": state.get("ml_result", {}),
        "physical_rules": state.get("physical_rules", []),
        "cascading_impact": state.get("cascading_impact", {}),
        "rag_context": state.get("rag_context", []),
        "similar_cases": state.get("ml_result", {}).get("similar_historical_cases", []),
    }, indent=2)
    rc = invoke_json(ROOT_CAUSE_SYSTEM, user, temperature=0.1)
    if rc.get("fallback"):
        rc = {"reasoning": "Fallback root cause based on ML predicted fault and telemetry evidence."}
    rc = _normalize_root_cause(rc, state.get("ml_result", {}))
    return {**state, "root_cause": rc}


def planner_node(state: AgentState) -> AgentState:
    ml = state.get("ml_result", {})
    spares = get_spares_for_fault(ml.get("predicted_fault", ""), ml.get("predicted_stand", "mill_level"))
    user = json.dumps({
        "query": state["query"],
        "conversation_context": state.get("conversation_context", ""),
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
        plan = {}
    plan = _normalize_plan(plan, ml)
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
    if state.get("surface") == "copilot" and state.get("route", {}).get("intent") != "report_generation":
        return _deterministic_copilot_reply(state)
    ml = state.get("ml_result", {})
    plan = state.get("maintenance_plan", {}) or {}
    rules = [r for r in state.get("physical_rules", []) if r.get("matched")]
    rag_docs = state.get("rag_context", [])
    mode = state.get("answer_mode", "concise")
    immediate_actions = plan.get("immediate_actions", []) if isinstance(plan, dict) else []
    action = _textify_action(immediate_actions[0]) if immediate_actions else "Review Steel Pilot recommendation."
    if mode == "concise":
        return (
            f"- **Diagnosis:** {ml.get('predicted_fault')} on {ml.get('asset_name')} ({ml.get('risk_level')} risk, score {ml.get('risk_score')}).\n"
            f"- **Why:** anomaly probability {ml.get('anomaly_probability')}; top evidence: {'; '.join(ml.get('evidence', [])[:3])}.\n"
            f"- **Rule check:** {rules[0].get('title') if rules else 'No strong physical rule triggered; inspect telemetry evidence.'}\n"
            f"- **Next action:** {action}\n"
            f"- **SOP evidence:** {', '.join([d.get('source','') for d in rag_docs[:3]]) or 'No SOP source retrieved.'}"
        )
    return _deterministic_detailed_report(state)


def report_node(state: AgentState) -> AgentState:
    user = json.dumps({
        "answer_mode": state.get("answer_mode", "concise"),
        "surface": state.get("surface", "dashboard"),
        "user_query": state["query"],
        "conversation_context": state.get("conversation_context", ""),
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
    lower_final = final.lower()
    copilot_surface = state.get("surface") == "copilot"
    report_requested = state.get("route", {}).get("intent") == "report_generation"
    final_word_count = len(final.split())
    if (
        final.startswith("Steel Pilot could not call")
        or (not copilot_surface and state.get("answer_mode") == "detailed" and "## diagnosis" not in lower_final and "**diagnosis**" not in lower_final)
        or (copilot_surface and not report_requested and ("## diagnosis" in lower_final or "## why it matters" in lower_final or "## cascading impact" in lower_final))
        or (copilot_surface and not report_requested and final_word_count > 220)
        or final.lstrip().startswith("{")
    ):
        final = _deterministic_report(state)
    return {**state, "final_answer": final}


def audit_node(state: AgentState) -> AgentState:
    ml = state.get("ml_result", {})
    plan = state.get("maintenance_plan", {}) or {}
    immediate = plan.get("immediate_actions", []) if isinstance(plan, dict) else []
    if isinstance(immediate, list) and immediate:
        first_action = _textify_action(immediate[0])
    elif immediate:
        first_action = _textify_action(immediate)
    else:
        first_action = "Review Steel Pilot recommendation."
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
    surface: str = "dashboard",
    conversation_context: str | None = None,
) -> dict[str, Any]:
    mode = "detailed" if str(answer_mode).lower().startswith("detail") else "concise"
    inputs: AgentState = {
        "query": query,
        "thread_id": thread_id,
        "row_index": row_index,
        "answer_mode": mode,
        "surface": surface,
        "conversation_context": conversation_context or "",
        "errors": [],
    }
    return app_graph.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
