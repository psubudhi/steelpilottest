from __future__ import annotations

from typing import Any, Callable

import pandas as pd


GOLD_TESTS: list[dict[str, Any]] = [
    {"id": "G01", "category": "Diagnosis", "query": "Diagnose the active alarm and give the next action.", "expects": ["ml", "risk", "actions", "trace"]},
    {"id": "G02", "category": "Priority", "query": "Which stand should be maintained first and why?", "expects": ["priority", "risk", "actions"]},
    {"id": "G03", "category": "Risk", "query": "Explain why the current alert is critical.", "expects": ["evidence", "risk", "trace"]},
    {"id": "G04", "category": "Urgency", "query": "What is the proxy RUL urgency and what should the shift engineer do?", "expects": ["rul", "actions"]},
    {"id": "G05", "category": "Physical Rules", "query": "Is this more likely mechanical or electrical?", "expects": ["rules", "root_cause", "evidence"]},
    {"id": "G06", "category": "Cascading", "query": "Can this issue affect upstream or downstream stands?", "expects": ["cascade", "evidence"]},
    {"id": "G07", "category": "SOP", "query": "Which SOP evidence supports the maintenance action?", "expects": ["rag", "actions"]},
    {"id": "G08", "category": "Report", "query": "Generate a shift handover style maintenance summary.", "expects": ["report", "risk", "actions"]},
    {"id": "G09", "category": "History", "query": "What similar historical cases were found?", "expects": ["similar", "evidence"]},
    {"id": "G10", "category": "Guardrail", "query": "What is the capital of France?", "expects": ["guardrail"]},
]


def score_agent_result(test: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("final_answer", ""))
    lower_answer = answer.lower()
    ml = result.get("ml_result", {}) or {}
    checks = {
        "final_answer_present": bool(answer.strip()),
        "ml_context_used": bool(ml),
        "risk_included": bool(ml.get("risk_level")) or "risk" in lower_answer,
        "rul_included": bool(ml.get("predicted_rul_band")) or "rul" in lower_answer or "urgency" in lower_answer,
        "evidence_included": bool(ml.get("evidence")) or "evidence" in lower_answer or "z-score" in lower_answer,
        "rag_citations_present": bool(result.get("rag_context")) or "sop" in lower_answer,
        "actions_present": bool(result.get("maintenance_plan")) or any(k in lower_answer for k in ["action", "inspect", "check", "monitor"]),
        "priority_present": bool(result.get("priority_board")) or "priority" in lower_answer or "maintain first" in lower_answer,
        "physical_rules_present": bool(result.get("physical_rules")) or "rule" in lower_answer or "mechanical" in lower_answer or "electrical" in lower_answer,
        "cascade_present": bool(result.get("cascading_impact")) or "cascade" in lower_answer or "upstream" in lower_answer or "downstream" in lower_answer,
        "trace_present": bool(result.get("decision_trace")) or "decided" in lower_answer,
        "similar_cases_present": bool(ml.get("similar_historical_cases")) or "similar" in lower_answer,
        "report_style_present": any(k in lower_answer for k in ["diagnosis", "recommendation", "maintenance", "handover"]),
        "guardrail_present": bool(result.get("guardrail")) or "designed for" in lower_answer and "steel" in lower_answer,
    }
    expectation_map = {
        "ml": "ml_context_used",
        "risk": "risk_included",
        "rul": "rul_included",
        "evidence": "evidence_included",
        "rag": "rag_citations_present",
        "actions": "actions_present",
        "priority": "priority_present",
        "rules": "physical_rules_present",
        "cascade": "cascade_present",
        "trace": "trace_present",
        "similar": "similar_cases_present",
        "report": "report_style_present",
        "root_cause": "report_style_present",
        "guardrail": "guardrail_present",
    }
    expected = test.get("expects", [])
    expected_checks = [expectation_map[e] for e in expected if e in expectation_map]
    expected_score = sum(bool(checks.get(c)) for c in expected_checks) / max(1, len(expected_checks))
    overall_score = (0.30 * float(checks["final_answer_present"]) + 0.70 * expected_score) * 100
    return {
        "id": test.get("id"),
        "category": test.get("category"),
        "query": test.get("query"),
        "expected": ", ".join(expected),
        "score": round(overall_score, 1),
        "pass": bool(overall_score >= 70),
        **checks,
    }


def run_gold_tests(
    answer_func: Callable[..., dict[str, Any]],
    selected_ids: list[str] | None = None,
    max_tests: int | None = None,
    thread_prefix: str = "steel-pilot-eval",
    row_index: int | None = None,
) -> pd.DataFrame:
    tests = GOLD_TESTS
    if selected_ids:
        selected = set(selected_ids)
        tests = [t for t in tests if t["id"] in selected]
    if max_tests is not None:
        tests = tests[: int(max_tests)]
    rows = []
    for test in tests:
        result = answer_func(test["query"], thread_id=f"{thread_prefix}-{test['id']}", row_index=row_index, answer_mode="concise")
        rows.append(score_agent_result(test, result))
    return pd.DataFrame(rows)
