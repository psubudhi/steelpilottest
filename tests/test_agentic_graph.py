from __future__ import annotations

import unittest

from src.agentic.graph import (
    _deterministic_report,
    _is_probably_in_domain,
    _normalize_plan,
)


class AgenticGraphTests(unittest.TestCase):
    def test_active_context_allows_follow_up_query(self) -> None:
        self.assertTrue(_is_probably_in_domain("What should we inspect first?", has_active_context=True))
        self.assertFalse(_is_probably_in_domain("Write a poem about rain", has_active_context=True))

    def test_normalize_plan_uses_raw_text_when_json_shape_is_bad(self) -> None:
        ml = {"predicted_fault": "bearing_fault", "risk_level": "high"}
        plan = _normalize_plan({"raw_text": "Inspect lubrication flow\nCheck bearing temperature"}, ml)
        self.assertIn("Inspect lubrication flow", plan["immediate_actions"])
        self.assertTrue(plan["safety_notes"])

    def test_detailed_report_has_sections(self) -> None:
        state = {
            "answer_mode": "detailed",
            "ml_result": {
                "predicted_fault": "bearing_fault",
                "asset_name": "Stand 3 Work Roll",
                "risk_level": "high",
                "risk_score": 72,
                "anomaly_probability": 0.91,
                "predicted_rul_band": "24-72h",
                "evidence": ["Torque rising", "Power rising"],
            },
            "maintenance_plan": {
                "immediate_actions": ["Inspect lubrication circuit"],
                "short_term_actions": ["Monitor bearing temperature"],
                "planned_actions": ["Schedule bearing inspection"],
            },
            "physical_rules": [{"matched": True, "title": "Torque and power rising together", "severity": "high", "explanation": "Mechanical load increase likely."}],
            "cascading_impact": {"primary_stand": "stand_3", "cascading_risk": "medium", "cascading_risk_score": 48},
            "rag_context": [{"source": "bearing_fault_sop.md", "retrieval_mode": "keyword_fallback"}],
            "root_cause": {"probable_root_causes": ["bearing wear"], "reasoning": "Telemetry aligns with mechanical load increase.", "uncertainty_notes": "Validate with inspection."},
        }
        report = _deterministic_report(state)
        self.assertIn("## Diagnosis", report)
        self.assertIn("## Cascading Impact", report)
        self.assertIn("## Root Cause Notes", report)


if __name__ == "__main__":
    unittest.main()
