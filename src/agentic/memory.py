from __future__ import annotations

from typing import Any

import pandas as pd

from .runtime_store import runtime_store


class FeedbackMemory:

    def save_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        return runtime_store.save_feedback(feedback)

    def load_feedback(self, limit: int = 200) -> list[dict[str, Any]]:
        df = runtime_store.feedback_display_df(limit=limit)
        return df.to_dict(orient="records")

    def to_dataframe(self, limit: int = 200) -> pd.DataFrame:
        return runtime_store.feedback_display_df(limit=limit)


feedback_memory = FeedbackMemory()
