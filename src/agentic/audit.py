from __future__ import annotations

from typing import Any

import pandas as pd

from .runtime_store import runtime_store


class DigitalLogbook:

    def save_event(self, event: dict[str, Any]) -> dict[str, Any]:
        return runtime_store.save_logbook_event(event)

    def load_events(self, limit: int = 200) -> list[dict[str, Any]]:
        return runtime_store.load_logbook_events(limit=limit)

    def get_event(self, log_id: str) -> dict[str, Any] | None:
        return runtime_store.get_logbook_event(log_id)

    def update_status(self, log_id: str, status: str) -> None:
        runtime_store.update_log_status(log_id, status)

    def to_dataframe(self, limit: int = 200) -> pd.DataFrame:
        return runtime_store.logbook_display_df(limit=limit)

    def summarize(self) -> dict[str, Any]:
        return runtime_store.summarize_logbook()


digital_logbook = DigitalLogbook()
