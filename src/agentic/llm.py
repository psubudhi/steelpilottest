from __future__ import annotations

import ast
import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model=settings.openai_model, temperature=temperature)


def _fallback_text(system: str, user: str, exc: Exception | None = None) -> str:
    return (
        "Steel Pilot could not call the configured LLM at this moment, so it returned a deterministic fallback.\n\n"
        "Please check OPENAI_API_KEY, OPENAI_MODEL, and internet/API access. The ML, telemetry, rules, "
        "drift, logbook, and feedback layers can still run locally.\n\n"
        f"LLM error: {type(exc).__name__ if exc else 'unknown'}"
    )


def invoke_text(system: str, user: str, temperature: float = 0.1) -> str:
    try:
        llm = get_llm(temperature=temperature)
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(resp.content)
    except Exception as exc:
        return _fallback_text(system, user, exc)


def _json_candidates(text: str) -> list[str]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    candidates: list[str] = [cleaned]
    start_positions = [idx for idx, ch in enumerate(cleaned) if ch in "[{"]
    for start in start_positions:
        opener = cleaned[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        for end in range(start, len(cleaned)):
            char = cleaned[end]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    candidates.append(cleaned[start : end + 1].strip())
                    break
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def _parse_candidate(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {"items": parsed}
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
    except Exception:
        return None
    return None


def invoke_json(system: str, user: str, temperature: float = 0.0) -> dict[str, Any]:
    try:
        llm = get_llm(temperature=temperature)
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = str(resp.content).strip()
        for candidate in _json_candidates(text):
            parsed = _parse_candidate(candidate)
            if parsed is not None:
                return parsed
        return {"raw_text": str(resp.content)}
    except Exception as exc:
        return {"llm_error": f"{type(exc).__name__}: {exc}", "fallback": True}
