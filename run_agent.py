from __future__ import annotations

import argparse
import json

from src.agentic.graph import answer_maintenance_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Steel Pilot V2 agent workflow from CLI")
    parser.add_argument("query", help="Maintenance query")
    parser.add_argument("--thread-id", default="steel-pilot-v2-cli")
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--answer-mode", choices=["concise", "detailed"], default="concise")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = answer_maintenance_query(
        args.query,
        thread_id=args.thread_id,
        row_index=args.row_index,
        answer_mode=args.answer_mode,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result.get("final_answer", "No answer generated."))


if __name__ == "__main__":
    main()
