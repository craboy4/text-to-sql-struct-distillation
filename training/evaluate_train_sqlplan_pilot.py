#!/usr/bin/env python3
"""Evaluate train-pilot SQL responses by SQLite result equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_minidev import execute_readonly_sql
from prepare_minidev_inference import SQL_BLOCK


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sql_from_response(response: str) -> str | None:
    match = SQL_BLOCK.search(response)
    return match.group(1).strip() if match and match.group(1).strip() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--databases", type=Path, default=Path("BIRD/train/train_databases"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=100000)
    args = parser.parse_args()
    prompts = {record["audit_id"]: record for record in read_jsonl(args.prompts)}
    responses = {record["audit_id"]: record["response"] for record in read_jsonl(args.responses)}
    if set(prompts) != set(responses):
        raise SystemExit("TRAIN_SQLPLAN_EVAL_FAILED: prompt and response IDs differ")
    correct = valid = 0
    rows = []
    for audit_id, prompt in prompts.items():
        database_path = args.databases / prompt["db_id"] / f"{prompt['db_id']}.sqlite"
        prediction = execute_readonly_sql(database_path, sql_from_response(responses[audit_id]), args.timeout_seconds, args.max_rows)
        gold = execute_readonly_sql(database_path, prompt["gold_sql"], args.timeout_seconds, args.max_rows)
        is_correct = prediction["status"] == "ok" and gold["status"] == "ok" and prediction["multiset"] == gold["multiset"]
        valid += int(prediction["status"] == "ok")
        correct += int(is_correct)
        rows.append({"audit_id": audit_id, "db_id": prompt["db_id"], "correct": bool(is_correct), "prediction_status": prediction["status"], "prediction_error": prediction.get("error")})
    payload = {"records": len(prompts), "execution_correct": correct, "ex": round(correct / len(prompts), 6), "valid_sql": valid, "valid_sql_rate": round(valid / len(prompts), 6), "per_record": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
