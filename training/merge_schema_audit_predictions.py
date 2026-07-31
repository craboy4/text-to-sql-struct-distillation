#!/usr/bin/env python3
"""Replace baseline predictions only where an audited rerun yielded an SQL block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_minidev_inference import SQL_BLOCK


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_sql(response: str) -> str | None:
    match = SQL_BLOCK.search(response)
    return match.group(1).strip() if match and match.group(1).strip() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--expanded-prompts", type=Path, required=True)
    parser.add_argument("--expanded-responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = read_jsonl(args.baseline_predictions)
    prompts = {record["audit_id"]: record for record in read_jsonl(args.expanded_prompts)}
    responses = {record["audit_id"]: record["response"] for record in read_jsonl(args.expanded_responses)}
    if set(prompts) != set(responses):
        raise SystemExit("SCHEMA_AUDIT_MERGE_FAILED: prompt and response audit IDs differ")
    replacement = {}
    for audit_id, prompt in prompts.items():
        sql = extract_sql(responses[audit_id])
        if sql is not None:
            replacement[prompt["question_id"]] = sql
    output = []
    replaced = 0
    for record in baseline:
        updated = dict(record)
        sql = replacement.get(record["question_id"])
        if sql is not None:
            updated["sql"] = sql
            replaced += 1
        output.append(updated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in output:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "baseline_predictions": len(baseline), "expanded_prompts": len(prompts), "sql_replacements": replaced}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
