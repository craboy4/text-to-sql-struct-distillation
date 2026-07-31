#!/usr/bin/env python3
"""Adapt question_id/SQL JSONL predictions for upstream Mini-Dev EX evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DELIMITER = "\t----- bird -----\t"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-prompts", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"AUTHOR_EX_ADAPT_FAILED: refusing to overwrite {args.output}")
    source = read_jsonl(args.upstream_prompts)
    predictions = read_jsonl(args.predictions)
    if len(source) != 500:
        raise SystemExit(f"AUTHOR_EX_ADAPT_FAILED: expected 500 upstream prompts, found {len(source)}")

    by_question_id: dict[int, str] = {}
    for line_number, prediction in enumerate(predictions, start=1):
        question_id, sql = prediction.get("question_id"), prediction.get("sql")
        if not isinstance(question_id, int) or not isinstance(sql, str):
            raise SystemExit(f"AUTHOR_EX_ADAPT_FAILED: prediction line {line_number} needs question_id and sql")
        if question_id in by_question_id:
            raise SystemExit(f"AUTHOR_EX_ADAPT_FAILED: duplicate question_id {question_id}")
        by_question_id[question_id] = sql

    result: dict[str, str] = {}
    missing: list[int] = []
    for index, prompt in enumerate(source):
        question_id, db_id = prompt.get("question_id"), prompt.get("db_id")
        if not isinstance(question_id, int) or not isinstance(db_id, str):
            raise SystemExit(f"AUTHOR_EX_ADAPT_FAILED: upstream prompt index {index} needs question_id and db_id")
        sql = by_question_id.get(question_id)
        if sql is None:
            missing.append(question_id)
            sql = " "
        result[str(index)] = f"{sql}{DELIMITER}{db_id}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"upstream_rows": len(source), "predictions": len(predictions), "missing_question_ids": missing}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
