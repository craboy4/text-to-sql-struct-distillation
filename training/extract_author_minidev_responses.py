#!/usr/bin/env python3
"""Extract SQL generated for upstream Mini-Dev prompts into evaluator inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_QUESTIONS = 500
XML_SQL = re.compile(r"<sql>\s*(.*?)\s*</sql>", flags=re.IGNORECASE | re.DOTALL)
SQL_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", flags=re.IGNORECASE | re.DOTALL)
RAW_SQL = re.compile(r"(?is)\b(?:WITH|SELECT)\b.*")


def fail(message: str) -> None:
    raise SystemExit(f"AUTHOR_MINIDEV_EXTRACT_FAILED: {message}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    fail(f"{path} line {line_number} is invalid JSON: {error}")
    return records


def extract_sql(response: str) -> str | None:
    match = XML_SQL.search(response)
    if match:
        return match.group(1).strip()
    match = SQL_BLOCK.search(response)
    if match:
        return match.group(1).strip()
    match = RAW_SQL.search(response)
    return match.group(0).strip() if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prompts", type=Path, required=True, help="Upstream mini_dev_prompt.jsonl")
    parser.add_argument("--responses", type=Path, required=True, help="run_minidev_api.py JSONL output")
    parser.add_argument("--predictions", type=Path, required=True, help="question_id/SQL JSONL for local evaluator")
    parser.add_argument("--author-predictions", type=Path, required=True, help="index/SQL JSON for upstream evaluator")
    parser.add_argument("--expected-questions", type=int, default=EXPECTED_QUESTIONS)
    args = parser.parse_args()

    if args.expected_questions < 1:
        fail("--expected-questions must be positive")
    if not args.source_prompts.is_file() or not args.responses.is_file():
        fail("source prompts and responses must exist")
    if args.predictions.exists() or args.author_predictions.exists():
        fail("refusing to overwrite existing prediction output")

    source = load_jsonl(args.source_prompts)
    if len(source) != args.expected_questions:
        fail(f"expected {args.expected_questions} source prompts, found {len(source)}")
    required = ("question_id", "db_id")
    if any(not all(key in item for key in required) for item in source):
        fail("source prompt records require question_id and db_id")

    sql_by_question_id: dict[int, str] = {}
    for line_number, record in enumerate(load_jsonl(args.responses), start=1):
        question_id = record.get("question_id")
        response = record.get("response")
        if not isinstance(question_id, int) or not isinstance(response, str):
            fail(f"response line {line_number} lacks integer question_id or text response")
        sql = extract_sql(response)
        if not sql:
            fail(f"response line {line_number} has no SQL code block or SELECT/WITH query")
        prior = sql_by_question_id.get(question_id)
        if prior is not None and prior != sql:
            fail(f"question_id {question_id} has incompatible repeated SQL responses")
        sql_by_question_id[question_id] = sql

    missing = [item["question_id"] for item in source if item["question_id"] not in sql_by_question_id]
    if missing:
        fail(f"missing responses for {len(missing)} prompts; first question_ids={missing[:10]}")

    canonical: dict[int, dict[str, Any]] = {}
    author_predictions: dict[str, str] = {}
    for index, item in enumerate(source):
        question_id = item["question_id"]
        sql = sql_by_question_id[question_id]
        record = {"question_id": question_id, "sql": sql}
        prior = canonical.setdefault(question_id, record)
        if prior["sql"] != sql:
            fail(f"question_id {question_id} maps to incompatible source predictions")
        author_predictions[str(index)] = f"{sql}\t----- bird -----\t{item['db_id']}"

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as destination:
        for record in canonical.values():
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    with args.author_predictions.open("w", encoding="utf-8") as destination:
        json.dump(author_predictions, destination, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "responses": len(sql_by_question_id),
                "canonical_predictions": len(canonical),
                "author_prediction_rows": len(author_predictions),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
