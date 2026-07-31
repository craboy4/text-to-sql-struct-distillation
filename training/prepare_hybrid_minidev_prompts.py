#!/usr/bin/env python3
"""Build Mini-Dev prompts that combine BIRD's layout with the local SQL contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_QUESTIONS = 500
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teacher_generation"))
from generate_teacher_data import SYSTEM_PROMPT  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"HYBRID_MINIDEV_PROMPTS_FAILED: {message}")


def user_prompt(item: dict[str, Any]) -> str:
    question = item.get("question")
    evidence = item.get("evidence")
    schema = item.get("schema")
    if not isinstance(question, str) or not isinstance(evidence, str) or not isinstance(schema, str):
        fail("source prompt requires text question, evidence, and schema")
    return f"""Task Overview:
You are a data science expert. Use the supplied SQLite schema, column meanings, and evidence to write the exact query requested by the question.

Database Engine:
SQLite

Database Schema:
{schema}

The schema defines tables, columns, primary keys, foreign keys, and relevant constraints.

Evidence:
{evidence.strip() or 'No additional evidence.'}

Question:
{question.strip()}

Instructions:
- Evidence is authoritative for stated mappings and constraints. Use only identifiers present in the schema.
- Return exactly the requested columns and preserve stored database values.
- Before finalizing the query, verify filters, joins, grouping, aggregation, ordering, limits, and tie behavior against the question.
- Follow the system response contract exactly: map schema items, state a short relational plan, then provide one executable read-only SQLite SELECT or WITH query inside the required <sql> block.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Upstream mini_dev_prompt.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=EXPECTED_QUESTIONS)
    args = parser.parse_args()
    if args.expected_questions < 1:
        fail("--expected-questions must be positive")
    if not args.source.is_file():
        fail(f"source prompt file does not exist: {args.source}")
    if args.output.exists():
        fail(f"refusing to overwrite existing output: {args.output}")

    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    with args.source.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                question_id = item["question_id"]
                db_id = item["db_id"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                fail(f"source line {line_number} is malformed: {error}")
            if not isinstance(question_id, int) or not isinstance(db_id, str) or question_id in seen_ids:
                fail(f"source line {line_number} has invalid or duplicate identifiers")
            user = user_prompt(item)
            records.append(
                {
                    "example_index": len(records) + 1,
                    "question_id": question_id,
                    "db_id": db_id,
                    "difficulty": item.get("difficulty"),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "prompt_strategy": "bird_layout_plus_structured_sql_contract",
                    "system_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                    "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
                    "user_characters": len(user),
                }
            )
            seen_ids.add(question_id)
    if len(records) != args.expected_questions:
        fail(f"expected {args.expected_questions} source prompts, found {len(records)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "prompts": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
