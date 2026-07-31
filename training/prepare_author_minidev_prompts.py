#!/usr/bin/env python3
"""Wrap the upstream Mini-Dev prompt artifact for the local API runner.

The ``prompt`` string is copied verbatim.  This script only supplies the
record shape expected by ``run_minidev_api.py``; it does not build or alter a
prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_QUESTIONS = 500


def fail(message: str) -> None:
    raise SystemExit(f"AUTHOR_MINIDEV_PROMPTS_FAILED: {message}")


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
    with args.source.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                question_id = item["question_id"]
                db_id = item["db_id"]
                prompt = item["prompt"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                fail(f"source line {line_number} is malformed: {error}")
            if not isinstance(question_id, int) or not isinstance(db_id, str) or not isinstance(prompt, str):
                fail(f"source line {line_number} has invalid question_id, db_id, or prompt")
            records.append(
                {
                    "example_index": len(records) + 1,
                    "question_id": question_id,
                    "db_id": db_id,
                    "difficulty": item.get("difficulty"),
                    "messages": [{"role": "user", "content": prompt}],
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_characters": len(prompt),
                }
            )
    if len(records) != args.expected_questions:
        fail(f"expected {args.expected_questions} source prompts, found {len(records)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"source": str(args.source), "output": str(args.output), "prompts": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
