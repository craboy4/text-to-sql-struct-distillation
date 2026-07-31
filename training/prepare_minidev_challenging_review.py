#!/usr/bin/env python3
"""Add a one-pass schema/plan/SQL review only to challenging Mini-Dev prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path, expected_records: int | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise SystemExit(f"CHALLENGING_REVIEW_FAILED: invalid JSON on line {line_number}: {error}") from error
    if expected_records is not None and len(records) != expected_records:
        raise SystemExit(f"CHALLENGING_REVIEW_FAILED: expected {expected_records} records in {path}, found {len(records)}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = load_jsonl(args.prompts, 500)
    responses = load_jsonl(args.responses)
    response_by_question = {
        response["question_id"]: response["response"]
        for response in responses
        if isinstance(response.get("question_id"), int) and isinstance(response.get("response"), str)
    }
    if len(response_by_question) != len(responses):
        raise SystemExit("CHALLENGING_REVIEW_FAILED: responses must provide one non-empty response per question_id")

    reviewed = 0
    output: list[dict[str, Any]] = []
    for prompt in prompts:
        messages = prompt.get("messages")
        if not isinstance(messages, list) or [message.get("role") for message in messages] != ["system", "user"]:
            raise SystemExit("CHALLENGING_REVIEW_FAILED: prompts must contain system/user messages")
        record = dict(prompt)
        if prompt.get("difficulty") == "challenging":
            candidate = response_by_question.get(prompt.get("question_id"))
            if not candidate:
                raise SystemExit(f"CHALLENGING_REVIEW_FAILED: missing candidate for question_id={prompt.get('question_id')}")
            review_user = (
                messages[1]["content"]
                + "\n\nCandidate solution to audit\n"
                + candidate.strip()
                + "\n\nIndependently verify the candidate against the task, evidence, and supplied schema. "
                "Check every linked table/column, literal, join, filter, aggregation, ordering, and tie rule. "
                "Correct any semantic or SQL mistake you find. Return a complete replacement response using the required "
                "three blocks only; do not discuss the review."
            )
            record["messages"] = [messages[0], {"role": "user", "content": review_user}]
            selection = dict(record.get("selection", {}))
            selection["review_mode"] = "challenging_candidate_audit"
            record["selection"] = selection
            record["prompt_characters"] = len(messages[0]["content"]) + len(review_user)
            reviewed += 1
        output.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in output:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(output), "challenging_reviews": reviewed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
