#!/usr/bin/env python3
"""Attach a generated SQLPlan to matching train prompts for final SQL generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-prompts", type=Path, required=True)
    parser.add_argument("--plan-responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-plan-characters", type=int, default=4000)
    args = parser.parse_args()
    if args.max_plan_characters < 1:
        raise SystemExit("TRAIN_SQLPLAN_FINAL_PREP_FAILED: plan limit must be positive")
    prompts = {record["audit_id"]: record for record in read_jsonl(args.baseline_prompts)}
    responses = {record["audit_id"]: record["response"] for record in read_jsonl(args.plan_responses)}
    if set(prompts) != set(responses):
        raise SystemExit("TRAIN_SQLPLAN_FINAL_PREP_FAILED: prompt and plan response IDs differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for audit_id, prompt in prompts.items():
            plan = responses[audit_id].strip()
            if len(plan) > args.max_plan_characters:
                plan = plan[: args.max_plan_characters]
            user = (
                prompt["messages"][1]["content"]
                + "\n\nInternal SQLPlan proposal\n"
                + plan
                + "\n\nUse this proposal as a checklist, but validate it independently against the task, evidence, and schema. Return the required three blocks only."
            )
            record = dict(prompt)
            record["messages"] = [prompt["messages"][0], {"role": "user", "content": user}]
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(prompts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
