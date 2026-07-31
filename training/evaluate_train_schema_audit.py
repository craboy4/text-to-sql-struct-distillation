#!/usr/bin/env python3
"""Evaluate schema-auditor table selection against held-out prompt labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_tables(response: str, candidates: list[str]) -> list[str] | None:
    match = re.search(r"\{.*\}", response.strip(), flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    values = payload.get("missing_tables") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return None
    canonical = {candidate.casefold(): candidate for candidate in candidates}
    selected = []
    for value in values:
        table = canonical.get(value.strip().casefold())
        if table is None:
            return None
        if table not in selected:
            selected.append(table)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = {record["audit_id"]: record for record in read_jsonl(args.prompts)}
    responses = {record["audit_id"]: record["response"] for record in read_jsonl(args.responses)}
    if set(prompts) != set(responses):
        raise SystemExit("SCHEMA_AUDITOR_EVAL_FAILED: prompt and response audit IDs differ")

    audits = []
    recall_hits = recall_total = controls = control_correct = exact_missing = missing_cases = valid = 0
    for audit_id, prompt in prompts.items():
        prediction = parse_tables(responses[audit_id], prompt["candidate_tables"])
        gold = set(prompt["gold_missing_tables"])
        if prediction is not None:
            valid += 1
        predicted = set(prediction or [])
        hits = len(gold & predicted)
        recall_hits += hits
        recall_total += len(gold)
        if gold:
            missing_cases += 1
            exact_missing += predicted == gold
        else:
            controls += 1
            control_correct += not predicted
        audits.append({"audit_id": audit_id, "gold_missing_tables": sorted(gold), "predicted_missing_tables": prediction, "valid_json": prediction is not None})
    payload = {
        "records": len(prompts),
        "valid_json_rate": round(valid / len(prompts), 6),
        "missing_table_cases": missing_cases,
        "missing_table_recall": round(recall_hits / recall_total, 6) if recall_total else None,
        "exact_missing_set_rate": round(exact_missing / missing_cases, 6) if missing_cases else None,
        "controls": controls,
        "control_no_expansion_rate": round(control_correct / controls, 6) if controls else None,
        "audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
