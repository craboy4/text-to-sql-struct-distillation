#!/usr/bin/env python3
"""Apply positive schema-auditor choices to a subset of Mini-Dev prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_train_schema_audit import parse_tables
from prepare_minidev_inference import DEFAULT_MINIDEV_ROOT, load_column_descriptions
from prepare_minidev_schemapack import build_user_prompt, closure_tables, load_database_schema, render_schema


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_task_evidence(user: str) -> tuple[str, str]:
    task_marker = "Task\n"
    evidence_marker = "\n\nEvidence\n"
    schema_marker = "\n\nFocused SQLite schema context with column meanings\n"
    if not (user.startswith(task_marker) and evidence_marker in user and schema_marker in user):
        raise ValueError("unexpected source user prompt contract")
    task, remainder = user[len(task_marker) :].split(evidence_marker, 1)
    evidence, _ = remainder.split(schema_marker, 1)
    return task, "" if evidence == "No additional evidence." else evidence


def expanded_columns(schema: Any, selection: dict[str, Any], requested_tables: list[str]) -> dict[str, set[str]]:
    columns = {table: set(names) for table, names in selection["selected_columns"].items()}
    original_tables = set(columns)
    closed_tables = closure_tables(sorted(original_tables | set(requested_tables)), schema.foreign_keys)
    for table in closed_tables:
        columns.setdefault(table, set())
    for table in requested_tables:
        columns[table].update(column.name for column in schema.columns_by_table[table])
    for foreign_key in schema.foreign_keys:
        if foreign_key.source_table in closed_tables and foreign_key.target_table in closed_tables:
            columns[foreign_key.source_table].add(foreign_key.source_column)
            columns[foreign_key.target_table].add(foreign_key.target_column)
    for table in closed_tables:
        for column in schema.columns_by_table[table]:
            if column.primary_key_rank:
                columns[table].add(column.name)
    return columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prompts", type=Path, required=True)
    parser.add_argument("--audit-prompts", type=Path, required=True)
    parser.add_argument("--audit-responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    parser.add_argument("--max-expanded-schema-characters", type=int, default=15000)
    args = parser.parse_args()
    if args.max_expanded_schema_characters < 1:
        raise SystemExit("MINIDEV_SCHEMA_AUDIT_APPLY_FAILED: schema character limit must be positive")
    source = {record["example_index"]: record for record in read_jsonl(args.source_prompts)}
    audit_prompts = {record["audit_id"]: record for record in read_jsonl(args.audit_prompts)}
    responses = {record["audit_id"]: record["response"] for record in read_jsonl(args.audit_responses)}
    if set(audit_prompts) != set(responses):
        raise SystemExit("MINIDEV_SCHEMA_AUDIT_APPLY_FAILED: prompt and response audit IDs differ")
    schemas: dict[str, Any] = {}
    output: list[dict[str, Any]] = []
    invalid = no_expansion = budget_skipped = 0
    for audit_id, audit_prompt in audit_prompts.items():
        prediction = parse_tables(responses[audit_id], audit_prompt["candidate_tables"])
        if prediction is None:
            invalid += 1
            continue
        if not prediction:
            no_expansion += 1
            continue
        prompt = source.get(audit_prompt["source_example_index"])
        if prompt is None:
            raise SystemExit(f"MINIDEV_SCHEMA_AUDIT_APPLY_FAILED: missing source prompt for audit_id={audit_id}")
        db_id = prompt["db_id"]
        if db_id not in schemas:
            database_dir = args.minidev_root / "dev_databases" / db_id
            schemas[db_id] = load_database_schema(
                database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir)
            )
        schema = schemas[db_id]
        columns = expanded_columns(schema, prompt["selection"], prediction)
        focused = render_schema(schema.columns_by_table, schema.foreign_keys, columns)
        if len(focused) > args.max_expanded_schema_characters:
            budget_skipped += 1
            continue
        task, evidence = split_task_evidence(prompt["messages"][1]["content"])
        output.append(
            {
                "audit_id": audit_id,
                "example_index": prompt["example_index"],
                "question_id": prompt["question_id"],
                "db_id": db_id,
                "messages": [prompt["messages"][0], {"role": "user", "content": build_user_prompt(task, evidence, focused)}],
                "schema_audit": {
                    "selected_tables": prediction,
                    "base_schema_characters": prompt["selection"]["schema_characters"],
                    "expanded_schema_characters": len(focused),
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in output:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "expanded_prompts": len(output), "no_expansion": no_expansion, "invalid": invalid, "budget_skipped": budget_skipped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
