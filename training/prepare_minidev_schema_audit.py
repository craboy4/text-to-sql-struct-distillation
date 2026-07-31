#!/usr/bin/env python3
"""Build schema-auditor prompts for retrieved Mini-Dev SchemaPacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_minidev_inference import DEFAULT_MINIDEV_ROOT, load_column_descriptions
from prepare_minidev_schemapack import load_database_schema, render_schema
from prepare_train_schema_audit_pilot import AUDITOR_SYSTEM, candidate_catalog, user_prompt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prompts", type=Path, required=True)
    parser.add_argument("--ranking-source-prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    parser.add_argument("--candidate-tables", type=int, default=12)
    args = parser.parse_args()
    if args.candidate_tables < 1:
        raise SystemExit("MINIDEV_SCHEMA_AUDIT_PREP_FAILED: --candidate-tables must be positive")
    source = load_jsonl(args.source_prompts)
    if len(source) != 500:
        raise SystemExit(f"MINIDEV_SCHEMA_AUDIT_PREP_FAILED: expected 500 source prompts, found {len(source)}")
    ranking_by_index: dict[int, list[str]] = {}
    if args.ranking_source_prompts is not None:
        ranking_source = load_jsonl(args.ranking_source_prompts)
        if len(ranking_source) != len(source):
            raise SystemExit("MINIDEV_SCHEMA_AUDIT_PREP_FAILED: ranking source must have the same record count")
        for prompt in ranking_source:
            ranking = prompt.get("selection", {}).get("ranked_tables")
            if isinstance(ranking, list) and all(isinstance(table, str) for table in ranking):
                ranking_by_index[prompt["example_index"]] = ranking
    schemas: dict[str, Any] = {}
    records = []
    for prompt in source:
        selection = prompt.get("selection")
        if not isinstance(selection, dict) or selection.get("mode") != "retrieved":
            continue
        db_id = prompt.get("db_id")
        if not isinstance(db_id, str):
            raise SystemExit("MINIDEV_SCHEMA_AUDIT_PREP_FAILED: source prompt lacks db_id")
        if db_id not in schemas:
            database_dir = args.minidev_root / "dev_databases" / db_id
            schemas[db_id] = load_database_schema(
                database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir)
            )
        schema = schemas[db_id]
        selected_columns = {table: set(columns) for table, columns in selection["selected_columns"].items()}
        selected = {table.casefold() for table in selected_columns}
        ranked = ranking_by_index.get(prompt["example_index"], selection.get("ranked_tables"))
        if not isinstance(ranked, list) or not all(isinstance(table, str) for table in ranked):
            raise SystemExit("MINIDEV_SCHEMA_AUDIT_PREP_FAILED: source prompt lacks ranked retrieval tables")
        candidates = [table for table in ranked if table.casefold() not in selected][: args.candidate_tables]
        if not candidates:
            continue
        user = prompt["messages"][1]["content"]
        task_marker = "Task\n"
        evidence_marker = "\n\nEvidence\n"
        schema_marker = "\n\nFocused SQLite schema context with column meanings\n"
        if not (user.startswith(task_marker) and evidence_marker in user and schema_marker in user):
            raise SystemExit("MINIDEV_SCHEMA_AUDIT_PREP_FAILED: unexpected source user prompt contract")
        task, remainder = user[len(task_marker) :].split(evidence_marker, 1)
        evidence, _ = remainder.split(schema_marker, 1)
        records.append(
            {
                "audit_id": prompt["example_index"],
                "source_example_index": prompt["example_index"],
                "question_id": prompt["question_id"],
                "db_id": db_id,
                "messages": [
                    {"role": "system", "content": AUDITOR_SYSTEM},
                    {
                        "role": "user",
                        "content": user_prompt(
                            task,
                            evidence if evidence != "No additional evidence." else "",
                            render_schema(schema.columns_by_table, schema.foreign_keys, selected_columns),
                            candidate_catalog(schema, candidates),
                        ),
                    },
                ],
                "candidate_tables": candidates,
                "selected_tables": selection["selected_tables"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(records), "source_prompts": len(source)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
