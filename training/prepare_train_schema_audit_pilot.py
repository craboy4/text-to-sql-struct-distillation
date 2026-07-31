#!/usr/bin/env python3
"""Build a gold-hidden BIRD-train pilot for LinkAlign-style schema auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from calibrate_schemapack_tables import DEFAULT_DATABASES, DEFAULT_TRAIN, gold_tables, load_examples
from prepare_minidev_inference import load_column_descriptions
from prepare_minidev_schemapack import DEFAULT_EMBEDDING_MODEL, QwenEmbedder, load_database_schema, select_schema


AUDITOR_SYSTEM = """You are a SQLite schema auditor. Decide whether the current focused schema is missing any table required to answer the task. Only choose from the candidate tables. Do not write SQL, plans, explanations, or Markdown. Return exactly one JSON object: {\"missing_tables\": [\"exact_candidate_table_name\", ...]}. Return an empty list if the current schema is sufficient. Do not include a table merely because it is related; include it only when the task requires reading or joining it."""


def candidate_catalog(schema: Any, names: list[str]) -> str:
    entries = []
    for table in names:
        columns = schema.columns_by_table[table]
        rendered_columns = []
        for column in columns:
            detail = f": {column.description}" if column.description else ""
            rendered_columns.append(f"{column.name} ({column.type_name}){detail}")
        entries.append(f"TABLE {table}\n" + "\n".join(f"- {column}" for column in rendered_columns))
    return "\n\n".join(entries)


def user_prompt(question: str, evidence: str, selected_schema: str, candidates: str) -> str:
    return f"""Task
{question}

Evidence
{evidence or 'No additional evidence.'}

Current focused SQLite schema
{selected_schema}

Candidate tables not currently included
{candidates}
"""


def stable_key(record: dict[str, str]) -> str:
    return hashlib.sha256(f"{record['db_id']}\n{record['question']}\n{record['SQL']}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--databases", type=Path, default=DEFAULT_DATABASES)
    parser.add_argument("--embedding-snapshots", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--schema-char-budget", type=int, default=9000)
    parser.add_argument("--top-tables", type=int, default=6)
    parser.add_argument("--columns-per-table", type=int, default=16)
    parser.add_argument("--fallback-full-schema-chars", type=int, default=15000)
    parser.add_argument("--evidence-column-boost", type=float, default=0.05)
    parser.add_argument("--candidate-tables", type=int, default=12)
    parser.add_argument("--max-missing", type=int, default=40)
    parser.add_argument("--max-controls", type=int, default=40)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.schema_char_budget, args.top_tables, args.columns_per_table, args.candidate_tables, args.max_missing, args.max_controls, args.embedding_batch_size) < 1:
        raise SystemExit("TRAIN_SCHEMA_AUDIT_PREP_FAILED: numeric arguments must be positive")
    if args.fallback_full_schema_chars < 0 or args.evidence_column_boost < 0:
        raise SystemExit("TRAIN_SCHEMA_AUDIT_PREP_FAILED: schema threshold and boost must be non-negative")

    examples = load_examples(args.train, 0)
    schemas: dict[str, Any] = {}
    caches: dict[str, dict[str, Any]] = defaultdict(dict)
    by_db: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        by_db[example["db_id"]].append(example)
    embedder = QwenEmbedder(args.embedding_snapshots, args.embedding_batch_size)
    for db_id, rows in by_db.items():
        database_dir = args.databases / db_id
        schemas[db_id] = load_database_schema(database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir))
        queries = [row["question"] + "\n" + row["evidence"] for row in rows]
        caches[db_id]["query_embeddings"] = dict(zip(queries, embedder.encode(queries)))

    prepared: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        schema = schemas[example["db_id"]]
        query = example["question"] + "\n" + example["evidence"]
        focused, selection = select_schema(
            schema,
            query,
            embedder,
            caches[example["db_id"]],
            args.schema_char_budget,
            args.top_tables,
            args.columns_per_table,
            args.fallback_full_schema_chars,
            example["evidence"],
            args.evidence_column_boost,
        )
        if selection["mode"] != "retrieved":
            continue
        selected = {table.casefold() for table in selection["selected_tables"]}
        required = gold_tables(example["SQL"], set(schema.columns_by_table))
        missing = sorted(table for table in required if table.casefold() not in selected)
        candidates = [table for table in selection["ranked_tables"] if table.casefold() not in selected][: args.candidate_tables]
        if not candidates:
            continue
        prepared.append(
            {
                "source": example,
                "schema": schema,
                "focused": focused,
                "selection": selection,
                "gold_missing_tables": missing,
                "candidate_tables": candidates,
                "key": stable_key(example),
            }
        )
        if index % 100 == 0 or index == len(examples):
            print(f"prepared={index}/{len(examples)}", flush=True)

    misses = sorted((record for record in prepared if record["gold_missing_tables"]), key=lambda record: record["key"])[: args.max_missing]
    controls = sorted((record for record in prepared if not record["gold_missing_tables"]), key=lambda record: record["key"])[: args.max_controls]
    sample = sorted(misses + controls, key=lambda record: record["key"])
    if not misses or not controls:
        raise SystemExit("TRAIN_SCHEMA_AUDIT_PREP_FAILED: pilot requires both missing-table and control examples")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for audit_id, record in enumerate(sample, start=1):
            source = record["source"]
            destination.write(
                json.dumps(
                    {
                        "audit_id": audit_id,
                        "db_id": source["db_id"],
                        "messages": [
                            {"role": "system", "content": AUDITOR_SYSTEM},
                            {
                                "role": "user",
                                "content": user_prompt(
                                    source["question"],
                                    source["evidence"],
                                    record["focused"],
                                    candidate_catalog(record["schema"], record["candidate_tables"]),
                                ),
                            },
                        ],
                        "gold_missing_tables": record["gold_missing_tables"],
                        "candidate_tables": record["candidate_tables"],
                        "selected_tables": record["selection"]["selected_tables"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({"output": str(args.output.resolve()), "records": len(sample), "missing_table_cases": len(misses), "controls": len(controls)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
