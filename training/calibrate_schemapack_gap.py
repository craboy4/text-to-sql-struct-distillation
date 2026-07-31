#!/usr/bin/env python3
"""Calibrate a selective extra-table SchemaPack expansion on BIRD train only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from calibrate_schemapack_tables import DEFAULT_DATABASES, DEFAULT_TRAIN, gold_columns, gold_tables, load_examples
from prepare_minidev_inference import load_column_descriptions
from prepare_minidev_schemapack import DEFAULT_EMBEDDING_MODEL, QwenEmbedder, load_database_schema, select_schema


def required_schema_covered(selection: dict[str, Any], tables: set[str], columns: set[tuple[str, str]]) -> tuple[bool, bool]:
    selected_tables = {table.casefold() for table in selection["selected_tables"]}
    selected_columns = {
        (table.casefold(), column.casefold())
        for table, names in selection["selected_columns"].items()
        for column in names
    }
    table_covered = tables <= selected_tables
    return table_covered, table_covered and columns <= selected_columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--databases", type=Path, default=DEFAULT_DATABASES)
    parser.add_argument("--embedding-snapshots", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--schema-char-budget", type=int, default=9000)
    parser.add_argument("--top-tables", type=int, default=6)
    parser.add_argument("--expanded-top-tables", type=int, default=7)
    parser.add_argument("--columns-per-table", type=int, default=16)
    parser.add_argument("--fallback-full-schema-chars", type=int, default=15000)
    parser.add_argument("--evidence-column-boost", type=float, default=0.05)
    parser.add_argument("--max-trigger-rate", type=float, default=0.35)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()
    if (
        min(args.schema_char_budget, args.top_tables, args.expanded_top_tables, args.columns_per_table, args.embedding_batch_size) < 1
        or args.expanded_top_tables <= args.top_tables
        or not 0 < args.max_trigger_rate <= 1
        or args.evidence_column_boost < 0
    ):
        raise SystemExit("SCHEMAPACK_GAP_CALIBRATION_FAILED: invalid numeric arguments")

    examples = load_examples(args.train, args.limit)
    schemas: dict[str, Any] = {}
    caches: dict[str, dict[str, Any]] = defaultdict(dict)
    examples_by_db: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        examples_by_db[example["db_id"]].append(example)
    embedder = QwenEmbedder(args.embedding_snapshots, args.embedding_batch_size)
    for db_id, database_examples in examples_by_db.items():
        database_dir = args.databases / db_id
        schemas[db_id] = load_database_schema(database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir))
        queries = [example["question"] + "\n" + example["evidence"] for example in database_examples]
        caches[db_id]["query_embeddings"] = dict(zip(queries, embedder.encode(queries)))

    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        schema = schemas[example["db_id"]]
        query = example["question"] + "\n" + example["evidence"]
        common = dict(
            schema_char_budget=args.schema_char_budget,
            columns_per_table=args.columns_per_table,
            fallback_full_schema_chars=args.fallback_full_schema_chars,
            evidence=example["evidence"],
            evidence_column_boost=args.evidence_column_boost,
        )
        _, base = select_schema(schema, query, embedder, caches[example["db_id"]], top_tables=args.top_tables, **common)
        _, expanded = select_schema(
            schema, query, embedder, caches[example["db_id"]], top_tables=args.expanded_top_tables, **common
        )
        required_tables = {table.casefold() for table in gold_tables(example["SQL"], set(schema.columns_by_table))}
        required_columns = {
            (table.casefold(), column.casefold())
            for table, column in gold_columns(example["SQL"], schema, {table for table in schema.columns_by_table if table.casefold() in required_tables})
        }
        base_table, base_schema = required_schema_covered(base, required_tables, required_columns)
        expanded_table, expanded_schema = required_schema_covered(expanded, required_tables, required_columns)
        gap = base.get("boundary_table_score_gap")
        records.append(
            {
                "gap": float(gap) if isinstance(gap, (float, int)) else None,
                "eligible": base["mode"] == "retrieved" and expanded["mode"] == "retrieved" and gap is not None,
                "base_table": base_table,
                "base_schema": base_schema,
                "expanded_table": expanded_table,
                "expanded_schema": expanded_schema,
                "base_characters": base["schema_characters"],
                "expanded_characters": expanded["schema_characters"],
            }
        )
        if index % 100 == 0 or index == len(examples):
            print(f"calibrated={index}/{len(examples)}", flush=True)

    eligible_gaps = sorted(record["gap"] for record in records if record["eligible"])
    if not eligible_gaps:
        raise SystemExit("SCHEMAPACK_GAP_CALIBRATION_FAILED: no eligible retrieved examples")
    thresholds = sorted({eligible_gaps[int((len(eligible_gaps) - 1) * quantile)] for quantile in np.linspace(0, 1, 21)})
    candidates = []
    for threshold in thresholds:
        triggered = [record["eligible"] and record["gap"] <= threshold for record in records]
        hybrid = [record["expanded_schema"] if use_expanded else record["base_schema"] for record, use_expanded in zip(records, triggered)]
        hybrid_tables = [record["expanded_table"] if use_expanded else record["base_table"] for record, use_expanded in zip(records, triggered)]
        hybrid_chars = [record["expanded_characters"] if use_expanded else record["base_characters"] for record, use_expanded in zip(records, triggered)]
        candidates.append(
            {
                "boundary_table_score_threshold": round(float(threshold), 8),
                "triggered_examples": sum(triggered),
                "trigger_rate": round(sum(triggered) / len(records), 6),
                "exact_schema_coverage": round(sum(hybrid) / len(records), 6),
                "exact_table_coverage": round(sum(hybrid_tables) / len(records), 6),
                "mean_schema_characters": round(sum(hybrid_chars) / len(records), 2),
            }
        )
    permitted = [candidate for candidate in candidates if candidate["trigger_rate"] <= args.max_trigger_rate]
    if not permitted:
        raise SystemExit("SCHEMAPACK_GAP_CALIBRATION_FAILED: no threshold fits max trigger rate")
    permitted.sort(
        key=lambda item: (-item["exact_schema_coverage"], -item["exact_table_coverage"], item["mean_schema_characters"], item["trigger_rate"])
    )
    base_schema = sum(record["base_schema"] for record in records) / len(records)
    base_tables = sum(record["base_table"] for record in records) / len(records)
    payload = {
        "split": "BIRD train filtered deterministic SHA-256 modulo-5 sample",
        "examples": len(records),
        "strategy": "selectively expand from top-k to top-(k+1) tables when the first retrieval boundary is ambiguous",
        "base": {
            "exact_schema_coverage": round(base_schema, 6),
            "exact_table_coverage": round(base_tables, 6),
            "mean_schema_characters": round(sum(record["base_characters"] for record in records) / len(records), 2),
        },
        "parameters": {
            "schema_char_budget": args.schema_char_budget,
            "top_tables": args.top_tables,
            "expanded_top_tables": args.expanded_top_tables,
            "columns_per_table": args.columns_per_table,
            "fallback_full_schema_chars": args.fallback_full_schema_chars,
            "evidence_column_boost": args.evidence_column_boost,
            "max_trigger_rate": args.max_trigger_rate,
        },
        "selected": permitted[0],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
