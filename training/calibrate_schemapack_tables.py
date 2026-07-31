#!/usr/bin/env python3
"""Choose SchemaPack retrieval parameters using BIRD-train table recall only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_minidev_inference import clean_text, load_column_descriptions
from prepare_minidev_schemapack import (
    DEFAULT_EMBEDDING_MODEL,
    QwenEmbedder,
    load_database_schema,
    select_schema,
)


DEFAULT_TRAIN = Path("BIRD/train_filtered/train.jsonl")
DEFAULT_DATABASES = Path("BIRD/train/train_databases")
TABLE_REFERENCE = re.compile(r'(?is)\b(?:from|join|update|into)\s+(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([a-zA-Z_][\w$]*))')
QUALIFIED_COLUMN = re.compile(r'(?is)(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([a-zA-Z_][\w$]*))\s*\.\s*(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([a-zA-Z_][\w$]*))')
STRING_LITERAL = re.compile(r"(?s)'(?:''|[^'])*'")
IDENTIFIER = re.compile(r"[a-zA-Z_][\w$]*")
SQL_WORDS = {
    "all", "and", "as", "asc", "avg", "between", "by", "case", "cast", "count", "cross", "desc", "distinct", "else",
    "end", "except", "exists", "from", "group", "having", "in", "inner", "intersect", "is", "join", "left", "like",
    "limit", "max", "min", "not", "null", "on", "or", "order", "outer", "right", "select", "sum", "then", "union",
    "using", "when", "where", "with",
}


def load_examples(path: Path, limit: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("db_id", "question", "SQL")):
                raise ValueError(f"train line {line_number} lacks db_id, question, or SQL")
            # A stable hash avoids using source order or Mini-Dev to choose parameters.
            identifier = f"{row['db_id']}\n{row['question']}\n{row['SQL']}"
            if int(hashlib.sha256(identifier.encode("utf-8")).hexdigest(), 16) % 5 != 0:
                continue
            records.append({key: clean_text(row.get(key)) for key in ("db_id", "question", "evidence", "SQL")})
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError("fixed train calibration split is empty")
    return records


def gold_tables(sql: str, known_tables: set[str]) -> set[str]:
    matches = {next(part for part in match.groups() if part is not None).casefold() for match in TABLE_REFERENCE.finditer(sql)}
    # Derived table names from CTEs can also follow FROM; retain physical tables only.
    table_by_name = {table.casefold(): table for table in known_tables}
    return {table_by_name[table] for table in matches if table in table_by_name}


def gold_columns(sql: str, schema: Any, required_tables: set[str]) -> set[tuple[str, str]]:
    """Extract physical SQL column references without accepting CTE aliases as tables."""
    table_by_name = {table.casefold(): table for table in schema.columns_by_table}
    aliases: dict[str, str] = {}
    for match in TABLE_REFERENCE.finditer(sql):
        raw_table = next(part for part in match.groups() if part is not None).casefold()
        if raw_table in table_by_name:
            aliases[raw_table] = table_by_name[raw_table]
            suffix = sql[match.end() :]
            alias_match = re.match(r'(?is)\s+(?:as\s+)?(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([a-zA-Z_][\w$]*))', suffix)
            if alias_match:
                raw_alias = next(part for part in alias_match.groups() if part is not None).casefold()
                if raw_alias not in SQL_WORDS:
                    aliases[raw_alias] = table_by_name[raw_table]
    for table in required_tables:
        aliases[table.casefold()] = table

    required: set[tuple[str, str]] = set()
    for match in QUALIFIED_COLUMN.finditer(sql):
        raw_owner = next(part for part in match.groups()[:4] if part is not None).casefold()
        raw_column = next(part for part in match.groups()[4:] if part is not None).casefold()
        table = aliases.get(raw_owner)
        if table is not None:
            columns = {column.name.casefold(): column.name for column in schema.columns_by_table[table]}
            if raw_column in columns:
                required.add((table, columns[raw_column]))

    masked = STRING_LITERAL.sub(" ", sql)
    per_column: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for table in required_tables:
        for column in schema.columns_by_table[table]:
            per_column[column.name.casefold()].append((table, column.name))
    for token in IDENTIFIER.findall(masked):
        key = token.casefold()
        if key not in SQL_WORDS and len(per_column.get(key, [])) == 1:
            required.add(per_column[key][0])
    return required


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--databases", type=Path, default=DEFAULT_DATABASES)
    parser.add_argument("--embedding-snapshots", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--schema-char-budgets", default="6000,7500,9000")
    parser.add_argument("--top-tables", default="4,6,8")
    parser.add_argument("--columns-per-table", default="8,10,12,16")
    parser.add_argument("--evidence-column-boosts", default="0,0.05,0.1,0.2")
    parser.add_argument("--value-match-boosts", default="0,0.05,0.1")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--detail-output", type=Path)
    args = parser.parse_args()

    budgets = [int(value) for value in args.schema_char_budgets.split(",")]
    table_limits = [int(value) for value in args.top_tables.split(",")]
    column_limits = [int(value) for value in args.columns_per_table.split(",")]
    evidence_boosts = [float(value) for value in args.evidence_column_boosts.split(",")]
    value_boosts = [float(value) for value in args.value_match_boosts.split(",")]
    if min(*budgets, *table_limits, *column_limits, args.embedding_batch_size) < 1 or min(*evidence_boosts, *value_boosts) < 0:
        raise SystemExit("SCHEMAPACK_CALIBRATION_FAILED: numeric arguments must be positive")

    examples = load_examples(args.train, args.limit)
    schemas: dict[str, Any] = {}
    caches: dict[str, dict[str, Any]] = defaultdict(dict)
    embedder = QwenEmbedder(args.embedding_snapshots, args.embedding_batch_size)
    configs = [
        (budget, table_limit, column_limit, evidence_boost, value_boost)
        for budget in budgets
        for table_limit in table_limits
        for column_limit in column_limits
        for evidence_boost in evidence_boosts
        for value_boost in value_boosts
    ]
    aggregate = {
        config: {
            "examples": 0,
            "covered_examples": 0,
            "gold_tables": 0,
            "covered_tables": 0,
            "covered_columns": 0,
            "exact_schema_examples": 0,
            "gold_columns": 0,
            "schema_characters": 0,
        }
        for config in configs
    }
    details: list[dict[str, Any]] = []
    examples_by_db: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        examples_by_db[example["db_id"]].append(example)

    for db_id, database_examples in examples_by_db.items():
        database_dir = args.databases / db_id
        sqlite_path = database_dir / f"{db_id}.sqlite"
        if not sqlite_path.is_file():
            raise ValueError(f"missing train database: {sqlite_path}")
        schemas[db_id] = load_database_schema(sqlite_path, load_column_descriptions(database_dir))
        queries = [example["question"] + "\n" + example["evidence"] for example in database_examples]
        vectors = embedder.encode(queries)
        caches[db_id]["query_embeddings"] = dict(zip(queries, vectors))

    for index, example in enumerate(examples, start=1):
        db_id = example["db_id"]
        schema = schemas[db_id]
        required = gold_tables(example["SQL"], set(schema.columns_by_table))
        required_columns = gold_columns(example["SQL"], schema, required)
        query = example["question"] + "\n" + example["evidence"]
        for config in configs:
            focused, selection = select_schema(
                schema, query, embedder, caches[db_id], *config[:3], evidence=example["evidence"],
                evidence_column_boost=config[3], value_match_boost=config[4]
            )
            selected = {table.casefold() for table in selection["selected_tables"]}
            normalized_tables = {table.casefold() for table in required}
            selected_columns = {
                (table.casefold(), column.casefold())
                for table, columns in selection["selected_columns"].items()
                for column in columns
            }
            normalized_columns = {(table.casefold(), column.casefold()) for table, column in required_columns}
            stats = aggregate[config]
            stats["examples"] += 1
            stats["covered_examples"] += normalized_tables <= selected
            stats["gold_tables"] += len(required)
            stats["covered_tables"] += len(normalized_tables & selected)
            stats["exact_schema_examples"] += normalized_tables <= selected and normalized_columns <= selected_columns
            stats["gold_columns"] += len(normalized_columns)
            stats["covered_columns"] += len(normalized_columns & selected_columns)
            stats["schema_characters"] += len(focused)
            if args.detail_output is not None:
                details.append(
                    {
                        "db_id": db_id,
                        "config": {
                            "schema_char_budget": config[0], "top_tables": config[1], "columns_per_table": config[2],
                            "evidence_column_boost": config[3],
                            "value_match_boost": config[4],
                        },
                        "mode": selection["mode"],
                        "full_schema_characters": selection["full_schema_characters"],
                        "schema_characters": selection["schema_characters"],
                        "selected_tables": len(selection["selected_tables"]),
                        "selected_columns": selection["selected_column_count"],
                        "exact_table_coverage": normalized_tables <= selected,
                        "exact_schema_coverage": normalized_tables <= selected and normalized_columns <= selected_columns,
                    }
                )
        if index % 100 == 0 or index == len(examples):
            print(f"calibrated={index}/{len(examples)}", flush=True)

    results = []
    for (budget, table_limit, column_limit, evidence_boost, value_boost), stats in aggregate.items():
        results.append(
            {
                "schema_char_budget": budget,
                "top_tables": table_limit,
                "columns_per_table": column_limit,
                "evidence_column_boost": evidence_boost,
                "value_match_boost": value_boost,
                "examples": stats["examples"],
                "exact_schema_coverage": round(stats["exact_schema_examples"] / stats["examples"], 6),
                "exact_table_coverage": round(stats["covered_examples"] / stats["examples"], 6),
                "column_recall": round(stats["covered_columns"] / stats["gold_columns"], 6),
                "table_recall": round(stats["covered_tables"] / stats["gold_tables"], 6),
                "mean_schema_characters": round(stats["schema_characters"] / stats["examples"], 2),
            }
        )
    results.sort(
        key=lambda row: (
            -row["exact_schema_coverage"],
            -row["column_recall"],
            -row["exact_table_coverage"],
            -row["table_recall"],
            row["mean_schema_characters"],
        )
    )
    payload = {
        "split": "BIRD train filtered deterministic SHA-256 modulo-5 sample",
        "examples": len(examples),
        "selection_rule": "maximize exact table-and-column coverage, then column/table recall, then minimize schema characters",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.detail_output is not None:
        args.detail_output.parent.mkdir(parents=True, exist_ok=True)
        with args.detail_output.open("w", encoding="utf-8") as destination:
            for detail in details:
                destination.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
