#!/usr/bin/env python3
"""Create a deterministic BIRD-train pilot for plan-then-SQL generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_minidev_inference import build_schema, build_user_prompt, clean_text, load_column_descriptions, load_sft_system_prompt


DEFAULT_TRAIN = Path("BIRD/train_filtered/train.jsonl")
DEFAULT_DATABASES = Path("BIRD/train/train_databases")
DEFAULT_SFT = Path("BIRD/teacher_generation/qwen_sft_messages.jsonl")

PLAN_SYSTEM = """You are the planning stage of a Text-to-SQL system. Ground every requested output, filter, aggregation, ordering rule, and join in the supplied SQLite schema and evidence. Return exactly one compact JSON object with these keys: tables, columns, joins, filters, aggregation, grouping, ordering_limit. Values must be arrays of exact schema identifiers or concise SQLite operations. Do not write SQL, Markdown, explanations, or any key other than these eight keys."""


def load_train(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("db_id", "question", "SQL")):
            raise ValueError(f"train line {line_number} lacks db_id, question, or SQL")
        rows.append({key: clean_text(row.get(key)) for key in ("db_id", "question", "evidence", "SQL")})
    return rows


def stable_key(row: dict[str, str]) -> str:
    return hashlib.sha256(f"{row['db_id']}\n{row['question']}\n{row['SQL']}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--databases", type=Path, default=DEFAULT_DATABASES)
    parser.add_argument("--sft-dataset", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--records", type=int, default=60)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-schema-characters", type=int, default=9000)
    args = parser.parse_args()
    if min(args.records, args.max_schema_characters) < 1 or args.start_offset < 0:
        raise SystemExit("TRAIN_SQLPLAN_PREP_FAILED: records and schema limit must be positive; offset must be non-negative")
    system = load_sft_system_prompt(args.sft_dataset)
    selected = []
    schema_cache: dict[str, str] = {}
    for row in sorted(load_train(args.train), key=stable_key):
        db_id = row["db_id"]
        schema = schema_cache.get(db_id)
        if schema is None:
            database_dir = args.databases / db_id
            schema = build_schema(database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir))
            schema_cache[db_id] = schema
        if len(schema) > args.max_schema_characters:
            continue
        if len(selected) < args.start_offset:
            selected.append((row, schema))
            continue
        selected.append((row, schema))
        if len(selected) == args.start_offset + args.records:
            break
    if len(selected) < args.start_offset + args.records:
        raise SystemExit(f"TRAIN_SQLPLAN_PREP_FAILED: only found {len(selected) - args.start_offset} schemas within budget after offset")
    selected = selected[args.start_offset :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "baseline_prompts.jsonl"
    plan_path = args.output_dir / "plan_prompts.jsonl"
    with baseline_path.open("w", encoding="utf-8") as baseline, plan_path.open("w", encoding="utf-8") as plans:
        for audit_id, (row, schema) in enumerate(selected, start=1):
            user = build_user_prompt(row["question"], row["evidence"], schema)
            metadata = {"audit_id": audit_id, "db_id": row["db_id"], "gold_sql": row["SQL"], "schema_characters": len(schema)}
            baseline.write(json.dumps({**metadata, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}, ensure_ascii=False) + "\n")
            plans.write(json.dumps({**metadata, "messages": [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": user}]}, ensure_ascii=False) + "\n")
    manifest = {
        "split": "BIRD train filtered, deterministic SHA-256 order, no Mini-Dev labels",
        "records": len(selected),
        "start_offset": args.start_offset,
        "max_schema_characters": args.max_schema_characters,
        "baseline_prompts": str(baseline_path.resolve()),
        "plan_prompts": str(plan_path.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
