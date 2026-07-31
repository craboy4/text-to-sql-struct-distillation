#!/usr/bin/env python3
"""Create a database-disjoint BIRD train execution-development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_HOLDOUT_DATABASES = (
    "hockey",
    "mondial_geo",
    "movie_3",
    "student_loan",
)
DEFAULT_LENGTH_EXCLUDED_DATABASES = ("works_cycles",)


class ValidationError(ValueError):
    """Raised when the source data cannot form a leak-free execution dev split."""


def fail(message: str) -> None:
    raise SystemExit(f"EXECUTION_DEV_FAILED: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_records(records: list[dict[str, Any]], seed: int, rows_per_database: int) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{seed}:{record['record_id']}".encode("ascii")).hexdigest(),
    )
    return ranked[:rows_per_database]


def load_source(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("record_id")
            source_record = record.get("source")
            messages = record.get("messages")
            roles = [message.get("role") for message in messages] if isinstance(messages, list) else []
            if not isinstance(record_id, int) or record_id in seen_ids:
                raise ValidationError(f"source line {line_number} has an invalid or duplicate record_id")
            if not isinstance(source_record, dict) or not all(
                isinstance(source_record.get(field), str) and source_record[field].strip()
                for field in ("db_id", "question", "gold_sql")
            ):
                raise ValidationError(f"source line {line_number} has incomplete BIRD source metadata")
            if roles != ["system", "user", "assistant"] or not all(
                isinstance(message.get("content"), str) for message in messages
            ):
                raise ValidationError(f"source line {line_number} has an invalid SFT message payload")
            seen_ids.add(record_id)
            records.append(record)
    if not records:
        raise ValidationError("source dataset is empty")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("BIRD/teacher_generation/teacher_sft_merged_two_stage.jsonl")
    )
    parser.add_argument("--database-root", type=Path, default=Path("BIRD/train/train_databases"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-database", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-databases", nargs="+", default=list(DEFAULT_HOLDOUT_DATABASES))
    parser.add_argument("--length-excluded-databases", nargs="+", default=list(DEFAULT_LENGTH_EXCLUDED_DATABASES))
    args = parser.parse_args()
    if args.rows_per_database < 1:
        fail("--rows-per-database must be positive")
    if args.output_dir.exists():
        fail(f"output directory already exists: {args.output_dir}")

    try:
        records = load_source(args.source)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        fail(str(error))

    holdout_databases = tuple(args.holdout_databases)
    length_excluded_databases = tuple(args.length_excluded_databases)
    if len(holdout_databases) != len(set(holdout_databases)):
        fail("holdout databases must be unique")
    if len(length_excluded_databases) != len(set(length_excluded_databases)):
        fail("length-excluded databases must be unique")
    if set(holdout_databases) & set(length_excluded_databases):
        fail("holdout and length-excluded databases must not overlap")
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_database[record["source"]["db_id"]].append(record)

    dev_records: list[dict[str, Any]] = []
    for db_id in holdout_databases:
        candidates = by_database.get(db_id, [])
        database_dir = args.database_root / db_id
        sqlite_path = database_dir / f"{db_id}.sqlite"
        if len(candidates) < args.rows_per_database:
            fail(f"database {db_id} has only {len(candidates)} source records")
        if not sqlite_path.is_file():
            fail(f"missing SQLite database: {sqlite_path}")
        dev_records.extend(sample_records(candidates, args.seed, args.rows_per_database))

    holdout_set = set(holdout_databases)
    length_excluded_set = set(length_excluded_databases)
    unknown_length_excluded = length_excluded_set - set(by_database)
    if unknown_length_excluded:
        fail(f"unknown length-excluded databases: {sorted(unknown_length_excluded)}")
    train_records = [
        record
        for record in records
        if record["source"]["db_id"] not in holdout_set | length_excluded_set
    ]
    if len(dev_records) != len({record["record_id"] for record in dev_records}):
        fail("sampled execution dev has duplicate record IDs")
    if any(record["source"]["db_id"] in holdout_set | length_excluded_set for record in train_records):
        fail("database-disjoint training split validation failed")

    args.output_dir.mkdir(parents=True)
    eval_root = args.output_dir / "eval_root"
    database_output = eval_root / "dev_databases"
    database_output.mkdir(parents=True)

    train_sft_path = args.output_dir / "train_sft_messages.jsonl"
    write_jsonl(train_sft_path, [{"messages": record["messages"]} for record in train_records])

    prompts: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    gold_lines: list[str] = []
    for index, record in enumerate(dev_records, 1):
        source_record = record["source"]
        db_id = source_record["db_id"]
        question_id = record["record_id"]
        prompts.append(
            {
                "example_index": index,
                "question_id": question_id,
                "db_id": db_id,
                "difficulty": "train_execution_dev",
                "messages": record["messages"][:2],
            }
        )
        examples.append(
            {
                "question_id": question_id,
                "db_id": db_id,
                "question": source_record["question"],
                "evidence": source_record.get("evidence", ""),
                "difficulty": "train_execution_dev",
            }
        )
        gold_lines.append(f"{source_record['gold_sql']}\t{db_id}")

    prompts_path = args.output_dir / "prompts.jsonl"
    write_jsonl(prompts_path, prompts)
    (eval_root / "mini_dev_sqlite.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (eval_root / "mini_dev_sqlite_gold.sql").write_text("\n".join(gold_lines) + "\n", encoding="utf-8")

    for db_id in holdout_databases:
        shutil.copytree(args.database_root / db_id, database_output / db_id)

    manifest = {
        "name": "BIRD train database-disjoint execution dev",
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "seed": args.seed,
        "rows_per_database": args.rows_per_database,
        "holdout_databases": list(holdout_databases),
        "length_excluded_databases": list(length_excluded_databases),
        "source_records": len(records),
        "training_records": len(train_records),
        "excluded_training_records": len(records) - len(train_records),
        "length_excluded_records": sum(len(by_database[db_id]) for db_id in length_excluded_databases),
        "execution_dev_records": len(dev_records),
        "train_sft_sha256": sha256_file(train_sft_path),
        "prompts_sha256": sha256_file(prompts_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
