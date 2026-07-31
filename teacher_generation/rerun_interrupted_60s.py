from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any

from generate_teacher_data import execute_sql, unordered_result_equivalent


def needs_rerun(record: dict[str, Any]) -> bool:
    return (
        record["teacher_execution"].get("error") == "interrupted"
        or record["gold_execution"].get("error") == "interrupted"
    )


def rerun_record(record: dict[str, Any], database_root: Path, timeout_seconds: float, max_rows: int) -> tuple[int, dict[str, Any]]:
    db_id = record["source"]["db_id"]
    sqlite_path = database_root / db_id / f"{db_id}.sqlite"
    previous_teacher = record["teacher_execution"]
    previous_gold = record["gold_execution"]
    teacher_execution = (
        execute_sql(sqlite_path, record["parsed"]["sql"], max_rows, timeout_seconds)
        if previous_teacher.get("error") == "interrupted"
        else previous_teacher
    )
    gold_execution = (
        execute_sql(sqlite_path, record["source"]["gold_sql"], max_rows, timeout_seconds)
        if previous_gold.get("error") == "interrupted"
        else previous_gold
    )
    equivalent = unordered_result_equivalent(teacher_execution, gold_execution)
    return record["record_id"], {
        "teacher_execution": teacher_execution,
        "gold_execution": gold_execution,
        "execution_equivalent_unordered": equivalent,
        "ready_for_sft": bool(record["parsed"]["format_ok"] and teacher_execution["ok"] and equivalent),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run only 10-second-interrupted BIRD SQL executions at a longer timeout.")
    parser.add_argument("--input", type=Path, default=Path("BIRD/teacher_generation/teacher_full_output_contract.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("BIRD/teacher_generation/teacher_full_output_contract_60s.jsonl"))
    parser.add_argument("--database-root", type=Path, default=Path("BIRD/train/train_databases"))
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-result-rows", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    candidates: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if needs_rerun(record):
                candidates.append(record)

    updates: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(rerun_record, record, args.database_root, args.timeout_seconds, args.max_result_rows) for record in candidates]
        for future in concurrent.futures.as_completed(futures):
            record_id, update = future.result()
            updates[record_id] = update
            print(f"record_id={record_id} equivalent={update['execution_equivalent_unordered']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open(encoding="utf-8") as source, temporary_output.open("w", encoding="utf-8") as destination:
        for line in source:
            record = json.loads(line)
            update = updates.get(record["record_id"])
            if update:
                record["execution_before_60s"] = {
                    "timeout_seconds": 10.0,
                    "teacher_execution": record["teacher_execution"],
                    "gold_execution": record["gold_execution"],
                    "execution_equivalent_unordered": record["execution_equivalent_unordered"],
                }
                record.update(update)
                record["execution_timeout_seconds"] = args.timeout_seconds
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_output, args.output)
    print(f"rerun_records={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
