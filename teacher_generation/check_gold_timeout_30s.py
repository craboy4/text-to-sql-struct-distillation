from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from generate_teacher_data import execute_sql


SOURCE = Path("BIRD/teacher_generation/teacher_full_output_contract.jsonl")
DATABASE_ROOT = Path("BIRD/train/train_databases")
OUTPUT = Path("BIRD/teacher_generation/gold_timeout_30s_report.json")


def run(item: tuple[int, Path, str]) -> tuple[int, dict]:
    record_id, sqlite_path, sql = item
    return record_id, execute_sql(sqlite_path, sql, max_rows=1_000, timeout_seconds=30.0)


def main() -> None:
    candidates = []
    with SOURCE.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["gold_execution"].get("error") == "interrupted":
                db_id = record["source"]["db_id"]
                candidates.append((record["record_id"], DATABASE_ROOT / db_id / f"{db_id}.sqlite", record["source"]["gold_sql"]))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run, candidate) for candidate in candidates]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    report = {
        "timeout_seconds": 30.0,
        "candidate_count": len(candidates),
        "success_count": sum(result["ok"] for _, result in results),
        "still_interrupted_record_ids": sorted(record_id for record_id, result in results if result.get("error") == "interrupted"),
        "other_failure_record_ids": sorted(record_id for record_id, result in results if not result["ok"] and result.get("error") != "interrupted"),
        "truncated_record_ids": sorted(record_id for record_id, result in results if result.get("truncated")),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
