from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from generate_teacher_data import execute_sql


SOURCE = Path("BIRD/teacher_generation/teacher_full_output_contract.jsonl")
DATABASE_ROOT = Path("BIRD/train/train_databases")
OUTPUT = Path("BIRD/teacher_generation/gold_timeout_60s_sequential.jsonl")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")

    candidates = []
    with SOURCE.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["gold_execution"].get("error") == "interrupted":
                db_id = record["source"]["db_id"]
                candidates.append((record["record_id"], db_id, DATABASE_ROOT / db_id / f"{db_id}.sqlite", record["source"]["gold_sql"]))

    with OUTPUT.open("w", encoding="utf-8") as destination:
        for ordinal, (record_id, db_id, sqlite_path, sql) in enumerate(candidates, start=1):
            result = execute_sql(sqlite_path, sql, max_rows=1_000, timeout_seconds=60.0)
            destination.write(json.dumps({"ordinal": ordinal, "record_id": record_id, "db_id": db_id, "execution": result}, ensure_ascii=False) + "\n")
            destination.flush()
            print(f"ordinal={ordinal}/{len(candidates)} record_id={record_id} ok={result['ok']} error={result['error']}")


if __name__ == "__main__":
    main()
