#!/usr/bin/env python3
"""Post-hoc table-structure diagnosis for frozen Mini-Dev predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibrate_schemapack_tables import gold_tables
from prepare_minidev_inference import DEFAULT_MINIDEV_ROOT, load_minidev_examples, load_minidev_gold_sql
from prepare_minidev_schemapack import load_database_schema
from prepare_minidev_inference import load_column_descriptions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    args = parser.parse_args()
    prompts = {record["example_index"]: record for record in read_jsonl(args.prompts)}
    predictions = {record["question_id"]: record["sql"] for record in read_jsonl(args.predictions)}
    evaluation = {record["example_index"]: record for record in read_jsonl(args.evaluation)}
    examples = load_minidev_examples(args.minidev_root)
    gold_sql = load_minidev_gold_sql(args.minidev_root, examples)
    if len(prompts) != len(examples) or len(evaluation) != len(examples):
        raise SystemExit("MINIDEV_STRUCTURE_DIAG_FAILED: prompts or evaluation do not cover all examples")
    schemas: dict[str, Any] = {}
    rows = []
    for index, (example, gold) in enumerate(zip(examples, gold_sql), start=1):
        db_id = example["db_id"]
        if db_id not in schemas:
            database_dir = args.minidev_root / "dev_databases" / db_id
            schemas[db_id] = load_database_schema(database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir))
        schema = schemas[db_id]
        known = set(schema.columns_by_table)
        gold_set = gold_tables(gold, known)
        prediction = predictions.get(example["question_id"], "")
        predicted_set = gold_tables(prediction, known) if prediction else set()
        selected = set(prompts[index]["selection"]["selected_tables"])
        correct = bool(evaluation[index]["correct"])
        rows.append(
            {
                "example_index": index,
                "question_id": example["question_id"],
                "difficulty": example.get("difficulty", "unknown"),
                "correct": correct,
                "gold_tables": sorted(gold_set),
                "predicted_tables": sorted(predicted_set),
                "prompt_covers_gold_tables": gold_set <= selected,
                "predicted_table_set_matches_gold": predicted_set == gold_set,
                "missing_gold_tables_in_prediction": sorted(gold_set - predicted_set),
                "extra_predicted_tables": sorted(predicted_set - gold_set),
            }
        )
    wrong = [row for row in rows if not row["correct"]]
    payload = {
        "total": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "wrong": len(wrong),
        "wrong_prompt_missing_gold_tables": sum(not row["prompt_covers_gold_tables"] for row in wrong),
        "wrong_prediction_table_set_matches_gold": sum(row["predicted_table_set_matches_gold"] for row in wrong),
        "wrong_prediction_omits_gold_table": sum(bool(row["missing_gold_tables_in_prediction"]) for row in wrong),
        "wrong_prediction_adds_extra_table": sum(bool(row["extra_predicted_tables"]) for row in wrong),
        "by_difficulty": {
            difficulty: {
                "total": len(bucket),
                "wrong": sum(not row["correct"] for row in bucket),
                "wrong_table_set_matches_gold": sum(not row["correct"] and row["predicted_table_set_matches_gold"] for row in bucket),
            }
            for difficulty in sorted({str(row["difficulty"]) for row in rows})
            for bucket in [[row for row in rows if str(row["difficulty"]) == difficulty]]
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
