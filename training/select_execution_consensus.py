#!/usr/bin/env python3
"""Select a challenging SQL candidate only when SQLite execution has a majority result."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluate_minidev import execute_readonly_sql, load_minidev, load_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate-2", type=Path, required=True)
    parser.add_argument("--candidate-3", type=Path, required=True)
    parser.add_argument("--minidev-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    base, second, third = (load_predictions(path) for path in (args.base, args.candidate_2, args.candidate_3))
    examples = load_minidev(args.minidev_root, 500)
    selected = dict(base)
    audits = []
    for example in examples:
        if example.difficulty != "challenging" or example.question_id not in base:
            continue
        database = args.minidev_root / "dev_databases" / example.db_id / f"{example.db_id}.sqlite"
        candidates = [base.get(example.question_id), second.get(example.question_id), third.get(example.question_id)]
        results = [execute_readonly_sql(database, sql or "", args.timeout_seconds, 100_000) for sql in candidates]
        groups: dict[str, list[int]] = defaultdict(list)
        for index, result in enumerate(results):
            if result["status"] == "ok":
                groups[str(result["signature"])].append(index)
        majority = [indexes for indexes in groups.values() if len(indexes) >= 2]
        chosen = 0
        if majority:
            best = max(majority, key=lambda indexes: (len(indexes), 0 in indexes))
            chosen = 0 if 0 in best else best[0]
            selected[example.question_id] = candidates[chosen] or base[example.question_id]
        audits.append({
            "question_id": example.question_id,
            "db_id": example.db_id,
            "chosen_candidate": chosen + 1,
            "majority": bool(majority),
            "statuses": [result["status"] for result in results],
            "signatures": [result.get("signature") for result in results],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for question_id, sql in sorted(selected.items()):
            destination.write(json.dumps({"question_id": question_id, "sql": sql}, ensure_ascii=False) + "\n")
    with args.audit_output.open("w", encoding="utf-8") as destination:
        for audit in audits:
            destination.write(json.dumps(audit, ensure_ascii=False) + "\n")
    print(json.dumps({"predictions": len(selected), "challenging": len(audits), "majority": sum(audit["majority"] for audit in audits), "changed": sum(audit["chosen_candidate"] != 1 for audit in audits)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
