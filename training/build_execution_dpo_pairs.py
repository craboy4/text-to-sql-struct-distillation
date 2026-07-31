#!/usr/bin/env python3
"""Build execution-verified Text-to-SQL DPO pairs from teacher repair traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ORIGINAL = Path("BIRD/teacher_generation/teacher_full_output_contract_60s.jsonl")
DEFAULT_REPAIRED = Path("BIRD/teacher_generation/repair_stage2_blind_derivation.jsonl")


def read_by_id(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("record_id")
            if not isinstance(record_id, int) or record_id in records:
                raise ValueError(f"{path}: invalid or duplicate record_id at line {line_number}")
            records[record_id] = row
    return records


def semantic_error(original: dict[str, Any]) -> bool:
    return bool(
        original.get("parsed", {}).get("format_ok")
        and original.get("teacher_execution", {}).get("ok")
        and original.get("gold_execution", {}).get("ok")
        and original.get("execution_equivalent_unordered") is False
    )


def valid_repair(repaired: dict[str, Any]) -> bool:
    return bool(
        repaired.get("accepted_for_sft")
        and isinstance(repaired.get("teacher", {}).get("raw_response"), str)
        and repaired["teacher"]["raw_response"].strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--repaired", type=Path, default=DEFAULT_REPAIRED)
    args = parser.parse_args()
    original = read_by_id(args.original)
    repaired = read_by_id(args.repaired)
    pairs = []
    skipped: dict[str, int] = {"missing_original": 0, "repair_not_accepted": 0, "not_semantic_error": 0, "invalid_original_response": 0, "identical_pair": 0}
    for record_id, repaired_row in repaired.items():
        original_row = original.get(record_id)
        if original_row is None:
            skipped["missing_original"] += 1
            continue
        if not valid_repair(repaired_row):
            skipped["repair_not_accepted"] += 1
            continue
        if not semantic_error(original_row):
            skipped["not_semantic_error"] += 1
            continue
        rejected = original_row.get("teacher", {}).get("raw_response")
        messages = original_row.get("messages")
        if not isinstance(rejected, str) or not rejected.strip() or not isinstance(messages, list) or [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            skipped["invalid_original_response"] += 1
            continue
        chosen = repaired_row["teacher"]["raw_response"]
        if chosen.strip() == rejected.strip():
            skipped["identical_pair"] += 1
            continue
        source = original_row["source"]
        pairs.append(
            {
                "record_id": record_id,
                "db_id": source["db_id"],
                "pair_type": "execution_wrong_teacher_vs_execution_verified_repair",
                "prompt": [{"role": "system", "content": messages[0]["content"]}, {"role": "user", "content": messages[1]["content"]}],
                "chosen": chosen,
                "rejected": rejected,
                "audit": {
                    "original_execution_equivalent": original_row["execution_equivalent_unordered"],
                    "repair_execution_equivalent": repaired_row["execution_equivalent_unordered"],
                    "repair_copied_intermediate_reasoning": repaired_row["copied_intermediate_reasoning"],
                },
            }
        )
    pairs.sort(key=lambda row: row["record_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for pair in pairs:
            destination.write(json.dumps(pair, ensure_ascii=False) + "\n")
    manifest = {
        "source": {
            "original": str(args.original.resolve()),
            "repaired": str(args.repaired.resolve()),
        },
        "pair_contract": "same original system/user prompt; rejected SQL is executable but result-inequivalent; chosen repair is execution-equivalent after blind SQL derivation",
        "pairs": len(pairs),
        "skipped": skipped,
        "output": str(args.output.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
