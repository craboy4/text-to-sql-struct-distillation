from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path
from typing import Any

from generate_teacher_data import (
    SYSTEM_PROMPT,
    chat_completion,
    execute_sql,
    load_env,
    parse_teacher_response,
    unordered_result_equivalent,
)


STAGE_ONE_SYSTEM = SYSTEM_PROMPT + """

For this planning-repair pass only, a Reference SQL answer is supplied in the user message.
It is authoritative. Derive schema_linking and query_plan that faithfully explain that answer's
semantics, then return a SQL query with the same result. Do not mention the reference answer.
"""

STAGE_TWO_SYSTEM = SYSTEM_PROMPT + """

For this SQL-derivation pass, the user supplies authoritative schema_linking and query_plan but
does not supply a reference SQL answer. Copy those two supplied blocks verbatim into your response,
then derive the SQL from them and the supplied schema. Do not add or remove requirements from the
given plan.
"""


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    serialized = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as output:
            output.write(serialized)


def load_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as source:
        for raw_line in source:
            if raw_line.strip():
                record = json.loads(raw_line)
                records[record["record_id"]] = record
    return records


def candidate_ids(records: dict[int, dict[str, Any]]) -> list[int]:
    return [
        record_id
        for record_id, record in records.items()
        if not record["ready_for_sft"]
        and record["gold_execution"]["ok"]
        and not record["gold_execution"].get("truncated")
    ]


def stage_one_user_prompt(record: dict[str, Any]) -> str:
    return (
        record["prompt"]["user"]
        + "\nReference SQL answer (authoritative for this planning-repair pass)\n"
        + record["source"]["gold_sql"].strip()
        + "\n"
    )


def stage_two_user_prompt(record: dict[str, Any], plan: dict[str, Any]) -> str:
    return (
        record["prompt"]["user"]
        + "\nAuthoritative intermediate reasoning\n"
        + "<schema_linking>\n"
        + plan["schema_linking"]
        + "\n</schema_linking>\n\n<query_plan>\n"
        + plan["query_plan"]
        + "\n</query_plan>\n"
    )


def plan_is_usable(parsed: dict[str, Any]) -> bool:
    return bool(parsed["format_ok"] and parsed["schema_linking"] and parsed["query_plan"])


def normalized(text: str | None) -> str:
    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines())


def run_stage_one(
    record: dict[str, Any], base_url: str, api_key: str, model: str, reasoning_effort: str
) -> dict[str, Any]:
    raw_response, usage, teacher_error = chat_completion(
        base_url,
        api_key,
        model,
        reasoning_effort,
        stage_one_user_prompt(record),
        STAGE_ONE_SYSTEM,
    )
    parsed = parse_teacher_response(raw_response) if raw_response else {
        "format_ok": False,
        "error": teacher_error,
        "schema_linking": None,
        "query_plan": None,
        "sql": None,
    }
    return {
        "record_id": record["record_id"],
        "source": record["source"],
        "prompt": {"system": STAGE_ONE_SYSTEM, "user": stage_one_user_prompt(record)},
        "teacher": {"model": model, "reasoning_effort": reasoning_effort, "raw_response": raw_response, "usage": usage},
        "parsed": parsed,
        "planning_usable": plan_is_usable(parsed),
    }


def run_stage_two(
    record: dict[str, Any],
    stage_one: dict[str, Any],
    database_root: Path,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    max_result_rows: int,
    sql_timeout_seconds: float,
) -> dict[str, Any]:
    plan = stage_one["parsed"]
    raw_response, usage, teacher_error = chat_completion(
        base_url,
        api_key,
        model,
        reasoning_effort,
        stage_two_user_prompt(record, plan),
        STAGE_TWO_SYSTEM,
    )
    parsed = parse_teacher_response(raw_response) if raw_response else {
        "format_ok": False,
        "error": teacher_error,
        "schema_linking": None,
        "query_plan": None,
        "sql": None,
    }
    sqlite_path = database_root / record["source"]["db_id"] / f"{record['source']['db_id']}.sqlite"
    execution = execute_sql(sqlite_path, parsed["sql"], max_result_rows, sql_timeout_seconds)
    equivalent = unordered_result_equivalent(execution, record["gold_execution"])
    copied_plan = (
        normalized(parsed["schema_linking"]) == normalized(plan["schema_linking"])
        and normalized(parsed["query_plan"]) == normalized(plan["query_plan"])
    )
    accepted = bool(parsed["format_ok"] and copied_plan and execution["ok"] and equivalent)
    return {
        "record_id": record["record_id"],
        "source": record["source"],
        "stage_one_record_id": stage_one["record_id"],
        "prompt": {"system": STAGE_TWO_SYSTEM, "user": stage_two_user_prompt(record, plan)},
        "teacher": {"model": model, "reasoning_effort": reasoning_effort, "raw_response": raw_response, "usage": usage},
        "parsed": parsed,
        "copied_intermediate_reasoning": copied_plan,
        "teacher_execution": execution,
        "gold_execution": record["gold_execution"],
        "execution_equivalent_unordered": equivalent,
        "accepted_for_sft": accepted,
    }


def clean_sft_record(original: dict[str, Any], repaired: dict[str, Any]) -> dict[str, Any]:
    raw_response = repaired["teacher"]["raw_response"]
    return {
        "record_id": original["record_id"],
        "source": original["source"],
        "prompt": {**original["prompt"], "system": SYSTEM_PROMPT},
        "teacher": repaired["teacher"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": original["prompt"]["user"]},
            {"role": "assistant", "content": raw_response},
        ],
        "parsed": repaired["parsed"],
        "teacher_execution": repaired["teacher_execution"],
        "gold_execution": original["gold_execution"],
        "execution_equivalent_unordered": repaired["execution_equivalent_unordered"],
        "ready_for_sft": True,
    }


def write_merged_sft(
    original_records: dict[int, dict[str, Any]], repaired_records: dict[int, dict[str, Any]], output: Path
) -> tuple[int, int]:
    repaired = {record_id: record for record_id, record in repaired_records.items() if record["accepted_for_sft"]}
    with output.open("w", encoding="utf-8") as target:
        for record_id in sorted(original_records):
            original = original_records[record_id]
            if record_id in repaired:
                record = clean_sft_record(original, repaired[record_id])
            else:
                record = {**original, "prompt": {**original["prompt"], "system": SYSTEM_PROMPT}}
                record["messages"] = [{**message} for message in record["messages"]]
                record["messages"][0]["content"] = SYSTEM_PROMPT
            if record["ready_for_sft"]:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
    return sum(record["ready_for_sft"] for record in original_records.values()), len(repaired)


def run_parallel(items: list[Any], workers: int, worker, output: Path) -> None:
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            append_jsonl(output, result, lock)
            if index % 25 == 0 or index == len(items):
                print(f"{output.name}: {index}/{len(items)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair non-SFT BIRD teacher records through gold-conditioned planning and blind SQL derivation.")
    parser.add_argument("--input", type=Path, default=Path("BIRD/teacher_generation/teacher_full_output_contract_60s.jsonl"))
    parser.add_argument("--database-root", type=Path, default=Path("BIRD/train/train_databases"))
    parser.add_argument("--env-file", type=Path, default=Path("BIRD/teacher_generation/teacher.env"))
    parser.add_argument("--stage-one-output", type=Path, default=Path("BIRD/teacher_generation/repair_stage1_gold_conditioned.jsonl"))
    parser.add_argument("--stage-two-output", type=Path, default=Path("BIRD/teacher_generation/repair_stage2_blind_derivation.jsonl"))
    parser.add_argument("--merged-sft-output", type=Path, default=Path("BIRD/teacher_generation/teacher_sft_merged_two_stage.jsonl"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-result-rows", type=int, default=1_000)
    parser.add_argument("--sql-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    load_env(args.env_file)
    import os

    base_url = os.environ["TEACHER_BASE_URL"]
    api_key = os.environ["TEACHER_API_KEY"]
    model = os.environ.get("TEACHER_MODEL", "gpt-5.6-terra")
    reasoning_effort = os.environ.get("TEACHER_REASONING_EFFORT", "high")
    original_records = load_records(args.input)
    selected_ids = candidate_ids(original_records)
    args.stage_one_output.parent.mkdir(parents=True, exist_ok=True)

    stage_one_records = load_records(args.stage_one_output)
    stage_one_pending = [original_records[record_id] for record_id in selected_ids if record_id not in stage_one_records]
    print(f"stage 1: {len(stage_one_records)}/{len(selected_ids)} already complete", flush=True)
    run_parallel(
        stage_one_pending,
        args.workers,
        lambda record: run_stage_one(record, base_url, api_key, model, reasoning_effort),
        args.stage_one_output,
    )

    stage_one_records = load_records(args.stage_one_output)
    stage_two_records = load_records(args.stage_two_output)
    stage_two_pending = [
        (original_records[record_id], stage_one_records[record_id])
        for record_id in selected_ids
        if record_id in stage_one_records
        and stage_one_records[record_id]["planning_usable"]
        and record_id not in stage_two_records
    ]
    print(f"stage 2: {len(stage_two_records)}/{len(stage_two_pending) + len(stage_two_records)} already complete", flush=True)
    run_parallel(
        stage_two_pending,
        args.workers,
        lambda pair: run_stage_two(
            pair[0], pair[1], args.database_root, base_url, api_key, model, reasoning_effort,
            args.max_result_rows, args.sql_timeout_seconds,
        ),
        args.stage_two_output,
    )

    stage_two_records = load_records(args.stage_two_output)
    existing, repaired = write_merged_sft(original_records, stage_two_records, args.merged_sft_output)
    print(f"merged SFT: existing={existing}, repaired={repaired}, total={existing + repaired}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
