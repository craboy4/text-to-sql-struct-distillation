#!/usr/bin/env python3
"""Run variable-size schema-auditor prompt sets through the configured teacher API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any


TEACHER_GENERATION = Path(__file__).resolve().parents[1] / "teacher_generation"
sys.path.insert(0, str(TEACHER_GENERATION))

from generate_teacher_data import chat_completion, load_env  # noqa: E402


def load_jsonl(path: Path, require_response: bool = False) -> list[dict[str, Any]]:
    records = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            audit_id = record.get("audit_id")
            if not isinstance(audit_id, int) or audit_id < 1 or audit_id in seen:
                raise ValueError(f"invalid or duplicate audit_id on line {line_number}")
            if require_response:
                if not isinstance(record.get("response"), str) or not record["response"].strip():
                    raise ValueError(f"missing response on line {line_number}")
            else:
                messages = record.get("messages")
                if not isinstance(messages, list) or [message.get("role") for message in messages] != ["system", "user"]:
                    raise ValueError(f"prompt line {line_number} must contain system/user messages")
            seen.add(audit_id)
            records.append(record)
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def call(record: dict[str, Any], base_url: str, api_key: str, model: str, reasoning_effort: str) -> dict[str, Any]:
    system, user = record["messages"]
    response, usage, error = chat_completion(
        base_url, api_key, model, reasoning_effort, user["content"], system["content"]
    )
    if response is None:
        raise RuntimeError(error or "teacher returned no response")
    return {"audit_id": record["audit_id"], "model": model, "reasoning_effort": reasoning_effort, "response": response, "usage": usage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("BIRD/teacher_generation/teacher.env"))
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("SCHEMA_AUDITOR_API_FAILED: --workers must be positive")
    prompts = load_jsonl(args.prompts)
    completed = {record["audit_id"] for record in load_jsonl(args.output, require_response=True)} if args.output.exists() else set()
    load_env(args.env_file)
    import os

    pending = [record for record in prompts if record["audit_id"] not in completed]
    print(f"completed={len(completed)} pending={len(pending)}", flush=True)
    failures: list[str] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor, args.output.open("a", encoding="utf-8") as destination:
        futures = {
            executor.submit(call, record, os.environ["TEACHER_BASE_URL"], os.environ["TEACHER_API_KEY"], args.model, args.reasoning_effort): record
            for record in pending
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = futures[future]
            try:
                response = future.result()
            except RuntimeError as error:
                failures.append(f"audit_id={record['audit_id']}: {error}")
            else:
                destination.write(json.dumps(response, ensure_ascii=False) + "\n")
                destination.flush()
            if count % 10 == 0 or count == len(pending):
                print(f"processed={count}/{len(pending)} failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(f"SCHEMA_AUDITOR_API_FAILED: {len(failures)} failures; rerun to resume. {failures[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
