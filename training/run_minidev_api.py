#!/usr/bin/env python3
"""Run SFT-matched Mini-Dev prompts through the configured teacher API.

Responses are appended only after a successful API call, so rerunning this
command resumes without emitting duplicate ``example_index`` records.
"""

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


EXPECTED_QUESTIONS = 500
DEFAULT_PROMPTS = Path("BIRD/training/minidev_inference_prompts.jsonl")
DEFAULT_ENV = Path("BIRD/teacher_generation/teacher.env")
DEFAULT_OUTPUT = Path("BIRD/training/minidev_terra_high_responses.jsonl")


class ValidationError(ValueError):
    """Raised for malformed local prompt or response records."""


def fail(message: str) -> None:
    raise SystemExit(f"MINIDEV_API_FAILED: {message}")


def load_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"prompt file does not exist: {path}")
    prompts: list[dict[str, Any]] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            index = record.get("example_index")
            messages = record.get("messages")
            if not isinstance(index, int) or index < 1:
                raise ValidationError(f"prompt line {line_number} has no positive integer example_index")
            if index in seen:
                raise ValidationError(f"prompt line {line_number} duplicates example_index {index}")
            roles = [message.get("role") for message in messages] if isinstance(messages, list) else []
            if roles not in (["system", "user"], ["user"]):
                raise ValidationError(f"prompt line {line_number} must contain system/user or user-only messages")
            if not all(isinstance(message.get("content"), str) for message in messages):
                raise ValidationError(f"prompt line {line_number} contains non-text message content")
            seen.add(index)
            prompts.append(record)
    if len(prompts) != EXPECTED_QUESTIONS:
        raise ValidationError(f"expected {EXPECTED_QUESTIONS} prompts, found {len(prompts)}")
    return sorted(prompts, key=lambda record: record["example_index"])


def completed_indexes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            index = record.get("example_index")
            response = record.get("response")
            if not isinstance(index, int) or not isinstance(response, str) or not response.strip():
                raise ValidationError(f"response line {line_number} is not a completed response record")
            if index in completed:
                raise ValidationError(f"response line {line_number} duplicates example_index {index}")
            completed.add(index)
    return completed


def canonical_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Call identical repeated Mini-Dev tasks once because EX is keyed by question_id."""
    by_question_id: dict[int, dict[str, Any]] = {}
    for prompt in prompts:
        question_id = prompt["question_id"]
        prior = by_question_id.setdefault(question_id, prompt)
        if prior is prompt:
            continue
        if prior["db_id"] != prompt["db_id"] or prior["messages"] != prompt["messages"]:
            raise ValidationError(f"question_id {question_id} refers to different benchmark prompts")
    return list(by_question_id.values())


def call(prompt: dict[str, Any], base_url: str, api_key: str, model: str, reasoning_effort: str) -> dict[str, Any]:
    messages = prompt["messages"]
    system_prompt = messages[0]["content"] if len(messages) == 2 else None
    user_prompt = messages[-1]["content"]
    response, usage, error = chat_completion(
        base_url,
        api_key,
        model,
        reasoning_effort,
        user_prompt,
        system_prompt,
    )
    if response is None:
        raise RuntimeError(error or "teacher returned no response")
    return {
        "example_index": prompt["example_index"],
        "question_id": prompt["question_id"],
        "db_id": prompt["db_id"],
        "model": model,
        "reasoning_effort": reasoning_effort,
        "response": response,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means all pending prompts")
    args = parser.parse_args()
    if args.workers < 1:
        fail("--workers must be at least 1")
    if args.limit < 0:
        fail("--limit must be non-negative")

    try:
        prompts = load_prompts(args.prompts)
        done = completed_indexes(args.output)
        load_env(args.env_file)
        import os

        base_url = os.environ["TEACHER_BASE_URL"]
        api_key = os.environ["TEACHER_API_KEY"]
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        fail(str(error))

    prompts = canonical_prompts(prompts)
    pending = [prompt for prompt in prompts if prompt["example_index"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"completed={len(done)} pending={len(pending)} workers={args.workers}", flush=True)

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor, args.output.open(
        "a", encoding="utf-8"
    ) as destination:
        futures = {
            executor.submit(call, prompt, base_url, api_key, args.model, args.reasoning_effort): prompt
            for prompt in pending
        }
        for completed_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            prompt = futures[future]
            try:
                record = future.result()
            except RuntimeError as error:
                failures.append(f"example_index={prompt['example_index']}: {error}")
            else:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                destination.flush()
            if completed_count % 10 == 0 or completed_count == len(pending):
                print(f"processed={completed_count}/{len(pending)} failures={len(failures)}", flush=True)

    if failures:
        fail(f"{len(failures)} API requests failed; rerun to resume. First failure: {failures[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
