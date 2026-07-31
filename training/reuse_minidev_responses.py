#!/usr/bin/env python3
"""Assemble responses only when their source prompt exactly matches a target prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_minidev_api import completed_indexes, load_prompts


def load_responses(path: Path) -> dict[int, dict[str, Any]]:
    responses: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index = record.get("example_index")
        if not isinstance(index, int) or not isinstance(record.get("response"), str):
            raise ValueError(f"invalid response record in {path}")
        responses[index] = record
    return responses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-prompts", type=Path, required=True)
    parser.add_argument("--source-prompts", type=Path, action="append", required=True)
    parser.add_argument("--source-responses", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if len(args.source_prompts) != len(args.source_responses):
        raise SystemExit("REUSE_RESPONSES_FAILED: source prompt and response counts differ")
    if args.output.exists() and completed_indexes(args.output):
        raise SystemExit(f"REUSE_RESPONSES_FAILED: output already contains responses: {args.output}")

    targets = {record["example_index"]: record for record in load_prompts(args.target_prompts)}
    sources = [
        ({record["example_index"]: record for record in load_prompts(prompt_path)}, load_responses(response_path))
        for prompt_path, response_path in zip(args.source_prompts, args.source_responses)
    ]
    selected: list[dict[str, Any]] = []
    missing: list[int] = []
    for index, target in sorted(targets.items()):
        match: dict[str, Any] | None = None
        for source_prompts, source_responses in sources:
            source_prompt = source_prompts[index]
            response = source_responses.get(index)
            if source_prompt["messages"] == target["messages"] and response is not None:
                match = response
                break
        if match is None:
            missing.append(index)
        else:
            selected.append(match)
    if missing and not args.allow_partial:
        raise SystemExit(
            f"REUSE_RESPONSES_FAILED: {len(missing)} target prompts have no exact source response; first indexes={missing[:10]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for response in selected:
            destination.write(json.dumps(response, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "responses": len(selected), "missing": len(missing), "first_missing": missing[:10]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
