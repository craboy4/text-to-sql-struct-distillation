#!/usr/bin/env python3
"""Force an independent candidate only for challenging Mini-Dev prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", type=int, required=True)
    args = parser.parse_args()
    prompts = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) != 500 or args.candidate_id < 2:
        raise SystemExit("INDEPENDENT_CHALLENGING_FAILED: require 500 prompts and candidate id >= 2")
    changed = 0
    output: list[dict[str, Any]] = []
    for prompt in prompts:
        record = dict(prompt)
        messages = record.get("messages")
        if not isinstance(messages, list) or [message.get("role") for message in messages] != ["system", "user"]:
            raise SystemExit("INDEPENDENT_CHALLENGING_FAILED: prompts must contain system/user messages")
        if record.get("difficulty") == "challenging":
            user = messages[1]["content"] + (
                f"\n\nIndependent candidate {args.candidate_id}\n"
                "Solve the target task independently from first principles. Do not assume any prior candidate answer."
            )
            record["messages"] = [messages[0], {"role": "user", "content": user}]
            selection = dict(record.get("selection", {}))
            selection["independent_candidate"] = args.candidate_id
            record["selection"] = selection
            record["prompt_characters"] = len(messages[0]["content"]) + len(user)
            changed += 1
        output.append(record)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in output:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(output), "changed": changed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
