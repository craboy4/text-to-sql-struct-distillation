from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("BIRD/teacher_generation/teacher_sft_merged_two_stage.jsonl")
OUTPUT = Path("BIRD/teacher_generation/qwen_sft_messages.jsonl")


def main() -> int:
    count = 0
    with SOURCE.open(encoding="utf-8") as source, OUTPUT.open("w", encoding="utf-8") as target:
        for raw_line in source:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            messages = record["messages"]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"Unexpected message roles for record {record['record_id']}")
            target.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    print(f"Exported {count} records to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
