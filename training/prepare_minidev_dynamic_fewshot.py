#!/usr/bin/env python3
"""Prepend one cross-database BIRD-train SQL pattern to challenging Mini-Dev prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from prepare_minidev_inference import DEFAULT_MINIDEV_ROOT, load_minidev_examples
from prepare_minidev_schemapack import DEFAULT_EMBEDDING_MODEL, QwenEmbedder


DEFAULT_TRAIN = Path("BIRD/train_filtered/train.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_demo(example: dict[str, str], max_characters: int) -> str | None:
    question = example["question"].strip()
    evidence = example.get("evidence", "").strip() or "No additional evidence."
    sql = example["SQL"].strip()
    text = f"Question\n{question}\n\nEvidence\n{evidence}\n\nSQL\n{sql}"
    return text if len(text) <= max_characters else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    parser.add_argument("--embedding-snapshots", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--max-demo-characters", type=int, default=1400)
    args = parser.parse_args()
    if min(args.embedding_batch_size, args.candidate_pool, args.max_demo_characters) < 1:
        raise SystemExit("DYNAMIC_FEWSHOT_FAILED: numeric arguments must be positive")

    prompts = load_jsonl(args.prompts)
    if len(prompts) != 500:
        raise SystemExit(f"DYNAMIC_FEWSHOT_FAILED: expected 500 prompts, found {len(prompts)}")
    train = [
        {key: str(row.get(key, "")) for key in ("db_id", "question", "evidence", "SQL")}
        for row in load_jsonl(args.train)
        if all(isinstance(row.get(key), str) and row[key].strip() for key in ("db_id", "question", "SQL"))
    ]
    if not train:
        raise SystemExit("DYNAMIC_FEWSHOT_FAILED: no valid train examples")
    minidev = {index: example for index, example in enumerate(load_minidev_examples(args.minidev_root), start=1)}
    challenging = [prompt for prompt in prompts if prompt.get("difficulty") == "challenging"]
    if not challenging:
        raise SystemExit("DYNAMIC_FEWSHOT_FAILED: no challenging prompts")

    embedder = QwenEmbedder(args.embedding_snapshots, args.embedding_batch_size)
    train_queries = [row["question"] + "\n" + row["evidence"] for row in train]
    train_embeddings = embedder.encode(train_queries)
    target_queries = [minidev[prompt["example_index"]]["question"] + "\n" + str(minidev[prompt["example_index"]].get("evidence", "")) for prompt in challenging]
    target_embeddings = embedder.encode(target_queries)

    selected: dict[int, dict[str, Any]] = {}
    for prompt, embedding in zip(challenging, target_embeddings):
        db_id = prompt["db_id"]
        ranking = np.argsort(-(train_embeddings @ embedding), kind="stable")
        demo = None
        for index in ranking[: args.candidate_pool]:
            candidate = train[int(index)]
            if candidate["db_id"] == db_id:
                continue
            content = compact_demo(candidate, args.max_demo_characters)
            if content is not None:
                demo = {"content": content, "db_id": candidate["db_id"], "similarity": round(float(train_embeddings[index] @ embedding), 6)}
                break
        if demo is None:
            raise SystemExit(f"DYNAMIC_FEWSHOT_FAILED: no compact cross-database demo for index={prompt['example_index']}")
        selected[prompt["example_index"]] = demo

    output = []
    for prompt in prompts:
        messages = prompt.get("messages")
        if not isinstance(messages, list) or [message.get("role") for message in messages] != ["system", "user"]:
            raise SystemExit("DYNAMIC_FEWSHOT_FAILED: prompts must contain system/user messages")
        record = dict(prompt)
        demo = selected.get(prompt["example_index"])
        if demo is not None:
            user = (
                "Reference training pattern\n"
                "Use only its relational reasoning pattern. Never reuse its database names, tables, columns, literal values, or SQL text.\n\n"
                + demo["content"]
                + "\n\nTarget task\n"
                + messages[1]["content"]
            )
            record["messages"] = [messages[0], {"role": "user", "content": user}]
            selection = dict(record.get("selection", {}))
            selection["dynamic_fewshot"] = {"source_db_id": demo["db_id"], "similarity": demo["similarity"]}
            record["selection"] = selection
            record["prompt_characters"] = len(messages[0]["content"]) + len(user)
        output.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in output:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(output), "fewshot_prompts": len(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
