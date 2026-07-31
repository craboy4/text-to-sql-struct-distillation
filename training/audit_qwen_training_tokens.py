#!/usr/bin/env python3
"""Count final SFT and DPO examples with the exact Qwen chat template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_TOKENIZER = Path(r"C:\Users\WangYiran\.cache\huggingface\hub\models--Qwen--Qwen3-8B\snapshots")


def lengths(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    def percentile(q: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]
    return {"examples": len(values), "mean": round(sum(values) / len(values), 2), "p50": percentile(.5), "p90": percentile(.9), "p99": percentile(.99), "max": ordered[-1]}


def token_length(tokenizer: AutoTokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=False
    )
    return len(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--dpo", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshots", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [path for path in args.tokenizer_snapshots.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise SystemExit("QWEN_TOKEN_AUDIT_FAILED: expected one tokenizer snapshot")
    tokenizer = AutoTokenizer.from_pretrained(candidates[0], local_files_only=True)
    sft_prompt = []
    sft_total = []
    for line in args.sft.open(encoding="utf-8"):
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        sft_prompt.append(token_length(tokenizer, messages[:2], True))
        sft_total.append(token_length(tokenizer, messages, False))
    dpo_prompt = []
    dpo_chosen = []
    dpo_rejected = []
    for line in args.dpo.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row["prompt"]
        dpo_prompt.append(token_length(tokenizer, prompt, True))
        dpo_chosen.append(token_length(tokenizer, prompt + [{"role": "assistant", "content": row["chosen"]}], False))
        dpo_rejected.append(token_length(tokenizer, prompt + [{"role": "assistant", "content": row["rejected"]}], False))
    payload = {"tokenizer": str(candidates[0]), "sft_prompt": lengths(sft_prompt), "sft_total": lengths(sft_total), "dpo_prompt": lengths(dpo_prompt), "dpo_chosen_total": lengths(dpo_chosen), "dpo_rejected_total": lengths(dpo_rejected)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
