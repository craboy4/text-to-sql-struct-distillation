#!/usr/bin/env python3
"""Batch-generate SFT-matched Mini-Dev responses with a local Qwen checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


EXPECTED_PROMPTS = 500


class ValidationError(ValueError):
    """Raised when prompt or resume data cannot be used safely."""


def fail(message: str) -> None:
    raise SystemExit(f"MINIDEV_LOCAL_FAILED: {message}")


def load_prompts(path: Path, expected_prompts: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    canonical_by_question_id: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = record.get("example_index")
            question_id = record.get("question_id")
            messages = record.get("messages")
            roles = [message.get("role") for message in messages] if isinstance(messages, list) else []
            if not isinstance(index, int) or index < 1 or index in seen_indexes:
                raise ValidationError(f"prompt line {line_number} has an invalid or duplicate example_index")
            if not isinstance(question_id, int) or roles not in (["system", "user"], ["user"]):
                raise ValidationError(f"prompt line {line_number} must contain question_id and user messages")
            if not all(isinstance(message.get("content"), str) for message in messages):
                raise ValidationError(f"prompt line {line_number} has non-text message content")
            prior = canonical_by_question_id.setdefault(question_id, record)
            if prior is not record and (prior.get("db_id") != record.get("db_id") or prior["messages"] != messages):
                raise ValidationError(f"question_id {question_id} maps to non-identical prompts")
            seen_indexes.add(index)
            records.append(record)
    if len(records) != expected_prompts:
        raise ValidationError(f"expected {expected_prompts} prompts, found {len(records)}")
    return list(canonical_by_question_id.values())


def completed_indexes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = record.get("example_index")
            response = record.get("response")
            if not isinstance(index, int) or not isinstance(response, str) or not response.strip() or index in done:
                raise ValidationError(f"response line {line_number} is malformed or duplicated")
            done.add(index)
    return done


def render_chat(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def seconds_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/text2sql_qwen3/models/Qwen3-8B"))
    parser.add_argument("--adapter", type=Path, help="Optional PEFT LoRA adapter to load over --model")
    parser.add_argument("--prompts", type=Path, default=Path("/root/autodl-tmp/text2sql_qwen3/eval/minidev_inference_prompts.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=30_720)
    parser.add_argument("--max-new-tokens", type=int, default=1_024)
    parser.add_argument("--expected-prompts", type=int, default=EXPECTED_PROMPTS)
    parser.add_argument("--limit", type=int, default=0, help="0 runs every pending unique Mini-Dev question")
    parser.add_argument("--enable-thinking", action="store_true", help="Use Qwen3's native thinking chat template")
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.max_input_tokens < 1
        or args.max_new_tokens < 1
        or args.expected_prompts < 1
        or args.limit < 0
    ):
        fail("batch size, token limits, and limit must be positive (or limit=0)")

    try:
        prompts = load_prompts(args.prompts, args.expected_prompts)
        done = completed_indexes(args.output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        fail(str(error))
    pending = [record for record in prompts if record["example_index"] not in done]
    if args.limit:
        pending = pending[: args.limit]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        fail("CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rendered: list[tuple[int, dict[str, Any], str]] = []
    for record in pending:
        text = render_chat(tokenizer, record["messages"], args.enable_thinking)
        length = len(tokenizer.encode(text, add_special_tokens=False))
        if length > args.max_input_tokens:
            raise ValidationError(
                f"example_index {record['example_index']} has {length} prompt tokens, above --max-input-tokens"
            )
        rendered.append((length, record, text))
    rendered.sort(key=lambda item: item[0])

    print(
        f"loading_model={args.model} pending_unique_questions={len(rendered)} batch_size={args.batch_size} "
        f"enable_thinking={args.enable_thinking}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to("cuda").eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    with torch.inference_mode(), args.output.open("a", encoding="utf-8") as destination:
        start = 0
        batch_size = args.batch_size
        while start < len(rendered):
            batch = rendered[start : start + batch_size]
            texts = [item[2] for item in batch]
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to("cuda")
            input_width = encoded["input_ids"].shape[1]
            try:
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                next_batch_size = max(1, batch_size // 2)
                print(
                    f"cuda_oom batch_size={batch_size} input_width={input_width} "
                    f"retry_batch_size={next_batch_size}",
                    flush=True,
                )
                del encoded
                torch.cuda.empty_cache()
                batch_size = next_batch_size
                continue
            for row, (_, record, _) in enumerate(batch):
                response_ids = generated[row, input_width:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                destination.write(
                    json.dumps(
                        {
                            "example_index": record["example_index"],
                            "question_id": record["question_id"],
                            "db_id": record["db_id"],
                            "model": str(args.model),
                            "response": response,
                            "prompt_tokens": int(encoded["attention_mask"][row].sum().item()),
                            "generated_tokens": int((response_ids != tokenizer.pad_token_id).sum().item()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            destination.flush()
            completed += len(batch)
            start += len(batch)
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            eta = (len(rendered) - completed) / rate if rate else 0.0
            memory_gib = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"completed={completed}/{len(rendered)} elapsed={seconds_text(elapsed)} "
                f"eta={seconds_text(eta)} examples_per_second={rate:.3f} peak_vram_gib={memory_gib:.1f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        fail(str(error))
