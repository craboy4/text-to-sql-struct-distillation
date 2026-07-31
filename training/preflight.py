#!/usr/bin/env python3
"""Validate the remote SFT payload and runtime before spending GPU time."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"PRECHECK_FAILED: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/text2sql_qwen3"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-records", type=int, default=6246)
    args = parser.parse_args()

    root = args.project_root
    dataset = args.dataset or root / "data" / "qwen_sft_messages.jsonl"
    model = root / "models" / "Qwen3-8B"
    minidev = root / "eval" / "MINIDEV"
    if not dataset.is_file():
        fail(f"missing dataset: {dataset}")
    if not (model / "config.json").is_file():
        fail(f"missing model config: {model / 'config.json'}")
    if not any(model.glob("*.safetensors")):
        fail(f"no safetensors files in {model}")
    if not (minidev / "mini_dev_sqlite.json").is_file():
        fail(f"missing Mini-Dev JSON: {minidev}")

    records = 0
    with dataset.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            messages = record.get("messages")
            roles = [message.get("role") for message in messages or []]
            if roles != ["system", "user", "assistant"]:
                fail(f"invalid message roles on dataset line {line_number}: {roles}")
            records += 1
    if args.expected_records < 1 or records != args.expected_records:
        fail(f"expected {args.expected_records} SFT records, found {records}")

    minidev_records = json.loads((minidev / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    gold_lines = [line for line in (minidev / "mini_dev_sqlite_gold.sql").read_text(encoding="utf-8").splitlines() if line]
    sqlite_count = len(list((minidev / "dev_databases").glob("*/*.sqlite")))
    if len(minidev_records) != 500 or len(gold_lines) != 500 or sqlite_count != 11:
        fail(f"unexpected Mini-Dev payload: records={len(minidev_records)} gold={len(gold_lines)} sqlite={sqlite_count}")

    try:
        import torch
    except ImportError as error:
        fail(f"cannot import torch: {error}")
    if not torch.cuda.is_available():
        fail("CUDA is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability < (12, 0):
        fail(f"expected Blackwell-class GPU, found CUDA capability {capability}")

    try:
        swift_check = subprocess.check_output(
            ["python", "-c", "from swift.pipelines import sft_main; print('available')"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"ms-swift SFT entrypoint is unavailable: {error}")
    if swift_check.splitlines()[-1:] != ["available"]:
        fail(f"unexpected ms-swift SFT check: {swift_check!r}")

    free_gib = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 1024**3
    if free_gib < 40:
        fail(f"only {free_gib:.1f} GiB free on data disk")

    print(json.dumps({
        "dataset_records": records,
        "dataset": str(dataset.resolve()),
        "minidev_records": len(minidev_records),
        "minidev_gold_lines": len(gold_lines),
        "minidev_sqlite_databases": sqlite_count,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_capability": capability,
        "torch": torch.__version__,
        "swift_sft": "available",
        "data_disk_free_gib": round(free_gib, 2),
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
