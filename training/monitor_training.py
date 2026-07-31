#!/usr/bin/env python3
"""Print checkpoint progress and an ETA from ms-swift TrainerState."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def seconds_text(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def newest_state(output_dir: Path) -> Path | None:
    candidates = list(output_dir.glob("checkpoint-*/trainer_state.json"))
    direct = output_dir / "trainer_state.json"
    if direct.is_file():
        candidates.append(direct)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def report(output_dir: Path) -> None:
    state_path = newest_state(output_dir)
    if state_path is None:
        print("status=waiting_for_first_checkpoint")
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = int(state.get("global_step", 0))
    total = int(state.get("max_steps", 0))
    epoch = state.get("epoch")
    start_file = output_dir / "run_started_at.txt"
    started_at = float(start_file.read_text(encoding="utf-8").strip()) if start_file.is_file() else state_path.stat().st_mtime
    elapsed = max(0.0, time.time() - started_at)
    eta = elapsed * (total - step) / step if step > 0 and total > step else 0.0 if total and step >= total else None
    checkpoint = state_path.parent.name if state_path.parent != output_dir else "root"
    percent = (100.0 * step / total) if total else 0.0
    print(f"status=running checkpoint={checkpoint}")
    print(f"step={step}/{total} progress={percent:.2f}% epoch={epoch}")
    print(f"elapsed={seconds_text(elapsed)} eta={seconds_text(eta)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/text2sql_qwen3/outputs/qwen3_8b_bird_lora"))
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    while True:
        report(args.output_dir)
        if not args.follow:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
