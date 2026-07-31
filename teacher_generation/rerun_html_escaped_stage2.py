from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path

from generate_teacher_data import load_env
from repair_with_gold_plan import load_records, run_stage_two, write_merged_sft


def main() -> int:
    root = Path("BIRD/teacher_generation")
    original = load_records(root / "teacher_full_output_contract_60s.jsonl")
    stage_one = load_records(root / "repair_stage1_gold_conditioned.jsonl")
    stage_two = load_records(root / "repair_stage2_blind_derivation.jsonl")
    output = root / "repair_stage2_html_escape_rerun.jsonl"
    rerun = load_records(output)
    ids = [
        record_id for record_id, record in stage_two.items()
        if not record["accepted_for_sft"]
        and record["teacher_execution"].get("error") in {"no such column: lt", "near \";\": syntax error"}
        and record_id not in rerun
    ]
    load_env(root / "teacher.env")
    kwargs = {
        "database_root": Path("BIRD/train/train_databases"),
        "base_url": os.environ["TEACHER_BASE_URL"],
        "api_key": os.environ["TEACHER_API_KEY"],
        "model": os.environ.get("TEACHER_MODEL", "gpt-5.6-terra"),
        "reasoning_effort": os.environ.get("TEACHER_REASONING_EFFORT", "high"),
        "max_result_rows": 1_000,
        "sql_timeout_seconds": 60.0,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor, output.open("a", encoding="utf-8") as target:
        futures = {executor.submit(run_stage_two, original[i], stage_one[i], **kwargs): i for i in ids}
        for future in concurrent.futures.as_completed(futures):
            target.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            target.flush()
    stage_two.update(load_records(output))
    existing, repaired = write_merged_sft(original, stage_two, root / "teacher_sft_merged_two_stage.jsonl")
    print(f"rerun={len(ids)} merged={existing + repaired} repaired={repaired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
