#!/usr/bin/env python3
"""Execute Mini-Dev SQLite predictions and report execution accuracy (EX)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


EXPECTED_QUESTIONS = 500
DEFAULT_MINIDEV_ROOT = Path("/root/autodl-tmp/text2sql_qwen3/eval/MINIDEV")


class MiniDevValidationError(ValueError):
    """Raised when the benchmark payload or prediction file is malformed."""


class QueryTimeoutError(RuntimeError):
    """Raised when SQLite's progress handler reaches the query deadline."""


class QueryResultLimitError(RuntimeError):
    """Raised when a query returns too many rows to compare safely."""


@dataclass(frozen=True)
class Example:
    index: int
    question_id: int
    db_id: str
    difficulty: str
    gold_sql: str


def fail(message: str) -> None:
    raise SystemExit(f"EVALUATION_FAILED: {message}")


def load_minidev(root: Path, expected_questions: int) -> list[Example]:
    questions_path = root / "mini_dev_sqlite.json"
    gold_path = root / "mini_dev_sqlite_gold.sql"
    database_root = root / "dev_databases"
    if not questions_path.is_file() or not gold_path.is_file() or not database_root.is_dir():
        raise MiniDevValidationError(
            f"expected mini_dev_sqlite.json, mini_dev_sqlite_gold.sql, and dev_databases under {root}"
        )

    try:
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MiniDevValidationError(f"invalid Mini-Dev JSON: {error}") from error
    if not isinstance(questions, list) or len(questions) != expected_questions:
        raise MiniDevValidationError(f"expected {expected_questions} questions, found {len(questions)}")

    gold_lines = [line for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(gold_lines) != expected_questions:
        raise MiniDevValidationError(f"expected {expected_questions} gold queries, found {len(gold_lines)}")

    examples: list[Example] = []
    for index, (question, gold_line) in enumerate(zip(questions, gold_lines), start=1):
        if not isinstance(question, dict):
            raise MiniDevValidationError(f"question {index} is not an object")
        question_id = question.get("question_id")
        db_id = question.get("db_id")
        if isinstance(question_id, bool) or not isinstance(question_id, int) or not isinstance(db_id, str):
            raise MiniDevValidationError(f"question {index} is missing an integer question_id or string db_id")
        try:
            gold_sql, gold_db_id = gold_line.rsplit("\t", 1)
        except ValueError as error:
            raise MiniDevValidationError(f"gold line {index} is missing its tab-separated db_id") from error
        if db_id != gold_db_id:
            raise MiniDevValidationError(
                f"Mini-Dev db mismatch at question {question_id}: JSON={db_id}, gold={gold_db_id}"
            )
        database_path = database_root / db_id / f"{db_id}.sqlite"
        if not database_path.is_file():
            raise MiniDevValidationError(f"missing SQLite database for question {question_id}: {database_path}")
        examples.append(
            Example(
                index=index,
                question_id=question_id,
                db_id=db_id,
                difficulty=str(question.get("difficulty", "unknown")),
                gold_sql=gold_sql,
            )
        )
    return examples


def parse_question_id(value: Any, source: str) -> int:
    if isinstance(value, bool):
        raise MiniDevValidationError(f"{source}: question_id must be an integer")
    try:
        question_id = int(value)
    except (TypeError, ValueError) as error:
        raise MiniDevValidationError(f"{source}: question_id must be an integer") from error
    if str(question_id) != str(value).strip() and not isinstance(value, int):
        raise MiniDevValidationError(f"{source}: question_id must not contain non-integer text")
    return question_id


def load_predictions(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise MiniDevValidationError(f"prediction file does not exist: {path}")
    predictions: dict[int, str] = {}
    sql_keys = ("sql", "SQL", "predicted_sql", "prediction")
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise MiniDevValidationError(f"prediction line {line_number} is not valid JSON: {error}") from error
            if not isinstance(payload, dict):
                raise MiniDevValidationError(f"prediction line {line_number} must be a JSON object")

            if "question_id" in payload:
                question_id = parse_question_id(payload["question_id"], f"prediction line {line_number}")
                sql = next((payload[key] for key in sql_keys if key in payload), None)
            elif len(payload) == 1:
                raw_question_id, sql = next(iter(payload.items()))
                question_id = parse_question_id(raw_question_id, f"prediction line {line_number}")
            else:
                raise MiniDevValidationError(
                    f"prediction line {line_number} needs question_id plus sql, or a one-key id-to-SQL mapping"
                )
            if not isinstance(sql, str):
                raise MiniDevValidationError(f"prediction line {line_number}: SQL must be a string")
            if question_id in predictions:
                raise MiniDevValidationError(f"duplicate prediction for question_id {question_id}")
            predictions[question_id] = sql
    return predictions


def remove_leading_comments(sql: str) -> str:
    remaining = sql.lstrip()
    while True:
        if remaining.startswith("--"):
            newline = remaining.find("\n")
            remaining = "" if newline < 0 else remaining[newline + 1 :].lstrip()
        elif remaining.startswith("/*"):
            end = remaining.find("*/", 2)
            if end < 0:
                return ""
            remaining = remaining[end + 2 :].lstrip()
        else:
            return remaining


def is_readonly_select(sql: str) -> bool:
    first_token = remove_leading_comments(sql).split(None, 1)
    return bool(first_token) and first_token[0].upper() in {"SELECT", "WITH"}


def normalize_value(value: Any) -> tuple[str, str]:
    if value is None:
        return ("null", "")
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, bool):
        return ("number", "1" if value else "0")
    if isinstance(value, int):
        return ("number", str(value))
    if isinstance(value, float):
        if math.isnan(value):
            return ("number", "nan")
        if math.isinf(value):
            return ("number", "infinity" if value > 0 else "-infinity")
        try:
            decimal = Decimal(str(value)).normalize()
        except InvalidOperation:
            return ("number", repr(value))
        if decimal == 0:
            decimal = Decimal(0)
        return ("number", format(decimal, "f"))
    return ("text", str(value))


def summarize_rows(rows: list[tuple[Any, ...]]) -> tuple[Counter[str], str]:
    multiset: Counter[str] = Counter()
    for row in rows:
        normalized = [normalize_value(value) for value in row]
        multiset[json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))] += 1
    canonical = json.dumps(sorted(multiset.items()), ensure_ascii=True, separators=(",", ":"))
    return multiset, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def execute_readonly_sql(database_path: Path, sql: str, timeout_seconds: float, max_rows: int) -> dict[str, Any]:
    if not is_readonly_select(sql):
        return {"status": "invalid_read_only", "error": "SQL must begin with SELECT or WITH"}

    deadline = time.monotonic() + timeout_seconds
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
        connection.execute("PRAGMA query_only = ON")
        allowed_actions = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            getattr(sqlite3, "SQLITE_RECURSIVE", -1),
        }
        connection.set_authorizer(
            lambda action, _arg1, _arg2, _db_name, _trigger: sqlite3.SQLITE_OK
            if action in allowed_actions
            else sqlite3.SQLITE_DENY
        )

        def enforce_timeout() -> int:
            return int(time.monotonic() >= deadline)

        connection.set_progress_handler(enforce_timeout, 1_000)
        cursor = connection.execute(sql)
        rows: list[tuple[Any, ...]] = []
        while True:
            batch = cursor.fetchmany(min(1_024, max_rows + 1 - len(rows)))
            if not batch:
                break
            rows.extend(batch)
            if len(rows) > max_rows:
                raise QueryResultLimitError(f"query returned more than {max_rows} rows")
        multiset, signature = summarize_rows(rows)
        return {
            "status": "ok",
            "row_count": len(rows),
            "signature": signature,
            "multiset": multiset,
        }
    except QueryResultLimitError as error:
        return {"status": "result_limit", "error": str(error)}
    except sqlite3.Error as error:
        if time.monotonic() >= deadline or "interrupted" in str(error).lower():
            return {"status": "timeout", "error": f"exceeded {timeout_seconds:g}s"}
        return {"status": "execution_error", "error": str(error)[:500]}
    finally:
        if connection is not None:
            connection.close()


def result_fields(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_status": result["status"],
        f"{prefix}_error": result.get("error"),
        f"{prefix}_row_count": result.get("row_count"),
        f"{prefix}_signature": result.get("signature"),
    }


def evaluate(
    examples: list[Example], predictions: dict[int, str], minidev_root: Path, timeout_seconds: float, max_rows: int, benchmark: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    known_ids = {example.question_id for example in examples}
    unexpected_ids = sorted(set(predictions) - known_ids)
    if unexpected_ids:
        raise MiniDevValidationError(f"predictions include unknown question_ids: {unexpected_ids[:10]}")

    database_root = minidev_root / "dev_databases"
    per_question: list[dict[str, Any]] = []
    valid_sql = 0
    correct = 0
    gold_failures = 0
    for position, example in enumerate(examples, start=1):
        record: dict[str, Any] = {
            "example_index": example.index,
            "question_id": example.question_id,
            "db_id": example.db_id,
            "difficulty": example.difficulty,
            "correct": False,
        }
        prediction = predictions.get(example.question_id)
        if prediction is None:
            record.update(result_fields("prediction", {"status": "missing"}))
            record.update(result_fields("gold", {"status": "not_run"}))
            per_question.append(record)
            continue

        database_path = database_root / example.db_id / f"{example.db_id}.sqlite"
        prediction_result = execute_readonly_sql(database_path, prediction, timeout_seconds, max_rows)
        record.update(result_fields("prediction", prediction_result))
        if prediction_result["status"] != "ok":
            record.update(result_fields("gold", {"status": "not_run"}))
            per_question.append(record)
            continue

        valid_sql += 1
        gold_result = execute_readonly_sql(database_path, example.gold_sql, timeout_seconds, max_rows)
        record.update(result_fields("gold", gold_result))
        if gold_result["status"] != "ok":
            gold_failures += 1
        else:
            record["correct"] = prediction_result["multiset"] == gold_result["multiset"]
            correct += int(record["correct"])
        per_question.append(record)
        if position % 50 == 0 or position == len(examples):
            print(f"evaluated={position}/{len(examples)}", flush=True)

    total = len(examples)
    by_difficulty: dict[str, dict[str, int]] = {}
    for record in per_question:
        bucket = by_difficulty.setdefault(record["difficulty"], {"total": 0, "valid_sql": 0, "correct": 0})
        bucket["total"] += 1
        bucket["valid_sql"] += int(record["prediction_status"] == "ok")
        bucket["correct"] += int(record["correct"])
    for bucket in by_difficulty.values():
        bucket["valid_sql_rate"] = round(bucket["valid_sql"] / bucket["total"], 6)
        bucket["ex"] = round(bucket["correct"] / bucket["total"], 6)

    summary = {
        "benchmark": benchmark,
        "total_questions": total,
        "predictions_provided": len(predictions),
        "valid_sql": valid_sql,
        "valid_sql_rate": round(valid_sql / total, 6),
        "execution_correct": correct,
        "ex": round(correct / total, 6),
        "gold_query_failures": gold_failures,
        "timeout_seconds": timeout_seconds,
        "max_rows": max_rows,
        "by_difficulty": by_difficulty,
    }
    return per_question, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL: question_id plus sql on each line")
    parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=EXPECTED_QUESTIONS)
    parser.add_argument("--benchmark", default="BIRD Mini-Dev SQLite")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=100_000)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.expected_questions < 1:
        fail("--timeout-seconds and --expected-questions must be positive")
    if args.max_rows <= 0:
        fail("--max-rows must be positive")

    try:
        examples = load_minidev(args.minidev_root, args.expected_questions)
        predictions = load_predictions(args.predictions)
        per_question, summary = evaluate(
            examples, predictions, args.minidev_root, args.timeout_seconds, args.max_rows, args.benchmark
        )
    except MiniDevValidationError as error:
        fail(str(error))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "per_question.jsonl"
    with results_path.open("w", encoding="utf-8") as output:
        for record in per_question:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary.update(
        {
            "minidev_root": str(args.minidev_root.resolve()),
            "predictions": str(args.predictions.resolve()),
            "per_question_results": str(results_path.resolve()),
        }
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
