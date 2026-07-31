#!/usr/bin/env python3
"""Prepare SFT-matched BIRD Mini-Dev data and extract SQL predictions.

The ``prepare`` command reads the actual SFT JSONL to reuse its system prompt
verbatim.  It writes JSONL records with ``question_id``, ``db_id``, and a
two-message chat payload suitable for a chat inference runner.

The ``prepare-sft-dev`` command writes a three-message, JSONL validation set
whose outer schema is exactly the same as the SFT dataset.  It is for
validation-loss compatibility only: its public gold SQL is an assistant target
and must never be supplied to Mini-Dev inference.

The ``extract`` command accepts JSONL inference responses.  Responses may
carry their ``question_id`` or may be in exactly the same order as the prompt
file.  It extracts the required ``<sql>`` block and writes the prediction
format consumed by ``evaluate_minidev.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote


EXPECTED_QUESTIONS = 500
DEFAULT_MINIDEV_ROOT = Path("BIRD/mini_dev/minidev/MINIDEV")
DEFAULT_SFT_DATASET = Path("BIRD/teacher_generation/qwen_sft_messages.jsonl")
SQL_BLOCK = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.IGNORECASE | re.DOTALL)


class ValidationError(ValueError):
    """Raised when an input cannot safely produce a benchmark prediction."""


def fail(message: str) -> None:
    raise SystemExit(f"MINIDEV_PREP_FAILED: {message}")


def load_sft_system_prompt(dataset: Path) -> str:
    if not dataset.is_file():
        raise ValidationError(f"SFT dataset does not exist: {dataset}")
    with dataset.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValidationError(f"invalid SFT JSON on line {line_number}: {error}") from error
            messages = record.get("messages")
            if not isinstance(messages, list) or [item.get("role") for item in messages] != [
                "system",
                "user",
                "assistant",
            ]:
                raise ValidationError(f"unexpected SFT message roles on line {line_number}")
            system = messages[0].get("content")
            if not isinstance(system, str) or not system.strip():
                raise ValidationError(f"SFT system message is empty on line {line_number}")
            return system
    raise ValidationError(f"SFT dataset is empty: {dataset}")


def clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def read_csv_rows(path: Path) -> list[dict[str, str | None]]:
    """BIRD descriptions are mostly UTF-8, with a few Windows-1252 files."""
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open(encoding=encoding, newline="") as source:
                return list(csv.DictReader(source))
        except UnicodeDecodeError as error:
            last_error = error
    raise ValidationError(f"could not decode column description CSV {path}: {last_error}")


def load_column_descriptions(database_dir: Path) -> dict[str, dict[str, str]]:
    description_dir = database_dir / "database_description"
    if not description_dir.is_dir():
        return {}

    descriptions: dict[str, dict[str, str]] = {}
    for csv_path in sorted(description_dir.glob("*.csv"), key=lambda path: path.name.casefold()):
        per_column: dict[str, str] = {}
        for row in read_csv_rows(csv_path):
            original_name = clean_text(row.get("original_column_name"))
            display_name = clean_text(row.get("column_name"))
            column_name = original_name or display_name
            if not column_name:
                continue
            details = clean_text(row.get("column_description"))
            values = clean_text(row.get("value_description"))
            if values:
                details = f"{details} Representative values: {values}".strip()
            if details:
                per_column[column_name.casefold()] = details
        descriptions[csv_path.stem.casefold()] = per_column
    return descriptions


def sqlite_uri(path: Path) -> str:
    normalized = str(path.resolve()).replace("\\", "/")
    return f"file:{quote(normalized)}?mode=ro"


def build_schema(sqlite_path: Path, descriptions: dict[str, dict[str, str]]) -> str:
    """Match the SFT schema shape: complete DDL followed by column meanings."""
    with sqlite3.connect(sqlite_uri(sqlite_path), uri=True) as connection:
        rows = connection.execute(
            """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
              AND sql IS NOT NULL
            ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
            """
        ).fetchall()
        if not rows:
            raise ValidationError(f"SQLite database has no visible schema: {sqlite_path}")

        parts: list[str] = []
        for _, table_name, ddl in rows:
            part = str(ddl).rstrip(";") + ";"
            escaped_name = str(table_name).replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{escaped_name}")').fetchall()
            table_descriptions = descriptions.get(str(table_name).casefold(), {})
            meanings = [
                (str(column[1]), table_descriptions.get(str(column[1]).casefold()))
                for column in columns
            ]
            meanings = [(column_name, detail) for column_name, detail in meanings if detail]
            if meanings:
                part += "\nColumn meanings:\n" + "\n".join(
                    f"- {table_name}.{column_name}: {detail}" for column_name, detail in meanings
                )
            parts.append(part)
    return "\n\n".join(parts)


def build_user_prompt(question: str, evidence: str, schema: str) -> str:
    evidence_text = evidence.strip() or "No additional evidence."
    return f"""Task
{question.strip()}

Evidence
{evidence_text}

Complete SQLite schema with column meanings
{schema}
"""


def build_simple_user_prompt(question: str, evidence: str, schema: str) -> str:
    evidence_text = evidence.strip() or "No additional evidence."
    return f"""SQLite database schema with column meanings
{schema}

Evidence
{evidence_text}

Question
{question.strip()}

Output only one executable SQLite SELECT query.
"""


def load_minidev_examples(root: Path) -> list[dict[str, Any]]:
    path = root / "mini_dev_sqlite.json"
    database_root = root / "dev_databases"
    if not path.is_file() or not database_root.is_dir():
        raise ValidationError(f"expected mini_dev_sqlite.json and dev_databases under {root}")
    try:
        examples = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid Mini-Dev JSON: {error}") from error
    if not isinstance(examples, list) or len(examples) != EXPECTED_QUESTIONS:
        raise ValidationError(f"expected {EXPECTED_QUESTIONS} Mini-Dev examples, found {len(examples)}")

    identities: dict[int, tuple[str, str, str]] = {}
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            raise ValidationError(f"Mini-Dev example {index} is not an object")
        question_id = example.get("question_id")
        db_id = example.get("db_id")
        question = example.get("question")
        if isinstance(question_id, bool) or not isinstance(question_id, int):
            raise ValidationError(f"Mini-Dev example {index} has no integer question_id")
        if not isinstance(db_id, str) or not db_id:
            raise ValidationError(f"Mini-Dev example {question_id} has no db_id")
        if not isinstance(question, str) or not question.strip():
            raise ValidationError(f"Mini-Dev example {question_id} has no question")
        identity = (db_id, question.strip(), clean_text(example.get("evidence")))
        previous = identities.setdefault(question_id, identity)
        if previous != identity:
            raise ValidationError(
                f"question_id {question_id} refers to different Mini-Dev tasks; "
                "evaluate_minidev.py cannot represent that safely"
            )
        sqlite_path = database_root / db_id / f"{db_id}.sqlite"
        if not sqlite_path.is_file():
            raise ValidationError(f"missing SQLite database for question {question_id}: {sqlite_path}")
    return examples


def load_minidev_gold_sql(root: Path, examples: list[dict[str, Any]]) -> list[str]:
    path = root / "mini_dev_sqlite_gold.sql"
    if not path.is_file():
        raise ValidationError(f"Mini-Dev gold SQL file does not exist: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != len(examples):
        raise ValidationError(f"expected {len(examples)} Mini-Dev gold queries, found {len(lines)}")

    gold_sql: list[str] = []
    for index, (example, line) in enumerate(zip(examples, lines), start=1):
        try:
            sql, db_id = line.rsplit("\t", 1)
        except ValueError as error:
            raise ValidationError(f"gold line {index} is missing its tab-separated db_id") from error
        if db_id != example["db_id"]:
            raise ValidationError(
                f"Mini-Dev db mismatch at question {example['question_id']}: "
                f"JSON={example['db_id']}, gold={db_id}"
            )
        if not re.match(r"(?is)^\s*(select|with)\b", sql):
            raise ValidationError(f"gold SQL for question {example['question_id']} is not a SELECT or WITH query")
        gold_sql.append(sql.strip())
    return gold_sql


def build_gold_target(gold_sql: str) -> str:
    """Create the required three-block target without inventing rationales."""
    return f"""<schema_linking>
- Use the complete SQLite schema and authoritative evidence supplied in the user message.
</schema_linking>

<query_plan>
1. Apply the task constraints and evidence to the supplied SQLite schema.
2. Return only the requested result columns and rows.
</query_plan>

<sql>
{gold_sql}
</sql>"""


def prepare(args: argparse.Namespace) -> int:
    system_prompt = load_sft_system_prompt(args.sft_dataset)
    examples = load_minidev_examples(args.minidev_root)
    database_root = args.minidev_root / "dev_databases"
    schema_cache: dict[str, str] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as destination:
        for index, example in enumerate(examples, start=1):
            db_id = example["db_id"]
            if db_id not in schema_cache:
                database_dir = database_root / db_id
                schema_cache[db_id] = build_schema(
                    database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir)
                )
            schema = schema_cache[db_id]
            user_prompt = build_user_prompt(example["question"], clean_text(example.get("evidence")), schema)
            record = {
                "example_index": index,
                "question_id": example["question_id"],
                "db_id": db_id,
                "difficulty": str(example.get("difficulty", "unknown")),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
                "prompt_sha256": hashlib.sha256((system_prompt + "\n" + user_prompt).encode("utf-8")).hexdigest(),
                "prompt_characters": len(system_prompt) + len(user_prompt),
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(examples),
                "databases": len(schema_cache),
                "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "max_prompt_characters": max(
                    len(system_prompt)
                    + len(build_user_prompt(example["question"], clean_text(example.get("evidence")), schema_cache[example["db_id"]]))
                    for example in examples
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def prepare_sft_dev(args: argparse.Namespace) -> int:
    """Write Mini-Dev labels in the exact JSONL shape expected by SFT loaders."""
    system_prompt = load_sft_system_prompt(args.sft_dataset)
    examples = load_minidev_examples(args.minidev_root)
    gold_sql = load_minidev_gold_sql(args.minidev_root, examples)
    database_root = args.minidev_root / "dev_databases"
    schema_cache: dict[str, str] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as destination:
        for example, sql in zip(examples, gold_sql):
            db_id = example["db_id"]
            if db_id not in schema_cache:
                database_dir = database_root / db_id
                schema_cache[db_id] = build_schema(
                    database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir)
                )
            user_prompt = build_user_prompt(example["question"], clean_text(example.get("evidence")), schema_cache[db_id])
            destination.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": build_gold_target(sql)},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(examples),
                "databases": len(schema_cache),
                "format": "SFT messages: system, user, assistant",
                "use": "validation loss only; do not use this file for Mini-Dev inference",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def prepare_simple(args: argparse.Namespace) -> int:
    """Write user-only prompts with schema, evidence, and question context."""
    examples = load_minidev_examples(args.minidev_root)
    database_root = args.minidev_root / "dev_databases"
    schema_cache: dict[str, str] = {}
    prompt_lengths: list[int] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as destination:
        for index, example in enumerate(examples, start=1):
            db_id = example["db_id"]
            if db_id not in schema_cache:
                database_dir = database_root / db_id
                schema_cache[db_id] = build_schema(
                    database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir)
                )
            user_prompt = build_simple_user_prompt(
                example["question"], clean_text(example.get("evidence")), schema_cache[db_id]
            )
            prompt_lengths.append(len(user_prompt))
            destination.write(
                json.dumps(
                    {
                        "example_index": index,
                        "question_id": example["question_id"],
                        "db_id": db_id,
                        "difficulty": str(example.get("difficulty", "unknown")),
                        "messages": [{"role": "user", "content": user_prompt}],
                        "schema_sha256": hashlib.sha256(schema_cache[db_id].encode("utf-8")).hexdigest(),
                        "prompt_characters": len(user_prompt),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(examples),
                "databases": len(schema_cache),
                "min_prompt_characters": min(prompt_lengths),
                "mean_prompt_characters": round(sum(prompt_lengths) / len(prompt_lengths), 2),
                "max_prompt_characters": max(prompt_lengths),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def parse_question_id(value: Any, source: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{source}: question_id must be an integer")
    try:
        question_id = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{source}: question_id must be an integer") from error
    if not isinstance(value, int) and str(question_id) != str(value).strip():
        raise ValidationError(f"{source}: invalid question_id {value!r}")
    return question_id


def response_content(payload: Any, source: str) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        raise ValidationError(f"{source}: response must be a JSON object or string")
    for key in ("response", "output", "generated_text", "prediction", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return message["content"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    raise ValidationError(f"{source}: could not find generated assistant text")


def extract_sql(content: str, source: str) -> str:
    match = SQL_BLOCK.search(content)
    sql = match.group(1).strip().strip("`").strip() if match else content.strip()
    fenced = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", sql, re.IGNORECASE | re.DOTALL)
    if fenced:
        sql = fenced.group(1).strip()
    if not re.match(r"(?is)^(select|with)\b", sql):
        # A malformed or truncated completion is an invalid prediction, not invalid benchmark input.
        return ""
    return sql


def load_prompt_records(path: Path, expected_prompts: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"prompt file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValidationError(f"prompt line {line_number} is invalid JSON: {error}") from error
            if not isinstance(record, dict) or "question_id" not in record or "db_id" not in record:
                raise ValidationError(f"prompt line {line_number} needs question_id and db_id")
            records.append(record)
    if len(records) != expected_prompts:
        raise ValidationError(f"expected {expected_prompts} prompt records, found {len(records)}")
    identities: dict[int, tuple[str, str]] = {}
    for record in records:
        question_id = parse_question_id(record["question_id"], "prompt")
        messages = record.get("messages")
        roles = [message.get("role") for message in messages] if isinstance(messages, list) else []
        if roles not in (["system", "user"], ["user"]):
            raise ValidationError(f"prompt question_id {question_id} needs system/user or user-only messages")
        user_content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
        if not isinstance(user_content, str):
            raise ValidationError(f"prompt question_id {question_id} has no user content")
        identity = (str(record["db_id"]), user_content)
        previous = identities.setdefault(question_id, identity)
        if previous != identity:
            raise ValidationError(
                f"question_id {question_id} refers to different prompts; "
                "evaluate_minidev.py cannot represent that safely"
            )
    return records


def extract(args: argparse.Namespace) -> int:
    if args.expected_prompts < 1:
        fail("--expected-prompts must be positive")
    prompts = load_prompt_records(args.prompts, args.expected_prompts)
    if not args.responses.is_file():
        raise ValidationError(f"response file does not exist: {args.responses}")
    prompt_by_id: dict[int, dict[str, Any]] = {}
    prompt_by_index: dict[int, dict[str, Any]] = {}
    for prompt in prompts:
        question_id = parse_question_id(prompt["question_id"], "prompt")
        prompt_by_id.setdefault(question_id, prompt)
        example_index = prompt.get("example_index")
        if isinstance(example_index, bool) or not isinstance(example_index, int):
            raise ValidationError(f"prompt question_id {question_id} has no integer example_index")
        prompt_by_index[example_index] = prompt
    predictions_by_id: dict[int, str] = {}
    prediction_source_index: dict[int, int] = {}
    fallback_index = 0

    with args.responses.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValidationError(f"response line {line_number} is invalid JSON: {error}") from error
            source_name = f"response line {line_number}"
            prompt: dict[str, Any]
            if isinstance(payload, dict) and "example_index" in payload:
                example_index = parse_question_id(payload["example_index"], source_name)
                prompt = prompt_by_index.get(example_index, {})
                if not prompt:
                    raise ValidationError(f"{source_name}: unknown example_index {example_index}")
                question_id = parse_question_id(prompt["question_id"], "prompt")
            elif isinstance(payload, dict) and "question_id" in payload:
                question_id = parse_question_id(payload["question_id"], source_name)
                prompt = prompt_by_id.get(question_id, {})
                if not prompt:
                    raise ValidationError(f"{source_name}: unknown question_id {question_id}")
            else:
                if fallback_index >= len(prompts):
                    raise ValidationError(f"{source_name}: more responses than prompts")
                prompt = prompts[fallback_index]
                question_id = parse_question_id(prompt["question_id"], "prompt")
                fallback_index += 1
            sql = extract_sql(response_content(payload, source_name), source_name)
            canonical_prompt = prompt_by_id[question_id]
            canonical_index = canonical_prompt["example_index"]
            source_index = prompt["example_index"]
            if question_id not in predictions_by_id or source_index == canonical_index:
                predictions_by_id[question_id] = sql
                prediction_source_index[question_id] = source_index
            elif source_index == prediction_source_index[question_id] and predictions_by_id[question_id] != sql:
                raise ValidationError(
                    f"{source_name}: duplicate response for canonical question_id {question_id} produced different SQL"
                )

    if not args.allow_partial and set(predictions_by_id) != set(prompt_by_id):
        missing = sorted(set(prompt_by_id) - set(predictions_by_id))
        raise ValidationError(f"missing predictions for {len(missing)} question_ids: {missing[:10]}")
    predictions: list[dict[str, Any]] = []
    emitted: set[int] = set()
    for prompt in prompts:
        question_id = parse_question_id(prompt["question_id"], "prompt")
        if question_id in emitted or question_id not in predictions_by_id:
            continue
        predictions.append({"question_id": question_id, "db_id": prompt["db_id"], "sql": predictions_by_id[question_id]})
        emitted.add(question_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for prediction in predictions:
            destination.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "predictions": len(predictions)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="write SFT-matched Mini-Dev chat messages")
    prepare_parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    prepare_parser.add_argument("--sft-dataset", type=Path, default=DEFAULT_SFT_DATASET)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.set_defaults(handler=prepare)

    sft_dev_parser = commands.add_parser(
        "prepare-sft-dev", help="write a three-message Mini-Dev validation-loss dataset"
    )
    sft_dev_parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    sft_dev_parser.add_argument("--sft-dataset", type=Path, default=DEFAULT_SFT_DATASET)
    sft_dev_parser.add_argument("--output", type=Path, required=True)
    sft_dev_parser.set_defaults(handler=prepare_sft_dev)

    simple_parser = commands.add_parser("prepare-simple", help="write user-only schema/evidence/question Mini-Dev prompts")
    simple_parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    simple_parser.add_argument("--output", type=Path, required=True)
    simple_parser.set_defaults(handler=prepare_simple)

    extract_parser = commands.add_parser("extract", help="extract <sql> blocks to evaluator prediction JSONL")
    extract_parser.add_argument("--prompts", type=Path, required=True)
    extract_parser.add_argument("--responses", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--expected-prompts", type=int, default=EXPECTED_QUESTIONS)
    extract_parser.add_argument("--allow-partial", action="store_true")
    extract_parser.set_defaults(handler=extract)

    args = parser.parse_args()
    try:
        return args.handler(args)
    except ValidationError as error:
        fail(str(error))


if __name__ == "__main__":
    sys.exit(main())
