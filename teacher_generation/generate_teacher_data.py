from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


SYSTEM_PROMPT = """You are a senior Text-to-SQL engineer. Solve the user task using only the supplied complete SQLite schema, column meanings, and evidence.

Your response must contain exactly three XML-like blocks in this order, with no Markdown fences, commentary, or text outside the blocks.

1. <schema_linking> answers: Which exact tables, columns, literal values, and join relationships are required? Map each meaningful phrase in the task or evidence to exact table.column identifiers. Include join keys when needed.
2. <query_plan> answers: How should those schema items be combined? State a short numbered relational plan covering filters, joins, grouping, aggregation, ordering, limits, and tie handling when applicable.
3. <sql> answers: What single executable SQLite query implements that plan? It must be one read-only SELECT statement; a WITH query is allowed. Use only supplied identifiers and no placeholders or explanation.

The evidence is authoritative for its stated mappings and constraints. DDL defines the physical schema and foreign keys; column meanings explain business semantics, abbreviations, and representative values. Preserve the requested output semantics, including ties, ordering, and limits.

In the SQL SELECT output, preserve stored database values.
Do not use CASE or convert flags/codes such as 0/1 into natural-language labels, unless the question explicitly requires an exact literal label.
Return only explicitly requested columns.
Inside the <sql> block, write literal SQLite operators: <, >, <=, >=, <>.
Never HTML-escape SQL characters. Never output &lt;, &gt;, or &amp;.

Here is an illustrative BIRD example. It demonstrates the response contract only: never reuse its tables, columns, values, or SQL unless they appear in the current task's supplied schema.

Example schema excerpt from BIRD database `book_publishing_company`:
CREATE TABLE sales (
    stor_id TEXT NOT NULL REFERENCES stores(stor_id),
    ord_date DATETIME NOT NULL,
    qty INTEGER NOT NULL,
    title_id TEXT NOT NULL REFERENCES titles(title_id)
);
CREATE TABLE stores (
    stor_id TEXT PRIMARY KEY,
    stor_name TEXT,
    city TEXT
);
Column meanings: `sales.qty` is the number of sales transactions; `sales.ord_date` is the date when an order was placed; `stores.stor_name` is the bookstore name.

Example task: For each store, return its name and total quantity sold during 1993. Keep only stores with at least 100 units sold, and rank them from the largest total to the smallest.
Example evidence: qty is the sales quantity; during 1993 means the year of sales.ord_date is 1993.

Example response:
<schema_linking>
- store name -> stores.stor_name
- store identity and join -> stores.stor_id = sales.stor_id
- total quantity sold -> SUM(sales.qty)
- during 1993 -> strftime('%Y', sales.ord_date) = '1993'
- at least 100 units -> HAVING SUM(sales.qty) >= 100
</schema_linking>

<query_plan>
1. Join stores to sales through stor_id.
2. Keep sales whose order date falls in 1993.
3. Group rows by store and sum qty for each store.
4. Retain groups with totals of at least 100 and sort totals descending.
</query_plan>

<sql>
SELECT s.stor_name, SUM(sa.qty) AS total_qty
FROM stores AS s
JOIN sales AS sa ON s.stor_id = sa.stor_id
WHERE strftime('%Y', sa.ord_date) = '1993'
GROUP BY s.stor_id, s.stor_name
HAVING SUM(sa.qty) >= 100
ORDER BY total_qty DESC;
</sql>"""


TAGGED_RESPONSE = re.compile(
    r"\A\s*<schema_linking>\s*(?P<linking>.*?)\s*</schema_linking>\s*"
    r"<query_plan>\s*(?P<plan>.*?)\s*</query_plan>\s*"
    r"<sql>\s*(?P<sql>.*?)\s*</sql>\s*\Z",
    re.DOTALL,
)


def load_env(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid environment line in {path}: {raw_line!r}")
        os.environ.setdefault(key.strip(), value.strip())


def load_column_meanings(path: Path) -> dict[tuple[str, str, str], str]:
    raw_meanings = json.loads(path.read_text(encoding="utf-8"))
    meanings: dict[tuple[str, str, str], str] = {}
    for key, description in raw_meanings.items():
        db_id, table_name, column_name = key.split("|", 2)
        meanings[(db_id, table_name, column_name)] = description
    return meanings


def sqlite_schema(sqlite_path: Path, db_id: str, column_meanings: dict[tuple[str, str, str], str]) -> str:
    uri = f"file:{quote(str(sqlite_path.resolve()).replace(os.sep, '/'))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
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
        schema_parts: list[str] = []
        for _, table_name, ddl in rows:
            part = ddl.rstrip(";") + ";"
            escaped_table_name = table_name.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{escaped_table_name}")').fetchall()
            descriptions = [
                (column[1], column_meanings[(db_id, table_name, column[1])])
                for column in columns
                if (db_id, table_name, column[1]) in column_meanings
            ]
            if descriptions:
                part += "\nColumn meanings:\n" + "\n".join(
                    f"- {table_name}.{column_name}: {description}"
                    for column_name, description in descriptions
                )
            schema_parts.append(part)
    return "\n\n".join(schema_parts)


def build_user_prompt(question: str, evidence: str, schema: str) -> str:
    evidence_text = evidence.strip() or "No additional evidence."
    return f"""Task
{question.strip()}

Evidence
{evidence_text}

Complete SQLite schema with column meanings
{schema}
"""


def parse_teacher_response(content: str) -> dict[str, Any]:
    match = TAGGED_RESPONSE.fullmatch(content)
    if not match:
        return {
            "format_ok": False,
            "error": "Response must contain only schema_linking, query_plan, and sql blocks in that order.",
            "schema_linking": None,
            "query_plan": None,
            "sql": None,
        }

    linking = match.group("linking").strip()
    plan = match.group("plan").strip()
    sql = match.group("sql").strip().strip("`").strip()
    errors: list[str] = []
    if not linking:
        errors.append("schema_linking is empty")
    if not plan:
        errors.append("query_plan is empty")
    if not sql:
        errors.append("sql is empty")
    if sql and not re.match(r"(?is)^\s*(select|with)\b", sql):
        errors.append("SQL is not a SELECT or WITH query")
    return {
        "format_ok": not errors,
        "error": "; ".join(errors) if errors else None,
        "schema_linking": linking or None,
        "query_plan": plan or None,
        "sql": sql or None,
    }


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value


def read_only_authorizer(action: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger: str | None) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def execute_sql(sqlite_path: Path, sql: str | None, max_rows: int, timeout_seconds: float) -> dict[str, Any]:
    if not sql:
        return {"ok": False, "error": "No SQL was parsed.", "columns": [], "rows": [], "truncated": False}
    candidate = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", candidate):
        return {"ok": False, "error": "Only SELECT or WITH statements may execute.", "columns": [], "rows": [], "truncated": False}

    deadline = time.monotonic() + timeout_seconds

    def progress_handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    uri = f"file:{quote(str(sqlite_path.resolve()).replace(os.sep, '/'))}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.set_authorizer(read_only_authorizer)
            connection.set_progress_handler(progress_handler, 1_000)
            cursor = connection.execute(candidate)
            columns = [description[0] for description in cursor.description or []]
            rows: list[list[Any]] = []
            truncated = False
            while True:
                batch = cursor.fetchmany(min(100, max_rows - len(rows) + 1))
                if not batch:
                    break
                for row in batch:
                    if len(rows) >= max_rows:
                        truncated = True
                        break
                    rows.append([json_value(value) for value in row])
                if truncated:
                    break
            return {
                "ok": True,
                "error": None,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
            }
    except sqlite3.Error as exc:
        return {"ok": False, "error": str(exc), "columns": [], "rows": [], "truncated": False}


def unordered_result_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool | None:
    if not left["ok"] or not right["ok"] or left.get("truncated") or right.get("truncated"):
        return None
    left_rows = Counter(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for row in left["rows"])
    right_rows = Counter(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for row in right["rows"])
    return left_rows == right_rows


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    user_prompt: str,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": ([{"role": "user", "content": user_prompt}]
                     if system_prompt is None
                     else [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]),
        "reasoning_effort": reasoning_effort,
    }
    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"].get("content")
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Teacher returned no textual assistant content.")
            return content, body.get("usage"), None
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            last_error = str(exc)
            if attempt < 3:
                time.sleep(attempt * 3)
    return None, None, last_error


def existing_record_ids(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    record_ids: set[int] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record_ids.add(json.loads(line)["record_id"])
    return record_ids


def build_record(
    record_id: int,
    source_record: dict[str, Any],
    sqlite_path: Path,
    schema: str,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    max_result_rows: int,
    sql_timeout_seconds: float,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(source_record["question"], source_record.get("evidence", ""), schema)
    raw_response, usage, teacher_error = chat_completion(base_url, api_key, model, reasoning_effort, user_prompt)
    parsed = parse_teacher_response(raw_response) if raw_response else {
        "format_ok": False,
        "error": teacher_error,
        "schema_linking": None,
        "query_plan": None,
        "sql": None,
    }
    teacher_execution = execute_sql(sqlite_path, parsed["sql"], max_result_rows, sql_timeout_seconds)
    gold_execution = execute_sql(sqlite_path, source_record["SQL"], max_result_rows, sql_timeout_seconds)
    equivalent = unordered_result_equivalent(teacher_execution, gold_execution)
    return {
        "record_id": record_id,
        "source": {
            "db_id": source_record["db_id"],
            "question": source_record["question"],
            "evidence": source_record.get("evidence", ""),
            "gold_sql": source_record["SQL"],
        },
        "prompt": {
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        },
        "teacher": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "raw_response": raw_response,
            "usage": usage,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
        ] if raw_response else None,
        "parsed": parsed,
        "teacher_execution": teacher_execution,
        "gold_execution": gold_execution,
        "execution_equivalent_unordered": equivalent,
        "ready_for_sft": bool(parsed["format_ok"] and teacher_execution["ok"] and equivalent),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate structured teacher data for filtered BIRD train.")
    parser.add_argument("--input", type=Path, default=Path("BIRD/train_filtered/train.jsonl"))
    parser.add_argument("--database-root", type=Path, default=Path("BIRD/train/train_databases"))
    parser.add_argument("--column-meanings", type=Path, default=Path("BIRD/train_filtered/train_column_meaning.json"))
    parser.add_argument("--output", type=Path, default=Path("BIRD/teacher_generation/teacher_pilot_10.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path("BIRD/teacher_generation/teacher.env"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-result-rows", type=int, default=1_000)
    parser.add_argument("--sql-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    load_env(args.env_file)
    base_url = os.environ["TEACHER_BASE_URL"]
    api_key = os.environ["TEACHER_API_KEY"]
    model = os.environ.get("TEACHER_MODEL", "gpt-5.6-terra")
    reasoning_effort = os.environ.get("TEACHER_REASONING_EFFORT", "high")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_record_ids(args.output)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    column_meanings = load_column_meanings(args.column_meanings)
    schema_cache: dict[str, str] = {}
    pending: list[tuple[int, dict[str, Any], Path, str]] = []
    with args.input.open(encoding="utf-8") as source:
        for record_id, raw_line in enumerate(source):
            if len(pending) >= args.limit:
                break
            if record_id in done:
                continue
            source_record = json.loads(raw_line)
            db_id = source_record["db_id"]
            sqlite_path = args.database_root / db_id / f"{db_id}.sqlite"
            if not sqlite_path.exists():
                raise FileNotFoundError(f"Missing SQLite database for {db_id}: {sqlite_path}")
            schema = schema_cache.setdefault(db_id, sqlite_schema(sqlite_path, db_id, column_meanings))
            pending.append((record_id, source_record, sqlite_path, schema))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor, args.output.open("a", encoding="utf-8") as destination:
        futures = [
            executor.submit(
                build_record,
                record_id,
                source_record,
                sqlite_path,
                schema,
                base_url,
                api_key,
                model,
                reasoning_effort,
                args.max_result_rows,
                args.sql_timeout_seconds,
            )
            for record_id, source_record, sqlite_path, schema in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            destination.flush()
            print(
                f"record_id={record['record_id']} format_ok={record['parsed']['format_ok']} "
                f"teacher_exec={record['teacher_execution']['ok']} equivalent={record['execution_equivalent_unordered']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
