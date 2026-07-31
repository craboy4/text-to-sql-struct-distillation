#!/usr/bin/env python3
"""Build retrieval-focused Mini-Dev SchemaPacks for the long SFT prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from prepare_minidev_inference import (
    DEFAULT_MINIDEV_ROOT,
    ValidationError,
    clean_text,
    load_column_descriptions,
    load_minidev_examples,
    load_sft_system_prompt,
    sqlite_uri,
)


DEFAULT_SFT_DATASET = Path("BIRD/teacher_generation/qwen_sft_messages.jsonl")
DEFAULT_EMBEDDING_MODEL = Path(
    r"C:\Users\WangYiran\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots"
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CAMEL_CASE = re.compile(r"([a-z])([A-Z])")
VALUE_LITERAL = re.compile(r"(?:'([^']{3,80})'|\"([^\"]{3,80})\")")
TABLE_RERANK_FEATURE_NAMES = (
    "table_bm25_rank",
    "table_dense_rank",
    "best_column_bm25_rank",
    "best_column_dense_rank",
    "evidence_column_match",
)


@dataclass(frozen=True)
class Column:
    table: str
    name: str
    type_name: str
    not_null: bool
    primary_key_rank: int
    description: str


@dataclass(frozen=True)
class ForeignKey:
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass
class DatabaseSchema:
    sqlite_path: Path
    columns_by_table: dict[str, list[Column]]
    foreign_keys: list[ForeignKey]
    table_docs: dict[str, str]
    column_docs: dict[tuple[str, str], str]
    full_schema: str


def terms(text: str) -> list[str]:
    expanded = CAMEL_CASE.sub(r"\1 \2", text).replace("_", " ")
    return TOKEN_PATTERN.findall(expanded.casefold())


def bm25_scores(query: str, documents: list[str]) -> np.ndarray:
    query_terms = terms(query)
    tokenized = [terms(document) for document in documents]
    document_frequency = Counter(term for document in tokenized for term in set(document))
    average_length = sum(len(document) for document in tokenized) / max(len(tokenized), 1)
    scores = np.zeros(len(documents), dtype=np.float32)
    for index, document in enumerate(tokenized):
        counts = Counter(document)
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document) / max(average_length, 1))
            scores[index] += idf * frequency * 2.5 / denominator
    return scores


def rrf_scores(bm25: np.ndarray, dense: np.ndarray, constant: int = 60) -> np.ndarray:
    scores = np.zeros(len(bm25), dtype=np.float32)
    for ranking in (np.argsort(-bm25), np.argsort(-dense)):
        for position, index in enumerate(ranking, start=1):
            scores[index] += 1 / (constant + position)
    return scores


class QwenEmbedder:
    def __init__(self, snapshots: Path, batch_size: int) -> None:
        if not snapshots.is_dir():
            raise ValidationError(f"embedding model snapshots directory does not exist: {snapshots}")
        candidates = [path for path in snapshots.iterdir() if path.is_dir()]
        if len(candidates) != 1:
            raise ValidationError(f"expected one embedding model snapshot under {snapshots}, found {len(candidates)}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(candidates[0], local_files_only=True, padding_side="left")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(candidates[0], local_files_only=True, torch_dtype=dtype).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden_states = self.model(**encoded).last_hidden_state[:, -1]
            normalized = torch.nn.functional.normalize(hidden_states.float(), p=2, dim=1).cpu().numpy()
            vectors.append(normalized)
            if (start // self.batch_size + 1) % 16 == 0 or start + len(batch) == len(texts):
                print(f"embedded={min(start + len(batch), len(texts))}/{len(texts)}", flush=True)
        return np.concatenate(vectors, axis=0)


def load_database_schema(sqlite_path: Path, descriptions: dict[str, dict[str, str]]) -> DatabaseSchema:
    columns_by_table: dict[str, list[Column]] = {}
    foreign_keys: list[ForeignKey] = []
    with sqlite3.connect(sqlite_uri(sqlite_path), uri=True) as connection:
        tables = connection.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
            ORDER BY name
            """
        ).fetchall()
        if not tables:
            raise ValidationError(f"SQLite database has no visible tables: {sqlite_path}")
        for table_name, _ in tables:
            escaped = str(table_name).replace('"', '""')
            table_descriptions = descriptions.get(str(table_name).casefold(), {})
            columns_by_table[str(table_name)] = [
                Column(
                    table=str(table_name),
                    name=str(row[1]),
                    type_name=str(row[2] or "TEXT"),
                    not_null=bool(row[3]),
                    primary_key_rank=int(row[5]),
                    description=table_descriptions.get(str(row[1]).casefold(), ""),
                )
                for row in connection.execute(f'PRAGMA table_info("{escaped}")')
            ]
            for row in connection.execute(f'PRAGMA foreign_key_list("{escaped}")'):
                foreign_keys.append(
                    ForeignKey(
                        source_table=str(table_name),
                        source_column=str(row[3]),
                        target_table=str(row[2]),
                        target_column=str(row[4]),
                    )
                )

    table_docs = {
        table: " ".join(
            [f"table {table}"]
            + [f"column {column.name} {column.description}" for column in columns]
        )
        for table, columns in columns_by_table.items()
    }
    column_docs = {
        (column.table, column.name): f"table {column.table} column {column.name} {column.description}"
        for columns in columns_by_table.values()
        for column in columns
    }
    return DatabaseSchema(
        sqlite_path=sqlite_path,
        columns_by_table=columns_by_table,
        foreign_keys=foreign_keys,
        table_docs=table_docs,
        column_docs=column_docs,
        full_schema=render_schema(
            columns_by_table,
            foreign_keys,
            {table: {column.name for column in columns} for table, columns in columns_by_table.items()},
        ),
    )


def render_schema(
    columns_by_table: dict[str, list[Column]], foreign_keys: list[ForeignKey], selected_columns: dict[str, set[str]]
) -> str:
    parts: list[str] = []
    for table_name in sorted(selected_columns):
        columns = [column for column in columns_by_table[table_name] if column.name in selected_columns[table_name]]
        if not columns:
            continue
        definitions = [
            f'    "{column.name}" {column.type_name}' + (" NOT NULL" if column.not_null else "")
            for column in columns
        ]
        primary_keys = sorted((column for column in columns if column.primary_key_rank), key=lambda column: column.primary_key_rank)
        if primary_keys:
            definitions.append("    PRIMARY KEY (" + ", ".join(f'"{column.name}"' for column in primary_keys) + ")")
        for foreign_key in foreign_keys:
            if (
                foreign_key.source_table == table_name
                and foreign_key.source_column in selected_columns[foreign_key.source_table]
                and foreign_key.target_table in selected_columns
                and foreign_key.target_column in selected_columns[foreign_key.target_table]
            ):
                definitions.append(
                    f'    FOREIGN KEY ("{foreign_key.source_column}") REFERENCES '
                    f'"{foreign_key.target_table}" ("{foreign_key.target_column}")'
                )
        part = f'CREATE TABLE "{table_name}" (\n' + ",\n".join(definitions) + "\n);"
        meanings = [column for column in columns if column.description]
        if meanings:
            part += "\nColumn meanings:\n" + "\n".join(
                f"- {table_name}.{column.name}: {column.description}" for column in meanings
            )
        parts.append(part)
    return "\n\n".join(parts)


def closure_tables(seed_tables: list[str], foreign_keys: list[ForeignKey]) -> set[str]:
    graph: dict[str, set[str]] = defaultdict(set)
    for foreign_key in foreign_keys:
        graph[foreign_key.source_table].add(foreign_key.target_table)
        graph[foreign_key.target_table].add(foreign_key.source_table)
    selected = {seed_tables[0]} if seed_tables else set()
    for target in seed_tables[1:]:
        queue: deque[tuple[str, list[str]]] = deque((node, [node]) for node in selected)
        visited = set(selected)
        path: list[str] | None = None
        while queue:
            node, current_path = queue.popleft()
            if node == target:
                path = current_path
                break
            for neighbor in graph[node] - visited:
                visited.add(neighbor)
                queue.append((neighbor, current_path + [neighbor]))
        selected.update(path or [target])
    return selected


def bidirectional_table_ranking(
    tables: list[str],
    column_keys: list[tuple[str, str]],
    table_scores: np.ndarray,
    column_scores: np.ndarray,
) -> list[str]:
    """Interleave table-first and column-first retrieval candidates."""
    table_rank = [tables[index] for index in np.argsort(-table_scores)]
    best_column_score: dict[str, float] = {}
    for index, (table_name, _) in enumerate(column_keys):
        best_column_score[table_name] = max(best_column_score.get(table_name, float("-inf")), float(column_scores[index]))
    column_table_rank = sorted(tables, key=lambda table_name: (-best_column_score[table_name], table_name))

    merged: list[str] = []
    for table_first, column_first in zip(table_rank, column_table_rank):
        for table_name in (table_first, column_first):
            if table_name not in merged:
                merged.append(table_name)
    return merged


def evidence_identifier_scores(evidence: str, column_keys: list[tuple[str, str]]) -> np.ndarray:
    """Return exact schema-identifier matches explicitly supplied by BIRD evidence."""
    normalized = evidence.casefold()
    return np.array(
        [
            float(bool(re.search(rf"(?<![a-z0-9_]){re.escape(column.casefold())}(?![a-z0-9_])", normalized)))
            for _, column in column_keys
        ],
        dtype=np.float32,
    )


def value_match_scores(schema: DatabaseSchema, text: str, column_keys: list[tuple[str, str]], cache: dict[str, Any]) -> np.ndarray:
    """Locate explicit quoted entities in SQLite and return their exact matching columns."""
    values = sorted(
        {
            next(part for part in match.groups() if part is not None).strip().casefold()
            for match in VALUE_LITERAL.finditer(text)
            if any(character.isalpha() for character in next(part for part in match.groups() if part is not None))
        }
    )
    if not values:
        return np.zeros(len(column_keys), dtype=np.float32)
    value_cache = cache.setdefault("value_match_scores", {})
    cache_key = tuple(values)
    if cache_key in value_cache:
        return value_cache[cache_key]
    value_index = cache.get("value_index")
    if value_index is None:
        value_index = defaultdict(set)
        key_by_column = {key: index for index, key in enumerate(column_keys)}
        with sqlite3.connect(sqlite_uri(schema.sqlite_path), uri=True) as connection:
            for table, columns in schema.columns_by_table.items():
                quoted_table = table.replace('"', '""')
                for column in columns:
                    if "TEXT" not in column.type_name.upper() and "CHAR" not in column.type_name.upper():
                        continue
                    quoted_column = column.name.replace('"', '""')
                    rows = connection.execute(
                        f'SELECT DISTINCT LOWER(TRIM(CAST("{quoted_column}" AS TEXT))) FROM "{quoted_table}" '
                        f'WHERE "{quoted_column}" IS NOT NULL LIMIT 500'
                    )
                    index = key_by_column[(table, column.name)]
                    for (candidate,) in rows:
                        if candidate is not None and 3 <= len(candidate) <= 80 and any(character.isalpha() for character in candidate):
                            value_index[str(candidate)].add(index)
        cache["value_index"] = value_index
    matched = set().union(*(value_index.get(value, set()) for value in values))
    scores = np.array([float(index in matched) for index in range(len(column_keys))], dtype=np.float32)
    value_cache[cache_key] = scores
    return scores


def rank_fractions(scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1], where 1 is the best rank within this schema."""
    if len(scores) <= 1:
        return np.ones(len(scores), dtype=np.float32)
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
    return (1 - ranks / (len(scores) - 1)).astype(np.float32)


def table_rerank_features(
    tables: list[str],
    column_keys: list[tuple[str, str]],
    table_bm25: np.ndarray,
    table_dense: np.ndarray,
    column_bm25: np.ndarray,
    column_dense: np.ndarray,
    evidence_scores: np.ndarray,
) -> np.ndarray:
    """Build bounded, query-local features for the optional train-only table reranker."""
    best_column_bm25 = {table: float("-inf") for table in tables}
    best_column_dense = {table: float("-inf") for table in tables}
    evidence_match = {table: 0.0 for table in tables}
    for index, (table, _) in enumerate(column_keys):
        best_column_bm25[table] = max(best_column_bm25[table], float(column_bm25[index]))
        best_column_dense[table] = max(best_column_dense[table], float(column_dense[index]))
        evidence_match[table] = max(evidence_match[table], float(evidence_scores[index]))
    return np.column_stack(
        (
            rank_fractions(table_bm25),
            rank_fractions(table_dense),
            rank_fractions(np.array([best_column_bm25[table] for table in tables], dtype=np.float32)),
            rank_fractions(np.array([best_column_dense[table] for table in tables], dtype=np.float32)),
            np.array([evidence_match[table] for table in tables], dtype=np.float32),
        )
    )


def reranked_table_scores(features: np.ndarray, reranker: dict[str, Any]) -> np.ndarray:
    names = tuple(reranker.get("feature_names", []))
    mean = np.array(reranker.get("mean", []), dtype=np.float32)
    scale = np.array(reranker.get("scale", []), dtype=np.float32)
    coefficients = np.array(reranker.get("coefficients", []), dtype=np.float32)
    if names != TABLE_RERANK_FEATURE_NAMES or not all(len(values) == features.shape[1] for values in (mean, scale, coefficients)):
        raise ValidationError("SCHEMAPACK_PREP_FAILED: invalid table reranker feature contract")
    if np.any(scale <= 0):
        raise ValidationError("SCHEMAPACK_PREP_FAILED: table reranker scales must be positive")
    return ((features - mean) / scale) @ coefficients + float(reranker.get("intercept", 0.0))


def select_schema(
    schema: DatabaseSchema,
    query: str,
    embedder: QwenEmbedder,
    cache: dict[str, Any],
    schema_char_budget: int,
    top_tables: int,
    columns_per_table: int,
    fallback_full_schema_chars: int = 0,
    evidence: str = "",
    evidence_column_boost: float = 0.0,
    value_match_boost: float = 0.0,
    table_reranker: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    tables = sorted(schema.columns_by_table)
    column_keys = sorted(schema.column_docs)
    full_within_budget = len(schema.full_schema) <= schema_char_budget
    full_fallback = fallback_full_schema_chars and len(schema.full_schema) >= fallback_full_schema_chars
    if full_within_budget or full_fallback:
        selected_columns = {
            table: {column.name for column in columns} for table, columns in schema.columns_by_table.items()
        }
        return schema.full_schema, {
            "mode": "full_within_budget" if full_within_budget else "full_fallback",
            "selected_tables": sorted(selected_columns),
            "selected_columns": {table: sorted(columns) for table, columns in selected_columns.items()},
            "selected_column_count": sum(len(columns) for columns in selected_columns.values()),
            "schema_characters": len(schema.full_schema),
            "full_schema_characters": len(schema.full_schema),
            "evidence_column_matches": 0,
            "value_column_matches": 0,
        }
    if "table_docs" not in cache:
        cache["table_docs"] = [schema.table_docs[table] for table in tables]
        cache["column_docs"] = [schema.column_docs[key] for key in column_keys]
        cache["table_embeddings"] = embedder.encode(cache["table_docs"])
        cache["column_embeddings"] = embedder.encode(cache["column_docs"])
    query_embeddings = cache.setdefault("query_embeddings", {})
    query_embedding = query_embeddings.get(query)
    if query_embedding is None:
        query_embedding = embedder.encode([query])[0]
        query_embeddings[query] = query_embedding
    table_bm25 = bm25_scores(query, cache["table_docs"])
    table_dense = cache["table_embeddings"] @ query_embedding
    column_bm25 = bm25_scores(query, cache["column_docs"])
    column_dense = cache["column_embeddings"] @ query_embedding
    table_scores = rrf_scores(table_bm25, table_dense)
    column_scores = rrf_scores(column_bm25, column_dense)
    evidence_scores = evidence_identifier_scores(evidence, column_keys)
    value_scores = value_match_scores(schema, query, column_keys, cache) if value_match_boost else np.zeros(len(column_keys))
    if evidence_column_boost:
        column_scores += evidence_column_boost * evidence_scores
        for index, (table_name, _) in enumerate(column_keys):
            table_scores[tables.index(table_name)] += evidence_column_boost * evidence_scores[index]
    if value_match_boost:
        column_scores += value_match_boost * value_scores
        for index, (table_name, _) in enumerate(column_keys):
            table_scores[tables.index(table_name)] += value_match_boost * value_scores[index]
    if table_reranker is None:
        table_rank = bidirectional_table_ranking(tables, column_keys, table_scores, column_scores)
    else:
        features = table_rerank_features(
            tables, column_keys, table_bm25, table_dense, column_bm25, column_dense, evidence_scores
        )
        table_rank = [tables[index] for index in np.argsort(-reranked_table_scores(features, table_reranker), kind="stable")]
    column_rank = [column_keys[index] for index in np.argsort(-column_scores)]
    for table_limit in range(min(top_tables, len(table_rank)), 0, -1):
        seed_tables = table_rank[:table_limit]
        boundary_gap = (
            None
            if table_limit == len(table_rank)
            else float(
                min(table_scores[tables.index(table)] for table in seed_tables)
                - max(table_scores[tables.index(table)] for table in table_rank[table_limit:])
            )
        )
        selected_tables = closure_tables(seed_tables, schema.foreign_keys)
        for column_limit in range(columns_per_table, 1, -1):
            selected_columns: dict[str, set[str]] = {table: set() for table in selected_tables}
            for table in selected_tables:
                ranked = [key for key in column_rank if key[0] == table]
                selected_columns[table].update(column for _, column in ranked[:column_limit])
            for table in selected_tables:
                for column in schema.columns_by_table[table]:
                    if column.primary_key_rank:
                        selected_columns[table].add(column.name)
            for foreign_key in schema.foreign_keys:
                if foreign_key.source_table in selected_tables and foreign_key.target_table in selected_tables:
                    selected_columns[foreign_key.source_table].add(foreign_key.source_column)
                    selected_columns[foreign_key.target_table].add(foreign_key.target_column)
            focused = render_schema(schema.columns_by_table, schema.foreign_keys, selected_columns)
            if len(focused) <= schema_char_budget:
                return focused, {
                    "mode": "retrieved",
                    "selected_tables": sorted(selected_tables),
                    "selected_columns": {table: sorted(columns) for table, columns in selected_columns.items()},
                    "selected_column_count": sum(len(columns) for columns in selected_columns.values()),
                    "schema_characters": len(focused),
                    "full_schema_characters": len(schema.full_schema),
                    "table_limit": table_limit,
                    "columns_per_table": column_limit,
                    "evidence_column_matches": int(evidence_scores.sum()),
                    "value_column_matches": int(value_scores.sum()),
                    # These retrieval-only diagnostics support train-calibrated
                    # schema-gap gating; they are never shown in the prompt.
                    "seed_tables": seed_tables,
                    "ranked_tables": table_rank,
                    "seed_table_score_range": round(
                        float(max(table_scores[tables.index(table)] for table in seed_tables)
                              - min(table_scores[tables.index(table)] for table in seed_tables)),
                        8,
                    ),
                    "boundary_table_score_gap": round(boundary_gap, 8) if boundary_gap is not None else None,
                }
    raise ValidationError(f"cannot fit a closed focused schema within {schema_char_budget} characters")


def build_user_prompt(question: str, evidence: str, schema: str) -> str:
    return f"""Task
{question.strip()}

Evidence
{evidence.strip() or 'No additional evidence.'}

Focused SQLite schema context with column meanings
{schema}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minidev-root", type=Path, default=DEFAULT_MINIDEV_ROOT)
    parser.add_argument("--sft-dataset", type=Path, default=DEFAULT_SFT_DATASET)
    parser.add_argument("--embedding-snapshots", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema-char-budget", type=int, default=6000)
    parser.add_argument("--top-tables", type=int, default=4)
    parser.add_argument("--columns-per-table", type=int, default=8)
    parser.add_argument("--fallback-full-schema-chars", type=int, default=0)
    parser.add_argument("--evidence-column-boost", type=float, default=0.0)
    parser.add_argument("--value-match-boost", type=float, default=0.0)
    parser.add_argument("--table-reranker", type=Path)
    parser.add_argument("--gap-boundary-table-score-threshold", type=float)
    parser.add_argument("--gap-expanded-top-tables", type=int)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.schema_char_budget, args.top_tables, args.columns_per_table, args.embedding_batch_size) < 1:
        raise SystemExit("SCHEMAPACK_PREP_FAILED: schema budget, table limit, column limit, and batch size must be positive")
    if args.fallback_full_schema_chars < 0:
        raise SystemExit("SCHEMAPACK_PREP_FAILED: fallback schema threshold must be non-negative")
    if min(args.evidence_column_boost, args.value_match_boost) < 0:
        raise SystemExit("SCHEMAPACK_PREP_FAILED: retrieval boosts must be non-negative")
    if (args.gap_boundary_table_score_threshold is None) != (args.gap_expanded_top_tables is None):
        raise SystemExit("SCHEMAPACK_PREP_FAILED: schema-gap threshold and expanded table count must be supplied together")
    if args.gap_expanded_top_tables is not None and args.gap_expanded_top_tables <= args.top_tables:
        raise SystemExit("SCHEMAPACK_PREP_FAILED: expanded table count must exceed --top-tables")
    table_reranker = None
    if args.table_reranker is not None:
        try:
            table_reranker = json.loads(args.table_reranker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"SCHEMAPACK_PREP_FAILED: could not load table reranker: {error}") from error

    examples = load_minidev_examples(args.minidev_root)
    system = load_sft_system_prompt(args.sft_dataset)
    compressed_system = system.replace(
        "supplied complete SQLite schema, column meanings, and evidence",
        "supplied focused SQLite schema context, column meanings, and evidence",
    )
    if compressed_system == system:
        raise SystemExit("SCHEMAPACK_PREP_FAILED: could not adapt SFT system prompt for focused schema context")
    database_root = args.minidev_root / "dev_databases"
    schemas: dict[str, DatabaseSchema] = {}
    caches: dict[str, dict[str, Any]] = defaultdict(dict)
    embedder = QwenEmbedder(args.embedding_snapshots, args.embedding_batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        db_id = example["db_id"]
        if db_id not in schemas:
            database_dir = database_root / db_id
            schemas[db_id] = load_database_schema(
                database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir)
            )
        evidence = clean_text(example.get("evidence"))
        query = example["question"] + "\n" + evidence
        focused_schema, selection = select_schema(
            schemas[db_id],
            query,
            embedder,
            caches[db_id],
            args.schema_char_budget,
            args.top_tables,
            args.columns_per_table,
            args.fallback_full_schema_chars,
            evidence,
            args.evidence_column_boost,
            args.value_match_boost,
            table_reranker,
        )
        gap = selection.get("boundary_table_score_gap")
        if (
            args.gap_boundary_table_score_threshold is not None
            and selection["mode"] == "retrieved"
            and isinstance(gap, (int, float))
            and gap <= args.gap_boundary_table_score_threshold
        ):
            initial_selection = selection
            focused_schema, selection = select_schema(
                schemas[db_id],
                query,
                embedder,
                caches[db_id],
                args.schema_char_budget,
                args.gap_expanded_top_tables,
                args.columns_per_table,
                args.fallback_full_schema_chars,
                evidence,
                args.evidence_column_boost,
                args.value_match_boost,
                table_reranker,
            )
            selection["schema_gap_expansion"] = {
                "initial_top_tables": args.top_tables,
                "expanded_top_tables": args.gap_expanded_top_tables,
                "initial_boundary_table_score_gap": gap,
                "initial_schema_characters": initial_selection["schema_characters"],
            }
        user = build_user_prompt(example["question"], evidence, focused_schema)
        records.append(
            {
                "example_index": index,
                "question_id": example["question_id"],
                "db_id": db_id,
                "difficulty": str(example.get("difficulty", "unknown")),
                "messages": [{"role": "system", "content": compressed_system}, {"role": "user", "content": user}],
                "schema_sha256": hashlib.sha256(focused_schema.encode("utf-8")).hexdigest(),
                "prompt_characters": len(compressed_system) + len(user),
                "selection": selection,
            }
        )
        if index % 50 == 0 or index == len(examples):
            print(f"prepared={index}/{len(examples)}", flush=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    schema_lengths = [record["selection"]["schema_characters"] for record in records]
    full_lengths = [record["selection"]["full_schema_characters"] for record in records]
    print(
        {
            "output": str(args.output.resolve()),
            "records": len(records),
            "mean_schema_characters": round(sum(schema_lengths) / len(schema_lengths), 2),
            "mean_full_schema_characters": round(sum(full_lengths) / len(full_lengths), 2),
            "mean_schema_reduction": round(1 - sum(schema_lengths) / sum(full_lengths), 6),
            "full_within_budget": sum(record["selection"]["mode"] == "full_within_budget" for record in records),
            "full_fallback": sum(record["selection"]["mode"] == "full_fallback" for record in records),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
