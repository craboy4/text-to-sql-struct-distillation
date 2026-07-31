#!/usr/bin/env python3
"""Train a train-only linear table reranker for SchemaPack."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from calibrate_schemapack_tables import DEFAULT_DATABASES, DEFAULT_TRAIN, gold_columns, gold_tables, load_examples
from prepare_minidev_inference import clean_text, load_column_descriptions
from prepare_minidev_schemapack import (
    DEFAULT_EMBEDDING_MODEL,
    TABLE_RERANK_FEATURE_NAMES,
    QwenEmbedder,
    bm25_scores,
    evidence_identifier_scores,
    load_database_schema,
    select_schema,
    table_rerank_features,
)


def is_validation_database(db_id: str, modulus: int, remainder: int) -> bool:
    return int(hashlib.sha256(db_id.encode("utf-8")).hexdigest(), 16) % modulus == remainder


def features_for_query(
    schema: Any, query: str, evidence: str, embedder: QwenEmbedder, cache: dict[str, Any]
) -> tuple[list[str], np.ndarray]:
    tables = sorted(schema.columns_by_table)
    column_keys = sorted(schema.column_docs)
    if "table_docs" not in cache:
        cache["table_docs"] = [schema.table_docs[table] for table in tables]
        cache["column_docs"] = [schema.column_docs[key] for key in column_keys]
        cache["table_embeddings"] = embedder.encode(cache["table_docs"])
        cache["column_embeddings"] = embedder.encode(cache["column_docs"])
    query_embedding = cache.setdefault("query_embeddings", {}).get(query)
    if query_embedding is None:
        query_embedding = embedder.encode([query])[0]
        cache["query_embeddings"][query] = query_embedding
    table_bm25 = bm25_scores(query, cache["table_docs"])
    table_dense = cache["table_embeddings"] @ query_embedding
    column_bm25 = bm25_scores(query, cache["column_docs"])
    column_dense = cache["column_embeddings"] @ query_embedding
    return tables, table_rerank_features(
        tables,
        column_keys,
        table_bm25,
        table_dense,
        column_bm25,
        column_dense,
        evidence_identifier_scores(evidence, column_keys),
    )


def fit_artifact(features: np.ndarray, labels: np.ndarray, c_value: float) -> dict[str, Any]:
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(C=c_value, class_weight="balanced", max_iter=500, random_state=0).fit(
        scaler.transform(features), labels
    )
    return {
        "feature_names": list(TABLE_RERANK_FEATURE_NAMES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "regularization_c": c_value,
    }


def coverage(
    examples: list[dict[str, str]],
    schemas: dict[str, Any],
    caches: dict[str, dict[str, Any]],
    embedder: QwenEmbedder,
    reranker: dict[str, Any] | None,
    schema_char_budget: int,
    top_tables: int,
    columns_per_table: int,
    evidence_column_boost: float,
) -> dict[str, float]:
    exact_tables = exact_schema = total_tables = covered_tables = 0
    for example in examples:
        schema = schemas[example["db_id"]]
        required_tables = gold_tables(example["SQL"], set(schema.columns_by_table))
        required_columns = gold_columns(example["SQL"], schema, required_tables)
        query = example["question"] + "\n" + example["evidence"]
        _, selection = select_schema(
            schema,
            query,
            embedder,
            caches[example["db_id"]],
            schema_char_budget,
            top_tables,
            columns_per_table,
            evidence="" if not evidence_column_boost else example["evidence"],
            evidence_column_boost=evidence_column_boost,
            table_reranker=reranker,
        )
        selected_tables = {table.casefold() for table in selection["selected_tables"]}
        selected_columns = {
            (table.casefold(), column.casefold())
            for table, columns in selection["selected_columns"].items()
            for column in columns
        }
        normalized_tables = {table.casefold() for table in required_tables}
        normalized_columns = {(table.casefold(), column.casefold()) for table, column in required_columns}
        exact_tables += normalized_tables <= selected_tables
        exact_schema += normalized_tables <= selected_tables and normalized_columns <= selected_columns
        total_tables += len(normalized_tables)
        covered_tables += len(normalized_tables & selected_tables)
    return {
        "examples": len(examples),
        "exact_table_coverage": round(exact_tables / len(examples), 6),
        "exact_schema_coverage": round(exact_schema / len(examples), 6),
        "table_recall": round(covered_tables / total_tables, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--databases", type=Path, default=DEFAULT_DATABASES)
    parser.add_argument("--embedding-snapshots", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--validation-db-modulus", type=int, default=5)
    parser.add_argument("--validation-db-remainders", default="0,1,2,3,4")
    parser.add_argument("--regularization-candidates", default="0.1,1,10")
    parser.add_argument("--schema-char-budget", type=int, default=9000)
    parser.add_argument("--top-tables", type=int, default=6)
    parser.add_argument("--columns-per-table", type=int, default=16)
    parser.add_argument("--evidence-column-boost", type=float, default=0.05)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.validation_db_modulus, args.schema_char_budget, args.top_tables, args.columns_per_table, args.embedding_batch_size) < 1:
        raise SystemExit("SCHEMAPACK_RERANKER_FAILED: numeric arguments must be positive")
    validation_remainders = [int(value) for value in args.validation_db_remainders.split(",")]
    if (
        not validation_remainders
        or not all(0 <= value < args.validation_db_modulus for value in validation_remainders)
        or args.evidence_column_boost < 0
    ):
        raise SystemExit("SCHEMAPACK_RERANKER_FAILED: invalid validation split or evidence boost")
    c_values = [float(value) for value in args.regularization_candidates.split(",")]
    if not c_values or min(c_values) <= 0:
        raise SystemExit("SCHEMAPACK_RERANKER_FAILED: regularization candidates must be positive")

    examples = load_examples(args.train, 0)
    schemas: dict[str, Any] = {}
    caches: dict[str, dict[str, Any]] = defaultdict(dict)
    examples_by_db: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        examples_by_db[example["db_id"]].append(example)
    embedder = QwenEmbedder(args.embedding_snapshots, args.embedding_batch_size)
    for db_id, database_examples in examples_by_db.items():
        database_dir = args.databases / db_id
        schemas[db_id] = load_database_schema(database_dir / f"{db_id}.sqlite", load_column_descriptions(database_dir))
        queries = [example["question"] + "\n" + example["evidence"] for example in database_examples]
        caches[db_id]["query_embeddings"] = dict(zip(queries, embedder.encode(queries)))

    feature_records: list[tuple[dict[str, str], np.ndarray, np.ndarray]] = []
    for index, example in enumerate(examples, start=1):
        db_id = example["db_id"]
        tables, features = features_for_query(
            schemas[db_id], example["question"] + "\n" + example["evidence"], example["evidence"], embedder, caches[db_id]
        )
        required = {table.casefold() for table in gold_tables(example["SQL"], set(schemas[db_id].columns_by_table))}
        labels = np.array([float(table.casefold() in required) for table in tables], dtype=np.int8)
        feature_records.append((example, features, labels))
        if index % 100 == 0:
            print(f"features={index}/{len(examples)}", flush=True)
    all_features = np.concatenate([features for _, features, _ in feature_records])
    all_labels = np.concatenate([labels for _, _, labels in feature_records])
    fold_reports = []
    aggregate: dict[float, list[dict[str, float]]] = {c_value: [] for c_value in c_values}
    baselines = []
    for remainder in validation_remainders:
        validation_examples = [
            example for example, _, _ in feature_records
            if is_validation_database(example["db_id"], args.validation_db_modulus, remainder)
        ]
        training_records = [
            (features, labels) for example, features, labels in feature_records
            if not is_validation_database(example["db_id"], args.validation_db_modulus, remainder)
        ]
        if not validation_examples or not training_records:
            raise SystemExit("SCHEMAPACK_RERANKER_FAILED: empty train or validation database group")
        baseline = coverage(
            validation_examples, schemas, caches, embedder, None, args.schema_char_budget, args.top_tables,
            args.columns_per_table, args.evidence_column_boost,
        )
        baselines.append(baseline)
        candidate_rows = []
        for c_value in c_values:
            artifact = fit_artifact(
                np.concatenate([features for features, _ in training_records]),
                np.concatenate([labels for _, labels in training_records]),
                c_value,
            )
            metrics = coverage(
                validation_examples, schemas, caches, embedder, artifact, args.schema_char_budget, args.top_tables,
                args.columns_per_table, args.evidence_column_boost,
            )
            aggregate[c_value].append(metrics)
            candidate_rows.append({"regularization_c": c_value, **metrics})
        fold_reports.append({
            "remainder": remainder,
            "validation_databases": sorted({example["db_id"] for example in validation_examples}),
            "baseline": baseline,
            "candidates": candidate_rows,
        })

    def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
        total_examples = sum(int(row["examples"]) for row in rows)
        return {
            "examples": total_examples,
            "exact_table_coverage": round(sum(row["exact_table_coverage"] * row["examples"] for row in rows) / total_examples, 6),
            "exact_schema_coverage": round(sum(row["exact_schema_coverage"] * row["examples"] for row in rows) / total_examples, 6),
            "table_recall": round(sum(row["table_recall"] * row["examples"] for row in rows) / total_examples, 6),
        }

    reports = [{"regularization_c": c_value, **average_metrics(rows)} for c_value, rows in aggregate.items()]
    reports.sort(key=lambda row: (-row["exact_schema_coverage"], -row["exact_table_coverage"], -row["table_recall"], row["regularization_c"]))
    selected_c = reports[0]["regularization_c"]
    artifact = fit_artifact(all_features, all_labels, selected_c)
    artifact["training_split"] = "BIRD train filtered deterministic SHA-256 modulo-5 sample"
    artifact["validation_split"] = f"five-fold db_id SHA-256 modulo {args.validation_db_modulus}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "examples": len(examples),
        "training_table_rows": len(all_labels),
        "baseline": average_metrics(baselines),
        "candidates": reports,
        "selected_regularization_c": selected_c,
        "folds": fold_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
