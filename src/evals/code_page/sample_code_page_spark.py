#!/usr/bin/env python3
"""Sample the fixed Eval9 10k code-page corpus with Spark.

The source-side computation deliberately has one Spark action:

    project score/text -> filter -> Bernoulli sample(seed=2026) -> collect()

The driver then performs a seeded shuffle of that bounded random subset and
keeps exactly 10,000 rows.

There is no content hashing and no deduplication.  Only the ``text`` column is
published locally.  ``sample_manifest.json`` is installed last and acts as the
completion marker for reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SAMPLE_FORMAT = "eval9-code-page-sample-v1"
SAMPLING_ALGORITHM = "spark-bernoulli-sample-driver-shuffle-v1"
SAMPLE_SIZE = 10_000
SAMPLE_SEED = 2026
SAMPLE_FRACTION = 0.001
MIN_EXCLUSIVE_SCORE = 7
MAX_INCLUSIVE_SCORE = 9
SOURCE_SCORE_COLUMN = "info.core.k2_hqs_score"
SOURCE_TEXT_COLUMN = "text"
FILTER_EXPRESSION = (
    "text IS NOT NULL AND length(trim(text)) > 0 "
    "AND info.core.k2_hqs_score > 7 "
    "AND info.core.k2_hqs_score <= 9"
)

DEFAULT_INPUT_PATH = os.environ.get("CODE_PAGE_INPUT", "data/code_page_v5")
DEFAULT_OUTPUT_DIR = os.environ.get("CODE_PAGE_OUTPUT", "data/code_page_retrieval")
PARQUET_NAME = "code_page_hqs8_9_10k.parquet"
JSONL_NAME = "code_page_hqs8_9_10k.jsonl"
MANIFEST_NAME = "sample_manifest.json"

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_SIZE = 1024

_SPARK_REPRODUCIBILITY_CONF_KEYS = (
    "spark.master",
    "spark.submit.deployMode",
    "spark.default.parallelism",
    "spark.sql.adaptive.enabled",
    "spark.sql.files.maxPartitionBytes",
    "spark.sql.files.openCostInBytes",
    "spark.sql.files.minPartitionNum",
    "spark.sql.files.maxPartitionNum",
    "spark.sql.shuffle.partitions",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Spark-sample exactly 10,000 non-empty HQS 8/9 code-page texts and "
            "atomically publish the one-column local Eval9 corpus."
        )
    )
    parser.add_argument(
        "--input-path",
        "--input",
        dest="input_path",
        default=DEFAULT_INPUT_PATH,
        help="Parquet dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory for the canonical sample and manifest.",
    )
    parser.add_argument(
        "--app-name",
        default="eval9-code-page-sample",
        help="Spark application name.",
    )
    parser.add_argument(
        "--master",
        default=None,
        help=(
            "Optional Spark master override. Omit under spark-submit/YARN; "
            "use local[2] for a local smoke test."
        ),
    )
    parser.add_argument(
        "--spark-log-level",
        default="WARN",
        choices=("ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"),
        help="Driver SparkContext log level.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=SAMPLE_FRACTION,
        help=(
            "Bernoulli fraction applied after filtering. The default 0.001 "
            "is expected to return roughly 60k rows from the current source, "
            "after which the driver shuffles and keeps exactly 10k."
        ),
    )
    jsonl_group = parser.add_mutually_exclusive_group()
    jsonl_group.add_argument(
        "--write-jsonl",
        "--jsonl",
        dest="write_jsonl",
        action="store_true",
        help="Also atomically publish the optional one-field JSONL mirror.",
    )
    jsonl_group.add_argument(
        "--no-jsonl",
        "--no-write-jsonl",
        dest="write_jsonl",
        action="store_false",
        help="Publish only the required Parquet file.",
    )
    parser.set_defaults(write_jsonl=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate and atomically replace this artifact. Without this "
            "flag, a complete matching manifest is validated and reused."
        ),
    )
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_from_epoch_millis(value: int | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path, *, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, published_name: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": published_name or path.name,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _script_record() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "path": "src/evals/code_page/sample_code_page_spark.py",
        "sha256": _file_sha256(script_path),
    }


def _manifest_with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("manifest_payload_sha256", None)
    result["manifest_payload_sha256"] = _payload_sha256(result)
    return result


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must contain one JSON object: {path}")
    expected_digest = payload.get("manifest_payload_sha256")
    if not isinstance(expected_digest, str):
        raise ValueError(f"manifest is missing manifest_payload_sha256: {path}")
    without_digest = dict(payload)
    without_digest.pop("manifest_payload_sha256", None)
    actual_digest = _payload_sha256(without_digest)
    if actual_digest != expected_digest:
        raise ValueError(
            f"manifest digest mismatch for {path}: "
            f"{actual_digest} != {expected_digest}"
        )
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _new_staging_path(output_dir: Path, final_name: str) -> Path:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{final_name}.",
        suffix=".tmp",
        dir=output_dir,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _spark_conf_value(spark: Any, key: str) -> str | None:
    try:
        return str(spark.conf.get(key))
    except Exception:
        try:
            return str(spark.sparkContext.getConf().get(key))
        except Exception:
            return None


def _spark_runtime(spark: Any) -> dict[str, Any]:
    context = spark.sparkContext
    return {
        "version": str(spark.version),
        "application_id": str(context.applicationId),
        "application_name": str(context.appName),
        "master": str(context.master),
        "configuration": {
            key: _spark_conf_value(spark, key)
            for key in _SPARK_REPRODUCIBILITY_CONF_KEYS
        },
    }


def _input_status_summary(spark: Any, input_path: str) -> dict[str, Any]:
    """Summarize Parquet file statuses without running a Spark action."""

    jvm = spark.sparkContext._jvm
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    root_path = jvm.org.apache.hadoop.fs.Path(input_path)
    root_fs = root_path.getFileSystem(hadoop_conf)
    matched = root_fs.globStatus(root_path)
    if matched is None or len(matched) == 0:
        raise FileNotFoundError(f"input path does not exist or match files: {input_path}")

    files: dict[str, tuple[int, int]] = {}

    def add_file(status: Any) -> None:
        qualified_path = str(
            status.getPath()
            .getFileSystem(hadoop_conf)
            .makeQualified(status.getPath())
            .toString()
        )
        if not qualified_path.lower().endswith(".parquet"):
            return
        files[qualified_path] = (
            int(status.getLen()),
            int(status.getModificationTime()),
        )

    for status in matched:
        if bool(status.isFile()):
            add_file(status)
            continue
        filesystem = status.getPath().getFileSystem(hadoop_conf)
        iterator = filesystem.listFiles(status.getPath(), True)
        while bool(iterator.hasNext()):
            add_file(iterator.next())

    if not files:
        raise FileNotFoundError(
            f"no Parquet data files found below input path: {input_path}"
        )

    ordered_statuses = [
        {"path": path, "bytes": size, "modification_time_ms": modification_time}
        for path, (size, modification_time) in sorted(files.items())
    ]
    sizes = [entry["bytes"] for entry in ordered_statuses]
    modification_times = [
        entry["modification_time_ms"] for entry in ordered_statuses
    ]
    return {
        "provided_path": input_path,
        "qualified_path": str(root_fs.makeQualified(root_path).toString()),
        "filesystem_uri": str(root_fs.getUri().toString()),
        "matched_root_count": len(matched),
        "parquet_file_count": len(ordered_statuses),
        "parquet_total_bytes": sum(sizes),
        "smallest_file_bytes": min(sizes),
        "largest_file_bytes": max(sizes),
        "earliest_modification_time_ms": min(modification_times),
        "latest_modification_time_ms": max(modification_times),
        "earliest_modification_time_utc": _utc_from_epoch_millis(
            min(modification_times)
        ),
        "latest_modification_time_utc": _utc_from_epoch_millis(
            max(modification_times)
        ),
        "file_status_digest": _payload_sha256(ordered_statuses),
    }


def _identity(
    *,
    input_path: str,
    input_status: Mapping[str, Any],
    spark_runtime: Mapping[str, Any],
    write_jsonl: bool,
    sample_fraction: float,
) -> dict[str, Any]:
    spark_configuration = dict(spark_runtime["configuration"])
    return {
        "input_path": input_path,
        "input_snapshot": {
            "parquet_file_count": int(input_status["parquet_file_count"]),
            "parquet_total_bytes": int(input_status["parquet_total_bytes"]),
            "file_status_digest": str(input_status["file_status_digest"]),
        },
        "source_projection": [SOURCE_SCORE_COLUMN, SOURCE_TEXT_COLUMN],
        "filter_expression": FILTER_EXPRESSION,
        "valid_quality_scores": [8, 9],
        "sampling": {
            "algorithm": SAMPLING_ALGORITHM,
            "seed": SAMPLE_SEED,
            "sample_size": SAMPLE_SIZE,
            "sample_fraction": float(sample_fraction),
            "content_hashing": False,
            "deduplication": False,
        },
        "spark_reproducibility_boundary": {
            "version": str(spark_runtime["version"]),
            "master": str(spark_runtime["master"]),
            "configuration": spark_configuration,
        },
        "outputs": {
            "parquet": PARQUET_NAME,
            "jsonl": JSONL_NAME if write_jsonl else None,
            "schema": [{"name": "text", "type": "string"}],
        },
        "code": _script_record(),
    }


def _assert_file_record(path: Path, record: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = int(record["bytes"])
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"file size mismatch for {path}: {actual_bytes} != {expected_bytes}"
        )
    expected_sha256 = str(record["sha256"])
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"file digest mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )


def _validate_texts(texts: Sequence[Any], *, expected_count: int) -> list[str]:
    if len(texts) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count:,} sampled texts, found {len(texts):,}"
        )
    validated: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"sample row {index} is not a string: {type(text).__name__}"
            )
        # Spark 3.3's trim(col) removes the default ASCII space character.
        # Mirror that exact source predicate instead of silently imposing a
        # broader Python-whitespace policy during driver-side validation.
        if not text.strip(" "):
            raise ValueError(f"sample row {index} is empty after Spark trim")
        validated.append(text)
    return validated


def _read_and_validate_parquet(
    path: Path,
    *,
    expected_count: int,
) -> list[str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required on the Spark driver to publish Eval9 locally"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    arrow_schema = parquet_file.schema_arrow
    if arrow_schema.names != ["text"]:
        raise ValueError(
            f"{path}: expected exactly one column named text, found "
            f"{arrow_schema.names}"
        )
    if not pa.types.is_string(arrow_schema.field("text").type):
        raise ValueError(
            f"{path}: expected text: string, found "
            f"{arrow_schema.field('text').type}"
        )
    if parquet_file.metadata.num_rows != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count:,} rows, found "
            f"{parquet_file.metadata.num_rows:,}"
        )
    table = pq.read_table(path)
    if table.column_names != ["text"]:
        raise ValueError(f"{path}: unexpected read-back columns {table.column_names}")
    return _validate_texts(
        table.column("text").to_pylist(),
        expected_count=expected_count,
    )


def _read_and_validate_jsonl(
    path: Path,
    *,
    expected_count: int,
) -> list[str]:
    texts: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or set(row) != {"text"}:
                raise ValueError(
                    f"{path}:{line_number}: expected only the text field"
                )
            texts.append(row["text"])
    return _validate_texts(texts, expected_count=expected_count)


def _validate_existing_artifact(
    *,
    output_dir: Path,
    manifest: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("format") != SAMPLE_FORMAT:
        raise ValueError(
            f"unexpected sample format: {manifest.get('format')!r} "
            f"(expected {SAMPLE_FORMAT!r})"
        )
    if manifest.get("complete") is not True:
        raise ValueError("existing sample manifest is not complete")
    if manifest.get("identity") != dict(expected_identity):
        raise ValueError(
            "existing sample identity differs from this invocation; "
            "use --overwrite or a different --output-dir"
        )
    if int(manifest.get("row_count", -1)) != SAMPLE_SIZE:
        raise ValueError(
            f"existing manifest row_count is not {SAMPLE_SIZE:,}: "
            f"{manifest.get('row_count')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("existing manifest files field must be an object")

    parquet_record = files.get("parquet")
    if not isinstance(parquet_record, Mapping):
        raise ValueError("existing manifest is missing files.parquet")
    parquet_path = output_dir / str(parquet_record.get("path"))
    _assert_file_record(parquet_path, parquet_record)
    parquet_texts = _read_and_validate_parquet(
        parquet_path,
        expected_count=SAMPLE_SIZE,
    )

    expected_jsonl = expected_identity["outputs"]["jsonl"]
    jsonl_record = files.get("jsonl")
    if expected_jsonl is None:
        if jsonl_record is not None:
            raise ValueError("manifest unexpectedly contains a JSONL output")
    else:
        if not isinstance(jsonl_record, Mapping):
            raise ValueError("existing manifest is missing files.jsonl")
        jsonl_path = output_dir / str(jsonl_record.get("path"))
        _assert_file_record(jsonl_path, jsonl_record)
        jsonl_texts = _read_and_validate_jsonl(
            jsonl_path,
            expected_count=SAMPLE_SIZE,
        )
        if jsonl_texts != parquet_texts:
            raise ValueError("existing JSONL text order/content differs from Parquet")

    return {
        "status": "reused",
        "manifest": str(output_dir / MANIFEST_NAME),
        "parquet": str(parquet_path),
        "jsonl": (
            str(output_dir / str(jsonl_record["path"]))
            if isinstance(jsonl_record, Mapping)
            else None
        ),
        "row_count": SAMPLE_SIZE,
    }


def _sample_texts(
    spark: Any,
    input_path: str,
    *,
    sample_fraction: float,
) -> list[str]:
    """Run the sole Spark action and return the ordered 10k text sample."""

    import random

    try:
        from pyspark.sql import functions as functions
        from pyspark.sql.types import ByteType, IntegerType, LongType, ShortType
        from pyspark.sql.types import StringType, StructField, StructType
    except ModuleNotFoundError as exc:
        raise RuntimeError("run this program with PySpark/spark-submit") from exc

    # Supplying the known, minimal nested schema avoids a separate Parquet
    # schema-inference job and prevents any source data column other than the
    # score leaf and text from entering the scan.
    source_schema = StructType(
        [
            StructField(
                "info",
                StructType(
                    [
                        StructField(
                            "core",
                            StructType(
                                [
                                    StructField(
                                        "k2_hqs_score",
                                        LongType(),
                                        nullable=True,
                                    )
                                ]
                            ),
                            nullable=True,
                        )
                    ]
                ),
                nullable=True,
            ),
            StructField("text", StringType(), nullable=True),
        ]
    )
    source = spark.read.schema(source_schema).parquet(input_path)
    projected = source.select(
        functions.col(SOURCE_SCORE_COLUMN).alias("k2_hqs_score"),
        functions.col(SOURCE_TEXT_COLUMN),
    )
    if projected.columns != ["k2_hqs_score", "text"]:
        raise AssertionError(
            f"unexpected source projection columns: {projected.columns}"
        )
    if not isinstance(projected.schema["text"].dataType, StringType):
        raise TypeError(
            f"source text column must be string, found "
            f"{projected.schema['text'].dataType.simpleString()}"
        )
    if not isinstance(
        projected.schema["k2_hqs_score"].dataType,
        (ByteType, ShortType, IntegerType, LongType),
    ):
        raise TypeError(
            f"source {SOURCE_SCORE_COLUMN} must be integral, found "
            f"{projected.schema['k2_hqs_score'].dataType.simpleString()}"
        )

    filtered = projected.where(
        functions.col("text").isNotNull()
        & (functions.length(functions.trim(functions.col("text"))) > 0)
        & (functions.col("k2_hqs_score") > MIN_EXCLUSIVE_SCORE)
        & (functions.col("k2_hqs_score") <= MAX_INCLUSIVE_SCORE)
    )
    sampled = filtered.sample(
        withReplacement=False,
        fraction=float(sample_fraction),
        seed=SAMPLE_SEED,
    ).select("text")

    # Do not add count(), take(), toPandas(), or any other source-side action.
    rows = sampled.collect()
    if len(rows) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Bernoulli sample returned only {len(rows):,} rows, fewer than "
            f"the required {SAMPLE_SIZE:,}; increase --sample-fraction and rerun"
        )
    texts = [row["text"] for row in rows]
    random.Random(SAMPLE_SEED).shuffle(texts)
    return _validate_texts(
        texts[:SAMPLE_SIZE],
        expected_count=SAMPLE_SIZE,
    )


def _write_parquet(path: Path, texts: Sequence[str]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required on the Spark driver to publish Eval9 locally"
        ) from exc

    table = pa.Table.from_arrays(
        [pa.array(texts, type=pa.string())],
        names=["text"],
    )
    if table.schema.names != ["text"] or not pa.types.is_string(
        table.schema.field("text").type
    ):
        raise AssertionError(f"unexpected output Arrow schema: {table.schema}")
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        use_dictionary=False,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        write_statistics=True,
    )
    _fsync_file(path)


def _write_jsonl(path: Path, texts: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(
                json.dumps(
                    {"text": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_sample(
    *,
    output_dir: Path,
    texts: Sequence[str],
    identity: Mapping[str, Any],
    input_status: Mapping[str, Any],
    spark_runtime: Mapping[str, Any],
    write_jsonl: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / PARQUET_NAME
    jsonl_path = output_dir / JSONL_NAME
    manifest_path = output_dir / MANIFEST_NAME

    if not overwrite:
        conflicting = [
            path
            for path in (parquet_path, jsonl_path, manifest_path)
            if path.exists()
        ]
        if conflicting:
            raise FileExistsError(
                "refusing to replace files without --overwrite: "
                + ", ".join(map(str, conflicting))
            )

    parquet_staging = _new_staging_path(output_dir, PARQUET_NAME)
    jsonl_staging = (
        _new_staging_path(output_dir, JSONL_NAME) if write_jsonl else None
    )
    try:
        _write_parquet(parquet_staging, texts)
        parquet_roundtrip = _read_and_validate_parquet(
            parquet_staging,
            expected_count=SAMPLE_SIZE,
        )
        if parquet_roundtrip != list(texts):
            raise ValueError("Parquet read-back order/content differs from collected rows")

        jsonl_roundtrip: list[str] | None = None
        if jsonl_staging is not None:
            _write_jsonl(jsonl_staging, texts)
            jsonl_roundtrip = _read_and_validate_jsonl(
                jsonl_staging,
                expected_count=SAMPLE_SIZE,
            )
            if jsonl_roundtrip != parquet_roundtrip:
                raise ValueError("JSONL read-back order/content differs from Parquet")

        files: dict[str, Any] = {
            "parquet": {
                **_file_record(
                    parquet_staging,
                    published_name=PARQUET_NAME,
                ),
                "row_count": SAMPLE_SIZE,
                "schema": [{"name": "text", "type": "string"}],
                "compression": PARQUET_COMPRESSION,
            }
        }
        if jsonl_staging is not None:
            files["jsonl"] = {
                **_file_record(
                    jsonl_staging,
                    published_name=JSONL_NAME,
                ),
                "row_count": SAMPLE_SIZE,
                "schema": [{"name": "text", "type": "string"}],
            }

        manifest = _manifest_with_digest(
            {
                "format": SAMPLE_FORMAT,
                "complete": True,
                "generated_at_utc": _utc_now(),
                "identity": dict(identity),
                "input": dict(input_status),
                "source_projection": [SOURCE_SCORE_COLUMN, SOURCE_TEXT_COLUMN],
                "filter_expression": FILTER_EXPRESSION,
                "valid_quality_scores": [8, 9],
                "sampling": {
                    "algorithm": SAMPLING_ALGORITHM,
                    "seed": SAMPLE_SEED,
                    "sample_size": SAMPLE_SIZE,
                    "sample_fraction": float(
                        identity["sampling"]["sample_fraction"]
                    ),
                    "spark_actions": ["collect"],
                    "content_hashing": False,
                    "deduplication": False,
                },
                "row_count": SAMPLE_SIZE,
                "schema": [{"name": "text", "type": "string"}],
                "spark": dict(spark_runtime),
                "code": _script_record(),
                "files": files,
            }
        )

        # Every staged payload has already passed a complete read-back check.
        # In overwrite mode, invalidate the old completion marker before any
        # payload replacement.  A hard failure can then leave only an
        # incomplete artifact, never stale metadata claiming completeness.
        if overwrite:
            manifest_path.unlink(missing_ok=True)
            _fsync_directory(output_dir)

        # Install data first and the manifest completion marker last.
        os.replace(parquet_staging, parquet_path)
        if jsonl_staging is not None:
            os.replace(jsonl_staging, jsonl_path)
        elif overwrite:
            jsonl_path.unlink(missing_ok=True)
        _atomic_write_json(manifest_path, manifest)
        _fsync_directory(output_dir)

        installed = _read_manifest(manifest_path)
        result = _validate_existing_artifact(
            output_dir=output_dir,
            manifest=installed,
            expected_identity=identity,
        )
        result["status"] = "created"
        return result
    finally:
        parquet_staging.unlink(missing_ok=True)
        if jsonl_staging is not None:
            jsonl_staging.unlink(missing_ok=True)


def _create_spark_session(args: argparse.Namespace) -> Any:
    try:
        from pyspark.sql import SparkSession
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PySpark is unavailable. Run this file with spark-submit or add "
            "$SPARK_HOME/python and its py4j zip to PYTHONPATH."
        ) from exc

    builder = (
        SparkSession.builder.appName(args.app_name)
        .config("spark.sql.optimizer.nestedSchemaPruning.enabled", "true")
    )
    if args.master:
        builder = builder.master(args.master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(args.spark_log_level)
    return spark


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < float(args.sample_fraction) <= 1.0:
        raise ValueError("--sample-fraction must be in (0, 1]")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spark = _create_spark_session(args)
    try:
        input_status_before = _input_status_summary(spark, args.input_path)
        spark_runtime = _spark_runtime(spark)
        identity = _identity(
            input_path=args.input_path,
            input_status=input_status_before,
            spark_runtime=spark_runtime,
            write_jsonl=bool(args.write_jsonl),
            sample_fraction=float(args.sample_fraction),
        )
        manifest_path = output_dir / MANIFEST_NAME

        if manifest_path.exists() and not args.overwrite:
            manifest = _read_manifest(manifest_path)
            return _validate_existing_artifact(
                output_dir=output_dir,
                manifest=manifest,
                expected_identity=identity,
            )

        if not args.overwrite:
            conflicting = [
                path
                for path in (
                    output_dir / PARQUET_NAME,
                    output_dir / JSONL_NAME,
                )
                if path.exists()
            ]
            if conflicting:
                raise FileExistsError(
                    "sample files exist without a reusable matching manifest; "
                    "use --overwrite or a different --output-dir: "
                    + ", ".join(map(str, conflicting))
                )

        print(
            json.dumps(
                {
                    "status": "sampling",
                    "input_path": args.input_path,
                    "input_parquet_files": input_status_before[
                        "parquet_file_count"
                    ],
                    "input_parquet_bytes": input_status_before[
                        "parquet_total_bytes"
                    ],
                    "sample_size": SAMPLE_SIZE,
                    "seed": SAMPLE_SEED,
                    "sample_fraction": float(args.sample_fraction),
                    "deduplication": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        texts = _sample_texts(
            spark,
            args.input_path,
            sample_fraction=float(args.sample_fraction),
        )

        # Detect a changing source tree before publishing the sampled artifact.
        input_status_after = _input_status_summary(spark, args.input_path)
        for key in (
            "parquet_file_count",
            "parquet_total_bytes",
            "file_status_digest",
        ):
            if input_status_after[key] != input_status_before[key]:
                raise RuntimeError(
                    "input Parquet file set changed while sampling "
                    f"({key}: {input_status_before[key]!r} -> "
                    f"{input_status_after[key]!r}); refusing to publish"
                )

        return _publish_sample(
            output_dir=output_dir,
            texts=texts,
            identity=identity,
            input_status=input_status_before,
            spark_runtime=spark_runtime,
            write_jsonl=bool(args.write_jsonl),
            overwrite=bool(args.overwrite),
        )
    finally:
        spark.stop()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
