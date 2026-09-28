#!/usr/bin/env python
"""Extract Eval 9 code-page features with four frozen SAEs on eight GPUs.

The canonical input is a 10,000-row Parquet file whose only column is
``text``.  Candidate documents are owned deterministically by
``document_index % world_size``.  Independent SAE seed pages are read from a
JSONL file with ``seed_id``, ``query_id``, and ``text`` fields.

Every selected 512-token window is forwarded through Qwen exactly once.  The
resulting target-layer hidden states are cached on CPU for the current atomic
shard and then consumed by BatchTopK/Token, Temporal, Mean-Chunk, and
Cross-Chunk in sequence.  Only one SAE encoder is resident on a GPU at a time,
and decoder weights are never materialized.

The script is deliberately strict.  It validates the input schema, the common
65,536-wide SAE dictionary, all sparse arrays, deterministic ownership, window
placement, merged document coverage, and every final artifact before publishing
``feature_manifest.json``.
"""

import argparse
import gc
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from chunk_saes.artifacts import (
    ensure_reusable_artifact,
    file_record,
    json_digest,
    model_metadata_fingerprint,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.evaluation_protocol import (
    FIXED_CHUNK_REPRESENTATION_PROTOCOL,
    fixed_chunk_protocol_metadata,
    full_dictionary_feature_widths,
    mean_after_threshold,
    validate_full_dictionary_width,
)
from chunk_saes.plot_style import METHODS
from chunk_saes.runtime import (
    all_gather_objects,
    bind_local_rank_cpu_affinity,
    broadcast_object,
    configure_cpu_threads,
    initialize_distributed,
)
from chunk_saes.utils import atomic_json_dump, log


FEATURE_FORMAT = "chunk-saes-eval9-code-page-features-v1"
RANK_FORMAT = "chunk-saes-eval9-code-page-rank-v1"
SHARD_FORMAT = "chunk-saes-eval9-code-page-shard-v1"
SEED_RANK_FORMAT = "chunk-saes-eval9-code-page-seed-rank-v1"
MERGED_FORMAT = "chunk-saes-eval9-code-page-merged-v1"
SEED_FORMAT = "chunk-saes-eval9-code-page-seeds-v1"
PROGRESS_FORMAT = "chunk-saes-eval9-code-page-progress-v1"
RUN_FORMAT = "chunk-saes-eval9-code-page-run-v1"

WINDOW_SIZE = 512
MAX_WINDOWS = 8
TOKEN_BUDGET = WINDOW_SIZE * MAX_WINDOWS
TOP_K = 128
DICTIONARY_WIDTH = 65_536
REQUIRED_METHODS = ("token", "temporal", "mean", "cross")

if tuple(METHODS) != REQUIRED_METHODS:
    raise RuntimeError(
        f"unexpected shared SAE method order: {tuple(METHODS)!r}"
    )

_ILLEGAL_CONTROL_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SeedRow:
    seed_index: int
    seed_id: str
    query_id: str
    text: str


@dataclass
class WindowPlan:
    item_indices: np.ndarray
    qwen_token_count: np.ndarray
    window_offsets: np.ndarray
    window_item_index: np.ndarray
    window_index: np.ndarray
    token_start: np.ndarray
    token_end: np.ndarray
    valid_tokens: np.ndarray
    sequences: list[list[int]]


@dataclass
class HiddenWindowCache:
    hidden: list[torch.Tensor]
    means: torch.Tensor


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract Eval 9 code-page BatchTopK/Temporal/Mean/Cross SAE "
            "features with deterministic torchrun rank sharding."
        )
    )
    p.add_argument(
        "--input-parquet",
        required=True,
        help="Canonical text-only code_page_hqs8_9_10k.parquet.",
    )
    p.add_argument(
        "--seeds-jsonl",
        required=True,
        help="Independent SAE seeds with seed_id, query_id, and text.",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument(
        "--output-dir",
        "--data-dir",
        dest="output_dir",
        required=True,
        help=(
            "Eval 9 data directory. Writes document_windows.jsonl and the "
            "features/ directory below it."
        ),
    )
    p.add_argument(
        "--checkpoint-selection",
        choices=("best", "final"),
        default="best",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--documents-per-shard",
        type=int,
        default=256,
        help=(
            "Atomic resume granularity within each rank. The default keeps "
            "checkpoint reloads low while bounding the CPU hidden cache."
        ),
    )
    p.add_argument(
        "--length-bucket-size",
        type=int,
        default=64,
        help="Bucket window lengths before batching to reduce padding.",
    )
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument(
        "--offload-embeddings-to-cpu",
        action="store_true",
        help="Keep the token embedding table on CPU if GPU memory is constrained.",
    )
    p.add_argument("--expected-documents", type=int, default=10_000)
    p.add_argument(
        "--expected-world-size",
        type=int,
        default=8,
        help="Defaults to the required eight torchrun ranks.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard this script's prior feature outputs and restart.",
    )
    p.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep atomic resume shards after the complete manifest is published.",
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.documents_per_shard <= 0:
        raise ValueError("--documents-per-shard must be positive")
    if not 1 <= args.length_bucket_size <= WINDOW_SIZE:
        raise ValueError(
            f"--length-bucket-size must be in [1, {WINDOW_SIZE}]"
        )
    if args.expected_documents <= 0:
        raise ValueError("--expected-documents must be positive")
    if args.expected_world_size <= 0:
        raise ValueError("--expected-world-size must be positive")


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _broadcast(value: Any, rank: int) -> Any:
    return broadcast_object(value, rank=rank)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_savez(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key].copy() for key in handle.files}


def _scalar_string(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, found shape={array.shape}")
    return str(array.item())


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, found shape={array.shape}")
    return int(array.item())


def _inspect_parquet(path: Path, expected_documents: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow.remove_metadata()
    if schema.names != ["text"]:
        raise ValueError(
            f"{path}: expected exactly one Parquet column named text, "
            f"found {schema.names}"
        )
    if not pa.types.is_string(schema.field("text").type):
        raise ValueError(
            f"{path}: expected text: string, found {schema.field('text').type}"
        )
    row_count = int(parquet.metadata.num_rows)
    if row_count != expected_documents:
        raise ValueError(
            f"{path}: expected {expected_documents} rows, found {row_count}"
        )
    return {
        "rows": row_count,
        "row_groups": int(parquet.metadata.num_row_groups),
        "schema": "text: string",
    }


def _load_document_texts(
    path: Path,
    *,
    expected_documents: int,
) -> list[str]:
    _inspect_parquet(path, expected_documents)
    table = pq.read_table(path, columns=["text"], use_threads=True)
    if table.num_rows != expected_documents or table.column_names != ["text"]:
        raise ValueError(
            f"{path}: inconsistent Parquet read, rows={table.num_rows}, "
            f"columns={table.column_names}"
        )
    texts = table.column("text").to_pylist()
    if len(texts) != expected_documents:
        raise ValueError(
            f"{path}: read {len(texts)} texts, expected {expected_documents}"
        )
    for document_index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(
                f"document_index={document_index} is not a non-null string"
            )
        if not text.strip():
            raise ValueError(f"document_index={document_index} is empty")
    return texts


def _load_seeds(path: Path) -> list[SeedRow]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[SeedRow] = []
    seen_seed_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            missing = [
                key
                for key in ("seed_id", "query_id", "text")
                if key not in payload
            ]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields {missing}"
                )
            seed_id = payload["seed_id"]
            query_id = payload["query_id"]
            text = payload["text"]
            if not isinstance(seed_id, str) or not seed_id.strip():
                raise ValueError(
                    f"{path}:{line_number}: seed_id must be a non-empty string"
                )
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(
                    f"{path}:{line_number}: query_id must be a non-empty string"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{path}:{line_number}: text must be a non-empty string"
                )
            if seed_id in seen_seed_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate seed_id={seed_id!r}"
                )
            seen_seed_ids.add(seed_id)
            rows.append(
                SeedRow(
                    seed_index=len(rows),
                    seed_id=seed_id,
                    query_id=query_id,
                    text=text,
                )
            )
    if not rows:
        raise ValueError(f"{path}: no seed rows")
    return rows


def _normalize_text(text: str) -> str:
    # Keep all printable content and ordinary tabs/newlines.  Only newline
    # conventions and control characters that cannot carry code-page semantics
    # are normalized.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _ILLEGAL_CONTROL_RE.sub(" ", text)


def _window_slices(token_count: int) -> list[tuple[int, int]]:
    if token_count <= 0:
        raise ValueError("cannot window an empty token sequence")
    if token_count <= TOKEN_BUDGET:
        return [
            (start, min(start + WINDOW_SIZE, token_count))
            for start in range(0, token_count, WINDOW_SIZE)
        ]
    starts = [
        int(round(index * (token_count - WINDOW_SIZE) / (MAX_WINDOWS - 1)))
        for index in range(MAX_WINDOWS)
    ]
    windows = [(start, start + WINDOW_SIZE) for start in starts]
    if len(set(starts)) != MAX_WINDOWS:
        raise AssertionError(
            f"uniform window formula produced duplicate starts for N={token_count}"
        )
    return windows


def _build_window_plan(
    *,
    item_indices: Sequence[int],
    texts: Sequence[str],
    tokenizer: Any,
) -> WindowPlan:
    if len(item_indices) != len(texts):
        raise ValueError("item indices and texts must have equal lengths")
    token_counts: list[int] = []
    offsets = [0]
    window_item_indices: list[int] = []
    window_indices: list[int] = []
    token_starts: list[int] = []
    token_ends: list[int] = []
    valid_tokens: list[int] = []
    sequences: list[list[int]] = []

    for item_index, raw_text in zip(item_indices, texts, strict=True):
        normalized = _normalize_text(raw_text)
        ids = tokenizer.encode(normalized, add_special_tokens=False)
        ids = [int(value) for value in ids]
        if not ids:
            raise ValueError(
                f"item_index={item_index} tokenizes to an empty sequence"
            )
        token_count = len(ids)
        windows = _window_slices(token_count)
        if not 1 <= len(windows) <= MAX_WINDOWS:
            raise AssertionError(
                f"item_index={item_index} has {len(windows)} windows"
            )
        for window_index, (start, end) in enumerate(windows):
            sequence = ids[start:end]
            if not sequence or len(sequence) > WINDOW_SIZE:
                raise AssertionError(
                    f"invalid window item={item_index} start={start} end={end}"
                )
            sequences.append(sequence)
            window_item_indices.append(int(item_index))
            window_indices.append(window_index)
            token_starts.append(start)
            token_ends.append(end)
            valid_tokens.append(len(sequence))
        token_counts.append(token_count)
        offsets.append(len(sequences))

    return WindowPlan(
        item_indices=np.asarray(item_indices, dtype=np.int64),
        qwen_token_count=np.asarray(token_counts, dtype=np.int64),
        window_offsets=np.asarray(offsets, dtype=np.int64),
        window_item_index=np.asarray(window_item_indices, dtype=np.int64),
        window_index=np.asarray(window_indices, dtype=np.int16),
        token_start=np.asarray(token_starts, dtype=np.int64),
        token_end=np.asarray(token_ends, dtype=np.int64),
        valid_tokens=np.asarray(valid_tokens, dtype=np.int16),
        sequences=sequences,
    )


def _empty_sparse(rows: int, *, value_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((rows, TOP_K), -1, dtype=np.int32),
        np.zeros((rows, TOP_K), dtype=value_dtype),
    )


def _store_tensor_topk(
    destination_indices: np.ndarray,
    destination_values: np.ndarray,
    rows: Sequence[int],
    indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    if indices.ndim != 2 or values.ndim != 2 or indices.shape != values.shape:
        raise ValueError("top-k tensors must have matching [B,K] shapes")
    if len(rows) != indices.shape[0]:
        raise ValueError("top-k batch row count mismatch")
    take = min(TOP_K, indices.shape[1])
    cpu_indices = indices[:, :take].detach().cpu().numpy().astype(np.int32)
    cpu_values = values[:, :take].detach().float().cpu().numpy().astype(np.float32)
    if not np.isfinite(cpu_values).all():
        raise FloatingPointError("SAE produced non-finite activations")
    if np.any(cpu_values < 0):
        raise FloatingPointError("thresholded SAE produced negative activations")
    for source_row, target_row in enumerate(rows):
        valid = cpu_values[source_row] > 0
        count = int(valid.sum())
        if count:
            destination_indices[target_row, :count] = cpu_indices[
                source_row, valid
            ]
            destination_values[target_row, :count] = cpu_values[
                source_row, valid
            ]


def _aggregate_document_sparse(
    *,
    window_indices: np.ndarray,
    window_values: np.ndarray,
    window_valid_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not (
        window_indices.shape == window_values.shape
        and window_indices.ndim == 2
        and window_indices.shape[1] == TOP_K
        and window_valid_tokens.shape == (window_indices.shape[0],)
    ):
        raise ValueError("invalid window sparse arrays for document aggregation")

    feature_parts: list[np.ndarray] = []
    contribution_parts: list[np.ndarray] = []
    total_weight = 0.0
    for row in range(window_indices.shape[0]):
        valid = window_indices[row] >= 0
        if not np.any(valid):
            continue
        features = window_indices[row, valid].astype(np.int64, copy=False)
        values = window_values[row, valid].astype(np.float32, copy=False)
        norm = float(np.linalg.norm(values))
        if not math.isfinite(norm) or norm <= 0:
            continue
        weight = float(window_valid_tokens[row])
        feature_parts.append(features)
        contribution_parts.append(values * (weight / norm))
        total_weight += weight

    output_indices = np.full(TOP_K, -1, dtype=np.int32)
    output_values = np.zeros(TOP_K, dtype=np.float32)
    if not feature_parts:
        return output_indices, output_values

    features = np.concatenate(feature_parts)
    contributions = np.concatenate(contribution_parts)
    unique_features, inverse = np.unique(features, return_inverse=True)
    sums = np.zeros(unique_features.shape[0], dtype=np.float32)
    np.add.at(sums, inverse, contributions)
    sums /= max(total_weight, 1.0)

    # Deterministic tie-break: descending activation, then ascending feature ID.
    order = np.lexsort((unique_features, -sums))
    order = order[:TOP_K]
    selected_features = unique_features[order]
    selected_values = sums[order]
    positive = selected_values > 0
    selected_features = selected_features[positive]
    selected_values = selected_values[positive]
    if selected_values.size:
        norm = float(np.linalg.norm(selected_values))
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError("invalid document-level sparse norm")
        selected_values = selected_values / norm
        count = selected_values.size
        output_indices[:count] = selected_features.astype(np.int32)
        output_values[:count] = selected_values.astype(np.float32)
    return output_indices, output_values


def _quantize_sparse(
    indices: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indices.shape != values.shape or indices.ndim != 2:
        raise ValueError("sparse indices and values must have matching matrices")
    output_indices = indices.astype(np.int32, copy=True)
    output_values = values.astype(np.float16)
    if not np.isfinite(output_values).all():
        raise FloatingPointError(
            "float16 sparse storage overflowed to a non-finite value"
        )
    valid = (output_indices >= 0) & (output_values > 0)
    output_indices[~valid] = -1
    output_values[~valid] = np.float16(0)
    nnz = valid.sum(axis=1).astype(np.int16)
    return output_indices, output_values, nnz


@torch.inference_mode()
def _cache_window_hidden(
    *,
    plan: WindowPlan,
    extractor: Any,
    batch_size: int,
    length_bucket_size: int,
    rank: int,
    label: str,
) -> HiddenWindowCache:
    window_count = len(plan.sequences)
    if window_count == 0:
        return HiddenWindowCache(
            hidden=[],
            means=torch.empty(
                (0, int(extractor.hidden_size)),
                dtype=extractor.dtype,
                device="cpu",
            ),
        )

    buckets: dict[int, list[int]] = {}
    for position, sequence in enumerate(plan.sequences):
        bucket = int(
            math.ceil(len(sequence) / length_bucket_size) * length_bucket_size
        )
        buckets.setdefault(bucket, []).append(position)

    hidden_cache: list[torch.Tensor | None] = [None] * window_count
    means_cache: list[torch.Tensor | None] = [None] * window_count
    completed_windows = 0
    for bucket in sorted(buckets):
        positions = buckets[bucket]
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            sequences = [plan.sequences[position] for position in batch_positions]
            layer_batch = extractor.forward_ids(sequences)
            means = layer_batch.means()
            for batch_row, target_position in enumerate(batch_positions):
                valid_length = len(sequences[batch_row])
                hidden_cache[target_position] = (
                    layer_batch.hidden[batch_row, :valid_length]
                    .detach()
                    .to("cpu", copy=True)
                    .contiguous()
                )
                means_cache[target_position] = (
                    means[batch_row]
                    .detach()
                    .to("cpu", copy=True)
                    .contiguous()
                )
            del means, layer_batch
            completed_windows += len(batch_positions)

        log(
            f"{label}: forwarded and cached {completed_windows}/"
            f"{window_count} windows "
            f"(length bucket <= {bucket})",
            rank=rank,
        )

    if any(value is None for value in hidden_cache):
        raise AssertionError(f"{label}: incomplete hidden cache")
    if any(value is None for value in means_cache):
        raise AssertionError(f"{label}: incomplete mean-hidden cache")
    return HiddenWindowCache(
        hidden=[value for value in hidden_cache if value is not None],
        means=torch.stack(
            [value for value in means_cache if value is not None],
            dim=0,
        ),
    )


def _load_single_encoder(
    *,
    mode: str,
    sae_set: Mapping[str, Any],
    device: torch.device,
    feature_widths: Mapping[str, int],
    rank: int,
) -> Any:
    if mode not in REQUIRED_METHODS:
        raise ValueError(f"unsupported SAE mode={mode!r}")
    # Import lazily so --help and CPU-only artifact checks do not import the
    # repository's custom Qwen3.5 modeling stack.
    from evals.document_linking.evaluate_lexical_controlled_document_linking import SparseEncoder

    checkpoint = Path(str(sae_set["modes"][mode]["checkpoint_path"]))
    log(
        f"loading the sole resident SAE encoder mode={mode} from {checkpoint}",
        rank=rank,
    )
    encoder = SparseEncoder(checkpoint, device)
    try:
        validate_full_dictionary_width(
            mode,
            encoder.dictionary_width,
            int(feature_widths[mode]),
        )
    except Exception:
        encoder.close()
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise
    return encoder


@torch.inference_mode()
def _encode_cached_mode(
    *,
    mode: str,
    plan: WindowPlan,
    cache: HiddenWindowCache,
    encoder: Any,
    batch_size: int,
    rank: int,
    label: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    window_count = len(plan.sequences)
    if len(cache.hidden) != window_count:
        raise ValueError("hidden cache length does not match the window plan")
    if cache.means.ndim != 2 or cache.means.shape[0] != window_count:
        raise ValueError("mean-hidden cache has an invalid shape")
    raw_window_indices, raw_window_values = _empty_sparse(
        window_count,
        value_dtype=np.float32,
    )
    device = encoder.device

    if mode in {"mean", "cross"}:
        for start in range(0, window_count, batch_size):
            stop = min(window_count, start + batch_size)
            means = cache.means[start:stop].to(
                device,
                dtype=encoder.weight.dtype,
                non_blocking=False,
            )
            indices, values = encoder.topk(means, TOP_K)
            _store_tensor_topk(
                raw_window_indices,
                raw_window_values,
                list(range(start, stop)),
                indices,
                values,
            )
            del means, indices, values
    elif mode in {"token", "temporal"}:
        for position, host_hidden in enumerate(cache.hidden):
            hidden = host_hidden.to(
                device,
                dtype=encoder.weight.dtype,
                non_blocking=False,
            )
            dense_codes = encoder.dense(hidden)
            aggregate = mean_after_threshold(dense_codes)
            del dense_codes, hidden
            values, indices = aggregate.topk(
                min(TOP_K, aggregate.shape[0])
            )
            _store_tensor_topk(
                raw_window_indices,
                raw_window_values,
                [position],
                indices.unsqueeze(0),
                values.unsqueeze(0),
            )
            del aggregate, indices, values
    else:
        raise ValueError(f"unsupported SAE mode={mode!r}")

    log(
        f"{label}: encoded {window_count} cached windows with mode={mode}",
        rank=rank,
    )

    item_count = plan.item_indices.shape[0]
    item_indices, item_values = _empty_sparse(
        item_count,
        value_dtype=np.float32,
    )
    for item_position in range(item_count):
        left = int(plan.window_offsets[item_position])
        right = int(plan.window_offsets[item_position + 1])
        row_indices, row_values = _aggregate_document_sparse(
            window_indices=raw_window_indices[left:right],
            window_values=raw_window_values[left:right],
            window_valid_tokens=plan.valid_tokens[left:right],
        )
        item_indices[item_position] = row_indices
        item_values[item_position] = row_values

    stored_window_indices, stored_window_values, window_nnz = (
        _quantize_sparse(
            raw_window_indices,
            raw_window_values,
        )
    )
    stored_item_indices, stored_item_values, item_nnz = _quantize_sparse(
        item_indices,
        item_values,
    )
    return (
        stored_window_indices,
        stored_window_values,
        window_nnz,
        stored_item_indices,
        stored_item_values,
        item_nnz,
    )


@torch.inference_mode()
def _encode_window_plan(
    *,
    plan: WindowPlan,
    extractor: Any,
    sae_set: Mapping[str, Any],
    feature_widths: Mapping[str, int],
    batch_size: int,
    length_bucket_size: int,
    rank: int,
    label: str,
) -> dict[
    str,
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
]:
    if not plan.sequences:
        if plan.item_indices.size:
            raise ValueError("items without token windows are not allowed")
        empty: dict[
            str,
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
        ] = {}
        for mode in REQUIRED_METHODS:
            window_indices, window_values = _empty_sparse(
                0, value_dtype=np.float32
            )
            item_indices, item_values = _empty_sparse(
                0, value_dtype=np.float32
            )
            empty[mode] = (
                *_quantize_sparse(window_indices, window_values),
                *_quantize_sparse(item_indices, item_values),
            )
        return empty

    cache = _cache_window_hidden(
        plan=plan,
        extractor=extractor,
        batch_size=batch_size,
        length_bucket_size=length_bucket_size,
        rank=rank,
        label=label,
    )
    encoded: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ] = {}
    device = torch.device(extractor.device)
    try:
        for mode in REQUIRED_METHODS:
            encoder = _load_single_encoder(
                mode=mode,
                sae_set=sae_set,
                device=device,
                feature_widths=feature_widths,
                rank=rank,
            )
            try:
                encoded[mode] = _encode_cached_mode(
                    mode=mode,
                    plan=plan,
                    cache=cache,
                    encoder=encoder,
                    batch_size=batch_size,
                    rank=rank,
                    label=label,
                )
            finally:
                encoder.close()
                del encoder
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                log(
                    f"{label}: unloaded SAE encoder mode={mode}",
                    rank=rank,
                )
    finally:
        cache.hidden.clear()
        del cache
        gc.collect()
    return encoded


def _collection_payload(
    *,
    plan: WindowPlan,
    encoded: Mapping[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ],
    kind: str,
    artifact_format: str,
    identity_digest: str,
    rank: int,
    world_size: int,
    shard_index: int,
    source_parquet_sha256: str | None = None,
    seeds_jsonl_sha256: str | None = None,
    seed_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    if kind not in {"document", "seed"}:
        raise ValueError(f"unsupported collection kind={kind!r}")
    item_key = "document_index" if kind == "document" else "seed_index"
    window_item_key = (
        "window_document_index" if kind == "document" else "window_seed_index"
    )
    payload: dict[str, np.ndarray] = {
        "artifact_format": np.asarray(artifact_format),
        "collection": np.asarray(kind),
        "identity_digest": np.asarray(identity_digest),
        "rank": np.asarray(rank, dtype=np.int16),
        "world_size": np.asarray(world_size, dtype=np.int16),
        "shard_index": np.asarray(shard_index, dtype=np.int32),
        "feature_width": np.asarray(DICTIONARY_WIDTH, dtype=np.int64),
        "top_k": np.asarray(TOP_K, dtype=np.int64),
        item_key: plan.item_indices.astype(np.int64, copy=False),
        "qwen_token_count": plan.qwen_token_count.astype(
            np.int64, copy=False
        ),
        "window_offsets": plan.window_offsets.astype(np.int64, copy=False),
        window_item_key: plan.window_item_index.astype(np.int64, copy=False),
        "window_index": plan.window_index.astype(np.int16, copy=False),
        "window_token_start": plan.token_start.astype(np.int64, copy=False),
        "window_token_end": plan.token_end.astype(np.int64, copy=False),
        "window_valid_tokens": plan.valid_tokens.astype(np.int16, copy=False),
    }
    if kind == "document":
        if source_parquet_sha256 is None:
            raise ValueError(
                "document payload requires source_parquet_sha256"
            )
        payload["source_parquet_sha256"] = np.asarray(
            source_parquet_sha256
        )
        payload["document_count"] = np.asarray(
            plan.item_indices.shape[0], dtype=np.int64
        )
    else:
        if seeds_jsonl_sha256 is None:
            raise ValueError("seed payload requires seeds_jsonl_sha256")
        payload["seeds_jsonl_sha256"] = np.asarray(seeds_jsonl_sha256)
    if kind == "seed":
        if seed_ids is None or query_ids is None:
            raise ValueError("seed payload requires seed_ids and query_ids")
        if not (
            len(seed_ids) == len(query_ids) == plan.item_indices.shape[0]
        ):
            raise ValueError("seed metadata length mismatch")
        payload["seed_id"] = np.asarray(seed_ids, dtype=np.str_)
        payload["query_id"] = np.asarray(query_ids, dtype=np.str_)

    for mode in REQUIRED_METHODS:
        (
            window_indices,
            window_values,
            window_nnz,
            item_indices,
            item_values,
            item_nnz,
        ) = encoded[mode]
        payload[f"{mode}_window_indices"] = window_indices
        payload[f"{mode}_window_values"] = window_values
        payload[f"{mode}_window_nnz"] = window_nnz
        payload[f"{mode}_{kind}_indices"] = item_indices
        payload[f"{mode}_{kind}_values"] = item_values
        payload[f"{mode}_{kind}_nnz"] = item_nnz
    return payload


def _expected_payload_keys(kind: str) -> set[str]:
    item_key = "document_index" if kind == "document" else "seed_index"
    window_item_key = (
        "window_document_index" if kind == "document" else "window_seed_index"
    )
    keys = {
        "artifact_format",
        "collection",
        "identity_digest",
        "rank",
        "world_size",
        "shard_index",
        "feature_width",
        "top_k",
        item_key,
        "qwen_token_count",
        "window_offsets",
        window_item_key,
        "window_index",
        "window_token_start",
        "window_token_end",
        "window_valid_tokens",
    }
    if kind == "document":
        keys.update(("source_parquet_sha256", "document_count"))
    else:
        keys.update(("seed_id", "query_id", "seeds_jsonl_sha256"))
    for mode in REQUIRED_METHODS:
        keys.update(
            (
                f"{mode}_window_indices",
                f"{mode}_window_values",
                f"{mode}_window_nnz",
                f"{mode}_{kind}_indices",
                f"{mode}_{kind}_values",
                f"{mode}_{kind}_nnz",
            )
        )
    return keys


def _validate_sparse_matrix(
    *,
    indices: np.ndarray,
    values: np.ndarray,
    nnz: np.ndarray,
    rows: int,
    name: str,
    require_unit_norm: bool,
) -> None:
    if indices.dtype != np.int32 or indices.shape != (rows, TOP_K):
        raise ValueError(
            f"{name} indices must be int32[{rows},{TOP_K}], "
            f"found {indices.dtype}{indices.shape}"
        )
    if values.dtype != np.float16 or values.shape != (rows, TOP_K):
        raise ValueError(
            f"{name} values must be float16[{rows},{TOP_K}], "
            f"found {values.dtype}{values.shape}"
        )
    if nnz.dtype != np.int16 or nnz.shape != (rows,):
        raise ValueError(
            f"{name} nnz must be int16[{rows}], found {nnz.dtype}{nnz.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    valid = indices >= 0
    if np.any(indices[valid] >= DICTIONARY_WIDTH):
        raise ValueError(f"{name} contains out-of-range feature indices")
    if np.any(indices[~valid] != -1):
        raise ValueError(f"{name} uses an invalid negative feature sentinel")
    if np.any(values[valid] <= 0) or np.any(values[~valid] != 0):
        raise ValueError(f"{name} index/value validity masks disagree")
    actual_nnz = valid.sum(axis=1).astype(np.int16)
    if not np.array_equal(nnz, actual_nnz):
        raise ValueError(f"{name} nnz does not match sparse contents")
    for row in range(rows):
        count = int(nnz[row])
        if count and not np.all(valid[row, :count]):
            raise ValueError(f"{name} row={row} has a hole before nnz")
        if np.any(valid[row, count:]):
            raise ValueError(f"{name} row={row} has data after nnz")
        if count > 1:
            active_values = values[row, :count].astype(np.float32)
            if np.any(active_values[:-1] < active_values[1:]):
                raise ValueError(
                    f"{name} row={row} is not sorted by descending value"
                )
            active_indices = indices[row, :count]
            if np.unique(active_indices).size != count:
                raise ValueError(
                    f"{name} row={row} contains duplicate feature indices"
                )
    if require_unit_norm and rows:
        norms = np.linalg.norm(values.astype(np.float32), axis=1)
        zero = nnz == 0
        if np.any(norms[zero] != 0):
            raise ValueError(f"{name} zero vectors have non-zero norms")
        if np.any(np.abs(norms[~zero] - 1.0) > 5e-3):
            raise ValueError(f"{name} non-zero vectors are not L2 normalized")


def _validate_collection_payload(
    payload: Mapping[str, np.ndarray],
    *,
    kind: str,
    identity_digest: str,
    expected_indices: Sequence[int] | None = None,
    expected_seed_ids: Sequence[str] | None = None,
    expected_query_ids: Sequence[str] | None = None,
    expected_rank: int | None = None,
    expected_world_size: int | None = None,
    allowed_formats: Sequence[str] | None = None,
) -> dict[str, int]:
    if set(payload) != _expected_payload_keys(kind):
        missing = sorted(_expected_payload_keys(kind) - set(payload))
        extra = sorted(set(payload) - _expected_payload_keys(kind))
        raise ValueError(
            f"{kind} payload schema mismatch: missing={missing}, extra={extra}"
        )
    artifact_format = _scalar_string(
        payload["artifact_format"], name="artifact_format"
    )
    if allowed_formats is not None and artifact_format not in allowed_formats:
        raise ValueError(
            f"unexpected {kind} artifact format={artifact_format!r}"
        )
    if _scalar_string(payload["collection"], name="collection") != kind:
        raise ValueError(f"payload collection is not {kind}")
    if (
        _scalar_string(payload["identity_digest"], name="identity_digest")
        != identity_digest
    ):
        raise ValueError(f"{kind} payload identity digest mismatch")
    rank = _scalar_integer(payload["rank"], name="rank")
    world_size = _scalar_integer(payload["world_size"], name="world_size")
    if expected_rank is not None and rank != expected_rank:
        raise ValueError(f"{kind} payload rank {rank} != {expected_rank}")
    if expected_world_size is not None and world_size != expected_world_size:
        raise ValueError(
            f"{kind} payload world_size {world_size} != {expected_world_size}"
        )
    if (
        _scalar_integer(payload["feature_width"], name="feature_width")
        != DICTIONARY_WIDTH
    ):
        raise ValueError("feature_width does not match the Eval 9 protocol")
    if _scalar_integer(payload["top_k"], name="top_k") != TOP_K:
        raise ValueError("top_k does not match the Eval 9 protocol")
    checksum_key = (
        "source_parquet_sha256"
        if kind == "document"
        else "seeds_jsonl_sha256"
    )
    checksum = _scalar_string(payload[checksum_key], name=checksum_key)
    if _SHA256_RE.fullmatch(checksum) is None:
        raise ValueError(f"{checksum_key} is not a SHA-256 hex digest")

    item_key = "document_index" if kind == "document" else "seed_index"
    window_item_key = (
        "window_document_index" if kind == "document" else "window_seed_index"
    )
    item_indices = payload[item_key]
    token_counts = payload["qwen_token_count"]
    offsets = payload["window_offsets"]
    if item_indices.dtype != np.int64 or item_indices.ndim != 1:
        raise ValueError(f"{item_key} must be a one-dimensional int64 array")
    item_count = item_indices.shape[0]
    if kind == "document":
        declared_count = _scalar_integer(
            payload["document_count"], name="document_count"
        )
        if declared_count != item_count:
            raise ValueError(
                f"document_count={declared_count} does not match "
                f"the {item_count} stored rows"
            )
    if token_counts.dtype != np.int64 or token_counts.shape != (item_count,):
        raise ValueError(
            "qwen_token_count must be int64 with one row per item"
        )
    if np.any(token_counts <= 0):
        raise ValueError("qwen_token_count must be positive")
    if offsets.dtype != np.int64 or offsets.shape != (item_count + 1,):
        raise ValueError(
            "window_offsets must be int64 with item_count + 1 entries"
        )
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("window_offsets must start at zero and be monotonic")
    window_count = int(offsets[-1])
    counts = np.diff(offsets)
    if np.any(counts < 1) or np.any(counts > MAX_WINDOWS):
        if item_count:
            raise ValueError("each item must have between one and eight windows")
        if window_count != 0:
            raise ValueError("empty collection cannot contain windows")

    expected_dtypes_and_shapes = {
        window_item_key: (np.dtype(np.int64), (window_count,)),
        "window_index": (np.dtype(np.int16), (window_count,)),
        "window_token_start": (np.dtype(np.int64), (window_count,)),
        "window_token_end": (np.dtype(np.int64), (window_count,)),
        "window_valid_tokens": (np.dtype(np.int16), (window_count,)),
    }
    for name, (dtype, shape) in expected_dtypes_and_shapes.items():
        value = payload[name]
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(
                f"{name} must be {dtype}{shape}, found {value.dtype}{value.shape}"
            )

    for item_position, item_index in enumerate(item_indices.tolist()):
        left = int(offsets[item_position])
        right = int(offsets[item_position + 1])
        if not np.all(payload[window_item_key][left:right] == item_index):
            raise ValueError(
                f"{window_item_key} disagrees with offsets for item={item_index}"
            )
        expected_window_index = np.arange(
            right - left, dtype=np.int16
        )
        if not np.array_equal(
            payload["window_index"][left:right], expected_window_index
        ):
            raise ValueError(
                f"window_index is not contiguous for item={item_index}"
            )
        expected_windows = _window_slices(int(token_counts[item_position]))
        actual_windows = list(
            zip(
                payload["window_token_start"][left:right].astype(int).tolist(),
                payload["window_token_end"][left:right].astype(int).tolist(),
                strict=True,
            )
        )
        if actual_windows != expected_windows:
            raise ValueError(
                f"window protocol mismatch for item={item_index}: "
                f"{actual_windows} != {expected_windows}"
            )
        expected_valid = np.asarray(
            [right_edge - left_edge for left_edge, right_edge in expected_windows],
            dtype=np.int16,
        )
        if not np.array_equal(
            payload["window_valid_tokens"][left:right], expected_valid
        ):
            raise ValueError(
                f"window_valid_tokens mismatch for item={item_index}"
            )

    if expected_indices is not None:
        expected = np.asarray(expected_indices, dtype=np.int64)
        if not np.array_equal(item_indices, expected):
            raise ValueError(
                f"{kind} indices differ from the deterministic expected order"
            )
    if (
        expected_rank is not None
        and expected_world_size is not None
        and rank >= 0
        and item_count
    ):
        if np.any(item_indices % expected_world_size != expected_rank):
            raise ValueError(
                f"{kind} payload violates index % world_size ownership"
            )

    if kind == "seed":
        for name in ("seed_id", "query_id"):
            values = payload[name]
            if values.ndim != 1 or values.shape != (item_count,):
                raise ValueError(f"{name} must have one string per seed")
            if values.dtype.kind not in {"U", "S"}:
                raise ValueError(f"{name} must use a non-object string dtype")
            if any(not str(value) for value in values.tolist()):
                raise ValueError(f"{name} contains an empty value")
        if expected_seed_ids is not None and payload["seed_id"].astype(str).tolist() != list(
            expected_seed_ids
        ):
            raise ValueError("seed_id order/content mismatch")
        if expected_query_ids is not None and payload["query_id"].astype(str).tolist() != list(
            expected_query_ids
        ):
            raise ValueError("query_id order/content mismatch")

    for mode in REQUIRED_METHODS:
        _validate_sparse_matrix(
            indices=payload[f"{mode}_window_indices"],
            values=payload[f"{mode}_window_values"],
            nnz=payload[f"{mode}_window_nnz"],
            rows=window_count,
            name=f"{mode}_window",
            require_unit_norm=False,
        )
        _validate_sparse_matrix(
            indices=payload[f"{mode}_{kind}_indices"],
            values=payload[f"{mode}_{kind}_values"],
            nnz=payload[f"{mode}_{kind}_nnz"],
            rows=item_count,
            name=f"{mode}_{kind}",
            require_unit_norm=True,
        )
    return {"items": item_count, "windows": window_count}


def _combine_payloads(
    payloads: Sequence[Mapping[str, np.ndarray]],
    *,
    kind: str,
    artifact_format: str,
    identity_digest: str,
    rank: int,
    world_size: int,
    shard_index: int = -1,
) -> dict[str, np.ndarray]:
    if not payloads:
        raise ValueError("cannot combine an empty payload list")
    item_key = "document_index" if kind == "document" else "seed_index"
    window_item_key = (
        "window_document_index" if kind == "document" else "window_seed_index"
    )
    item_counts = [payload[item_key].shape[0] for payload in payloads]
    window_counts = [
        int(payload["window_offsets"][-1]) for payload in payloads
    ]
    first = payloads[0]
    for name in ("feature_width", "top_k"):
        expected_value = _scalar_integer(first[name], name=name)
        for payload in payloads[1:]:
            if _scalar_integer(payload[name], name=name) != expected_value:
                raise ValueError(
                    f"cannot combine payloads with different {name} values"
                )
    checksum_key = (
        "source_parquet_sha256"
        if kind == "document"
        else "seeds_jsonl_sha256"
    )
    expected_checksum = _scalar_string(
        first[checksum_key], name=checksum_key
    )
    for payload in payloads[1:]:
        if (
            _scalar_string(payload[checksum_key], name=checksum_key)
            != expected_checksum
        ):
            raise ValueError(
                f"cannot combine payloads with different {checksum_key}"
            )
    output: dict[str, np.ndarray] = {
        "artifact_format": np.asarray(artifact_format),
        "collection": np.asarray(kind),
        "identity_digest": np.asarray(identity_digest),
        "rank": np.asarray(rank, dtype=np.int16),
        "world_size": np.asarray(world_size, dtype=np.int16),
        "shard_index": np.asarray(shard_index, dtype=np.int32),
        "feature_width": np.asarray(
            _scalar_integer(first["feature_width"], name="feature_width"),
            dtype=np.int64,
        ),
        "top_k": np.asarray(
            _scalar_integer(first["top_k"], name="top_k"),
            dtype=np.int64,
        ),
        checksum_key: np.asarray(expected_checksum),
        item_key: np.concatenate(
            [payload[item_key] for payload in payloads], axis=0
        ),
        "qwen_token_count": np.concatenate(
            [payload["qwen_token_count"] for payload in payloads], axis=0
        ),
        window_item_key: np.concatenate(
            [payload[window_item_key] for payload in payloads], axis=0
        ),
        "window_index": np.concatenate(
            [payload["window_index"] for payload in payloads], axis=0
        ),
        "window_token_start": np.concatenate(
            [payload["window_token_start"] for payload in payloads], axis=0
        ),
        "window_token_end": np.concatenate(
            [payload["window_token_end"] for payload in payloads], axis=0
        ),
        "window_valid_tokens": np.concatenate(
            [payload["window_valid_tokens"] for payload in payloads], axis=0
        ),
    }
    combined_counts = np.concatenate(
        [np.diff(payload["window_offsets"]) for payload in payloads]
    )
    output["window_offsets"] = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(combined_counts, dtype=np.int64),
        )
    )
    if int(output["window_offsets"][-1]) != sum(window_counts):
        raise AssertionError("combined window offsets are inconsistent")
    if output[item_key].shape[0] != sum(item_counts):
        raise AssertionError("combined item count is inconsistent")

    if kind == "document":
        output["document_count"] = np.asarray(
            sum(item_counts), dtype=np.int64
        )
    else:
        output["seed_id"] = np.concatenate(
            [payload["seed_id"] for payload in payloads], axis=0
        ).astype(np.str_)
        output["query_id"] = np.concatenate(
            [payload["query_id"] for payload in payloads], axis=0
        ).astype(np.str_)

    for mode in REQUIRED_METHODS:
        for scope in ("window", kind):
            output[f"{mode}_{scope}_indices"] = np.concatenate(
                [payload[f"{mode}_{scope}_indices"] for payload in payloads],
                axis=0,
            )
            output[f"{mode}_{scope}_values"] = np.concatenate(
                [payload[f"{mode}_{scope}_values"] for payload in payloads],
                axis=0,
            )
            output[f"{mode}_{scope}_nnz"] = np.concatenate(
                [payload[f"{mode}_{scope}_nnz"] for payload in payloads],
                axis=0,
            )
    return output


def _reorder_payload(
    payload: Mapping[str, np.ndarray],
    *,
    kind: str,
    order: np.ndarray,
    artifact_format: str,
    identity_digest: str,
    rank: int,
    world_size: int,
) -> dict[str, np.ndarray]:
    item_key = "document_index" if kind == "document" else "seed_index"
    window_item_key = (
        "window_document_index" if kind == "document" else "window_seed_index"
    )
    item_count = payload[item_key].shape[0]
    order = np.asarray(order, dtype=np.int64)
    if order.shape != (item_count,) or not np.array_equal(
        np.sort(order), np.arange(item_count, dtype=np.int64)
    ):
        raise ValueError("reorder must be a permutation of every item row")

    window_slices = [
        slice(
            int(payload["window_offsets"][position]),
            int(payload["window_offsets"][position + 1]),
        )
        for position in order.tolist()
    ]
    counts = np.asarray(
        [window_slice.stop - window_slice.start for window_slice in window_slices],
        dtype=np.int64,
    )
    window_order = np.concatenate(
        [
            np.arange(window_slice.start, window_slice.stop, dtype=np.int64)
            for window_slice in window_slices
        ]
    )
    output: dict[str, np.ndarray] = {
        "artifact_format": np.asarray(artifact_format),
        "collection": np.asarray(kind),
        "identity_digest": np.asarray(identity_digest),
        "rank": np.asarray(rank, dtype=np.int16),
        "world_size": np.asarray(world_size, dtype=np.int16),
        "shard_index": np.asarray(-1, dtype=np.int32),
        "feature_width": payload["feature_width"].copy(),
        "top_k": payload["top_k"].copy(),
        item_key: payload[item_key][order],
        "qwen_token_count": payload["qwen_token_count"][order],
        "window_offsets": np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum(counts, dtype=np.int64),
            )
        ),
        window_item_key: payload[window_item_key][window_order],
        "window_index": payload["window_index"][window_order],
        "window_token_start": payload["window_token_start"][window_order],
        "window_token_end": payload["window_token_end"][window_order],
        "window_valid_tokens": payload["window_valid_tokens"][window_order],
    }
    if kind == "document":
        output["source_parquet_sha256"] = payload[
            "source_parquet_sha256"
        ].copy()
        output["document_count"] = np.asarray(
            item_count, dtype=np.int64
        )
    else:
        output["seeds_jsonl_sha256"] = payload[
            "seeds_jsonl_sha256"
        ].copy()
        output["seed_id"] = payload["seed_id"][order].astype(np.str_)
        output["query_id"] = payload["query_id"][order].astype(np.str_)
    for mode in REQUIRED_METHODS:
        output[f"{mode}_window_indices"] = payload[
            f"{mode}_window_indices"
        ][window_order]
        output[f"{mode}_window_values"] = payload[
            f"{mode}_window_values"
        ][window_order]
        output[f"{mode}_window_nnz"] = payload[f"{mode}_window_nnz"][
            window_order
        ]
        output[f"{mode}_{kind}_indices"] = payload[
            f"{mode}_{kind}_indices"
        ][order]
        output[f"{mode}_{kind}_values"] = payload[
            f"{mode}_{kind}_values"
        ][order]
        output[f"{mode}_{kind}_nnz"] = payload[f"{mode}_{kind}_nnz"][
            order
        ]
    return output


def _document_shard_indices(
    *,
    document_count: int,
    rank: int,
    world_size: int,
    documents_per_shard: int,
) -> list[list[int]]:
    owned = list(range(rank, document_count, world_size))
    return [
        owned[start : start + documents_per_shard]
        for start in range(0, len(owned), documents_per_shard)
    ]


def _shard_path(rank_work_dir: Path, shard_index: int) -> Path:
    return rank_work_dir / f"document-shard-{shard_index:05d}.npz"


def _progress_path(rank_work_dir: Path) -> Path:
    return rank_work_dir / "progress.json"


def _write_rank_progress(
    *,
    rank_work_dir: Path,
    identity_digest: str,
    rank: int,
    world_size: int,
    expected_indices: Sequence[int],
    shard_paths: Sequence[Path],
    complete: bool,
) -> None:
    records = []
    completed_documents = 0
    completed_windows = 0
    for shard_index, path in enumerate(shard_paths):
        if not path.is_file():
            continue
        payload = _load_npz(path)
        stats = _validate_collection_payload(
            payload,
            kind="document",
            identity_digest=identity_digest,
            expected_rank=rank,
            expected_world_size=world_size,
            allowed_formats=(SHARD_FORMAT,),
        )
        records.append(
            {
                "shard_index": shard_index,
                "documents": stats["items"],
                "windows": stats["windows"],
                "file": file_record(path, relative_to=rank_work_dir),
            }
        )
        completed_documents += stats["items"]
        completed_windows += stats["windows"]
    atomic_json_dump(
        {
            "format": PROGRESS_FORMAT,
            "complete": bool(complete),
            "identity_digest": identity_digest,
            "rank": rank,
            "world_size": world_size,
            "expected_documents": len(expected_indices),
            "expected_document_indices_digest": json_digest(
                [int(value) for value in expected_indices]
            ),
            "completed_documents": completed_documents,
            "completed_windows": completed_windows,
            "shards": records,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        _progress_path(rank_work_dir),
    )


def _extract_document_shard(
    *,
    shard_index: int,
    document_indices: Sequence[int],
    document_texts: Sequence[str],
    extractor: Any,
    sae_set: Mapping[str, Any],
    feature_widths: Mapping[str, int],
    args: argparse.Namespace,
    identity_digest: str,
    source_parquet_sha256: str,
    rank: int,
    world_size: int,
) -> dict[str, np.ndarray]:
    texts = [document_texts[index] for index in document_indices]
    plan = _build_window_plan(
        item_indices=document_indices,
        texts=texts,
        tokenizer=extractor.tokenizer,
    )
    encoded = _encode_window_plan(
        plan=plan,
        extractor=extractor,
        sae_set=sae_set,
        feature_widths=feature_widths,
        batch_size=args.batch_size,
        length_bucket_size=args.length_bucket_size,
        rank=rank,
        label=f"document shard {shard_index:05d}",
    )
    payload = _collection_payload(
        plan=plan,
        encoded=encoded,
        kind="document",
        artifact_format=SHARD_FORMAT,
        identity_digest=identity_digest,
        rank=rank,
        world_size=world_size,
        shard_index=shard_index,
        source_parquet_sha256=source_parquet_sha256,
    )
    _validate_collection_payload(
        payload,
        kind="document",
        identity_digest=identity_digest,
        expected_indices=document_indices,
        expected_rank=rank,
        expected_world_size=world_size,
        allowed_formats=(SHARD_FORMAT,),
    )
    return payload


def _extract_seed_rank(
    *,
    seeds: Sequence[SeedRow],
    owned_seed_indices: Sequence[int],
    extractor: Any,
    sae_set: Mapping[str, Any],
    feature_widths: Mapping[str, int],
    args: argparse.Namespace,
    identity_digest: str,
    seeds_jsonl_sha256: str,
    rank: int,
    world_size: int,
) -> dict[str, np.ndarray]:
    selected = [seeds[index] for index in owned_seed_indices]
    plan = _build_window_plan(
        item_indices=owned_seed_indices,
        texts=[row.text for row in selected],
        tokenizer=extractor.tokenizer,
    )
    encoded = _encode_window_plan(
        plan=plan,
        extractor=extractor,
        sae_set=sae_set,
        feature_widths=feature_widths,
        batch_size=args.batch_size,
        length_bucket_size=args.length_bucket_size,
        rank=rank,
        label="seed pages",
    )
    payload = _collection_payload(
        plan=plan,
        encoded=encoded,
        kind="seed",
        artifact_format=SEED_RANK_FORMAT,
        identity_digest=identity_digest,
        rank=rank,
        world_size=world_size,
        shard_index=-1,
        seeds_jsonl_sha256=seeds_jsonl_sha256,
        seed_ids=[row.seed_id for row in selected],
        query_ids=[row.query_id for row in selected],
    )
    _validate_collection_payload(
        payload,
        kind="seed",
        identity_digest=identity_digest,
        expected_indices=owned_seed_indices,
        expected_seed_ids=[row.seed_id for row in selected],
        expected_query_ids=[row.query_id for row in selected],
        expected_rank=rank,
        expected_world_size=world_size,
        allowed_formats=(SEED_RANK_FORMAT,),
    )
    return payload


def _close_extractor(extractor: Any) -> None:
    if extractor is not None:
        extractor.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rank_file(features_dir: Path, rank: int) -> Path:
    return features_dir / f"rank{rank:03d}.npz"


def _seed_rank_file(work_dir: Path, rank: int) -> Path:
    return work_dir / f"seed-rank{rank:03d}.npz"


def _validate_rank_file(
    path: Path,
    *,
    identity_digest: str,
    expected_indices: Sequence[int],
    rank: int,
    world_size: int,
) -> dict[str, int]:
    return _validate_collection_payload(
        _load_npz(path),
        kind="document",
        identity_digest=identity_digest,
        expected_indices=expected_indices,
        expected_rank=rank,
        expected_world_size=world_size,
        allowed_formats=(RANK_FORMAT,),
    )


def _validate_seed_rank_file(
    path: Path,
    *,
    identity_digest: str,
    seeds: Sequence[SeedRow],
    owned_seed_indices: Sequence[int],
    rank: int,
    world_size: int,
) -> dict[str, int]:
    selected = [seeds[index] for index in owned_seed_indices]
    return _validate_collection_payload(
        _load_npz(path),
        kind="seed",
        identity_digest=identity_digest,
        expected_indices=owned_seed_indices,
        expected_seed_ids=[row.seed_id for row in selected],
        expected_query_ids=[row.query_id for row in selected],
        expected_rank=rank,
        expected_world_size=world_size,
        allowed_formats=(SEED_RANK_FORMAT,),
    )


def _assemble_rank_file(
    *,
    shard_paths: Sequence[Path],
    output_path: Path,
    identity_digest: str,
    expected_indices: Sequence[int],
    rank: int,
    world_size: int,
) -> dict[str, int]:
    payloads = []
    cursor = 0
    for shard_index, path in enumerate(shard_paths):
        payload = _load_npz(path)
        shard_count = payload["document_index"].shape[0]
        shard_expected = expected_indices[cursor : cursor + shard_count]
        _validate_collection_payload(
            payload,
            kind="document",
            identity_digest=identity_digest,
            expected_indices=shard_expected,
            expected_rank=rank,
            expected_world_size=world_size,
            allowed_formats=(SHARD_FORMAT,),
        )
        if _scalar_integer(payload["shard_index"], name="shard_index") != shard_index:
            raise ValueError(f"{path}: shard_index metadata mismatch")
        cursor += shard_count
        payloads.append(payload)
    if cursor != len(expected_indices):
        raise ValueError(
            f"rank={rank} shards cover {cursor} documents, "
            f"expected {len(expected_indices)}"
        )
    combined = _combine_payloads(
        payloads,
        kind="document",
        artifact_format=RANK_FORMAT,
        identity_digest=identity_digest,
        rank=rank,
        world_size=world_size,
    )
    stats = _validate_collection_payload(
        combined,
        kind="document",
        identity_digest=identity_digest,
        expected_indices=expected_indices,
        expected_rank=rank,
        expected_world_size=world_size,
        allowed_formats=(RANK_FORMAT,),
    )
    _atomic_savez(output_path, combined)
    _validate_rank_file(
        output_path,
        identity_digest=identity_digest,
        expected_indices=expected_indices,
        rank=rank,
        world_size=world_size,
    )
    return stats


def _merge_documents(
    *,
    features_dir: Path,
    identity_digest: str,
    document_count: int,
    world_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    rank_payloads = []
    for rank in range(world_size):
        expected = list(range(rank, document_count, world_size))
        path = _rank_file(features_dir, rank)
        payload = _load_npz(path)
        _validate_collection_payload(
            payload,
            kind="document",
            identity_digest=identity_digest,
            expected_indices=expected,
            expected_rank=rank,
            expected_world_size=world_size,
            allowed_formats=(RANK_FORMAT,),
        )
        rank_payloads.append(payload)
    concatenated = _combine_payloads(
        rank_payloads,
        kind="document",
        artifact_format=MERGED_FORMAT,
        identity_digest=identity_digest,
        rank=-1,
        world_size=world_size,
    )
    order = np.argsort(
        concatenated["document_index"], kind="stable"
    ).astype(np.int64)
    merged = _reorder_payload(
        concatenated,
        kind="document",
        order=order,
        artifact_format=MERGED_FORMAT,
        identity_digest=identity_digest,
        rank=-1,
        world_size=world_size,
    )
    stats = _validate_collection_payload(
        merged,
        kind="document",
        identity_digest=identity_digest,
        expected_indices=list(range(document_count)),
        expected_rank=-1,
        expected_world_size=world_size,
        allowed_formats=(MERGED_FORMAT,),
    )
    return merged, stats


def _merge_seeds(
    *,
    work_dir: Path,
    identity_digest: str,
    seeds: Sequence[SeedRow],
    world_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    rank_payloads = []
    for rank in range(world_size):
        owned = list(range(rank, len(seeds), world_size))
        selected = [seeds[index] for index in owned]
        path = _seed_rank_file(work_dir, rank)
        payload = _load_npz(path)
        _validate_collection_payload(
            payload,
            kind="seed",
            identity_digest=identity_digest,
            expected_indices=owned,
            expected_seed_ids=[row.seed_id for row in selected],
            expected_query_ids=[row.query_id for row in selected],
            expected_rank=rank,
            expected_world_size=world_size,
            allowed_formats=(SEED_RANK_FORMAT,),
        )
        rank_payloads.append(payload)
    concatenated = _combine_payloads(
        rank_payloads,
        kind="seed",
        artifact_format=SEED_FORMAT,
        identity_digest=identity_digest,
        rank=-1,
        world_size=world_size,
    )
    order = np.argsort(concatenated["seed_index"], kind="stable").astype(
        np.int64
    )
    merged = _reorder_payload(
        concatenated,
        kind="seed",
        order=order,
        artifact_format=SEED_FORMAT,
        identity_digest=identity_digest,
        rank=-1,
        world_size=world_size,
    )
    stats = _validate_collection_payload(
        merged,
        kind="seed",
        identity_digest=identity_digest,
        expected_indices=list(range(len(seeds))),
        expected_seed_ids=[row.seed_id for row in seeds],
        expected_query_ids=[row.query_id for row in seeds],
        expected_rank=-1,
        expected_world_size=world_size,
        allowed_formats=(SEED_FORMAT,),
    )
    return merged, stats


def _document_window_rows(
    merged: Mapping[str, np.ndarray],
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    document_indices = merged["document_index"]
    offsets = merged["window_offsets"]
    for document_position, document_index in enumerate(
        document_indices.tolist()
    ):
        left = int(offsets[document_position])
        right = int(offsets[document_position + 1])
        token_count = int(merged["qwen_token_count"][document_position])
        for window_position in range(left, right):
            rows.append(
                {
                    "document_index": int(document_index),
                    "qwen_token_count": token_count,
                    "window_index": int(
                        merged["window_index"][window_position]
                    ),
                    "token_start": int(
                        merged["window_token_start"][window_position]
                    ),
                    "token_end": int(
                        merged["window_token_end"][window_position]
                    ),
                    "valid_tokens": int(
                        merged["window_valid_tokens"][window_position]
                    ),
                }
            )
    return rows


def _validate_document_windows(
    path: Path,
    merged: Mapping[str, np.ndarray],
) -> None:
    expected_rows = _document_window_rows(merged)
    actual_rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"{path}:{line_number}: blank lines are not allowed"
                )
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            expected_keys = {
                "document_index",
                "qwen_token_count",
                "window_index",
                "token_start",
                "token_end",
                "valid_tokens",
            }
            if set(payload) != expected_keys:
                raise ValueError(
                    f"{path}:{line_number}: window schema mismatch"
                )
            actual_rows.append(
                {key: int(payload[key]) for key in sorted(expected_keys)}
            )
    normalized_expected = [
        {key: int(row[key]) for key in sorted(row)} for row in expected_rows
    ]
    if actual_rows != normalized_expected:
        raise ValueError(
            f"{path}: document window rows do not match merged_features.npz"
        )


def _zero_vector_counts(
    payload: Mapping[str, np.ndarray],
    *,
    kind: str,
) -> dict[str, dict[str, int]]:
    return {
        mode: {
            "window": int(np.sum(payload[f"{mode}_window_nnz"] == 0)),
            kind: int(np.sum(payload[f"{mode}_{kind}_nnz"] == 0)),
        }
        for mode in REQUIRED_METHODS
    }


def _build_identity(
    *,
    args: argparse.Namespace,
    world_size: int,
    parquet_info: Mapping[str, Any],
    parquet_record: Mapping[str, Any],
    seed_record: Mapping[str, Any],
    seeds: Sequence[SeedRow],
    sae_set: Mapping[str, Any],
    model_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    feature_widths = full_dictionary_feature_widths(
        sae_set, modes=REQUIRED_METHODS
    )
    script_path = Path(__file__).resolve()
    script_record = file_record(script_path)
    return {
        "input_parquet": {
            **dict(parquet_record),
            **dict(parquet_info),
            "path": str(Path(args.input_parquet).expanduser().resolve()),
        },
        "seeds_jsonl": {
            **dict(seed_record),
            "rows": len(seeds),
            "seed_ids_digest": json_digest([row.seed_id for row in seeds]),
            "query_ids_digest": json_digest([row.query_id for row in seeds]),
            "path": str(Path(args.seeds_jsonl).expanduser().resolve()),
        },
        "extractor_script": {
            **script_record,
            "path": str(script_path),
        },
        "model": dict(model_fingerprint),
        "layer": int(args.layer),
        "sae_set_digest": str(sae_set["artifact_digest"]),
        "checkpoint_selection": str(args.checkpoint_selection),
        "sae_modes": list(REQUIRED_METHODS),
        "feature_widths": feature_widths,
        "top_k": TOP_K,
        "world_size": world_size,
        "ownership": "document_index % world_size",
        "model_dtype": str(args.model_dtype),
        "attention_implementation": str(args.attn_implementation),
        "offload_embeddings_to_cpu": bool(
            args.offload_embeddings_to_cpu
        ),
        "batch_size": int(args.batch_size),
        "documents_per_shard": int(args.documents_per_shard),
        "length_bucket_size": int(args.length_bucket_size),
        "text_normalization": (
            "CRLF/CR_to_LF_and_replace_C0_C1_controls_except_tab_newline"
        ),
        "windowing": {
            "tokenizer_add_special_tokens": False,
            "window_size": WINDOW_SIZE,
            "maximum_windows": MAX_WINDOWS,
            "continuous_until_tokens": TOKEN_BUDGET,
            "long_document_starts": (
                "round(i * (N - 512) / 7), i=0..7"
            ),
            "rounding": "Python round (nearest, ties-to-even)",
            "independent_forward": True,
            "kv_cache": False,
        },
        "representation_protocol": fixed_chunk_protocol_metadata(
            feature_widths=feature_widths
        ),
        "page_aggregation": {
            "window_representation": "threshold_then_top128",
            "window_normalization": "L2_nonzero",
            "window_weight": "valid_token_count",
            "combination": "weighted_mean",
            "document_truncation": "top128",
            "document_normalization": "L2_nonzero",
            "zero_vectors": "explicit_nnz_zero",
        },
        "memory_policy": {
            "qwen_forward_per_window": 1,
            "hidden_shared_across_all_four_saes": True,
            "hidden_cache": "CPU_per_atomic_shard",
            "sae_encoder_residency": "one_mode_at_a_time",
            "checkpoint_tensors_loaded": [
                "encoder_weight",
                "encoder_bias",
                "pre_bias_or_legacy_decoder_bias",
                "threshold",
                "activation_scale",
            ],
            "decoder_weight_loaded": False,
            "dense_token_codes": "one SAE and one window at a time",
        },
    }


def _validate_sae_set(
    *,
    sae_set: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, int]:
    modes = sae_set.get("modes")
    common = sae_set.get("common")
    if not isinstance(modes, Mapping) or tuple(modes) != REQUIRED_METHODS:
        raise ValueError(
            f"Eval 9 requires exactly {REQUIRED_METHODS}, found "
            f"{tuple(modes) if isinstance(modes, Mapping) else modes!r}"
        )
    if not isinstance(common, Mapping):
        raise ValueError("invalid SAE artifact common metadata")
    if int(common.get("layer", -1)) != args.layer:
        raise ValueError(
            f"SAE layer={common.get('layer')} does not match --layer={args.layer}"
        )
    checkpoint_model = Path(str(common.get("model", ""))).expanduser().resolve()
    requested_model = Path(args.model).expanduser().resolve()
    if checkpoint_model != requested_model:
        raise ValueError(
            f"SAE model={checkpoint_model} does not match --model={requested_model}"
        )
    if int(common.get("dict_size", 0)) != DICTIONARY_WIDTH:
        raise ValueError(
            f"Eval 9 requires dictionary width {DICTIONARY_WIDTH}, "
            f"found {common.get('dict_size')}"
        )
    if int(common.get("k", 0)) != TOP_K:
        raise ValueError(
            f"Eval 9 requires SAE K={TOP_K}, found {common.get('k')}"
        )
    feature_widths = full_dictionary_feature_widths(
        sae_set, modes=REQUIRED_METHODS
    )
    for mode in REQUIRED_METHODS:
        validate_full_dictionary_width(
            mode, int(feature_widths[mode]), DICTIONARY_WIDTH
        )
    return feature_widths


def _prepare_run(
    *,
    args: argparse.Namespace,
    world_size: int,
) -> dict[str, Any]:
    input_path = Path(args.input_parquet).expanduser().resolve()
    seeds_path = Path(args.seeds_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    features_dir = output_dir / "features"
    work_dir = features_dir / ".work"
    windows_path = output_dir / "document_windows.jsonl"
    manifest_path = features_dir / "feature_manifest.json"

    parquet_info = _inspect_parquet(input_path, args.expected_documents)
    seeds = _load_seeds(seeds_path)
    sae_set = resolve_sae_artifact_set(
        args.sae_root,
        selection=args.checkpoint_selection,
        modes=REQUIRED_METHODS,
    )
    feature_widths = _validate_sae_set(sae_set=sae_set, args=args)
    model_fingerprint = model_metadata_fingerprint(args.model)
    identity = _build_identity(
        args=args,
        world_size=world_size,
        parquet_info=parquet_info,
        parquet_record=file_record(input_path),
        seed_record=file_record(seeds_path),
        seeds=seeds,
        sae_set=sae_set,
        model_fingerprint=model_fingerprint,
    )
    identity_digest = json_digest(identity)

    if args.overwrite:
        if features_dir.exists():
            shutil.rmtree(features_dir)
        if windows_path.exists():
            windows_path.unlink()

    existing = None
    if manifest_path.is_file() and not args.overwrite:
        existing = ensure_reusable_artifact(
            manifest_path,
            expected_format=FEATURE_FORMAT,
            expected_identity=identity,
        )
        _validate_complete_outputs(
            features_dir=features_dir,
            windows_path=windows_path,
            identity_digest=identity_digest,
            document_count=args.expected_documents,
            seeds=seeds,
            world_size=world_size,
        )

    if existing is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        run_path = work_dir / "run_manifest.json"
        if run_path.is_file():
            run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
            if (
                run_manifest.get("format") != RUN_FORMAT
                or run_manifest.get("identity") != identity
                or run_manifest.get("identity_digest") != identity_digest
            ):
                raise ValueError(
                    f"incomplete feature directory belongs to another run: "
                    f"{features_dir}; use --overwrite"
                )
        else:
            unknown_entries = [
                path
                for path in features_dir.iterdir()
                if path.name != ".work"
            ]
            unknown_work_entries = (
                list(work_dir.iterdir()) if work_dir.exists() else []
            )
            if unknown_entries or unknown_work_entries:
                raise ValueError(
                    f"unidentified incomplete output at {features_dir}; "
                    "use --overwrite"
                )
            atomic_json_dump(
                {
                    "format": RUN_FORMAT,
                    "complete": False,
                    "identity": identity,
                    "identity_digest": identity_digest,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                run_path,
            )

    return {
        "skip": existing is not None,
        "input_path": str(input_path),
        "seeds_path": str(seeds_path),
        "output_dir": str(output_dir),
        "features_dir": str(features_dir),
        "work_dir": str(work_dir),
        "windows_path": str(windows_path),
        "manifest_path": str(manifest_path),
        "identity": identity,
        "identity_digest": identity_digest,
        "sae_set": sae_set,
        "feature_widths": feature_widths,
        "seed_count": len(seeds),
        "seed_ids_digest": json_digest([row.seed_id for row in seeds]),
        "query_ids_digest": json_digest([row.query_id for row in seeds]),
    }


def _validate_complete_outputs(
    *,
    features_dir: Path,
    windows_path: Path,
    identity_digest: str,
    document_count: int,
    seeds: Sequence[SeedRow],
    world_size: int,
) -> None:
    for rank in range(world_size):
        _validate_rank_file(
            _rank_file(features_dir, rank),
            identity_digest=identity_digest,
            expected_indices=list(range(rank, document_count, world_size)),
            rank=rank,
            world_size=world_size,
        )
    merged = _load_npz(features_dir / "merged_features.npz")
    _validate_collection_payload(
        merged,
        kind="document",
        identity_digest=identity_digest,
        expected_indices=list(range(document_count)),
        expected_rank=-1,
        expected_world_size=world_size,
        allowed_formats=(MERGED_FORMAT,),
    )
    seed_payload = _load_npz(features_dir / "seed_features.npz")
    _validate_collection_payload(
        seed_payload,
        kind="seed",
        identity_digest=identity_digest,
        expected_indices=list(range(len(seeds))),
        expected_seed_ids=[row.seed_id for row in seeds],
        expected_query_ids=[row.query_id for row in seeds],
        expected_rank=-1,
        expected_world_size=world_size,
        allowed_formats=(SEED_FORMAT,),
    )
    _validate_document_windows(windows_path, merged)


def _run_rank(
    *,
    args: argparse.Namespace,
    context: Mapping[str, Any],
    rank: int,
    world_size: int,
    local_rank: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    features_dir = Path(str(context["features_dir"]))
    work_dir = Path(str(context["work_dir"]))
    rank_work_dir = work_dir / f"rank{rank:03d}"
    rank_work_dir.mkdir(parents=True, exist_ok=True)
    identity_digest = str(context["identity_digest"])
    document_count = int(args.expected_documents)
    expected_document_indices = list(range(rank, document_count, world_size))
    shard_indices = _document_shard_indices(
        document_count=document_count,
        rank=rank,
        world_size=world_size,
        documents_per_shard=args.documents_per_shard,
    )
    shard_paths = [
        _shard_path(rank_work_dir, shard_index)
        for shard_index in range(len(shard_indices))
    ]
    rank_path = _rank_file(features_dir, rank)

    seeds = _load_seeds(Path(str(context["seeds_path"])))
    if len(seeds) != int(context["seed_count"]):
        raise ValueError("seed count changed after run initialization")
    if json_digest([row.seed_id for row in seeds]) != context["seed_ids_digest"]:
        raise ValueError("seed IDs changed after run initialization")
    if json_digest([row.query_id for row in seeds]) != context["query_ids_digest"]:
        raise ValueError("seed query IDs changed after run initialization")
    owned_seed_indices = list(range(rank, len(seeds), world_size))
    seed_rank_path = _seed_rank_file(work_dir, rank)

    rank_complete = False
    if rank_path.is_file():
        _validate_rank_file(
            rank_path,
            identity_digest=identity_digest,
            expected_indices=expected_document_indices,
            rank=rank,
            world_size=world_size,
        )
        rank_complete = True

    missing_document_shards: list[int] = []
    if not rank_complete:
        for shard_index, (path, expected) in enumerate(
            zip(shard_paths, shard_indices, strict=True)
        ):
            if path.is_file():
                payload = _load_npz(path)
                _validate_collection_payload(
                    payload,
                    kind="document",
                    identity_digest=identity_digest,
                    expected_indices=expected,
                    expected_rank=rank,
                    expected_world_size=world_size,
                    allowed_formats=(SHARD_FORMAT,),
                )
                if (
                    _scalar_integer(
                        payload["shard_index"], name="shard_index"
                    )
                    != shard_index
                ):
                    raise ValueError(f"{path}: shard index mismatch")
            else:
                missing_document_shards.append(shard_index)

    seed_complete = False
    if seed_rank_path.is_file():
        _validate_seed_rank_file(
            seed_rank_path,
            identity_digest=identity_digest,
            seeds=seeds,
            owned_seed_indices=owned_seed_indices,
            rank=rank,
            world_size=world_size,
        )
        seed_complete = True

    extractor: Any = None
    document_texts: list[str] | None = None
    try:
        if missing_document_shards or not seed_complete:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Eval 9 feature extraction requires CUDA"
                )
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only "
                    f"{torch.cuda.device_count()} CUDA devices are visible"
                )
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            # This import depends on the repository's Qwen3.5-enabled
            # transformers environment and is intentionally delayed until a
            # rank genuinely needs model inference.
            from chunk_saes.modeling import TargetLayerExtractor

            log(
                "loading truncated Qwen target-layer extractor; "
                "each selected window will be forwarded once",
                rank=rank,
            )
            extractor = TargetLayerExtractor(
                args.model,
                args.layer,
                str(device),
                dtype=args.model_dtype,
                attn_implementation=args.attn_implementation,
                offload_embeddings_to_cpu=args.offload_embeddings_to_cpu,
            )
            expected_activation_dim = int(
                context["sae_set"]["common"]["activation_dim"]
            )
            if extractor.hidden_size != expected_activation_dim:
                raise ValueError(
                    f"model hidden size {extractor.hidden_size} does not match "
                    f"SAE activation_dim {expected_activation_dim}"
                )
        if missing_document_shards:
            document_texts = _load_document_texts(
                Path(str(context["input_path"])),
                expected_documents=document_count,
            )
            for shard_index in missing_document_shards:
                shard_started = time.perf_counter()
                payload = _extract_document_shard(
                    shard_index=shard_index,
                    document_indices=shard_indices[shard_index],
                    document_texts=document_texts,
                    extractor=extractor,
                    sae_set=context["sae_set"],
                    feature_widths=context["feature_widths"],
                    args=args,
                    identity_digest=identity_digest,
                    source_parquet_sha256=context["identity"][
                        "input_parquet"
                    ]["sha256"],
                    rank=rank,
                    world_size=world_size,
                )
                path = shard_paths[shard_index]
                _atomic_savez(path, payload)
                _validate_collection_payload(
                    _load_npz(path),
                    kind="document",
                    identity_digest=identity_digest,
                    expected_indices=shard_indices[shard_index],
                    expected_rank=rank,
                    expected_world_size=world_size,
                    allowed_formats=(SHARD_FORMAT,),
                )
                _write_rank_progress(
                    rank_work_dir=rank_work_dir,
                    identity_digest=identity_digest,
                    rank=rank,
                    world_size=world_size,
                    expected_indices=expected_document_indices,
                    shard_paths=shard_paths,
                    complete=False,
                )
                log(
                    f"published document shard {shard_index + 1}/"
                    f"{len(shard_paths)} in "
                    f"{time.perf_counter() - shard_started:.1f}s",
                    rank=rank,
                )

        if not rank_complete:
            rank_stats = _assemble_rank_file(
                shard_paths=shard_paths,
                output_path=rank_path,
                identity_digest=identity_digest,
                expected_indices=expected_document_indices,
                rank=rank,
                world_size=world_size,
            )
            _write_rank_progress(
                rank_work_dir=rank_work_dir,
                identity_digest=identity_digest,
                rank=rank,
                world_size=world_size,
                expected_indices=expected_document_indices,
                shard_paths=shard_paths,
                complete=True,
            )
        else:
            rank_stats = _validate_rank_file(
                rank_path,
                identity_digest=identity_digest,
                expected_indices=expected_document_indices,
                rank=rank,
                world_size=world_size,
            )

        if not seed_complete:
            seed_payload = _extract_seed_rank(
                seeds=seeds,
                owned_seed_indices=owned_seed_indices,
                extractor=extractor,
                sae_set=context["sae_set"],
                feature_widths=context["feature_widths"],
                args=args,
                identity_digest=identity_digest,
                seeds_jsonl_sha256=context["identity"]["seeds_jsonl"][
                    "sha256"
                ],
                rank=rank,
                world_size=world_size,
            )
            _atomic_savez(seed_rank_path, seed_payload)
        seed_stats = _validate_seed_rank_file(
            seed_rank_path,
            identity_digest=identity_digest,
            seeds=seeds,
            owned_seed_indices=owned_seed_indices,
            rank=rank,
            world_size=world_size,
        )
    finally:
        document_texts = None
        _close_extractor(extractor)

    elapsed = time.perf_counter() - started
    log(
        f"rank complete: documents={rank_stats['items']} "
        f"windows={rank_stats['windows']} seeds={seed_stats['items']} "
        f"seed_windows={seed_stats['windows']} elapsed={elapsed:.1f}s",
        rank=rank,
    )
    return {
        "rank": rank,
        "documents": rank_stats["items"],
        "windows": rank_stats["windows"],
        "seeds": seed_stats["items"],
        "seed_windows": seed_stats["windows"],
        "elapsed_seconds": elapsed,
    }


def _publish_final(
    *,
    args: argparse.Namespace,
    context: Mapping[str, Any],
    rank_summaries: Sequence[Mapping[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    features_dir = Path(str(context["features_dir"]))
    work_dir = Path(str(context["work_dir"]))
    windows_path = Path(str(context["windows_path"]))
    identity_digest = str(context["identity_digest"])
    seeds = _load_seeds(Path(str(context["seeds_path"])))

    merged, document_stats = _merge_documents(
        features_dir=features_dir,
        identity_digest=identity_digest,
        document_count=args.expected_documents,
        world_size=world_size,
    )
    merged_path = features_dir / "merged_features.npz"
    _atomic_savez(merged_path, merged)
    _validate_collection_payload(
        _load_npz(merged_path),
        kind="document",
        identity_digest=identity_digest,
        expected_indices=list(range(args.expected_documents)),
        expected_rank=-1,
        expected_world_size=world_size,
        allowed_formats=(MERGED_FORMAT,),
    )

    seed_features, seed_stats = _merge_seeds(
        work_dir=work_dir,
        identity_digest=identity_digest,
        seeds=seeds,
        world_size=world_size,
    )
    seed_path = features_dir / "seed_features.npz"
    _atomic_savez(seed_path, seed_features)
    _validate_collection_payload(
        _load_npz(seed_path),
        kind="seed",
        identity_digest=identity_digest,
        expected_indices=list(range(len(seeds))),
        expected_seed_ids=[row.seed_id for row in seeds],
        expected_query_ids=[row.query_id for row in seeds],
        expected_rank=-1,
        expected_world_size=world_size,
        allowed_formats=(SEED_FORMAT,),
    )

    _atomic_write_jsonl(windows_path, _document_window_rows(merged))
    _validate_document_windows(windows_path, merged)

    files: dict[str, Any] = {
        f"rank{rank:03d}": file_record(
            _rank_file(features_dir, rank), relative_to=features_dir
        )
        for rank in range(world_size)
    }
    files.update(
        {
            "merged_features": file_record(
                merged_path, relative_to=features_dir
            ),
            "seed_features": file_record(
                seed_path, relative_to=features_dir
            ),
            "document_windows": file_record(
                windows_path, relative_to=features_dir
            ),
        }
    )
    feature_widths = {
        str(mode): int(width)
        for mode, width in context["feature_widths"].items()
    }
    manifest = write_artifact_manifest(
        {
            "format": FEATURE_FORMAT,
            "complete": True,
            "identity": context["identity"],
            "identity_digest": identity_digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents": document_stats["items"],
            "document_windows": document_stats["windows"],
            "seeds": seed_stats["items"],
            "seed_windows": seed_stats["windows"],
            "world_size": world_size,
            "rank_summaries": sorted(
                [dict(summary) for summary in rank_summaries],
                key=lambda row: int(row["rank"]),
            ),
            "feature_widths": feature_widths,
            "top_k": TOP_K,
            "representation_protocol": fixed_chunk_protocol_metadata(
                feature_widths=feature_widths
            ),
            "representation_protocol_name": (
                FIXED_CHUNK_REPRESENTATION_PROTOCOL
            ),
            "windowing": context["identity"]["windowing"],
            "page_aggregation": context["identity"]["page_aggregation"],
            "memory_policy": context["identity"]["memory_policy"],
            "storage_schema": {
                "feature_index": "int32",
                "feature_value": "float16",
                "nnz": "int16",
                "item_and_offset": "int64",
                "window_index_and_valid_tokens": "int16",
                "invalid_feature_index": -1,
            },
            "zero_vectors": {
                "documents": _zero_vector_counts(
                    merged, kind="document"
                ),
                "seeds": _zero_vector_counts(seed_features, kind="seed"),
            },
            "files": files,
        },
        Path(str(context["manifest_path"])),
    )
    ensure_reusable_artifact(
        Path(str(context["manifest_path"])),
        expected_format=FEATURE_FORMAT,
        expected_identity=context["identity"],
    )
    _validate_complete_outputs(
        features_dir=features_dir,
        windows_path=windows_path,
        identity_digest=identity_digest,
        document_count=args.expected_documents,
        seeds=seeds,
        world_size=world_size,
    )
    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    return manifest


def main() -> None:
    args = parser().parse_args()
    _validate_args(args)
    configure_cpu_threads(default=4)
    rank, world_size, local_rank = initialize_distributed()
    try:
        if world_size != args.expected_world_size:
            raise ValueError(
                f"Eval 9 expected world_size={args.expected_world_size}, "
                f"found {world_size}. Launch with torchrun --nproc_per_node "
                f"{args.expected_world_size}."
            )
        bind_local_rank_cpu_affinity(
            local_rank=local_rank,
            local_world_size=int(
                os.environ.get("LOCAL_WORLD_SIZE", world_size)
            ),
        )
        if rank == 0:
            context = _prepare_run(args=args, world_size=world_size)
        else:
            context = None
        context = _broadcast(context, rank)
        if bool(context["skip"]):
            log(
                f"reusing verified complete Eval 9 features at "
                f"{context['features_dir']}",
                rank=rank,
                main_only=True,
            )
            _barrier()
            return

        _barrier()
        summary = _run_rank(
            args=args,
            context=context,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
        summaries = all_gather_objects(summary, world_size=world_size)
        _barrier()
        if rank == 0:
            manifest = _publish_final(
                args=args,
                context=context,
                rank_summaries=summaries,
                world_size=world_size,
            )
            log(
                "published merged_features.npz, seed_features.npz, "
                "document_windows.jsonl, and feature_manifest.json "
                f"(artifact_digest={manifest['artifact_digest']})",
                rank=rank,
                main_only=True,
            )
        _barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
