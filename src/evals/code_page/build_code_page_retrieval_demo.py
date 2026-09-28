#!/usr/bin/env python
"""Build the Eval-9 code-page retrieval results and static review demo.

The script deliberately keeps the two retrieval branches independent:

* the keyword baseline applies the locked literal/boolean rule to every
  candidate text; and
* each SAE averages the same locked seed pages in its own feature space and
  performs cosine retrieval over every candidate page.

Inputs are immutable artifacts: a one-column Parquet candidate pool, merged
candidate SAE features, seed SAE features, and ``queries.locked.json``.
Outputs are deterministic JSON/HTML/CSV artifacts with a per-query blind
mapping.  Existing complete outputs are reused only when their full identity
and every recorded checksum match; replacement requires ``--overwrite``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chunk_saes.plot_style import METHOD_COLORS as SHARED_METHOD_COLORS

OUTPUT_FORMAT = "chunk-saes-code-page-retrieval-demo-v1"
RESULT_FORMAT = "chunk-saes-code-page-retrieval-result-v1"
METHOD_MAP_FORMAT = "chunk-saes-code-page-blind-method-map-v1"
MERGED_FEATURE_FORMAT = "chunk-saes-eval9-code-page-merged-v1"
SEED_FEATURE_FORMAT = "chunk-saes-eval9-code-page-seeds-v1"
FEATURE_MANIFEST_FORMAT = "chunk-saes-eval9-code-page-features-v1"
METHODS = ("token", "temporal", "mean", "cross")
METHOD_LABELS = {
    "token": "BatchTopK SAE",
    "temporal": "Temporal SAE",
    "mean": "Mean-Chunk SAE",
    "cross": "Cross-Chunk SAE",
}
METHOD_COLORS = {
    method: SHARED_METHOD_COLORS[method] for method in METHODS
}
QUERY_METHODS = ("keyword", *METHODS)
METHOD_NAMES = {"keyword": "Keyword baseline", **METHOD_LABELS}
METHOD_PALETTE = {"keyword": "#6b7280", **METHOD_COLORS}
SAFE_QUERY_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
WORD_EDGE_LEFT = r"(?<![A-Za-z0-9_])"
WORD_EDGE_RIGHT = r"(?![A-Za-z0-9_])"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build independent keyword and four-SAE retrieval results plus "
            "a deterministic static normal/blind Eval-9 demo."
        )
    )
    p.add_argument(
        "--parquet",
        "--documents",
        dest="parquet",
        required=True,
        help="Canonical one-column text Parquet candidate pool.",
    )
    p.add_argument(
        "--merged-features",
        required=True,
        help="Merged candidate document/window sparse features NPZ.",
    )
    p.add_argument(
        "--seed-features",
        required=True,
        help="Sparse features for the independently locked seed pages.",
    )
    p.add_argument(
        "--queries",
        required=True,
        help="Locked query configuration (normally queries.locked.json).",
    )
    p.add_argument(
        "--seeds-jsonl",
        "--seeds",
        dest="seeds_jsonl",
        help=(
            "Locked seed texts JSONL ({seed_id,query_id,text}). If omitted, "
            "seeds.jsonl beside queries.locked.json is used when present."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--html-initial-results",
        type=int,
        default=10,
        help="Results shown before expanding each HTML method panel.",
    )
    p.add_argument("--snippet-chars", type=int, default=1200)
    p.add_argument("--seed-display-chars", type=int, default=6000)
    p.add_argument("--shared-features", type=int, default=8)
    p.add_argument("--blind-seed", type=int, default=2026)
    p.add_argument("--expected-documents", type=int, default=10_000)
    p.add_argument("--expected-feature-width", type=int, default=65_536)
    p.add_argument("--expected-vector-k", type=int, default=128)
    p.add_argument("--feature-norm-tolerance", type=float, default=0.005)
    p.add_argument(
        "--parquet-sha256",
        help="Optional expected SHA-256 for the candidate Parquet.",
    )
    p.add_argument(
        "--merged-features-sha256",
        help="Optional expected SHA-256 for merged_features.npz.",
    )
    p.add_argument(
        "--seed-features-sha256",
        help="Optional expected SHA-256 for seed_features.npz.",
    )
    p.add_argument(
        "--queries-sha256",
        help="Optional expected SHA-256 for queries.locked.json.",
    )
    p.add_argument(
        "--seeds-jsonl-sha256",
        help="Optional expected SHA-256 for seeds.jsonl.",
    )
    p.add_argument(
        "--feature-manifest",
        help=(
            "Optional feature_manifest.json. If omitted, a sibling manifest "
            "beside merged_features.npz is automatically verified when present."
        ),
    )
    reuse_group = p.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse",
        action="store_true",
        help=(
            "Reuse a complete checksum-verified output with identical identity "
            "(this is also the default behavior)."
        ),
    )
    reuse_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Transactionally replace generated outputs with a rebuilt set.",
    )
    return p


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def input_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def output_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def validate_expected_sha256(
    record: Mapping[str, Any],
    expected: str | None,
    label: str,
) -> None:
    if expected is None:
        return
    normalized = expected.strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label}: expected checksum is not a SHA-256 hex digest")
    if record["sha256"] != normalized:
        raise ValueError(
            f"{label}: SHA-256 mismatch: {record['sha256']} != {normalized}"
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def scalar_from_array(array: np.ndarray, key: str) -> Any:
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError(f"{key}: expected a scalar, found shape {value.shape}")
    return value.reshape(()).item()


def decode_string(value: Any, label: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label}: invalid UTF-8 bytes") from exc
    if isinstance(value, (str, np.str_)):
        return str(value)
    raise ValueError(f"{label}: expected a string, found {type(value).__name__}")


def decode_string_vector(array: np.ndarray, label: str) -> list[str]:
    values = np.asarray(array)
    if values.ndim != 1:
        raise ValueError(f"{label}: expected rank 1, found shape {values.shape}")
    if values.dtype.kind not in {"U", "S"}:
        raise ValueError(
            f"{label}: expected fixed-width UTF-8/Unicode strings, "
            f"found dtype {values.dtype}; object/pickle arrays are forbidden"
        )
    return [decode_string(value, f"{label}[{index}]") for index, value in enumerate(values)]


class NpzBundle:
    """Safe, non-pickle NPZ reader with explicit alias diagnostics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._archive = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"cannot safely load NPZ {path}: {exc}") from exc
        self.keys = tuple(self._archive.files)

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> "NpzBundle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def first(
        self,
        aliases: Sequence[str],
        *,
        required: bool = True,
        label: str,
    ) -> tuple[str, np.ndarray] | None:
        matches = [key for key in aliases if key in self._archive.files]
        if not matches:
            if required:
                raise ValueError(
                    f"{self.path}: missing {label}; accepted keys are "
                    f"{list(aliases)}; present keys are {list(self.keys)}"
                )
            return None
        if len(matches) > 1:
            first = np.asarray(self._archive[matches[0]])
            for key in matches[1:]:
                other = np.asarray(self._archive[key])
                if first.shape != other.shape or not np.array_equal(first, other):
                    raise ValueError(
                        f"{self.path}: conflicting aliases for {label}: {matches}"
                    )
        key = matches[0]
        try:
            return key, np.asarray(self._archive[key])
        except ValueError as exc:
            raise ValueError(
                f"{self.path}:{key}: unsafe object array or invalid payload"
            ) from exc

    def optional_scalar(self, aliases: Sequence[str], label: str) -> Any | None:
        found = self.first(aliases, required=False, label=label)
        if found is None:
            return None
        key, value = found
        return scalar_from_array(value, f"{self.path}:{key}")


def validate_embedded_checksum(
    bundle: NpzBundle,
    aliases: Sequence[str],
    expected: str,
    label: str,
) -> None:
    value = bundle.optional_scalar(aliases, label)
    if value is None:
        return
    actual = decode_string(value, f"{bundle.path}:{label}").lower()
    if SHA256_RE.fullmatch(actual) is None:
        raise ValueError(f"{bundle.path}:{label}: invalid embedded SHA-256")
    if actual != expected:
        raise ValueError(
            f"{bundle.path}:{label}: embedded checksum mismatch: "
            f"{actual} != {expected}"
        )


def validate_declared_integer(
    bundle: NpzBundle,
    aliases: Sequence[str],
    expected: int,
    label: str,
) -> None:
    value = bundle.optional_scalar(aliases, label)
    if value is not None and int(value) != expected:
        raise ValueError(
            f"{bundle.path}:{label}: declared {int(value)}, expected {expected}"
        )


@dataclass(frozen=True)
class SparseRows:
    indices: np.ndarray
    values: np.ndarray
    normalized_values: np.ndarray
    norms: np.ndarray
    width: int
    k: int

    @property
    def rows(self) -> int:
        return int(self.indices.shape[0])

    def dense_mean_query(self, rows: Sequence[int]) -> np.ndarray:
        if not rows:
            raise ValueError("cannot create an SAE query without seed rows")
        query = np.zeros(self.width, dtype=np.float64)
        for row in rows:
            if row < 0 or row >= self.rows:
                raise IndexError(row)
            if self.norms[row] <= 0:
                raise ValueError(f"seed feature row {row} is a zero vector")
            valid = self.indices[row] >= 0
            query[self.indices[row, valid]] += self.normalized_values[row, valid]
        query /= float(len(rows))
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("the averaged SAE seed query is a zero/non-finite vector")
        query /= norm
        return query.astype(np.float32)

    def cosine_scores(self, query: np.ndarray) -> np.ndarray:
        if query.shape != (self.width,):
            raise ValueError(
                f"query shape {query.shape} does not match feature width {self.width}"
            )
        valid = self.indices >= 0
        safe_indices = np.where(valid, self.indices, 0)
        scores = np.sum(
            query[safe_indices] * self.normalized_values * valid,
            axis=1,
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scores)):
            raise ValueError("cosine retrieval produced non-finite scores")
        if np.any(scores < -1e-6) or np.any(scores > 1.0005):
            raise ValueError(
                "cosine scores fall outside the expected non-negative [0, 1] range"
            )
        return scores

    def shared_features(
        self,
        row: int,
        query: np.ndarray,
        limit: int,
    ) -> list[dict[str, Any]]:
        valid = self.indices[row] >= 0
        feature_ids = self.indices[row, valid]
        document_values = self.normalized_values[row, valid]
        contributions = query[feature_ids] * document_values
        order = sorted(
            range(len(feature_ids)),
            key=lambda index: (-float(contributions[index]), int(feature_ids[index])),
        )
        result = []
        for index in order:
            contribution = float(contributions[index])
            if contribution <= 0:
                continue
            feature_id = int(feature_ids[index])
            result.append(
                {
                    "feature_id": feature_id,
                    "query_value": float(query[feature_id]),
                    "document_value": float(document_values[index]),
                    "cosine_contribution": contribution,
                }
            )
            if len(result) >= limit:
                break
        return result


def load_sparse_rows(
    bundle: NpzBundle,
    *,
    indices_aliases: Sequence[str],
    values_aliases: Sequence[str],
    rows: int,
    width: int,
    expected_k: int,
    norm_tolerance: float,
    require_unit_norm: bool,
    nnz_aliases: Sequence[str] = (),
    require_storage_dtypes: bool = True,
    label: str,
) -> SparseRows:
    indices_key, raw_indices = bundle.first(
        indices_aliases, required=True, label=f"{label} feature indices"
    )  # type: ignore[misc]
    values_key, raw_values = bundle.first(
        values_aliases, required=True, label=f"{label} feature values"
    )  # type: ignore[misc]
    if raw_indices.ndim != 2 or raw_values.ndim != 2:
        raise ValueError(
            f"{bundle.path}:{label}: fixed Top-K arrays must be rank 2; "
            f"found {raw_indices.shape} and {raw_values.shape}"
        )
    if raw_indices.shape != raw_values.shape:
        raise ValueError(
            f"{bundle.path}:{label}: index/value shape mismatch: "
            f"{raw_indices.shape} != {raw_values.shape}"
        )
    if raw_indices.shape != (rows, expected_k):
        raise ValueError(
            f"{bundle.path}:{label}: expected shape {(rows, expected_k)}, "
            f"found {raw_indices.shape}"
        )
    if require_storage_dtypes and raw_indices.dtype != np.dtype(np.int32):
        raise ValueError(
            f"{bundle.path}:{indices_key}: indices must be int32, "
            f"found {raw_indices.dtype}"
        )
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"{bundle.path}:{indices_key}: indices must be integers, "
            f"found {raw_indices.dtype}"
        )
    if require_storage_dtypes and raw_values.dtype != np.dtype(np.float16):
        raise ValueError(
            f"{bundle.path}:{values_key}: values must be float16, "
            f"found {raw_values.dtype}"
        )
    if raw_values.dtype.kind not in {"f"}:
        raise ValueError(
            f"{bundle.path}:{values_key}: values must be floating point, "
            f"found {raw_values.dtype}"
        )
    indices = raw_indices.astype(np.int64, copy=False)
    values = raw_values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{bundle.path}:{label}: feature values contain NaN/Inf")
    if np.any(indices < -1):
        raise ValueError(f"{bundle.path}:{label}: feature indices below -1")
    valid = indices >= 0
    if np.any(indices[valid] >= width):
        maximum = int(indices[valid].max())
        raise ValueError(
            f"{bundle.path}:{label}: feature id {maximum} exceeds width {width}"
        )
    if np.any(values[valid] <= 0):
        raise ValueError(
            f"{bundle.path}:{label}: valid feature ids must have positive values"
        )
    if np.any(values[~valid] != 0):
        raise ValueError(
            f"{bundle.path}:{label}: padded feature ids (-1) must have zero values"
        )
    actual_nnz = valid.sum(axis=1)
    if np.any(
        valid
        != (
            np.arange(expected_k, dtype=np.int64)[None, :]
            < actual_nnz[:, None]
        )
    ):
        raise ValueError(
            f"{bundle.path}:{label}: active features must be contiguous before padding"
        )
    if nnz_aliases:
        found_nnz = bundle.first(
            nnz_aliases,
            required=True,
            label=f"{label} nnz",
        )
        assert found_nnz is not None
        nnz_key, raw_nnz = found_nnz
        if raw_nnz.dtype != np.dtype(np.int16) or raw_nnz.shape != (rows,):
            raise ValueError(
                f"{bundle.path}:{nnz_key}: expected int16 shape {(rows,)}, "
                f"found {raw_nnz.dtype} {raw_nnz.shape}"
            )
        if not np.array_equal(raw_nnz.astype(np.int64), actual_nnz):
            raise ValueError(
                f"{bundle.path}:{nnz_key}: nnz does not match sparse contents"
            )
    if expected_k > 1 and np.any(
        values[:, :-1] + 1e-7 < values[:, 1:]
    ):
        raise ValueError(
            f"{bundle.path}:{label}: values are not sorted descending"
        )
    sorted_ids = np.sort(np.where(valid, indices, width), axis=1)
    if expected_k > 1 and np.any(
        (sorted_ids[:, 1:] == sorted_ids[:, :-1])
        & (sorted_ids[:, 1:] < width)
    ):
        raise ValueError(f"{bundle.path}:{label}: duplicate feature ids within a row")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError(f"{bundle.path}:{label}: feature norms are non-finite")
    nonzero = norms > 0
    if require_unit_norm and np.any(
        np.abs(norms[nonzero] - 1.0) > norm_tolerance
    ):
        bad = np.flatnonzero(
            nonzero & (np.abs(norms - 1.0) > norm_tolerance)
        )[:5]
        examples = [(int(index), float(norms[index])) for index in bad]
        raise ValueError(
            f"{bundle.path}:{label}: nonzero rows are not L2-normalized within "
            f"tolerance {norm_tolerance}; examples={examples}"
        )
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[nonzero] = values[nonzero] / norms[nonzero, None]
    return SparseRows(
        indices=indices,
        values=values,
        normalized_values=normalized,
        norms=norms,
        width=width,
        k=expected_k,
    )


def feature_aliases(mode: str, kind: str, role: str) -> tuple[str, ...]:
    if role == "candidate":
        stems = (
            f"{mode}_document",
            f"{mode}_doc",
            f"{mode}_page",
            mode,
        )
    elif role == "seed":
        stems = (
            f"{mode}_seed",
            f"{mode}_document",
            f"{mode}_doc",
            mode,
        )
    elif role == "window":
        stems = (f"{mode}_window",)
    else:
        raise ValueError(role)
    return tuple(f"{stem}_{kind}" for stem in stems)


def nnz_aliases(mode: str, role: str) -> tuple[str, ...]:
    return feature_aliases(mode, "nnz", role)


@dataclass(frozen=True)
class WindowFeatures:
    offsets: np.ndarray
    methods: Mapping[str, SparseRows]
    token_starts: np.ndarray | None
    token_ends: np.ndarray | None
    valid_tokens: np.ndarray | None
    texts: tuple[str, ...] | None

    def best_window(
        self,
        method: str,
        document_index: int,
        query: np.ndarray,
    ) -> dict[str, Any] | None:
        start = int(self.offsets[document_index])
        stop = int(self.offsets[document_index + 1])
        if start == stop:
            return None
        rows = self.methods[method]
        valid = rows.indices[start:stop] >= 0
        safe_indices = np.where(valid, rows.indices[start:stop], 0)
        scores = np.sum(
            query[safe_indices]
            * rows.normalized_values[start:stop]
            * valid,
            axis=1,
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"{method} window cosine produced non-finite scores for "
                f"document {document_index}"
            )
        local_order = np.lexsort(
            (np.arange(stop - start, dtype=np.int64), -scores)
        )
        local = int(local_order[0])
        global_index = start + local
        result: dict[str, Any] = {
            "window_index": local,
            "global_window_index": global_index,
            "similarity": float(scores[local]),
        }
        if self.token_starts is not None:
            result["token_start"] = int(self.token_starts[global_index])
        if self.token_ends is not None:
            result["token_end"] = int(self.token_ends[global_index])
        if self.valid_tokens is not None:
            result["valid_tokens"] = int(self.valid_tokens[global_index])
        if self.texts is not None:
            result["text"] = self.texts[global_index]
        return result


@dataclass(frozen=True)
class CandidateFeatures:
    document_indices: np.ndarray
    methods: Mapping[str, SparseRows]
    windows: WindowFeatures | None
    identity_digest: str
    world_size: int


@dataclass(frozen=True)
class SeedFeatures:
    seed_ids: tuple[str, ...]
    query_ids: tuple[str, ...] | None
    methods: Mapping[str, SparseRows]
    texts: Mapping[str, str]
    candidate_document_indices: Mapping[str, int]
    identity_digest: str
    world_size: int


def required_npz_scalar_string(
    bundle: NpzBundle,
    aliases: Sequence[str],
    label: str,
) -> str:
    found = bundle.first(aliases, required=True, label=label)
    assert found is not None
    key, raw = found
    return decode_string(
        scalar_from_array(raw, f"{bundle.path}:{key}"),
        f"{bundle.path}:{key}",
    )


def required_npz_scalar_int(
    bundle: NpzBundle,
    aliases: Sequence[str],
    label: str,
) -> int:
    found = bundle.first(aliases, required=True, label=label)
    assert found is not None
    key, raw = found
    value = scalar_from_array(raw, f"{bundle.path}:{key}")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{bundle.path}:{key}: expected integer, found bool")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{bundle.path}:{key}: expected integer scalar") from exc
    if isinstance(value, (float, np.floating)) and float(value) != integer:
        raise ValueError(f"{bundle.path}:{key}: non-integral scalar {value}")
    return integer


def validate_feature_artifact_header(
    bundle: NpzBundle,
    *,
    expected_format: str,
    expected_collection: str,
) -> tuple[str, int]:
    artifact_format = required_npz_scalar_string(
        bundle, ("artifact_format",), "feature artifact format"
    )
    if artifact_format != expected_format:
        raise ValueError(
            f"{bundle.path}: expected artifact_format={expected_format!r}, "
            f"found {artifact_format!r}"
        )
    collection = required_npz_scalar_string(
        bundle, ("collection",), "feature collection"
    )
    if collection != expected_collection:
        raise ValueError(
            f"{bundle.path}: expected collection={expected_collection!r}, "
            f"found {collection!r}"
        )
    identity_digest = required_npz_scalar_string(
        bundle, ("identity_digest",), "feature identity digest"
    )
    if SHA256_RE.fullmatch(identity_digest) is None:
        raise ValueError(
            f"{bundle.path}: identity_digest is not a SHA-256 hex digest"
        )
    rank = required_npz_scalar_int(bundle, ("rank",), "feature rank")
    if rank != -1:
        raise ValueError(
            f"{bundle.path}: merged feature artifact must have rank=-1, found {rank}"
        )
    world_size = required_npz_scalar_int(
        bundle, ("world_size",), "feature world size"
    )
    if world_size <= 0:
        raise ValueError(f"{bundle.path}: world_size must be positive")
    shard_index = required_npz_scalar_int(
        bundle, ("shard_index",), "feature shard index"
    )
    if shard_index != -1:
        raise ValueError(
            f"{bundle.path}: merged feature artifact must have shard_index=-1"
        )
    return identity_digest, world_size


def optional_int_vector(
    bundle: NpzBundle,
    aliases: Sequence[str],
    *,
    length: int,
    label: str,
) -> np.ndarray | None:
    found = bundle.first(aliases, required=False, label=label)
    if found is None:
        return None
    key, raw = found
    if raw.shape != (length,) or raw.dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"{bundle.path}:{key}: expected integer shape {(length,)}, "
            f"found {raw.dtype} {raw.shape}"
        )
    return raw.astype(np.int64, copy=False)


def required_int_vector(
    bundle: NpzBundle,
    aliases: Sequence[str],
    *,
    length: int,
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray:
    found = bundle.first(aliases, required=True, label=label)
    assert found is not None
    key, raw = found
    expected_dtype = np.dtype(dtype)
    if raw.shape != (length,) or raw.dtype != expected_dtype:
        raise ValueError(
            f"{bundle.path}:{key}: expected {expected_dtype} shape {(length,)}, "
            f"found {raw.dtype} {raw.shape}"
        )
    return raw


def expected_window_slices(token_count: int) -> list[tuple[int, int]]:
    if token_count <= 0:
        raise ValueError("token count must be positive")
    if token_count <= 4096:
        return [
            (start, min(start + 512, token_count))
            for start in range(0, token_count, 512)
        ]
    return [
        (
            round(index * (token_count - 512) / 7),
            round(index * (token_count - 512) / 7) + 512,
        )
        for index in range(8)
    ]


def validate_window_metadata(
    *,
    path: Path,
    item_label: str,
    item_indices: np.ndarray,
    token_counts: np.ndarray,
    offsets: np.ndarray,
    window_item_indices: np.ndarray,
    window_indices: np.ndarray,
    token_starts: np.ndarray,
    token_ends: np.ndarray,
    valid_tokens: np.ndarray,
) -> None:
    if np.any(token_counts <= 0):
        raise ValueError(f"{path}: qwen_token_count must be positive")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{path}: window_offsets must start at zero and be monotonic")
    counts = np.diff(offsets)
    if np.any(counts < 1) or np.any(counts > 8):
        raise ValueError(f"{path}: each {item_label} must have 1..8 windows")
    for position, item_index in enumerate(item_indices.tolist()):
        start = int(offsets[position])
        stop = int(offsets[position + 1])
        if not np.all(window_item_indices[start:stop] == item_index):
            raise ValueError(
                f"{path}: window {item_label} index disagrees with offsets "
                f"for {item_label}={item_index}"
            )
        if not np.array_equal(
            window_indices[start:stop],
            np.arange(stop - start, dtype=window_indices.dtype),
        ):
            raise ValueError(
                f"{path}: window_index is not contiguous for "
                f"{item_label}={item_index}"
            )
        expected = expected_window_slices(int(token_counts[position]))
        actual = list(
            zip(
                token_starts[start:stop].astype(int).tolist(),
                token_ends[start:stop].astype(int).tolist(),
                strict=True,
            )
        )
        if actual != expected:
            raise ValueError(
                f"{path}: window protocol mismatch for {item_label}={item_index}: "
                f"{actual} != {expected}"
            )
        expected_valid = np.asarray(
            [right - left for left, right in expected],
            dtype=valid_tokens.dtype,
        )
        if not np.array_equal(valid_tokens[start:stop], expected_valid):
            raise ValueError(
                f"{path}: window_valid_tokens mismatch for "
                f"{item_label}={item_index}"
            )


def optional_text_vector(
    bundle: NpzBundle,
    aliases: Sequence[str],
    *,
    length: int,
    label: str,
) -> tuple[str, ...] | None:
    found = bundle.first(aliases, required=False, label=label)
    if found is None:
        return None
    key, raw = found
    values = decode_string_vector(raw, f"{bundle.path}:{key}")
    if len(values) != length:
        raise ValueError(
            f"{bundle.path}:{key}: expected {length} strings, found {len(values)}"
        )
    return tuple(values)


def load_candidate_features(
    path: Path,
    *,
    document_count: int,
    width: int,
    expected_k: int,
    norm_tolerance: float,
    parquet_sha256: str,
) -> CandidateFeatures:
    with NpzBundle(path) as bundle:
        identity_digest, world_size = validate_feature_artifact_header(
            bundle,
            expected_format=MERGED_FEATURE_FORMAT,
            expected_collection="document",
        )
        validate_embedded_checksum(
            bundle,
            (
                "source_parquet_sha256",
                "candidate_parquet_sha256",
                "documents_file_sha256",
            ),
            parquet_sha256,
            "source Parquet SHA-256",
        )
        validate_declared_integer(
            bundle,
            ("document_count", "num_documents", "n_documents"),
            document_count,
            "document count",
        )
        validate_declared_integer(
            bundle,
            ("feature_width", "dictionary_width", "dict_size"),
            width,
            "feature width",
        )
        found = bundle.first(
            ("document_index", "document_indices"),
            required=True,
            label="candidate document indices",
        )
        assert found is not None
        document_key, raw_document_indices = found
        if (
            raw_document_indices.shape != (document_count,)
            or raw_document_indices.dtype.kind not in {"i", "u"}
        ):
            raise ValueError(
                f"{path}:{document_key}: expected integer shape "
                f"{(document_count,)}, found "
                f"{raw_document_indices.dtype} {raw_document_indices.shape}"
            )
        document_indices = raw_document_indices.astype(np.int64, copy=False)
        expected_indices = np.arange(document_count, dtype=np.int64)
        if not np.array_equal(document_indices, expected_indices):
            raise ValueError(
                f"{path}:{document_key}: document indices must be exactly "
                f"0..{document_count - 1} in row order"
            )
        token_counts = required_int_vector(
            bundle,
            ("qwen_token_count",),
            length=document_count,
            dtype=np.dtype(np.int64),
            label="document Qwen token counts",
        )
        methods: dict[str, SparseRows] = {}
        for mode in METHODS:
            validate_declared_integer(
                bundle,
                (
                    f"{mode}_feature_width",
                    f"{mode}_dictionary_width",
                ),
                width,
                f"{mode} feature width",
            )
            methods[mode] = load_sparse_rows(
                bundle,
                indices_aliases=feature_aliases(mode, "indices", "candidate"),
                values_aliases=feature_aliases(mode, "values", "candidate"),
                rows=document_count,
                width=width,
                expected_k=expected_k,
                norm_tolerance=norm_tolerance,
                require_unit_norm=True,
                nnz_aliases=nnz_aliases(mode, "candidate"),
                label=f"{mode} candidate documents",
            )

        offsets_found = bundle.first(
            ("window_offsets", "document_window_offsets"),
            required=False,
            label="document window offsets",
        )
        windows: WindowFeatures | None = None
        any_window_array = any(
            any(alias in bundle.keys for alias in feature_aliases(mode, "indices", "window"))
            or any(alias in bundle.keys for alias in feature_aliases(mode, "values", "window"))
            for mode in METHODS
        )
        if offsets_found is None and any_window_array:
            raise ValueError(f"{path}: window arrays exist without window_offsets")
        if offsets_found is not None:
            offsets_key, raw_offsets = offsets_found
            if (
                raw_offsets.shape != (document_count + 1,)
                or raw_offsets.dtype != np.dtype(np.int64)
            ):
                raise ValueError(
                    f"{path}:{offsets_key}: expected integer shape "
                    f"{(document_count + 1,)}, found "
                    f"{raw_offsets.dtype} {raw_offsets.shape}"
                )
            offsets = raw_offsets.astype(np.int64, copy=False)
            window_count = int(offsets[-1])
            window_methods = {
                mode: load_sparse_rows(
                    bundle,
                    indices_aliases=feature_aliases(mode, "indices", "window"),
                    values_aliases=feature_aliases(mode, "values", "window"),
                    rows=window_count,
                    width=width,
                    expected_k=expected_k,
                    norm_tolerance=norm_tolerance,
                    require_unit_norm=False,
                    nnz_aliases=nnz_aliases(mode, "window"),
                    label=f"{mode} windows",
                )
                for mode in METHODS
            }
            window_document_indices = required_int_vector(
                bundle,
                ("window_document_index", "window_document_indices"),
                length=window_count,
                dtype=np.dtype(np.int64),
                label="window document indices",
            )
            window_indices = required_int_vector(
                bundle,
                ("window_index", "window_indices"),
                length=window_count,
                dtype=np.dtype(np.int16),
                label="within-document window indices",
            )
            token_starts = required_int_vector(
                bundle,
                ("window_token_start", "window_token_starts", "token_starts"),
                length=window_count,
                dtype=np.dtype(np.int64),
                label="window token starts",
            )
            token_ends = required_int_vector(
                bundle,
                ("window_token_end", "window_token_ends", "token_ends"),
                length=window_count,
                dtype=np.dtype(np.int64),
                label="window token ends",
            )
            valid_tokens = required_int_vector(
                bundle,
                ("window_valid_tokens", "valid_tokens"),
                length=window_count,
                dtype=np.dtype(np.int16),
                label="window valid token counts",
            )
            window_texts = optional_text_vector(
                bundle,
                ("window_texts", "window_text"),
                length=window_count,
                label="window texts",
            )
            validate_window_metadata(
                path=path,
                item_label="document",
                item_indices=document_indices,
                token_counts=token_counts,
                offsets=offsets,
                window_item_indices=window_document_indices,
                window_indices=window_indices,
                token_starts=token_starts,
                token_ends=token_ends,
                valid_tokens=valid_tokens,
            )
            windows = WindowFeatures(
                offsets=offsets,
                methods=window_methods,
                token_starts=token_starts,
                token_ends=token_ends,
                valid_tokens=valid_tokens,
                texts=window_texts,
            )
        return CandidateFeatures(
            document_indices=document_indices,
            methods=methods,
            windows=windows,
            identity_digest=identity_digest,
            world_size=world_size,
        )


def load_seed_features(
    path: Path,
    *,
    width: int,
    expected_k: int,
    norm_tolerance: float,
    queries_sha256: str,
    seeds_jsonl_sha256: str | None,
) -> SeedFeatures:
    with NpzBundle(path) as bundle:
        identity_digest, world_size = validate_feature_artifact_header(
            bundle,
            expected_format=SEED_FEATURE_FORMAT,
            expected_collection="seed",
        )
        validate_embedded_checksum(
            bundle,
            ("queries_file_sha256", "locked_queries_sha256"),
            queries_sha256,
            "locked queries SHA-256",
        )
        if seeds_jsonl_sha256 is not None:
            validate_embedded_checksum(
                bundle,
                ("seeds_jsonl_sha256", "seed_texts_sha256"),
                seeds_jsonl_sha256,
                "locked seed JSONL SHA-256",
            )
        validate_declared_integer(
            bundle,
            ("feature_width", "dictionary_width", "dict_size"),
            width,
            "feature width",
        )
        found = bundle.first(
            ("seed_ids", "seed_id"),
            required=True,
            label="seed ids",
        )
        assert found is not None
        seed_key, raw_seed_ids = found
        seed_ids = tuple(decode_string_vector(raw_seed_ids, f"{path}:{seed_key}"))
        if not seed_ids:
            raise ValueError(f"{path}:{seed_key}: no seed rows")
        if any(not seed_id.strip() for seed_id in seed_ids):
            raise ValueError(f"{path}:{seed_key}: empty seed id")
        if len(set(seed_ids)) != len(seed_ids):
            raise ValueError(f"{path}:{seed_key}: duplicate seed ids")
        seed_indices = required_int_vector(
            bundle,
            ("seed_index", "seed_indices"),
            length=len(seed_ids),
            dtype=np.dtype(np.int64),
            label="seed indices",
        )
        if not np.array_equal(
            seed_indices, np.arange(len(seed_ids), dtype=np.int64)
        ):
            raise ValueError(
                f"{path}: seed_index must be exactly 0..{len(seed_ids) - 1}"
            )
        token_counts = required_int_vector(
            bundle,
            ("qwen_token_count",),
            length=len(seed_ids),
            dtype=np.dtype(np.int64),
            label="seed Qwen token counts",
        )
        query_ids_found = bundle.first(
            ("query_ids", "query_id"),
            required=False,
            label="seed query ids",
        )
        query_ids: tuple[str, ...] | None = None
        if query_ids_found is not None:
            query_key, raw_query_ids = query_ids_found
            query_ids = tuple(
                decode_string_vector(
                    raw_query_ids, f"{path}:{query_key}"
                )
            )
            if len(query_ids) != len(seed_ids):
                raise ValueError(
                    f"{path}:{query_key}: expected {len(seed_ids)} query ids, "
                    f"found {len(query_ids)}"
                )
            if any(not query_id.strip() for query_id in query_ids):
                raise ValueError(f"{path}:{query_key}: empty query id")
        methods: dict[str, SparseRows] = {}
        for mode in METHODS:
            validate_declared_integer(
                bundle,
                (
                    f"{mode}_feature_width",
                    f"{mode}_dictionary_width",
                ),
                width,
                f"{mode} feature width",
            )
            methods[mode] = load_sparse_rows(
                bundle,
                indices_aliases=feature_aliases(mode, "indices", "seed"),
                values_aliases=feature_aliases(mode, "values", "seed"),
                rows=len(seed_ids),
                width=width,
                expected_k=expected_k,
                norm_tolerance=norm_tolerance,
                require_unit_norm=True,
                nnz_aliases=nnz_aliases(mode, "seed"),
                label=f"{mode} seeds",
            )
        offsets_found = bundle.first(
            ("window_offsets",),
            required=True,
            label="seed window offsets",
        )
        assert offsets_found is not None
        offsets_key, raw_offsets = offsets_found
        if (
            raw_offsets.dtype != np.dtype(np.int64)
            or raw_offsets.shape != (len(seed_ids) + 1,)
        ):
            raise ValueError(
                f"{path}:{offsets_key}: expected int64 shape "
                f"{(len(seed_ids) + 1,)}, found "
                f"{raw_offsets.dtype} {raw_offsets.shape}"
            )
        offsets = raw_offsets.astype(np.int64, copy=False)
        window_count = int(offsets[-1])
        window_seed_indices = required_int_vector(
            bundle,
            ("window_seed_index", "window_seed_indices"),
            length=window_count,
            dtype=np.dtype(np.int64),
            label="window seed indices",
        )
        window_indices = required_int_vector(
            bundle,
            ("window_index", "window_indices"),
            length=window_count,
            dtype=np.dtype(np.int16),
            label="within-seed window indices",
        )
        token_starts = required_int_vector(
            bundle,
            ("window_token_start",),
            length=window_count,
            dtype=np.dtype(np.int64),
            label="seed window token starts",
        )
        token_ends = required_int_vector(
            bundle,
            ("window_token_end",),
            length=window_count,
            dtype=np.dtype(np.int64),
            label="seed window token ends",
        )
        valid_tokens = required_int_vector(
            bundle,
            ("window_valid_tokens",),
            length=window_count,
            dtype=np.dtype(np.int16),
            label="seed window valid token counts",
        )
        validate_window_metadata(
            path=path,
            item_label="seed",
            item_indices=seed_indices,
            token_counts=token_counts,
            offsets=offsets,
            window_item_indices=window_seed_indices,
            window_indices=window_indices,
            token_starts=token_starts,
            token_ends=token_ends,
            valid_tokens=valid_tokens,
        )
        for mode in METHODS:
            load_sparse_rows(
                bundle,
                indices_aliases=feature_aliases(mode, "indices", "window"),
                values_aliases=feature_aliases(mode, "values", "window"),
                rows=window_count,
                width=width,
                expected_k=expected_k,
                norm_tolerance=norm_tolerance,
                require_unit_norm=False,
                nnz_aliases=nnz_aliases(mode, "window"),
                label=f"{mode} seed windows",
            )
        texts_vector = optional_text_vector(
            bundle,
            ("seed_texts", "seed_text"),
            length=len(seed_ids),
            label="seed texts",
        )
        candidate_indices = optional_int_vector(
            bundle,
            (
                "candidate_document_indices",
                "seed_document_indices",
                "candidate_document_index",
            ),
            length=len(seed_ids),
            label="seed candidate document indices",
        )
        if candidate_indices is not None and np.any(candidate_indices < -1):
            raise ValueError(f"{path}: candidate seed document indices must be >= -1")
        return SeedFeatures(
            seed_ids=seed_ids,
            query_ids=query_ids,
            methods=methods,
            texts=(
                dict(zip(seed_ids, texts_vector, strict=True))
                if texts_vector is not None
                else {}
            ),
            candidate_document_indices=(
                {
                    seed_id: int(index)
                    for seed_id, index in zip(
                        seed_ids, candidate_indices, strict=True
                    )
                    if int(index) >= 0
                }
                if candidate_indices is not None
                else {}
            ),
            identity_digest=identity_digest,
            world_size=world_size,
        )


def load_documents(path: Path, expected_documents: int) -> list[str]:
    schema = pq.read_schema(path)
    if schema.names != ["text"] or len(schema) != 1:
        raise ValueError(
            f"{path}: expected exactly one Parquet column named 'text', "
            f"found schema {schema}"
        )
    field = schema.field("text")
    if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
        raise ValueError(f"{path}: text must be string/large_string, found {field.type}")
    table = pq.read_table(path, columns=["text"])
    if table.num_rows != expected_documents:
        raise ValueError(
            f"{path}: expected {expected_documents} rows, found {table.num_rows}"
        )
    texts = table.column("text").to_pylist()
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"{path}: text[{index}] is null or non-string")
        if not text.strip():
            raise ValueError(f"{path}: text[{index}] is empty after stripping")
    return texts


@dataclass(frozen=True)
class Atom:
    mode: str
    value: str
    case_sensitive: bool
    label: str
    regex: re.Pattern[str]

    def evaluate(self, text: str) -> tuple[bool, int, list[tuple[int, int]], list[str]]:
        count = 0
        spans: list[tuple[int, int]] = []
        for match in self.regex.finditer(text):
            count += 1
            if len(spans) < 32:
                spans.append((match.start(), match.end()))
        return count > 0, count, spans, ([self.label] if count else [])

    def normalized(self) -> dict[str, Any]:
        return {
            "type": self.mode,
            "value": self.value,
            "case_sensitive": self.case_sensitive,
            "label": self.label,
        }


@dataclass(frozen=True)
class RuleNode:
    kind: str
    atom: Atom | None = None
    children: tuple["RuleNode", ...] = ()

    def evaluate(self, text: str) -> tuple[bool, int, list[tuple[int, int]], list[str]]:
        if self.kind == "atom":
            assert self.atom is not None
            return self.atom.evaluate(text)
        evaluations = [child.evaluate(text) for child in self.children]
        if self.kind == "all":
            matched = all(item[0] for item in evaluations)
        elif self.kind == "any":
            matched = any(item[0] for item in evaluations)
        else:
            raise AssertionError(self.kind)
        if not matched:
            return False, 0, [], []
        count = sum(item[1] for item in evaluations if item[0])
        spans = sorted(
            (span for item in evaluations if item[0] for span in item[2]),
            key=lambda span: (span[0], span[1]),
        )[:32]
        labels = sorted(
            {label for item in evaluations if item[0] for label in item[3]}
        )
        return True, count, spans, labels

    def normalized(self) -> dict[str, Any]:
        if self.kind == "atom":
            assert self.atom is not None
            return self.atom.normalized()
        return {
            "op": self.kind,
            "children": [child.normalized() for child in self.children],
        }


@dataclass(frozen=True)
class KeywordClause:
    clause_id: str
    node: RuleNode


@dataclass(frozen=True)
class KeywordEvaluation:
    matched_clause_count: int
    occurrence_count: int
    matched_clause_ids: tuple[str, ...]
    matched_atoms: tuple[str, ...]
    spans: tuple[tuple[int, int], ...]

    @property
    def matched(self) -> bool:
        return self.matched_clause_count > 0


@dataclass(frozen=True)
class KeywordRule:
    clauses: tuple[KeywordClause, ...]
    matched_clause_mode: str = "top_level_clauses"

    def evaluate(self, text: str) -> KeywordEvaluation:
        clause_ids: list[str] = []
        occurrence_count = 0
        spans: list[tuple[int, int]] = []
        atoms: set[str] = set()
        for clause in self.clauses:
            matched, count, clause_spans, labels = clause.node.evaluate(text)
            if not matched:
                continue
            clause_ids.append(clause.clause_id)
            occurrence_count += count
            spans.extend(clause_spans)
            atoms.update(labels)
        matched_clause_count = (
            len(atoms)
            if clause_ids and self.matched_clause_mode == "matched_atoms"
            else len(clause_ids)
        )
        return KeywordEvaluation(
            matched_clause_count=matched_clause_count,
            occurrence_count=occurrence_count,
            matched_clause_ids=tuple(clause_ids),
            matched_atoms=tuple(sorted(atoms)),
            spans=tuple(sorted(spans, key=lambda span: (span[0], span[1]))[:32]),
        )

    def normalized(self) -> dict[str, Any]:
        return {
            "matched_clause_mode": self.matched_clause_mode,
            "clauses": [
                {
                    "clause_id": clause.clause_id,
                    "rule": clause.node.normalized(),
                }
                for clause in self.clauses
            ]
        }


def compile_atom(raw: Any, label_hint: str) -> RuleNode:
    if isinstance(raw, str):
        mode = "phrase"
        value = raw
        case_sensitive = False
        label = raw
    elif isinstance(raw, Mapping):
        atom_keys = [key for key in ("phrase", "literal", "word", "regex") if key in raw]
        if len(atom_keys) != 1:
            raise ValueError(
                f"{label_hint}: atom requires exactly one of phrase/literal/word/regex"
            )
        key = atom_keys[0]
        mode = "phrase" if key == "literal" else key
        value = raw[key]
        if not isinstance(value, str):
            raise ValueError(f"{label_hint}.{key}: expected string")
        case_sensitive = bool(raw.get("case_sensitive", False))
        label_value = raw.get("label", raw.get("id", value))
        if not isinstance(label_value, str) or not label_value.strip():
            raise ValueError(f"{label_hint}: invalid atom label")
        label = label_value
    else:
        raise ValueError(f"{label_hint}: expected string or rule object")
    if not value:
        raise ValueError(f"{label_hint}: empty keyword value")
    if len(value) > 512:
        raise ValueError(f"{label_hint}: keyword/regex exceeds 512 characters")
    flags = 0 if case_sensitive else re.IGNORECASE
    if mode == "regex":
        pattern = value
    elif mode == "word":
        pattern = WORD_EDGE_LEFT + re.escape(value) + WORD_EDGE_RIGHT
    else:
        pattern = re.escape(value)
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"{label_hint}: invalid regex: {exc}") from exc
    if regex.search("") is not None:
        raise ValueError(f"{label_hint}: keyword regex may not match an empty string")
    return RuleNode(
        kind="atom",
        atom=Atom(
            mode=mode,
            value=value,
            case_sensitive=case_sensitive,
            label=label,
            regex=regex,
        ),
    )


def as_rule_items(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, Mapping)):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
        if not items:
            raise ValueError(f"{label}: empty rule list")
        return items
    raise ValueError(f"{label}: expected a rule or non-empty rule list")


def parse_rule_node(raw: Any, label: str) -> RuleNode:
    if isinstance(raw, str):
        return compile_atom(raw, label)
    if isinstance(raw, Sequence) and not isinstance(
        raw, (str, bytes, bytearray, Mapping)
    ):
        children = tuple(
            parse_rule_node(item, f"{label}[{index}]")
            for index, item in enumerate(raw)
        )
        if not children:
            raise ValueError(f"{label}: empty any-rule")
        return RuleNode(kind="any", children=children)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}: invalid rule node {raw!r}")
    if any(key in raw for key in ("phrase", "literal", "word", "regex")):
        return compile_atom(raw, label)
    op_value = raw.get("op", raw.get("operator", raw.get("type")))
    if isinstance(op_value, str) and op_value.lower() in {"or", "any", "and", "all"}:
        op = "any" if op_value.lower() in {"or", "any"} else "all"
        children_raw = None
        for key in ("children", "clauses", "terms", "rules"):
            if key in raw:
                children_raw = raw[key]
                break
        if children_raw is None:
            raise ValueError(f"{label}: {op_value} rule lacks children")
        children = tuple(
            parse_rule_node(item, f"{label}.{op}[{index}]")
            for index, item in enumerate(as_rule_items(children_raw, label))
        )
        return RuleNode(kind=op, children=children)

    all_value = None
    any_value = None
    for key in ("all_of", "all", "required", "must_all", "must"):
        if key in raw:
            all_value = raw[key]
            break
    for key in ("any_of", "any", "one_of", "must_any"):
        if key in raw:
            any_value = raw[key]
            break
    if all_value is None and any_value is None:
        raise ValueError(f"{label}: unrecognized rule object keys {sorted(raw)}")
    children: list[RuleNode] = []
    if all_value is not None:
        children.extend(
            parse_rule_node(item, f"{label}.all[{index}]")
            for index, item in enumerate(as_rule_items(all_value, f"{label}.all"))
        )
    if any_value is not None:
        any_children = tuple(
            parse_rule_node(item, f"{label}.any[{index}]")
            for index, item in enumerate(as_rule_items(any_value, f"{label}.any"))
        )
        children.append(RuleNode(kind="any", children=any_children))
    return RuleNode(kind="all", children=tuple(children))


def parse_keyword_rule(raw: Any, query_id: str) -> KeywordRule:
    label = f"query {query_id}.keyword_rule"
    # Preferred Eval-9 schema:
    #
    #   {"any": ["literal A", ...], "all": ["literal B", ...]}
    #
    # Every ``all`` literal is required.  ``any`` contributes no condition
    # when empty; otherwise at least one of its literals is required.  Plain
    # strings compile to case-insensitive literal substring searches.
    if isinstance(raw, Mapping) and set(raw).issubset({"any", "all"}) and raw:
        preferred_children: list[RuleNode] = []
        all_items = raw.get("all", [])
        any_items = raw.get("any", [])
        if not isinstance(all_items, list) or not isinstance(any_items, list):
            raise ValueError(
                f"{label}: preferred schema requires 'all' and 'any' arrays"
            )
        if any(not isinstance(item, str) for item in (*all_items, *any_items)):
            raise ValueError(
                f"{label}: preferred schema accepts only literal string phrases"
            )
        preferred_children.extend(
            compile_atom(item, f"{label}.all[{index}]")
            for index, item in enumerate(all_items)
        )
        if any_items:
            preferred_children.append(
                RuleNode(
                    kind="any",
                    children=tuple(
                        compile_atom(item, f"{label}.any[{index}]")
                        for index, item in enumerate(any_items)
                    ),
                )
            )
        if not preferred_children:
            raise ValueError(
                f"{label}: 'all' and 'any' may not both be empty"
            )
        return KeywordRule(
            clauses=(
                KeywordClause(
                    clause_id="rule",
                    node=RuleNode(kind="all", children=tuple(preferred_children)),
                ),
            ),
            matched_clause_mode="matched_atoms",
        )
    clause_items: list[Any]
    if isinstance(raw, Mapping) and "clauses" in raw and not (
        isinstance(raw.get("op"), str)
        and str(raw.get("op")).lower() in {"and", "all"}
    ):
        clause_items = as_rule_items(raw["clauses"], f"{label}.clauses")
    elif isinstance(raw, Mapping):
        op = raw.get("op", raw.get("operator"))
        if isinstance(op, str) and op.lower() in {"or", "any"}:
            children = raw.get("children", raw.get("terms", raw.get("rules")))
            clause_items = as_rule_items(children, f"{label}.children")
        else:
            only_any = None
            if not any(
                key in raw
                for key in (
                    "all_of",
                    "all",
                    "required",
                    "must_all",
                    "must",
                    "phrase",
                    "literal",
                    "word",
                    "regex",
                )
            ):
                for key in ("any_of", "any", "one_of"):
                    if key in raw:
                        only_any = raw[key]
                        break
            clause_items = (
                as_rule_items(only_any, f"{label}.any")
                if only_any is not None
                else [raw]
            )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        clause_items = as_rule_items(raw, label)
    else:
        clause_items = [raw]
    clauses: list[KeywordClause] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(clause_items):
        explicit_id: Any = None
        node_input = item
        if isinstance(item, Mapping):
            explicit_id = item.get("clause_id")
            if explicit_id is not None and "rule" in item:
                node_input = item["rule"]
        clause_id = (
            str(explicit_id)
            if explicit_id is not None
            else f"clause_{index + 1:02d}"
        )
        if not clause_id.strip() or clause_id in seen_ids:
            raise ValueError(f"{label}: invalid/duplicate clause id {clause_id!r}")
        seen_ids.add(clause_id)
        clauses.append(
            KeywordClause(
                clause_id=clause_id,
                node=parse_rule_node(node_input, f"{label}.{clause_id}"),
            )
        )
    if not clauses:
        raise ValueError(f"{label}: no keyword clauses")
    return KeywordRule(clauses=tuple(clauses))


@dataclass(frozen=True)
class SeedSpec:
    seed_id: str
    text: str
    source: str
    candidate_document_index: int | None


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    name: str
    status: str
    keyword_rule: KeywordRule
    seeds: tuple[SeedSpec, ...]
    sae_available: bool
    unavailable_reason: str | None
    excluded_document_indices: tuple[int, ...]


def validate_query_artifact(payload: Any, path: Path) -> None:
    if not isinstance(payload, Mapping):
        return
    if "format" in payload and (
        not isinstance(payload["format"], str) or not payload["format"].strip()
    ):
        raise ValueError(f"{path}: format must be a non-empty string")
    expected_digest = payload.get("artifact_digest")
    if expected_digest is not None:
        if not isinstance(expected_digest, str) or SHA256_RE.fullmatch(expected_digest) is None:
            raise ValueError(f"{path}: invalid artifact_digest")
        without_digest = dict(payload)
        without_digest.pop("artifact_digest", None)
        actual = json_digest(without_digest)
        if actual != expected_digest:
            raise ValueError(
                f"{path}: artifact_digest mismatch: {actual} != {expected_digest}"
            )
    if "locked" in payload and payload["locked"] is not True:
        raise ValueError(f"{path}: locked must be the JSON boolean true")


@dataclass(frozen=True)
class LockedSeedText:
    seed_id: str
    query_id: str
    text: str
    line_number: int


def load_locked_seed_texts(path: Path) -> dict[str, LockedSeedText]:
    records: dict[str, LockedSeedText] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            seed_id = raw.get("seed_id")
            query_id = raw.get("query_id")
            text = raw.get("text")
            if not isinstance(seed_id, str) or not seed_id.strip():
                raise ValueError(f"{path}:{line_number}: invalid seed_id")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"{path}:{line_number}: invalid query_id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: invalid/empty text")
            if seed_id in records:
                previous = records[seed_id]
                raise ValueError(
                    f"{path}:{line_number}: duplicate seed_id {seed_id!r}; "
                    f"first seen on line {previous.line_number}"
                )
            records[seed_id] = LockedSeedText(
                seed_id=seed_id,
                query_id=query_id,
                text=text,
                line_number=line_number,
            )
    if not records:
        raise ValueError(f"{path}: no seed records")
    return records


def query_seed_entries(raw_query: Mapping[str, Any], query_id: str) -> list[dict[str, Any]]:
    raw: Any = None
    for key in ("sae_seeds", "seeds", "sae_seed_ids", "seed_ids"):
        if key in raw_query:
            raw = raw_query[key]
            break
    if raw is None:
        return []
    if isinstance(raw, (str, Mapping)):
        raw_items = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        raw_items = list(raw)
    else:
        raise ValueError(f"query {query_id}: invalid SAE seed list")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if isinstance(item, str):
            entries.append({"seed_id": item})
        elif isinstance(item, Mapping):
            seed_id = item.get("seed_id", item.get("id"))
            if not isinstance(seed_id, str):
                raise ValueError(f"query {query_id}: seed {index} lacks seed_id")
            entry = dict(item)
            entry["seed_id"] = seed_id
            entries.append(entry)
        else:
            raise ValueError(f"query {query_id}: invalid seed entry {index}")

    supplemental = raw_query.get(
        "sae_seed_texts", raw_query.get("seed_texts")
    )
    if supplemental is not None:
        if isinstance(supplemental, Mapping):
            for entry in entries:
                if entry["seed_id"] in supplemental:
                    entry.setdefault("text", supplemental[entry["seed_id"]])
        elif isinstance(supplemental, Sequence) and not isinstance(
            supplemental, (str, bytes, bytearray)
        ):
            texts = list(supplemental)
            if len(texts) != len(entries):
                raise ValueError(
                    f"query {query_id}: seed text count {len(texts)} does not "
                    f"match seed id count {len(entries)}"
                )
            for entry, text in zip(entries, texts, strict=True):
                entry.setdefault("text", text)
        else:
            raise ValueError(f"query {query_id}: invalid sae_seed_texts")
    return entries


def load_queries(
    path: Path,
    *,
    seed_features: SeedFeatures,
    documents: Sequence[str],
    candidate_parquet_path: Path,
    locked_seed_texts: Mapping[str, LockedSeedText] | None,
    locked_seed_path: Path | None,
) -> tuple[list[QuerySpec], Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_query_artifact(payload, path)
    if isinstance(payload, Mapping) and payload.get("candidate_pool") is not None:
        declared_pool = payload["candidate_pool"]
        if not isinstance(declared_pool, str) or not declared_pool.strip():
            raise ValueError(f"{path}: candidate_pool must be a non-empty path")
        declared_path = Path(declared_pool).expanduser()
        if not declared_path.is_absolute():
            declared_path = path.parent / declared_path
        if declared_path.resolve() != candidate_parquet_path.resolve():
            raise ValueError(
                f"{path}: candidate_pool resolves to {declared_path.resolve()}, "
                f"not requested {candidate_parquet_path.resolve()}"
            )
    if ".locked." not in path.name and not (
        isinstance(payload, Mapping) and payload.get("locked") is True
    ):
        raise ValueError(
            f"{path}: locked query input must use a .locked. filename or locked=true"
        )
    if isinstance(payload, list):
        raw_queries = payload
    elif isinstance(payload, Mapping):
        raw_queries = payload.get("queries", payload.get("topics"))
        if raw_queries is None:
            raise ValueError(f"{path}: expected a 'queries' array")
    else:
        raise ValueError(f"{path}: expected a JSON object or array")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError(f"{path}: queries must be a non-empty array")
    seed_rows = {seed_id: index for index, seed_id in enumerate(seed_features.seed_ids)}
    del seed_rows  # existence is checked below; row mapping is rebuilt during retrieval.
    queries: list[QuerySpec] = []
    seen_query_ids: set[str] = set()
    referenced_seed_ids: list[str] = []
    for query_number, raw_query in enumerate(raw_queries):
        if not isinstance(raw_query, Mapping):
            raise ValueError(f"{path}: queries[{query_number}] is not an object")
        query_id = raw_query.get("query_id", raw_query.get("id"))
        if not isinstance(query_id, str) or SAFE_QUERY_ID.fullmatch(query_id) is None:
            raise ValueError(
                f"{path}: query id must match {SAFE_QUERY_ID.pattern!r}; "
                f"found {query_id!r}"
            )
        if query_id in seen_query_ids:
            raise ValueError(f"{path}: duplicate query_id {query_id}")
        seen_query_ids.add(query_id)
        name_value = raw_query.get(
            "name",
            raw_query.get("title", raw_query.get("topic", query_id)),
        )
        if not isinstance(name_value, str) or not name_value.strip():
            raise ValueError(f"query {query_id}: invalid display name")
        keyword_raw = raw_query.get(
            "keyword_rule", raw_query.get("keywords")
        )
        if keyword_raw is None:
            raise ValueError(f"query {query_id}: missing keyword_rule")
        keyword_rule = parse_keyword_rule(keyword_raw, query_id)
        status_value = raw_query.get("status", "ready")
        status = str(status_value)
        explicit_unavailable = (
            status.lower() in {"insufficient_seed", "disabled", "unavailable"}
            or raw_query.get("enabled") is False
        )
        raw_seed_entries = query_seed_entries(raw_query, query_id)
        if not explicit_unavailable and not 1 <= len(raw_seed_entries) <= 3:
            raise ValueError(
                f"query {query_id}: active SAE query requires 1..3 locked seeds, "
                f"found {len(raw_seed_entries)}"
            )
        if len(raw_seed_entries) > 3:
            raise ValueError(f"query {query_id}: more than three SAE seeds")
        query_texts: dict[str, str] = {}
        query_sources: dict[str, str] = {}
        query_candidate_indices: dict[str, int] = {}
        seed_ids: list[str] = []
        for entry in raw_seed_entries:
            seed_id = str(entry["seed_id"])
            if not seed_id.strip() or seed_id in seed_ids:
                raise ValueError(
                    f"query {query_id}: empty or duplicate seed id {seed_id!r}"
                )
            seed_ids.append(seed_id)
            if seed_id in referenced_seed_ids:
                raise ValueError(
                    f"query {query_id}: seed {seed_id!r} is already referenced "
                    "by another query"
                )
            referenced_seed_ids.append(seed_id)
            if seed_id not in seed_features.seed_ids:
                raise ValueError(
                    f"query {query_id}: seed {seed_id!r} is absent from "
                    f"{len(seed_features.seed_ids)} seed feature rows"
                )
            if seed_features.query_ids is not None:
                feature_row = seed_features.seed_ids.index(seed_id)
                feature_query_id = seed_features.query_ids[feature_row]
                if feature_query_id != query_id:
                    raise ValueError(
                        f"query {query_id}: seed_features.npz assigns "
                        f"{seed_id!r} to query {feature_query_id!r}"
                    )
            if "text" in entry:
                if not isinstance(entry["text"], str) or not entry["text"].strip():
                    raise ValueError(f"query {query_id}: invalid text for seed {seed_id}")
                query_texts[seed_id] = entry["text"]
            source_value = entry.get(
                "source", raw_query.get("seed_source", "independent external seed")
            )
            query_sources[seed_id] = str(source_value)
            document_index = entry.get(
                "document_index", entry.get("candidate_document_index")
            )
            if document_index is not None:
                query_candidate_indices[seed_id] = int(document_index)

        seeds: list[SeedSpec] = []
        for seed_id in seed_ids:
            jsonl_seed = (
                locked_seed_texts.get(seed_id)
                if locked_seed_texts is not None
                else None
            )
            if locked_seed_texts is not None and jsonl_seed is None:
                raise ValueError(
                    f"query {query_id}: seed {seed_id!r} is absent from "
                    f"{locked_seed_path}"
                )
            if jsonl_seed is not None and jsonl_seed.query_id != query_id:
                raise ValueError(
                    f"query {query_id}: seed {seed_id!r} belongs to query "
                    f"{jsonl_seed.query_id!r} in {locked_seed_path}:"
                    f"{jsonl_seed.line_number}"
                )
            npz_text = seed_features.texts.get(seed_id)
            query_text = query_texts.get(seed_id)
            jsonl_text = jsonl_seed.text if jsonl_seed is not None else None
            text_sources = [
                (source, value)
                for source, value in (
                    ("seeds.jsonl", jsonl_text),
                    ("queries.locked.json", query_text),
                    ("seed_features.npz", npz_text),
                )
                if value is not None
            ]
            if text_sources:
                authoritative_source, text = text_sources[0]
                for other_source, other_text in text_sources[1:]:
                    if other_text != text:
                        raise ValueError(
                            f"query {query_id}: seed text mismatch for {seed_id} "
                            f"between {authoritative_source} and {other_source}"
                        )
            else:
                text = None
            if text is None or not text.strip():
                raise ValueError(
                    f"query {query_id}: seed {seed_id} has no displayable locked text "
                    "in seeds.jsonl, queries.locked.json, or seed_features.npz"
                )
            npz_document_index = seed_features.candidate_document_indices.get(seed_id)
            query_document_index = query_candidate_indices.get(seed_id)
            if (
                npz_document_index is not None
                and query_document_index is not None
                and npz_document_index != query_document_index
            ):
                raise ValueError(
                    f"query {query_id}: candidate document index mismatch for "
                    f"seed {seed_id}"
                )
            candidate_document_index = (
                query_document_index
                if query_document_index is not None
                else npz_document_index
            )
            source = (
                f"{locked_seed_path.name}:line {jsonl_seed.line_number}"
                if jsonl_seed is not None and locked_seed_path is not None
                else query_sources[seed_id]
            )
            source_lower = source.lower()
            if (
                ("candidate" in source_lower or "pool" in source_lower)
                and candidate_document_index is None
            ):
                raise ValueError(
                    f"query {query_id}: seed {seed_id} claims candidate-pool "
                    "provenance but lacks document_index"
                )
            if candidate_document_index is not None:
                if not 0 <= candidate_document_index < len(documents):
                    raise ValueError(
                        f"query {query_id}: seed {seed_id} candidate index "
                        f"{candidate_document_index} is out of range"
                    )
                if documents[candidate_document_index] != text:
                    raise ValueError(
                        f"query {query_id}: in-pool seed {seed_id} text does not "
                        f"equal candidate document {candidate_document_index}"
                    )
            seeds.append(
                SeedSpec(
                    seed_id=seed_id,
                    text=text,
                    source=source,
                    candidate_document_index=candidate_document_index,
                )
            )
        extra_exclusions = raw_query.get("exclude_document_indices", [])
        if isinstance(extra_exclusions, int):
            extra_exclusions = [extra_exclusions]
        if not isinstance(extra_exclusions, Sequence) or isinstance(
            extra_exclusions, (str, bytes, bytearray)
        ):
            raise ValueError(f"query {query_id}: invalid exclude_document_indices")
        exclusions = {
            int(index)
            for index in extra_exclusions
        }
        exclusions.update(
            seed.candidate_document_index
            for seed in seeds
            if seed.candidate_document_index is not None
        )
        if any(index < 0 or index >= len(documents) for index in exclusions):
            raise ValueError(f"query {query_id}: excluded document index out of range")
        reason_value = raw_query.get("unavailable_reason")
        unavailable_reason = (
            str(reason_value)
            if reason_value is not None
            else ("insufficient_seed" if explicit_unavailable else None)
        )
        queries.append(
            QuerySpec(
                query_id=query_id,
                name=name_value,
                status=status,
                keyword_rule=keyword_rule,
                seeds=tuple(seeds),
                sae_available=not explicit_unavailable,
                unavailable_reason=unavailable_reason,
                excluded_document_indices=tuple(sorted(exclusions)),
            )
        )
    feature_seed_set = set(seed_features.seed_ids)
    referenced_seed_set = set(referenced_seed_ids)
    if referenced_seed_set != feature_seed_set:
        raise ValueError(
            f"{path}: locked query seed set differs from seed_features.npz; "
            f"missing_in_queries={sorted(feature_seed_set - referenced_seed_set)}, "
            f"missing_in_features={sorted(referenced_seed_set - feature_seed_set)}"
        )
    if locked_seed_texts is not None:
        jsonl_seed_set = set(locked_seed_texts)
        if referenced_seed_set != jsonl_seed_set:
            raise ValueError(
                f"{path}: locked query seed set differs from {locked_seed_path}; "
                f"unreferenced_jsonl={sorted(jsonl_seed_set - referenced_seed_set)}, "
                f"missing_jsonl={sorted(referenced_seed_set - jsonl_seed_set)}"
            )
    return queries, payload


def display_safe(text: str) -> str:
    return "".join(
        character
        if character in {"\n", "\t"} or ord(character) >= 32
        else "\ufffd"
        for character in text
    )


def truncate_for_display(text: str, limit: int) -> tuple[str, bool]:
    cleaned = display_safe(text)
    if len(cleaned) <= limit:
        return cleaned, False
    return cleaned[:limit].rstrip() + "\n…", True


def make_snippet(
    text: str,
    spans: Sequence[tuple[int, int]],
    limit: int,
) -> str:
    if limit <= 0:
        return ""
    cleaned = display_safe(text)
    if len(cleaned) <= limit:
        return cleaned
    if spans:
        focus_start, focus_end = spans[0]
        focus_start = min(max(0, focus_start), len(cleaned))
        focus_end = min(max(focus_start, focus_end), len(cleaned))
        start = max(0, focus_start - limit // 3)
        end = min(len(cleaned), max(focus_end + limit // 3, start + limit))
        start = max(0, end - limit)
    else:
        start, end = 0, limit
    prefix = "…\n" if start > 0 else ""
    suffix = "\n…" if end < len(cleaned) else ""
    return prefix + cleaned[start:end] + suffix


def keyword_result_entry(
    document_index: int,
    rank: int,
    evaluation: KeywordEvaluation,
    text: str,
    snippet_chars: int,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "document_index": document_index,
        "matched_clause_count": evaluation.matched_clause_count,
        "occurrence_count": evaluation.occurrence_count,
        "matched_clause_ids": list(evaluation.matched_clause_ids),
        "matched_atoms": list(evaluation.matched_atoms),
        "snippet": make_snippet(text, evaluation.spans, snippet_chars),
    }


def top_query_features(query: np.ndarray, limit: int = 24) -> list[dict[str, Any]]:
    nonzero = np.flatnonzero(query > 0)
    ordered = sorted(nonzero, key=lambda index: (-float(query[index]), int(index)))
    return [
        {"feature_id": int(index), "value": float(query[index])}
        for index in ordered[:limit]
    ]


def build_query_result(
    query: QuerySpec,
    *,
    documents: Sequence[str],
    candidates: CandidateFeatures,
    seeds: SeedFeatures,
    top_k: int,
    snippet_chars: int,
    shared_feature_limit: int,
    input_checksums: Mapping[str, str],
) -> dict[str, Any]:
    keyword_evaluations = [
        query.keyword_rule.evaluate(text) for text in documents
    ]
    keyword_hit_indices = [
        index
        for index, evaluation in enumerate(keyword_evaluations)
        if evaluation.matched
    ]
    keyword_hit_indices.sort(
        key=lambda index: (
            -keyword_evaluations[index].matched_clause_count,
            -keyword_evaluations[index].occurrence_count,
            index,
        )
    )
    keyword_results = [
        keyword_result_entry(
            document_index=index,
            rank=rank,
            evaluation=keyword_evaluations[index],
            text=documents[index],
            snippet_chars=snippet_chars,
        )
        for rank, index in enumerate(keyword_hit_indices, start=1)
    ]
    method_results: dict[str, Any] = {
        "keyword": {
            "method_id": "keyword",
            "method_name": METHOD_NAMES["keyword"],
            "available": True,
            "retrieval": "locked keyword rule over the complete candidate pool",
            "sorting": (
                "matched_clause_count descending, occurrence_count descending, "
                "document_index ascending"
            ),
            "total_hits": len(keyword_results),
            "html_result_limit": min(top_k, len(keyword_results)),
            "results": keyword_results,
        }
    }
    seed_rows = {
        seed_id: index for index, seed_id in enumerate(seeds.seed_ids)
    }
    if query.sae_available:
        selected_seed_rows = [seed_rows[seed.seed_id] for seed in query.seeds]
        excluded = set(query.excluded_document_indices)
        available_count = len(documents) - len(excluded)
        if available_count < top_k:
            raise ValueError(
                f"query {query.query_id}: only {available_count} candidates remain "
                f"after exclusions, fewer than top_k={top_k}"
            )
        for mode in METHODS:
            query_vector = seeds.methods[mode].dense_mean_query(selected_seed_rows)
            scores = candidates.methods[mode].cosine_scores(query_vector)
            if excluded:
                scores = scores.copy()
                scores[list(excluded)] = -np.inf
            order = np.lexsort(
                (
                    candidates.document_indices,
                    -scores,
                )
            )
            selected = [
                int(index)
                for index in order
                if math.isfinite(float(scores[index]))
            ][:top_k]
            if len(selected) != top_k:
                raise ValueError(
                    f"query {query.query_id}/{mode}: retrieved "
                    f"{len(selected)} != {top_k}"
                )
            retrieved: list[dict[str, Any]] = []
            for rank, document_index in enumerate(selected, start=1):
                keyword_evaluation = keyword_evaluations[document_index]
                matched_window = (
                    candidates.windows.best_window(
                        mode, document_index, query_vector
                    )
                    if candidates.windows is not None
                    else None
                )
                result: dict[str, Any] = {
                    "rank": rank,
                    "document_index": document_index,
                    "similarity": float(scores[document_index]),
                    "contains_keyword": keyword_evaluation.matched,
                    "keyword_matched_clause_count": (
                        keyword_evaluation.matched_clause_count
                    ),
                    "keyword_occurrence_count": keyword_evaluation.occurrence_count,
                    "keyword_matched_atoms": list(
                        keyword_evaluation.matched_atoms
                    ),
                    "snippet": make_snippet(
                        documents[document_index],
                        keyword_evaluation.spans,
                        snippet_chars,
                    ),
                    "matched_window": matched_window,
                    "shared_features": candidates.methods[mode].shared_features(
                        document_index,
                        query_vector,
                        shared_feature_limit,
                    ),
                }
                retrieved.append(result)
            method_results[mode] = {
                "method_id": mode,
                "method_name": METHOD_NAMES[mode],
                "available": True,
                "retrieval": (
                    "mean of the same locked seed pages in this SAE space; "
                    "cosine over the complete candidate pool"
                ),
                "sorting": "similarity descending, document_index ascending",
                "seed_ids": [seed.seed_id for seed in query.seeds],
                "excluded_document_indices": list(
                    query.excluded_document_indices
                ),
                "query_nonzero_features": int(np.count_nonzero(query_vector)),
                "query_top_features": top_query_features(query_vector),
                "results": retrieved,
            }
    else:
        for mode in METHODS:
            method_results[mode] = {
                "method_id": mode,
                "method_name": METHOD_NAMES[mode],
                "available": False,
                "unavailable_reason": query.unavailable_reason,
                "results": [],
            }
    return {
        "format": RESULT_FORMAT,
        "query": {
            "query_id": query.query_id,
            "name": query.name,
            "status": query.status,
            "keyword_rule": query.keyword_rule.normalized(),
            "sae_available": query.sae_available,
            "sae_seed_ids": [seed.seed_id for seed in query.seeds],
            "excluded_document_indices": list(query.excluded_document_indices),
        },
        "candidate_count": len(documents),
        "top_k": top_k,
        "input_sha256": dict(input_checksums),
        "seeds": [
            {
                "seed_id": seed.seed_id,
                "text": seed.text,
                "source": seed.source,
                "candidate_document_index": seed.candidate_document_index,
            }
            for seed in query.seeds
        ],
        "methods": method_results,
    }


def blind_mapping(query_id: str, blind_seed: int) -> dict[str, Any]:
    shuffled = sorted(
        QUERY_METHODS,
        key=lambda method: (
            hashlib.sha256(
                f"{blind_seed}\0{query_id}\0{method}".encode("utf-8")
            ).digest(),
            method,
        ),
    )
    blind_to_method = {
        chr(ord("A") + index): method
        for index, method in enumerate(shuffled)
    }
    method_to_blind = {
        method: blind for blind, method in blind_to_method.items()
    }
    return {
        "blind_to_method": blind_to_method,
        "method_to_blind": method_to_blind,
    }


def esc(value: Any) -> str:
    return html.escape(display_safe(str(value)), quote=True)


def page_shell(
    title: str,
    body: str,
    *,
    index_page: bool = False,
    navigation_href: str | None = "../index.html",
) -> str:
    navigation = (
        ""
        if index_page or navigation_href is None
        else f'<a class="back-link" href="{esc(navigation_href)}">← All topics</a>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; base-uri 'none'; form-action 'none'">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #252932;
      --muted: #626b78;
      --line: #d8dee8;
      --paper: #ffffff;
      --wash: #f5f7fa;
      --accent: #174ea6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--wash);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{ width: min(1500px, calc(100% - 32px)); margin: 24px auto 72px; }}
    h1, h2, h3, h4 {{ line-height: 1.2; }}
    a {{ color: var(--accent); }}
    .back-link {{ display: inline-block; margin-bottom: 14px; font-weight: 700; }}
    .hero, .seed-box, .method-panel, .topic-card, .notice {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 2px 9px rgba(20, 34, 55, 0.05);
    }}
    .hero {{ padding: 22px 24px; margin-bottom: 18px; }}
    .hero h1 {{ margin: 0 0 8px; }}
    .subtle, .metadata {{ color: var(--muted); }}
    .toolbar {{
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
      margin: 14px 0 20px;
    }}
    button, .button {{
      border: 1px solid #aeb8c6; background: white; color: var(--ink);
      border-radius: 9px; padding: 8px 12px; font-weight: 700; cursor: pointer;
      text-decoration: none;
    }}
    button.active {{ background: var(--ink); color: white; border-color: var(--ink); }}
    .blind-warning {{
      padding: 10px 12px; border-radius: 9px;
      background: #fff8df; border: 1px solid #ead58a; font-weight: 700;
    }}
    .seed-grid, .topic-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 12px;
    }}
    .seed-box, .topic-card {{ padding: 15px; }}
    .seed-box h3, .topic-card h2 {{ margin-top: 0; }}
    pre {{
      margin: 8px 0 0; padding: 12px; background: #f7f8fa;
      border: 1px solid #e2e6ed; border-radius: 9px;
      white-space: pre-wrap; overflow-wrap: anywhere;
      font: 12.5px/1.48 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      max-height: 28rem; overflow: auto;
    }}
    #method-grid {{ display: grid; gap: 18px; }}
    .method-panel {{
      padding: 0; overflow: hidden; border-top: 6px solid var(--method-color);
    }}
    .method-header {{
      padding: 16px 18px; border-bottom: 1px solid var(--line);
      display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px;
    }}
    .method-header h2 {{ margin: 0; }}
    .result-list {{ display: grid; gap: 12px; padding: 14px; }}
    .result {{
      border: 1px solid var(--line); border-radius: 10px;
      padding: 13px; background: #fff;
    }}
    .result-head {{
      display: flex; flex-wrap: wrap; justify-content: space-between;
      gap: 8px; font-weight: 800;
    }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
    .badge {{
      display: inline-block; padding: 2px 7px; border-radius: 999px;
      background: #eef1f5; color: #44505f; font-size: 12px; font-weight: 700;
    }}
    .feature-list {{
      margin: 8px 0 0; padding-left: 20px; font: 12px/1.45 ui-monospace, monospace;
    }}
    details.more-results {{ margin: 0 14px 14px; }}
    details.more-results > summary {{
      cursor: pointer; font-weight: 800; padding: 10px;
      background: #f3f5f8; border-radius: 8px;
    }}
    .notice {{ padding: 14px 16px; margin: 12px 0; }}
    .status {{ font-weight: 800; }}
    .topic-card .links {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid var(--line); padding: 9px; text-align: left; }}
    @media (max-width: 680px) {{
      main {{ width: min(100% - 18px, 1500px); margin-top: 10px; }}
      .hero {{ padding: 16px; }}
      .result-list {{ padding: 9px; }}
    }}
  </style>
</head>
<body>
<main>
  {navigation}
  {body}
</main>
</body>
</html>
"""


def render_shared_features(features: Sequence[Mapping[str, Any]]) -> str:
    if not features:
        return '<span class="subtle">No positive shared feature contribution.</span>'
    items = "".join(
        "<li>"
        f"feature {int(item['feature_id'])}: "
        f"{float(item['cosine_contribution']):.6f}"
        "</li>"
        for item in features
    )
    return f'<ul class="feature-list">{items}</ul>'


def render_result_card(
    method: str,
    result: Mapping[str, Any],
    *,
    blind: bool,
) -> str:
    rank = int(result["rank"])
    document_index = int(result["document_index"])
    if blind:
        metric = ""
        badges = ""
        explanation = ""
    elif method == "keyword":
        metric = (
            f"{int(result['matched_clause_count'])} clauses · "
            f"{int(result['occurrence_count'])} occurrences"
        )
        badges = "".join(
            f'<span class="badge">{esc(atom)}</span>'
            for atom in result.get("matched_atoms", [])
        )
        explanation = ""
    else:
        metric = f"cosine {float(result['similarity']):.6f}"
        contains = bool(result.get("contains_keyword", False))
        badges = (
            '<span class="badge normal-only">'
            + ("contains locked keyword" if contains else "no locked keyword")
            + "</span>"
        )
        matched_window = result.get("matched_window")
        window_html = ""
        if isinstance(matched_window, Mapping):
            window_bits = [f"window {int(matched_window['window_index'])}"]
            if "token_start" in matched_window and "token_end" in matched_window:
                window_bits.append(
                    f"tokens {int(matched_window['token_start'])}:"
                    f"{int(matched_window['token_end'])}"
                )
            window_bits.append(
                f"window cosine {float(matched_window['similarity']):.6f}"
            )
            window_text = matched_window.get("text")
            window_html = (
                '<div class="normal-only metadata">'
                + esc(" · ".join(window_bits))
                + "</div>"
            )
            if isinstance(window_text, str):
                shown, truncated = truncate_for_display(window_text, 1200)
                window_html += (
                    '<div class="normal-only"><strong>Best window text'
                    + (" (truncated)" if truncated else "")
                    + f":</strong><pre>{esc(shown)}</pre></div>"
                )
        explanation = (
            '<div class="normal-only"><strong>Shared feature contributions</strong>'
            + render_shared_features(result.get("shared_features", []))
            + window_html
            + "</div>"
        )
    return f"""
<article class="result">
  <div class="result-head">
    <span>Rank {rank} · document_index {document_index}</span>
    <span>{esc(metric)}</span>
  </div>
  <div class="badges">{badges}</div>
  <pre>{esc(result.get("snippet", ""))}</pre>
  {explanation}
</article>
"""


def render_method_panel(
    method: str,
    method_payload: Mapping[str, Any],
    *,
    blind_label: str,
    initial_results: int,
    top_k: int,
    blind: bool,
) -> str:
    all_results = list(method_payload.get("results", []))
    display_results = all_results[:top_k]
    initial = display_results[:initial_results]
    remainder = display_results[initial_results:]
    available = bool(method_payload.get("available", False))
    if not available:
        contents = (
            '<div class="notice">Unavailable: '
            + esc(method_payload.get("unavailable_reason", "unspecified"))
            + "</div>"
        )
    elif not display_results:
        contents = '<div class="notice">No matching result.</div>'
    else:
        contents = '<div class="result-list">' + "".join(
            render_result_card(method, result, blind=blind) for result in initial
        ) + "</div>"
        if remainder:
            contents += (
                '<details class="more-results"><summary>'
                f"Show ranks {initial_results + 1}–{len(display_results)}"
                '</summary><div class="result-list">'
                + "".join(
                    render_result_card(method, result, blind=blind)
                    for result in remainder
                )
                + "</div></details>"
            )
    count_text = (
        f"{int(method_payload.get('total_hits', len(all_results)))} literal hits"
        if method == "keyword"
        else f"{len(display_results)} retrieved pages"
    )
    return f"""
<section class="method-panel"
         style="--method-color: {'#6b7280' if blind else METHOD_PALETTE[method]}">
  <header class="method-header">
    <h2>{'Method ' + esc(blind_label) if blind else esc(METHOD_NAMES[method])}</h2>
    {'' if blind else '<span class="metadata">' + esc(count_text) + '</span>'}
  </header>
  {contents}
</section>
"""


def render_query_page(
    query: QuerySpec,
    result: Mapping[str, Any],
    mapping: Mapping[str, Any],
    *,
    initial_results: int,
    top_k: int,
    seed_display_chars: int,
    blind: bool,
) -> str:
    seed_cards = []
    for seed in query.seeds:
        shown, truncated = truncate_for_display(seed.text, seed_display_chars)
        source = f"Source: {seed.source}"
        if seed.candidate_document_index is not None:
            source += f" · candidate document {seed.candidate_document_index}"
        seed_cards.append(
            '<article class="seed-box">'
            f"<h3>{esc(seed.seed_id)}</h3>"
            f'<div class="metadata">{esc(source)}</div>'
            f"<pre>{esc(shown)}</pre>"
            + (
                '<div class="metadata">Seed text truncated for HTML display; '
                "the locked input remains authoritative.</div>"
                if truncated
                else ""
            )
            + "</article>"
        )
    seed_section = (
        '<section><h2>Locked independent SAE seeds</h2>'
        '<div class="seed-grid">'
        + "".join(seed_cards)
        + "</div></section>"
        if seed_cards
        else '<div class="notice">No usable independent SAE seed is locked.</div>'
    )
    method_to_blind = mapping["method_to_blind"]
    ordered_methods = (
        [
            mapping["blind_to_method"][blind_label]
            for blind_label in sorted(mapping["blind_to_method"])
        ]
        if blind
        else list(QUERY_METHODS)
    )
    panels = []
    for method in ordered_methods:
        panels.append(
            render_method_panel(
                method,
                result["methods"][method],
                blind_label=method_to_blind[method],
                initial_results=initial_results,
                top_k=top_k,
                blind=blind,
            )
        )
    toolbar = (
        '<span class="blind-warning">Method-blind view: identities, method-specific '
        "scores, keyword badges, matched windows, and SAE feature explanations "
        "are omitted. Freeze the review before consulting the unblinding artifact.</span>"
        if blind
        else '<a class="button" href="blind/'
        + esc(query.query_id)
        + '.html">Blind A–E view</a>'
        '<a class="button" href="../results/'
        + esc(query.query_id)
        + '.json">Result JSON</a>'
    )
    description = (
        "Five result lists are presented in deterministic per-topic A–E order. "
        "All lists use the same candidate pool."
        if blind
        else (
            "Keyword and SAE retrieval are independent. The keyword rule scans "
            "all candidates; each SAE uses the same locked seed texts in its own "
            "space and scans the same complete candidate pool."
        )
    )
    body = f"""
<section class="hero">
  <h1>{esc(query.name)}</h1>
  <div class="metadata">query_id: {esc(query.query_id)} · status: {esc(query.status)}</div>
  <p>{esc(description)}</p>
</section>
<div class="toolbar">
  {toolbar}
</div>
{seed_section}
<section>
  <h2>Retrieval results</h2>
  <div id="method-grid">
    {''.join(panels)}
  </div>
</section>
"""
    return page_shell(
        f"Eval-9 · {query.name}{' · Blind' if blind else ''}",
        body,
        navigation_href="index.html" if blind else "../index.html",
    )


def render_index_page(
    queries: Sequence[QuerySpec],
    results: Mapping[str, Mapping[str, Any]],
) -> str:
    cards = []
    for query in queries:
        keyword_hits = int(
            results[query.query_id]["methods"]["keyword"]["total_hits"]
        )
        cards.append(
            '<article class="topic-card">'
            f"<h2>{esc(query.name)}</h2>"
            f'<div class="metadata">query_id: {esc(query.query_id)}</div>'
            f'<p class="normal-only">{keyword_hits} keyword hits · '
            f"SAE {'available' if query.sae_available else 'unavailable'}</p>"
            '<div class="links">'
            f'<a class="button" href="pages/{esc(query.query_id)}.html">'
            "Normal view</a>"
            f'<a class="button" href="pages/blind/{esc(query.query_id)}.html">'
            "Blind A–E view</a>"
            f'<a class="button" href="results/{esc(query.query_id)}.json">'
            "JSON</a>"
            "</div></article>"
        )
    body = f"""
<section class="hero">
  <h1>Eval-9 Code Page Same-Topic Retrieval</h1>
  <p>
    Five independent result lists are shown per topic: one locked keyword
    baseline and four SAE cosine-retrieval methods. Use blind A–E pages for
    review; keep <code>method_map.json</code> away from reviewers until labels
    are frozen.
  </p>
</section>
<div class="topic-grid">{''.join(cards)}</div>
"""
    return page_shell("Eval-9 Code Page Retrieval", body, index_page=True)


def render_blind_index_page(queries: Sequence[QuerySpec]) -> str:
    cards = "".join(
        '<article class="topic-card">'
        f"<h2>{esc(query.name)}</h2>"
        f'<div class="metadata">query_id: {esc(query.query_id)}</div>'
        '<div class="links">'
        f'<a class="button" href="{esc(query.query_id)}.html">'
        "Open blind A–E review</a>"
        "</div></article>"
        for query in queries
    )
    body = f"""
<section class="hero">
  <h1>Eval-9 Blind A–E Review</h1>
    <p>
      Method identities, score types, keyword diagnostics, matched-window
      metadata, and feature explanations are omitted. Complete the review
      template before consulting the unblinding artifact.
    </p>
</section>
<div class="topic-grid">{cards}</div>
"""
    return page_shell(
        "Eval-9 Blind Review",
        body,
        index_page=True,
        navigation_href=None,
    )


def render_readme(
    queries: Sequence[QuerySpec],
    results: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        "# Eval 9 Code Page Retrieval",
        "",
        "- Candidate documents: 10,000",
        f"- Topics: {len(queries)}",
        "- Methods: Keyword, Token SAE, Temporal SAE, Mean-Chunk SAE, Cross-Chunk SAE",
        "- Normal demo: `index.html`",
        "- Blind demo: `pages/blind/index.html`",
        "",
        "## Query coverage and lexical overlap of SAE Top-20",
        "",
        "| Topic | Keyword hits | Token | Temporal | Mean | Cross |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for query in queries:
        payload = results[query.query_id]["methods"]
        overlaps = []
        for method in METHODS:
            overlaps.append(
                sum(
                    bool(row.get("contains_keyword", False))
                    for row in payload[method]["results"]
                )
            )
        lines.append(
            f"| {query.name} | {int(payload['keyword']['total_hits'])} | "
            f"{overlaps[0]}/20 | {overlaps[1]}/20 | {overlaps[2]}/20 | "
            f"{overlaps[3]}/20 |"
        )
    lines.extend(
        [
            "",
            "The keyword branch and the four SAE branches independently search "
            "the complete candidate pool. Keyword results do not select or "
            "filter SAE seeds or candidates.",
            "",
            "Keyword overlap is diagnostic only. A result can be topically "
            "relevant without containing the literal keyword, and a literal "
            "match can still be off-topic. Use the normal or blind pages for "
            "qualitative review.",
            "",
        ]
    )
    return "\n".join(lines)


def write_review_template(
    path: Path,
    *,
    queries: Sequence[QuerySpec],
    results: Mapping[str, Mapping[str, Any]],
    method_map: Mapping[str, Any],
    top_k: int,
) -> None:
    fieldnames = [
        "query_id",
        "query_name",
        "blind_method",
        "rank",
        "document_index",
        "same_topic_0_or_1",
        "useful_for_long_task_0_or_1",
        "duplicate_or_mirror_0_or_1",
        "generic_or_template_noise_0_or_1",
        "reviewer_confidence_1_to_3",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for query in queries:
            mapping = method_map["queries"][query.query_id]["blind_to_method"]
            for blind_label in sorted(mapping):
                method = mapping[blind_label]
                method_results = results[query.query_id]["methods"][method]["results"]
                for result in list(method_results)[:top_k]:
                    writer.writerow(
                        {
                            "query_id": query.query_id,
                            "query_name": query.name,
                            "blind_method": blind_label,
                            "rank": int(result["rank"]),
                            "document_index": int(result["document_index"]),
                            "same_topic_0_or_1": "",
                            "useful_for_long_task_0_or_1": "",
                            "duplicate_or_mirror_0_or_1": "",
                            "generic_or_template_noise_0_or_1": "",
                            "reviewer_confidence_1_to_3": "",
                            "notes": "",
                        }
                    )


def artifact_digest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("artifact_digest", None)
    result["artifact_digest"] = json_digest(result)
    return result


def read_and_validate_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest is not an object")
    if manifest.get("format") != OUTPUT_FORMAT:
        raise ValueError(
            f"{path}: expected format {OUTPUT_FORMAT}, found {manifest.get('format')}"
        )
    expected = manifest.get("artifact_digest")
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"{path}: invalid artifact_digest")
    without_digest = dict(manifest)
    without_digest.pop("artifact_digest", None)
    actual = json_digest(without_digest)
    if actual != expected:
        raise ValueError(f"{path}: artifact digest mismatch: {actual} != {expected}")
    if manifest.get("complete") is not True:
        raise ValueError(f"{path}: artifact is not complete")
    return manifest


def verify_output_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"{root}: output manifest has no file records")
    for key, raw_record in files.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{root}: invalid file record {key}")
        relative = raw_record.get("path")
        if not isinstance(relative, str):
            raise ValueError(f"{root}: file record {key} lacks a path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{root}: unsafe output path {relative}")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(raw_record.get("bytes", -1)):
            raise ValueError(f"{path}: output byte-size mismatch")
        if file_sha256(path) != raw_record.get("sha256"):
            raise ValueError(f"{path}: output SHA-256 mismatch")


def verified_manifest_file(
    manifest_path: Path,
    raw_record: Any,
    label: str,
) -> Path:
    if not isinstance(raw_record, Mapping):
        raise ValueError(f"{manifest_path}: invalid {label} file record")
    relative = raw_record.get("path")
    if not isinstance(relative, str):
        raise ValueError(f"{manifest_path}: {label} file record lacks path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{manifest_path}: unsafe {label} path {relative!r}")
    path = manifest_path.parent / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(raw_record.get("bytes", -1)):
        raise ValueError(f"{path}: byte-size mismatch against {manifest_path}")
    expected_sha256 = raw_record.get("sha256")
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ValueError(f"{manifest_path}: invalid SHA-256 for {label}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{path}: SHA-256 mismatch against {manifest_path}")
    return path.resolve()


def validate_feature_manifest(
    path: Path,
    *,
    merged_features_path: Path,
    seed_features_path: Path,
    parquet_record: Mapping[str, Any],
    seeds_jsonl_record: Mapping[str, Any] | None,
    candidates: CandidateFeatures,
    seeds: SeedFeatures,
    expected_documents: int,
    expected_width: int,
    expected_k: int,
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{path}: feature manifest must be a JSON object")
    if manifest.get("format") != FEATURE_MANIFEST_FORMAT:
        raise ValueError(
            f"{path}: expected format={FEATURE_MANIFEST_FORMAT!r}, "
            f"found {manifest.get('format')!r}"
        )
    expected_digest = manifest.get("artifact_digest")
    if not isinstance(expected_digest, str) or SHA256_RE.fullmatch(
        expected_digest
    ) is None:
        raise ValueError(f"{path}: invalid artifact_digest")
    without_digest = dict(manifest)
    without_digest.pop("artifact_digest", None)
    actual_digest = json_digest(without_digest)
    if actual_digest != expected_digest:
        raise ValueError(
            f"{path}: artifact_digest mismatch: "
            f"{actual_digest} != {expected_digest}"
        )
    if manifest.get("complete") is not True:
        raise ValueError(f"{path}: feature manifest is not complete")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{path}: feature manifest files must be an object")
    merged_record = files.get("merged_features")
    seed_record = files.get("seed_features")
    actual_merged = verified_manifest_file(path, merged_record, "merged_features")
    actual_seed = verified_manifest_file(path, seed_record, "seed_features")
    if actual_merged != merged_features_path.resolve():
        raise ValueError(
            f"{path}: merged feature record points to {actual_merged}, "
            f"not requested {merged_features_path}"
        )
    if actual_seed != seed_features_path.resolve():
        raise ValueError(
            f"{path}: seed feature record points to {actual_seed}, "
            f"not requested {seed_features_path}"
        )
    if int(manifest.get("documents", -1)) != expected_documents:
        raise ValueError(f"{path}: document count mismatch")
    if int(manifest.get("seeds", -1)) != len(seeds.seed_ids):
        raise ValueError(f"{path}: seed count mismatch")
    if int(manifest.get("top_k", -1)) != expected_k:
        raise ValueError(f"{path}: Top-K mismatch")
    feature_widths = manifest.get("feature_widths")
    if not isinstance(feature_widths, Mapping) or {
        str(mode): int(feature_widths.get(mode, -1)) for mode in METHODS
    } != {mode: expected_width for mode in METHODS}:
        raise ValueError(f"{path}: feature widths do not match Eval-9 settings")
    if manifest.get("representation_protocol_name") != "fixed-independent-chunk-v1":
        raise ValueError(f"{path}: unexpected representation protocol")
    representation = manifest.get("representation_protocol")
    if not isinstance(representation, Mapping):
        raise ValueError(f"{path}: representation_protocol is missing")
    if (
        representation.get("name") != "fixed-independent-chunk-v1"
        or representation.get("feature_scope") != "complete_dictionary"
        or representation.get("shared_hidden_states_across_methods") is not True
        or representation.get("token_temporal_aggregation")
        != "mean_after_threshold"
    ):
        raise ValueError(f"{path}: representation protocol metadata is incompatible")
    protocol_widths = representation.get("feature_widths")
    if not isinstance(protocol_widths, Mapping) or {
        str(mode): int(protocol_widths.get(mode, -1)) for mode in METHODS
    } != {mode: expected_width for mode in METHODS}:
        raise ValueError(f"{path}: protocol feature widths are incompatible")
    identity_digest = manifest.get("identity_digest")
    if identity_digest != candidates.identity_digest or identity_digest != seeds.identity_digest:
        raise ValueError(f"{path}: feature identity_digest mismatch")
    if int(manifest.get("world_size", -1)) != candidates.world_size:
        raise ValueError(f"{path}: feature world_size mismatch")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{path}: feature manifest identity must be an object")
    input_parquet = identity.get("input_parquet")
    if not isinstance(input_parquet, Mapping):
        raise ValueError(f"{path}: identity.input_parquet is missing")
    if (
        int(input_parquet.get("bytes", -1)) != int(parquet_record["bytes"])
        or input_parquet.get("sha256") != parquet_record["sha256"]
    ):
        raise ValueError(f"{path}: source Parquet checksum/size mismatch")
    seed_identity = identity.get("seeds_jsonl")
    if seeds_jsonl_record is not None:
        if not isinstance(seed_identity, Mapping):
            raise ValueError(f"{path}: identity.seeds_jsonl is missing")
        if (
            int(seed_identity.get("bytes", -1)) != int(seeds_jsonl_record["bytes"])
            or seed_identity.get("sha256") != seeds_jsonl_record["sha256"]
        ):
            raise ValueError(f"{path}: seeds.jsonl checksum/size mismatch")
    return dict(manifest)


def build_identity(
    args: argparse.Namespace,
    input_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "inputs": {
            key: {
                "path": record["path"],
                "bytes": int(record["bytes"]),
                "sha256": record["sha256"],
            }
            for key, record in sorted(input_records.items())
        },
        "script": {
            "path": str(script_path),
            "sha256": file_sha256(script_path),
        },
        "parameters": {
            "top_k": args.top_k,
            "html_initial_results": args.html_initial_results,
            "snippet_chars": args.snippet_chars,
            "seed_display_chars": args.seed_display_chars,
            "shared_features": args.shared_features,
            "blind_seed": args.blind_seed,
            "expected_documents": args.expected_documents,
            "expected_feature_width": args.expected_feature_width,
            "expected_vector_k": args.expected_vector_k,
            "feature_norm_tolerance": args.feature_norm_tolerance,
            "methods": list(QUERY_METHODS),
        },
    }


GENERATED_TOP_LEVEL = (
    "results",
    "pages",
    "index.html",
    "README.md",
    "method_map.json",
    "review_template.csv",
    "retrieval_demo_manifest.json",
)


def publish_staging(staging: Path, output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    if not output_dir.is_dir():
        raise ValueError(f"{output_dir}: output path exists and is not a directory")
    collisions = [
        name for name in GENERATED_TOP_LEVEL if (output_dir / name).exists()
    ]
    if collisions and not overwrite:
        raise FileExistsError(
            f"{output_dir}: generated outputs already exist ({collisions}); "
            "use --overwrite after verifying the mismatch"
        )
    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.backup.",
            dir=output_dir.parent,
        )
    )
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        # Invalidate the previous complete marker before replacing its files.
        ordered = [
            "retrieval_demo_manifest.json",
            "results",
            "pages",
            "index.html",
            "README.md",
            "method_map.json",
            "review_template.csv",
        ]
        for name in ordered:
            destination = output_dir / name
            source = staging / name
            backup_path = backup / name
            if destination.exists():
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup_path)
                moved_old.append(name)
            if name != "retrieval_demo_manifest.json":
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                moved_new.append(name)
        manifest_source = staging / "retrieval_demo_manifest.json"
        manifest_destination = output_dir / "retrieval_demo_manifest.json"
        os.replace(manifest_source, manifest_destination)
        moved_new.append("retrieval_demo_manifest.json")
    except Exception:
        for name in reversed(moved_new):
            destination = output_dir / name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
        for name in reversed(moved_old):
            backup_path = backup / name
            destination = output_dir / name
            if backup_path.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup_path, destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "top_k": args.top_k,
        "html_initial_results": args.html_initial_results,
        "snippet_chars": args.snippet_chars,
        "seed_display_chars": args.seed_display_chars,
        "shared_features": args.shared_features,
        "expected_documents": args.expected_documents,
        "expected_feature_width": args.expected_feature_width,
        "expected_vector_k": args.expected_vector_k,
    }
    for label, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{label.replace('_', '-')} must be positive")
    if args.html_initial_results > args.top_k:
        raise ValueError("--html-initial-results may not exceed --top-k")
    if not 0 < args.feature_norm_tolerance < 0.5:
        raise ValueError("--feature-norm-tolerance must be in (0, 0.5)")
    if args.top_k > args.expected_documents:
        raise ValueError("--top-k may not exceed --expected-documents")


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    queries_path = Path(args.queries).expanduser().resolve()
    merged_features_path = Path(args.merged_features).expanduser().resolve()
    feature_manifest_path = (
        Path(args.feature_manifest).expanduser().resolve()
        if args.feature_manifest
        else merged_features_path.with_name("feature_manifest.json")
    )
    use_feature_manifest = (
        args.feature_manifest is not None or feature_manifest_path.is_file()
    )
    seeds_jsonl_path = (
        Path(args.seeds_jsonl).expanduser().resolve()
        if args.seeds_jsonl
        else queries_path.with_name("seeds.jsonl")
    )
    use_seeds_jsonl = args.seeds_jsonl is not None or seeds_jsonl_path.is_file()
    if args.seeds_jsonl_sha256 is not None and not use_seeds_jsonl:
        raise FileNotFoundError(
            "--seeds-jsonl-sha256 was provided but no seeds.jsonl is available"
        )
    paths = {
        "parquet": Path(args.parquet).expanduser().resolve(),
        "merged_features": merged_features_path,
        "seed_features": Path(args.seed_features).expanduser().resolve(),
        "queries_locked": queries_path,
    }
    if use_seeds_jsonl:
        paths["seeds_jsonl"] = seeds_jsonl_path
    if use_feature_manifest:
        paths["feature_manifest"] = feature_manifest_path
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir in paths.values():
        raise ValueError("output directory may not be one of the input files")
    input_records = {
        key: input_record(path) for key, path in paths.items()
    }
    validate_expected_sha256(
        input_records["parquet"], args.parquet_sha256, "candidate Parquet"
    )
    validate_expected_sha256(
        input_records["merged_features"],
        args.merged_features_sha256,
        "merged features",
    )
    validate_expected_sha256(
        input_records["seed_features"],
        args.seed_features_sha256,
        "seed features",
    )
    validate_expected_sha256(
        input_records["queries_locked"],
        args.queries_sha256,
        "locked queries",
    )
    if use_seeds_jsonl:
        validate_expected_sha256(
            input_records["seeds_jsonl"],
            args.seeds_jsonl_sha256,
            "locked seed texts",
        )
    identity = build_identity(args, input_records)
    manifest_path = output_dir / "retrieval_demo_manifest.json"
    if manifest_path.is_file() and not args.overwrite:
        existing = read_and_validate_manifest(manifest_path)
        if existing.get("identity") != identity:
            raise ValueError(
                f"{manifest_path}: existing artifact identity differs from this run; "
                "use --overwrite or a new output directory"
            )
        verify_output_files(output_dir, existing)
        print(
            json.dumps(
                {
                    "status": "reused",
                    "output_dir": str(output_dir),
                    "artifact_digest": existing["artifact_digest"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return

    documents = load_documents(paths["parquet"], args.expected_documents)
    candidates = load_candidate_features(
        paths["merged_features"],
        document_count=len(documents),
        width=args.expected_feature_width,
        expected_k=args.expected_vector_k,
        norm_tolerance=args.feature_norm_tolerance,
        parquet_sha256=input_records["parquet"]["sha256"],
    )
    seeds = load_seed_features(
        paths["seed_features"],
        width=args.expected_feature_width,
        expected_k=args.expected_vector_k,
        norm_tolerance=args.feature_norm_tolerance,
        queries_sha256=input_records["queries_locked"]["sha256"],
        seeds_jsonl_sha256=(
            input_records["seeds_jsonl"]["sha256"]
            if use_seeds_jsonl
            else None
        ),
    )
    if candidates.identity_digest != seeds.identity_digest:
        raise ValueError(
            "merged_features.npz and seed_features.npz have different "
            "identity_digest values"
        )
    if candidates.world_size != seeds.world_size:
        raise ValueError(
            "merged_features.npz and seed_features.npz have different "
            "world_size values"
        )
    if use_feature_manifest:
        validate_feature_manifest(
            paths["feature_manifest"],
            merged_features_path=paths["merged_features"],
            seed_features_path=paths["seed_features"],
            parquet_record=input_records["parquet"],
            seeds_jsonl_record=input_records.get("seeds_jsonl"),
            candidates=candidates,
            seeds=seeds,
            expected_documents=args.expected_documents,
            expected_width=args.expected_feature_width,
            expected_k=args.expected_vector_k,
        )
    locked_seed_texts = (
        load_locked_seed_texts(paths["seeds_jsonl"])
        if use_seeds_jsonl
        else None
    )
    queries, query_payload = load_queries(
        paths["queries_locked"],
        seed_features=seeds,
        documents=documents,
        candidate_parquet_path=paths["parquet"],
        locked_seed_texts=locked_seed_texts,
        locked_seed_path=paths.get("seeds_jsonl"),
    )
    del query_payload

    input_checksums = {
        key: str(record["sha256"]) for key, record in input_records.items()
    }
    results: dict[str, dict[str, Any]] = {}
    for index, query in enumerate(queries, start=1):
        print(
            f"[eval9-demo] query {index}/{len(queries)}: {query.query_id}",
            flush=True,
        )
        results[query.query_id] = build_query_result(
            query,
            documents=documents,
            candidates=candidates,
            seeds=seeds,
            top_k=args.top_k,
            snippet_chars=args.snippet_chars,
            shared_feature_limit=args.shared_features,
            input_checksums=input_checksums,
        )

    method_map = artifact_digest_payload({
        "format": METHOD_MAP_FORMAT,
        "blind_seed": args.blind_seed,
        "mapping_algorithm": (
            "sort methods by SHA-256(blind_seed NUL query_id NUL method_id)"
        ),
        "methods": [
            {"method_id": method, "method_name": METHOD_NAMES[method]}
            for method in QUERY_METHODS
        ],
        "queries": {
            query.query_id: blind_mapping(query.query_id, args.blind_seed)
            for query in queries
        },
    })

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging.",
            dir=output_dir.parent,
        )
    )
    try:
        for query in queries:
            write_json(
                staging / "results" / f"{query.query_id}.json",
                results[query.query_id],
            )
            write_text(
                staging / "pages" / f"{query.query_id}.html",
                render_query_page(
                    query,
                    results[query.query_id],
                    method_map["queries"][query.query_id],
                    initial_results=args.html_initial_results,
                    top_k=args.top_k,
                    seed_display_chars=args.seed_display_chars,
                    blind=False,
                ),
            )
            write_text(
                staging / "pages" / "blind" / f"{query.query_id}.html",
                render_query_page(
                    query,
                    results[query.query_id],
                    method_map["queries"][query.query_id],
                    initial_results=args.html_initial_results,
                    top_k=args.top_k,
                    seed_display_chars=args.seed_display_chars,
                    blind=True,
                ),
            )
        write_text(staging / "index.html", render_index_page(queries, results))
        write_text(staging / "README.md", render_readme(queries, results))
        write_text(
            staging / "pages" / "blind" / "index.html",
            render_blind_index_page(queries),
        )
        write_json(staging / "method_map.json", method_map)
        write_review_template(
            staging / "review_template.csv",
            queries=queries,
            results=results,
            method_map=method_map,
            top_k=args.top_k,
        )
        generated_paths = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != "retrieval_demo_manifest.json"
        )
        files = {
            path.relative_to(staging).as_posix(): output_record(path, staging)
            for path in generated_paths
        }
        manifest = artifact_digest_payload(
            {
                "format": OUTPUT_FORMAT,
                "complete": True,
                "identity": identity,
                "summary": {
                    "documents": len(documents),
                    "queries": len(queries),
                    "sae_available_queries": sum(
                        query.sae_available for query in queries
                    ),
                    "methods": list(QUERY_METHODS),
                    "top_k": args.top_k,
                    "blind_seed": args.blind_seed,
                },
                "files": files,
            }
        )
        write_json(staging / "retrieval_demo_manifest.json", manifest)
        publish_staging(staging, output_dir, args.overwrite)
        published = read_and_validate_manifest(manifest_path)
        verify_output_files(output_dir, published)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "built",
                "output_dir": str(output_dir),
                "queries": len(queries),
                "documents": len(documents),
                "artifact_digest": published["artifact_digest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
