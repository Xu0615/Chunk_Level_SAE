from __future__ import annotations

import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import time
from array import array
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .data import iter_jsonl
from .utils import atomic_json_dump, content_hash, stable_u64


SAMPLE_PLAN_FORMAT = "chunk-saes-sample-plan-v2"
SAMPLE_PLAN_SHARD_FORMAT = "chunk-saes-sample-plan-shard-v2"
PLAN_ROW_HASH_VERSION = "chunk-saes-plan-row-v2"
DOCUMENT_PRIORITY_VERSION = "blake2b-document-priority-v2"
BOUNDARY_VERSION = "blake2b-nonoverlap-boundary-v2"
CATALOG_WORKER_FORMAT = "chunk-saes-token-catalog-worker-v1"
SQLITE_BUSY_TIMEOUT_MS = 120_000
SQLITE_COMMIT_RETRIES = 8


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hex_or_sha256(value: str) -> bytes:
    stripped = value.strip().lower()
    if len(stripped) == 64:
        try:
            return bytes.fromhex(stripped)
        except ValueError:
            pass
    return hashlib.sha256(value.encode("utf-8", errors="replace")).digest()


def _keyed_digest(namespace: str, seed: int, *parts: object, digest_size: int = 16) -> bytes:
    digest = hashlib.blake2b(digest_size=digest_size)
    digest.update(namespace.encode("ascii"))
    digest.update(struct.pack("<q", int(seed)))
    for part in parts:
        if isinstance(part, bytes):
            payload = part
        else:
            payload = str(part).encode("utf-8", errors="replace")
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.digest()


def _int32_bytes(values: Sequence[int]) -> bytes:
    packed = array("i", (int(value) for value in values))
    if packed.itemsize != 4:
        raise RuntimeError("This platform does not provide 32-bit array('i')")
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _int32_tuple(payload: bytes) -> tuple[int, ...]:
    values = array("i")
    values.frombytes(payload)
    if values.itemsize != 4:
        raise RuntimeError("This platform does not provide 32-bit array('i')")
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(int(value) for value in values)


def tensor_payload_sha256(tensors: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes without rereading a saved file."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)))
        byte_view = tensor.view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedDocument:
    """One restartable prepared-document-stream record.

    ``stream_id`` and ``ordinal`` are execution locators only. They are never
    included in logical plan hashes, so changing rank/world-size sharding does
    not change the plan as long as the canonical document set is unchanged.
    """

    stream_id: int
    ordinal: int
    doc_id: str
    source: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class PlanRow:
    pair_id: int
    occurrence_start: int
    doc_hash: bytes
    content_hash: bytes
    source: str
    input_ids_a: tuple[int, ...]
    input_ids_b: tuple[int, ...]
    start_a: int
    start_b: int
    document_token_count: int
    document_reuse_index: int = 0
    execution_rank: int = 0

    def __post_init__(self) -> None:
        if self.pair_id < 0 or self.occurrence_start < 0:
            raise ValueError("pair_id and occurrence_start must be non-negative")
        if self.execution_rank < 0:
            raise ValueError("execution_rank must be non-negative")
        if len(self.doc_hash) != 32 or len(self.content_hash) != 32:
            raise ValueError("doc_hash and content_hash must each contain 32 bytes")
        if not self.input_ids_a or not self.input_ids_b:
            raise ValueError("Both chunks must contain at least one token")
        if self.start_a < 0 or self.start_b != self.start_a + len(self.input_ids_a):
            raise ValueError("A/B chunks must be adjacent and have valid starts")
        if self.start_b + len(self.input_ids_b) > self.document_token_count:
            raise ValueError("Chunk span exceeds document token count")
        if any(not -(2**31) <= int(token) < 2**31 for token in self.input_ids):
            raise ValueError("Token IDs must fit int32")

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.input_ids_a + self.input_ids_b

    @property
    def length_a(self) -> int:
        return len(self.input_ids_a)

    @property
    def length_b(self) -> int:
        return len(self.input_ids_b)

    @property
    def token_count(self) -> int:
        return self.length_a + self.length_b

    @property
    def occurrence_stop(self) -> int:
        return self.occurrence_start + self.token_count


def plan_row_hash(row: PlanRow) -> bytes:
    digest = hashlib.sha256()
    digest.update(PLAN_ROW_HASH_VERSION.encode("ascii"))
    digest.update(
        struct.pack(
            "<QQqqqqq",
            row.pair_id,
            row.occurrence_start,
            row.start_a,
            row.start_b,
            row.length_a,
            row.length_b,
            row.document_token_count,
        )
    )
    digest.update(struct.pack("<qq", row.document_reuse_index, row.execution_rank))
    digest.update(row.doc_hash)
    digest.update(row.content_hash)
    source = row.source.encode("utf-8", errors="replace")
    digest.update(struct.pack("<Q", len(source)))
    digest.update(source)
    digest.update(_int32_bytes(row.input_ids))
    return digest.digest()


def compute_plan_digest(identity: dict, rows_digest: str) -> str:
    digest = hashlib.sha256()
    digest.update(SAMPLE_PLAN_FORMAT.encode("ascii"))
    digest.update(_canonical_json(identity))
    digest.update(bytes.fromhex(rows_digest))
    return digest.hexdigest()


def _subset_sum_indices(weights: Sequence[int], target: int, order: Sequence[int]) -> list[int] | None:
    if target == 0:
        return []
    gcd = math.gcd(*weights)
    if target % gcd:
        return None
    unit_target = target // gcd
    unit_weights = [weight // gcd for weight in weights]
    previous: list[tuple[int, int] | None] = [None] * (unit_target + 1)
    reachable = [False] * (unit_target + 1)
    reachable[0] = True
    for index in order:
        weight = unit_weights[index]
        for total in range(unit_target, weight - 1, -1):
            if not reachable[total] and reachable[total - weight]:
                reachable[total] = True
                previous[total] = (total - weight, index)
    if not reachable[unit_target]:
        return None
    selected: list[int] = []
    cursor = unit_target
    while cursor:
        item = previous[cursor]
        if item is None:
            raise RuntimeError("Broken subset-sum predecessor chain")
        cursor, index = item
        selected.append(index)
    return selected


def _unbounded_sum_indices(weights: Sequence[int], target: int, order: Sequence[int]) -> list[int] | None:
    if target == 0:
        return []
    gcd = math.gcd(*weights)
    if target % gcd:
        return None
    unit_target = target // gcd
    unit_weights = [weight // gcd for weight in weights]
    infinity = unit_target + 1
    best = [infinity] * (unit_target + 1)
    previous: list[tuple[int, int] | None] = [None] * (unit_target + 1)
    best[0] = 0
    for total in range(1, unit_target + 1):
        for index in order:
            weight = unit_weights[index]
            if weight <= total and best[total - weight] + 1 < best[total]:
                best[total] = best[total - weight] + 1
                previous[total] = (total - weight, index)
    if best[unit_target] == infinity:
        return None
    selected: list[int] = []
    cursor = unit_target
    while cursor:
        item = previous[cursor]
        if item is None:
            raise RuntimeError("Broken exact-sum predecessor chain")
        cursor, index = item
        selected.append(index)
    return selected


def _rotated_indices(count: int, seed_material: bytes) -> list[int]:
    if count == 0:
        return []
    offset = int.from_bytes(seed_material[:8], "little") % count
    forward = bool(seed_material[8] & 1)
    values = list(range(count))
    if not forward:
        values.reverse()
    return values[offset:] + values[:offset]


def solve_exact_cell_counts(
    target_tokens: int,
    sources: Sequence[str],
    lengths: Sequence[int],
    *,
    seed: int,
) -> dict[tuple[str, int, int], int]:
    """Return exact source/length-pair counts without rounding the token budget.

    The solver first allocates complete source×length-pair rounds. It then uses
    a deterministic subset sum, preferably independently per source, so formal
    1B-token plans retain exact source balance and residual cells differ by at
    most one. If the requested total is not representable, it fails loudly.
    """

    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    canonical_sources = sorted(set(sources))
    canonical_lengths = sorted(set(int(length) for length in lengths))
    if not canonical_sources:
        raise ValueError("At least one source is required")
    if not canonical_lengths or canonical_lengths[0] <= 0:
        raise ValueError("Chunk lengths must be positive")

    length_pairs = [(length_a, length_b) for length_a in canonical_lengths for length_b in canonical_lengths]
    weights = [length_a + length_b for length_a, length_b in length_pairs]
    gcd = math.gcd(*weights)
    if target_tokens % gcd:
        raise ValueError(
            f"target_tokens={target_tokens} is not representable by chunk lengths "
            f"{canonical_lengths}; token-count gcd is {gcd}"
        )

    counts = {
        (source, length_a, length_b): 0
        for source in canonical_sources
        for length_a, length_b in length_pairs
    }
    per_source_round = sum(weights)

    # Best case: every source receives exactly the same token budget.
    if target_tokens % len(canonical_sources) == 0:
        source_budget = target_tokens // len(canonical_sources)
        source_solutions: dict[str, tuple[int, list[int]]] = {}
        all_sources_solved = True
        for source in canonical_sources:
            rounds, remainder = divmod(source_budget, per_source_round)
            order = _rotated_indices(
                len(weights),
                _keyed_digest("cell-residual-order", seed, source, digest_size=16),
            )
            selected = _subset_sum_indices(weights, remainder, order)
            if selected is None:
                selected = _unbounded_sum_indices(weights, remainder, order)
            if selected is None:
                all_sources_solved = False
                break
            source_solutions[source] = (rounds, selected)
        if all_sources_solved:
            for source, (rounds, selected) in source_solutions.items():
                for length_a, length_b in length_pairs:
                    counts[(source, length_a, length_b)] = rounds
                for index in selected:
                    length_a, length_b = length_pairs[index]
                    counts[(source, length_a, length_b)] += 1
            assert sum((a + b) * count for (_source, a, b), count in counts.items()) == target_tokens
            return counts

    # General fallback: complete global rounds followed by an exact residual.
    cells = [
        (source, length_a, length_b)
        for length_index, (length_a, length_b) in enumerate(length_pairs)
        for source in sorted(
            canonical_sources,
            key=lambda item: _keyed_digest(
                "source-residual-order", seed + length_index, item, digest_size=16
            ),
        )
    ]
    cell_weights = [length_a + length_b for _source, length_a, length_b in cells]
    global_round = sum(cell_weights)
    rounds, remainder = divmod(target_tokens, global_round)
    for cell in counts:
        counts[cell] = rounds
    order = list(range(len(cells)))
    selected = _subset_sum_indices(cell_weights, remainder, order)
    if selected is None:
        selected = _unbounded_sum_indices(cell_weights, remainder, order)
    if selected is None:
        raise ValueError(
            f"target_tokens={target_tokens} cannot be represented exactly by "
            f"source×length cells for lengths={canonical_lengths}"
        )
    for index in selected:
        counts[cells[index]] += 1
    realized = sum((length_a + length_b) * count for (_source, length_a, length_b), count in counts.items())
    if realized != target_tokens:
        raise RuntimeError(f"Exact token solver produced {realized}, expected {target_tokens}")
    return counts


def solve_proportional_cell_counts(
    target_tokens: int,
    sources: Sequence[str],
    lengths: Sequence[int],
    source_token_weights: Mapping[str, int],
    *,
    seed: int,
) -> dict[tuple[str, int, int], int]:
    """Allocate exact pair cells in proportion to externally supplied source mass.

    Pair weights are multiples of the length-set gcd. Source budgets are first
    rounded down to that gcd, then the remaining gcd units are assigned by a
    deterministic largest-remainder rule. Each source budget is solved with
    the same exact cell solver, so the global target remains exact.
    """

    canonical_sources = sorted(set(sources))
    if not canonical_sources:
        raise ValueError("At least one source is required")
    weights = {source: int(source_token_weights.get(source, 0)) for source in canonical_sources}
    if any(value <= 0 for value in weights.values()):
        missing = [source for source, value in weights.items() if value <= 0]
        raise ValueError(f"proportional source weights must be positive: {missing}")
    canonical_lengths = sorted(set(int(length) for length in lengths))
    pair_weights = [a + b for a in canonical_lengths for b in canonical_lengths]
    unit = math.gcd(*pair_weights)
    if target_tokens % unit:
        raise ValueError(
            f"target_tokens={target_tokens} is not representable by chunk lengths; gcd={unit}"
        )
    total_weight = sum(weights.values())
    units_total = target_tokens // unit
    quotients_and_remainders = {
        source: divmod(units_total * value, total_weight)
        for source, value in weights.items()
    }
    base_units = {
        source: quotient
        for source, (quotient, _remainder) in quotients_and_remainders.items()
    }
    remaining = units_total - sum(base_units.values())
    order = sorted(
        canonical_sources,
        key=lambda source: (
            -quotients_and_remainders[source][1],
            _keyed_digest("proportional-source-remainder", seed, source, digest_size=16),
        ),
    )
    for source in order[:remaining]:
        base_units[source] += 1
    budgets = {source: units * unit for source, units in base_units.items()}
    zero_budget_sources = [source for source, budget in budgets.items() if budget == 0]
    if zero_budget_sources:
        raise ValueError(
            "target token budget is too small to represent every proportional source: "
            f"{zero_budget_sources}"
        )
    counts: dict[tuple[str, int, int], int] = {}
    for source in canonical_sources:
        solved = solve_exact_cell_counts(
            budgets[source],
            [source],
            canonical_lengths,
            seed=seed,
        )
        counts.update(solved)
    realized = sum((a + b) * count for (_source, a, b), count in counts.items())
    if realized != target_tokens:
        raise RuntimeError(
            f"proportional source solver produced {realized}, expected {target_tokens}"
        )
    return counts


def _tokenize(tokenizer, text: str) -> list[int]:
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        if isinstance(encoded, Mapping):
            if "input_ids" not in encoded:
                raise ValueError("Tokenizer mapping result has no input_ids field")
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if encoded and isinstance(encoded[0], (list, tuple)):
            if len(encoded) != 1:
                raise ValueError("Tokenizer returned batched input_ids for one document")
            encoded = encoded[0]
        return [int(token) for token in encoded]
    if hasattr(tokenizer, "encode"):
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    raise TypeError("Tokenizer must be callable or provide encode()")


def iter_prepared_document_records(root: str | Path) -> Iterator[PreparedDocument]:
    """Read every prepared document shard, independent of its physical rank."""
    root = Path(root)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("ranks")
    if not isinstance(entries, list):
        entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Prepared document manifest has no ranks/shards: {root}")
    ordered = sorted(entries, key=lambda item: str(item["path"]))

    def records(stream_id: int, item: dict) -> Iterator[PreparedDocument]:
        path = root / item["path"]
        for ordinal, record in enumerate(iter_jsonl(path)):
            text = record.get("text")
            if not isinstance(text, str):
                continue
            digest = str(
                record.get("text_hash")
                or record.get("content_hash")
                or content_hash(text)
            )
            yield PreparedDocument(
                stream_id=stream_id,
                ordinal=ordinal,
                doc_id=str(record.get("doc_id") or f"content:{digest}"),
                source=str(record.get("source") or "unknown"),
                text=text,
                content_hash=digest,
            )

    order_seed = (manifest.get("document_order") or {}).get("seed")
    if order_seed is None:
        for stream_id, item in enumerate(ordered):
            yield from records(stream_id, item)
        return
    streams = [records(stream_id, item) for stream_id, item in enumerate(ordered)]
    yield from heapq.merge(
        *streams,
        key=lambda document: (
            stable_u64(document.content_hash, int(order_seed)),
            document.content_hash,
        ),
    )


class SamplePlanWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        source_to_id: dict[str, int],
        target_tokens: int,
        shard_token_limit: int,
    ) -> None:
        if shard_token_limit <= 0:
            raise ValueError("shard_token_limit must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_to_id = dict(source_to_id)
        self.target_tokens = int(target_tokens)
        self.shard_token_limit = int(shard_token_limit)
        self.rows: list[tuple[PlanRow, bytes]] = []
        self.buffered_tokens = 0
        self.shards: list[dict] = []
        self.pairs = 0
        self.tokens = 0
        self.source_tokens: Counter[str] = Counter()
        self.source_pairs: Counter[str] = Counter()
        self.cell_counts: Counter[tuple[str, int, int]] = Counter()

    def add(self, row: PlanRow) -> bytes:
        if row.source not in self.source_to_id:
            raise ValueError(f"Unknown source in plan row: {row.source}")
        if row.token_count > self.shard_token_limit:
            raise ValueError(
                f"One pair contains {row.token_count} tokens, exceeding shard limit "
                f"{self.shard_token_limit}"
            )
        if self.rows and self.buffered_tokens + row.token_count > self.shard_token_limit:
            self._flush()
        digest = plan_row_hash(row)
        self.rows.append((row, digest))
        self.buffered_tokens += row.token_count
        self.pairs += 1
        self.tokens += row.token_count
        self.source_pairs[row.source] += 1
        self.source_tokens[row.source] += row.token_count
        self.cell_counts[(row.source, row.length_a, row.length_b)] += 1
        return digest

    def _flush(self) -> None:
        if not self.rows:
            return
        rows = [item[0] for item in self.rows]
        row_hashes = [item[1] for item in self.rows]
        input_ids: list[int] = []
        chunk_offsets = [0]
        for row in rows:
            input_ids.extend(row.input_ids_a)
            chunk_offsets.append(len(input_ids))
            input_ids.extend(row.input_ids_b)
            chunk_offsets.append(len(input_ids))
        tensors = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int32),
            "chunk_offsets": torch.tensor(chunk_offsets, dtype=torch.int64),
            "pair_id": torch.tensor([row.pair_id for row in rows], dtype=torch.int64),
            "occurrence_start": torch.tensor(
                [row.occurrence_start for row in rows], dtype=torch.int64
            ),
            "doc_hash": torch.tensor(
                [list(row.doc_hash) for row in rows], dtype=torch.uint8
            ),
            "content_hash": torch.tensor(
                [list(row.content_hash) for row in rows], dtype=torch.uint8
            ),
            "source_id": torch.tensor(
                [self.source_to_id[row.source] for row in rows], dtype=torch.int32
            ),
            "start_a": torch.tensor([row.start_a for row in rows], dtype=torch.int64),
            "start_b": torch.tensor([row.start_b for row in rows], dtype=torch.int64),
            "length_a": torch.tensor([row.length_a for row in rows], dtype=torch.int32),
            "length_b": torch.tensor([row.length_b for row in rows], dtype=torch.int32),
            "document_token_count": torch.tensor(
                [row.document_token_count for row in rows], dtype=torch.int64
            ),
            "document_reuse_index": torch.tensor(
                [row.document_reuse_index for row in rows], dtype=torch.int32
            ),
            "execution_rank": torch.tensor(
                [row.execution_rank for row in rows], dtype=torch.int32
            ),
            "row_hash": torch.tensor([list(value) for value in row_hashes], dtype=torch.uint8),
        }
        payload_sha256 = tensor_payload_sha256(tensors)
        shard_id = len(self.shards)
        final_path = self.root / f"plan-{shard_id:06d}.safetensors"
        partial_path = self.root / f".{final_path.name}.partial"
        partial_path.unlink(missing_ok=True)
        save_file(
            tensors,
            str(partial_path),
            metadata={"format": SAMPLE_PLAN_SHARD_FORMAT},
        )
        os.replace(partial_path, final_path)
        self.shards.append(
            {
                "path": final_path.name,
                "pairs": len(rows),
                "tokens": len(input_ids),
                "pair_id_min": min(row.pair_id for row in rows),
                "pair_id_max": max(row.pair_id for row in rows),
                "payload_sha256": payload_sha256,
                "bytes": final_path.stat().st_size,
            }
        )
        self.rows.clear()
        self.buffered_tokens = 0

    def finish(
        self,
        *,
        identity: dict,
        rows_digest: str,
        extra: dict | None = None,
    ) -> dict:
        self._flush()
        if self.tokens != self.target_tokens:
            raise ValueError(
                f"Refusing to publish inexact plan: realized={self.tokens}, "
                f"target={self.target_tokens}"
            )
        plan_digest = compute_plan_digest(identity, rows_digest)
        manifest = {
            "format": SAMPLE_PLAN_FORMAT,
            "complete": True,
            "identity": identity,
            "plan_digest": plan_digest,
            "rows_digest": rows_digest,
            "target_token_occurrences": self.target_tokens,
            "token_occurrences": self.tokens,
            "pairs": self.pairs,
            "rank_independent": True,
            "occurrence_ids": {
                "start": 0,
                "stop": self.tokens,
                "contiguous": True,
            },
            "source_pairs": dict(sorted(self.source_pairs.items())),
            "source_tokens": dict(sorted(self.source_tokens.items())),
            "cell_counts": {
                f"{source}\t{length_a}\t{length_b}": count
                for (source, length_a, length_b), count in sorted(self.cell_counts.items())
            },
            "shards": self.shards,
            **(extra or {}),
        }
        atomic_json_dump(manifest, self.root / "manifest.json")
        return manifest


def _load_plan_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != SAMPLE_PLAN_SHARD_FORMAT:
            raise ValueError(f"Not a sample-plan v2 shard: {path}")
        return {name: handle.get_tensor(name) for name in handle.keys()}


def load_sample_plan_manifest(
    root: str | Path,
    *,
    verify_logical_coverage: bool = True,
    verify_payload_checksums: bool = False,
) -> dict:
    root = Path(root)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != SAMPLE_PLAN_FORMAT:
        raise ValueError(
            f"Expected {SAMPLE_PLAN_FORMAT}, found {manifest.get('format')!r}; "
            "legacy/v1 activation-cache manifests cannot be used as plans"
        )
    if not manifest.get("complete"):
        raise ValueError(f"Sample plan is incomplete: {root}")
    if manifest.get("token_occurrences") != manifest.get("target_token_occurrences"):
        raise ValueError("Sample plan token count does not exactly match its target")
    if manifest.get("corpus_position_overlap_policy") != "forbidden":
        raise ValueError("Sample plan does not forbid cross-pair corpus-position overlap")
    if manifest.get("corpus_position_overlap_verified") is not True:
        raise ValueError("Sample plan lacks verified non-overlap evidence")
    if manifest.get("unique_corpus_token_positions") != manifest.get(
        "token_occurrences"
    ):
        raise ValueError("Sample plan corpus positions are not one-to-one with occurrences")
    if compute_plan_digest(manifest["identity"], manifest["rows_digest"]) != manifest.get(
        "plan_digest"
    ):
        raise ValueError("Sample plan digest does not match identity/rows digest")
    for item in manifest.get("shards", []):
        if not (root / item["path"]).is_file():
            raise FileNotFoundError(f"Missing sample plan shard: {root / item['path']}")
    if not verify_logical_coverage and not verify_payload_checksums:
        return manifest

    pairs = int(manifest["pairs"])
    seen = torch.zeros(pairs, dtype=torch.bool)
    row_hashes = torch.empty((pairs, 32), dtype=torch.uint8)
    occurrence_starts = torch.empty(pairs, dtype=torch.int64)
    token_counts = torch.empty(pairs, dtype=torch.int64)
    physical_pairs = 0
    physical_tokens = 0
    for item in manifest["shards"]:
        tensors = _load_plan_tensors(root / item["path"])
        if verify_payload_checksums and tensor_payload_sha256(tensors) != item["payload_sha256"]:
            raise ValueError(f"Plan shard payload checksum mismatch: {item['path']}")
        pair_id = tensors["pair_id"].long()
        if pair_id.numel() != int(item["pairs"]):
            raise ValueError(f"Plan shard pair count mismatch: {item['path']}")
        if bool(((pair_id < 0) | (pair_id >= pairs)).any()):
            raise ValueError(f"Out-of-range pair ID in {item['path']}")
        if bool(seen[pair_id].any()):
            raise ValueError(f"Duplicate pair ID in sample plan shard {item['path']}")
        seen[pair_id] = True
        row_hashes[pair_id] = tensors["row_hash"]
        occurrence_starts[pair_id] = tensors["occurrence_start"]
        counts = tensors["length_a"].long() + tensors["length_b"].long()
        token_counts[pair_id] = counts
        physical_pairs += pair_id.numel()
        physical_tokens += int(counts.sum())
    if physical_pairs != pairs or not bool(seen.all()):
        raise ValueError("Sample plan pair IDs do not form a complete bijection")
    expected_starts = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(token_counts[:-1], dim=0)]
    )
    if not torch.equal(occurrence_starts, expected_starts):
        raise ValueError("Sample plan occurrence ranges are not contiguous in pair-id order")
    if physical_tokens != int(manifest["token_occurrences"]):
        raise ValueError("Sample plan physical token total does not match manifest")
    digest = hashlib.sha256()
    digest.update(memoryview(row_hashes.contiguous().numpy()))
    if digest.hexdigest() != manifest["rows_digest"]:
        raise ValueError("Sample plan logical row digest mismatch")
    return manifest


def iter_sample_plan_rows(
    root: str | Path,
    *,
    rank: int | None = None,
    world_size: int = 1,
    verify_rows: bool = True,
) -> Iterator[PlanRow]:
    root = Path(root)
    manifest = load_sample_plan_manifest(root, verify_logical_coverage=False)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank is not None and not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} outside [0, {world_size})")
    planned_world_size = int(
        manifest.get("identity", {}).get("execution_world_size", 1)
    )
    if rank is not None and planned_world_size > 1 and world_size != planned_world_size:
        raise ValueError(
            f"sample plan was balanced for world_size={planned_world_size}, "
            f"but execution requested world_size={world_size}"
        )
    sources = list(manifest["identity"]["sources"])
    # Shards are split by token volume, not logical pair order. Open them lazily
    # using each shard's pair-id lower bound, then merge the sorted streams.
    # This lets the background extraction prefetcher start forwarding before
    # every multi-GB plan shard has been materialized.
    def shard_rows(item: dict) -> Iterator[PlanRow]:
        tensors = _load_plan_tensors(root / item["path"])
        offsets = tensors["chunk_offsets"].long()
        pair_ids = tensors["pair_id"].long()
        ordering = torch.argsort(pair_ids)
        for raw_index in ordering.tolist():
            raw_pair_id = int(pair_ids[raw_index])
            raw_execution_rank = int(tensors["execution_rank"][raw_index])
            if rank is not None:
                assigned_rank = (
                    raw_execution_rank
                    if planned_world_size > 1
                    else raw_pair_id % world_size
                )
                if assigned_rank != rank:
                    continue
            start_a_offset = int(offsets[2 * raw_index])
            start_b_offset = int(offsets[2 * raw_index + 1])
            stop_b_offset = int(offsets[2 * raw_index + 2])
            source_id = int(tensors["source_id"][raw_index])
            row = PlanRow(
                pair_id=raw_pair_id,
                occurrence_start=int(tensors["occurrence_start"][raw_index]),
                doc_hash=bytes(tensors["doc_hash"][raw_index].tolist()),
                content_hash=bytes(tensors["content_hash"][raw_index].tolist()),
                source=sources[source_id],
                input_ids_a=tuple(
                    int(value)
                    for value in tensors["input_ids"][
                        start_a_offset:start_b_offset
                    ].tolist()
                ),
                input_ids_b=tuple(
                    int(value)
                    for value in tensors["input_ids"][
                        start_b_offset:stop_b_offset
                    ].tolist()
                ),
                start_a=int(tensors["start_a"][raw_index]),
                start_b=int(tensors["start_b"][raw_index]),
                document_token_count=int(tensors["document_token_count"][raw_index]),
                document_reuse_index=int(
                    tensors["document_reuse_index"][raw_index]
                ),
                execution_rank=raw_execution_rank,
            )
            if verify_rows:
                expected = bytes(tensors["row_hash"][raw_index].tolist())
                if plan_row_hash(row) != expected:
                    raise ValueError(
                        f"Plan row hash mismatch for pair_id={raw_pair_id}"
                    )
            yield row

    items = list(manifest["shards"])
    if not all("pair_id_min" in item for item in items):
        yield from heapq.merge(
            *(shard_rows(item) for item in items),
            key=lambda row: row.pair_id,
        )
        return

    unopened = iter(
        sorted(
            items,
            key=lambda item: (int(item["pair_id_min"]), str(item["path"])),
        )
    )
    next_item = next(unopened, None)
    active: list[tuple[int, int, PlanRow, Iterator[PlanRow]]] = []
    serial = 0

    def open_item(item: dict) -> None:
        nonlocal serial
        iterator = shard_rows(item)
        first = next(iterator, None)
        if first is not None:
            heapq.heappush(active, (first.pair_id, serial, first, iterator))
            serial += 1

    while next_item is not None or active:
        if not active:
            assert next_item is not None
            open_item(next_item)
            next_item = next(unopened, None)
            continue
        while (
            next_item is not None
            and int(next_item["pair_id_min"]) <= active[0][0]
        ):
            open_item(next_item)
            next_item = next(unopened, None)
            if not active:
                break
        if not active:
            continue
        _pair_id, item_serial, row, iterator = heapq.heappop(active)
        yield row
        following = next(iterator, None)
        if following is not None:
            heapq.heappush(
                active,
                (following.pair_id, item_serial, following, iterator),
            )


def _initialize_workspace(path: Path) -> sqlite3.Connection:
    # Long read-only audits or filesystem stalls must not make the sole planner
    # writer fail immediately at commit time. SQLite's default busy timeout is
    # only five seconds, which is too short for a multi-tens-of-GB planner DB.
    connection = sqlite3.connect(path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1_000)
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-1048576")
    connection.executescript(
        """
        CREATE TABLE documents (
            content_hash BLOB PRIMARY KEY,
            canonical_key BLOB NOT NULL,
            doc_hash BLOB NOT NULL,
            doc_id TEXT NOT NULL,
            source TEXT NOT NULL,
            stream_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            sample_priority BLOB NOT NULL,
            allocated_tokens INTEGER NOT NULL DEFAULT 0,
            selection_count INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID;
        CREATE TABLE selections (
            selection_id INTEGER PRIMARY KEY,
            stream_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            source TEXT NOT NULL,
            content_hash BLOB NOT NULL,
            doc_hash BLOB NOT NULL,
            length_a INTEGER NOT NULL,
            length_b INTEGER NOT NULL,
            reuse_index INTEGER NOT NULL,
            planned_start_a INTEGER,
            execution_rank INTEGER,
            priority BLOB NOT NULL,
            order_key BLOB NOT NULL,
            pair_id INTEGER,
            occurrence_start INTEGER
        );
        CREATE INDEX selections_locator
            ON selections(stream_id, ordinal, selection_id);
        CREATE INDEX selections_content_hash
            ON selections(content_hash);
        CREATE INDEX selections_pair_id
            ON selections(pair_id);

        CREATE TABLE row_hashes (
            pair_id INTEGER PRIMARY KEY,
            row_hash BLOB NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE materialized_rows (
            pair_id INTEGER PRIMARY KEY,
            occurrence_start INTEGER NOT NULL,
            doc_hash BLOB NOT NULL,
            content_hash BLOB NOT NULL,
            source TEXT NOT NULL,
            input_ids_a BLOB NOT NULL,
            input_ids_b BLOB NOT NULL,
            start_a INTEGER NOT NULL,
            start_b INTEGER NOT NULL,
            document_token_count INTEGER NOT NULL,
            document_reuse_index INTEGER NOT NULL,
            execution_rank INTEGER NOT NULL,
            row_hash BLOB NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE materialized_locators (
            stream_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY(stream_id, ordinal)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _commit_with_retry(connection: sqlite3.Connection) -> None:
    """Commit large materialization batches across transient SQLite lock waits."""
    delay = 0.25
    for attempt in range(SQLITE_COMMIT_RETRIES):
        try:
            connection.commit()
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error).lower() or attempt + 1 >= SQLITE_COMMIT_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 8.0)


def _finalize_catalog_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS documents_source_tokens
            ON documents(source, token_count);
        CREATE INDEX IF NOT EXISTS documents_source_priority
            ON documents(source, sample_priority, content_hash);
        """
    )
    connection.commit()


def _batch_token_lengths(tokenizer, texts: Sequence[str]) -> list[int]:
    if not texts:
        return []
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_length=True,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("Batched tokenizer result must be a mapping")
    lengths = encoded.get("length")
    if lengths is None:
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("Batched tokenizer result has no input_ids field")
        lengths = [len(values) for values in input_ids]
    if hasattr(lengths, "tolist"):
        lengths = lengths.tolist()
    result = [int(value) for value in lengths]
    if len(result) != len(texts):
        raise ValueError(
            f"Batched tokenizer returned {len(result)} lengths for {len(texts)} documents"
        )
    return result


def _catalog_worker(
    *,
    document_path: str,
    stream_id: int,
    output_path: str,
    tokenizer_path: str,
    excluded_sources: tuple[str, ...],
    sample_seed: int,
    batch_size: int,
    batch_chars: int,
    tokenizer_threads: int,
    catalog_identity: dict,
) -> dict:
    if tokenizer_threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.model_max_length = 10**12
    output = Path(output_path)
    metadata_path = output.with_name(f"{output.name}.manifest.json")
    metadata_path.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    connection = sqlite3.connect(output)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute(
        """
        CREATE TABLE catalog_rows (
            content_hash BLOB NOT NULL,
            canonical_key BLOB NOT NULL,
            doc_hash BLOB NOT NULL,
            doc_id TEXT NOT NULL,
            source TEXT NOT NULL,
            stream_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            sample_priority BLOB NOT NULL
        )
        """
    )
    excluded_set = set(excluded_sources)
    scanned = 0
    excluded = 0
    inserted = 0
    pending: list[tuple[int, str, str, str, str]] = []
    pending_chars = 0

    def flush() -> None:
        nonlocal inserted, pending_chars
        if not pending:
            return
        lengths = _batch_token_lengths(tokenizer, [item[4] for item in pending])
        rows = []
        for (ordinal, digest, doc_id, source, _text), token_count in zip(
            pending, lengths, strict=True
        ):
            content_digest = _hex_or_sha256(digest)
            doc_digest = hashlib.sha256(
                doc_id.encode("utf-8", errors="replace")
            ).digest()
            canonical_key = hashlib.sha256(
                source.encode("utf-8", errors="replace")
                + b"\0"
                + doc_id.encode("utf-8", errors="replace")
            ).digest()
            sample_priority = _keyed_digest(
                DOCUMENT_PRIORITY_VERSION,
                sample_seed,
                content_digest,
                digest_size=16,
            )
            rows.append(
                (
                    content_digest,
                    canonical_key,
                    doc_digest,
                    doc_id,
                    source,
                    stream_id,
                    ordinal,
                    token_count,
                    sample_priority,
                )
            )
        connection.executemany(
            """
            INSERT INTO catalog_rows
            (content_hash, canonical_key, doc_hash, doc_id, source, stream_id,
             ordinal, token_count, sample_priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
        inserted += len(rows)
        pending.clear()
        pending_chars = 0

    try:
        for ordinal, record in enumerate(iter_jsonl(document_path)):
            text = record.get("text")
            if not isinstance(text, str):
                continue
            scanned += 1
            source = str(record.get("source") or "unknown")
            if source in excluded_set:
                excluded += 1
                continue
            digest = str(
                record.get("text_hash")
                or record.get("content_hash")
                or content_hash(text)
            )
            doc_id = str(record.get("doc_id") or f"content:{digest}")
            text_chars = len(text)
            if pending and (
                len(pending) >= batch_size
                or pending_chars + text_chars > batch_chars
            ):
                flush()
            pending.append((ordinal, digest, doc_id, source, text))
            pending_chars += text_chars
        flush()
    finally:
        connection.close()
    result = {
        "stream_id": stream_id,
        "document_path": document_path,
        "database_path": str(output),
        "documents_scanned": scanned,
        "excluded_records": excluded,
        "rows": inserted,
        "bytes": output.stat().st_size,
    }
    atomic_json_dump(
        {
            "format": CATALOG_WORKER_FORMAT,
            "complete": True,
            "identity": catalog_identity,
            "result": result,
        },
        metadata_path,
    )
    return result


def _load_reusable_catalog_worker(output: Path, expected_identity: dict) -> dict | None:
    metadata_path = output.with_name(f"{output.name}.manifest.json")
    if not output.is_file() or not metadata_path.is_file():
        return None
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        result = metadata["result"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        metadata.get("format") != CATALOG_WORKER_FORMAT
        or metadata.get("complete") is not True
        or metadata.get("identity") != expected_identity
        or int(result.get("bytes", -1)) != output.stat().st_size
    ):
        return None
    reused = dict(result)
    reused["database_path"] = str(output)
    reused["reused"] = True
    return reused


def _catalog_summary(connection: sqlite3.Connection, stats: dict) -> dict:
    corpus_digest = hashlib.sha256()
    source_documents: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    for content_digest, source, token_count in connection.execute(
        "SELECT content_hash, source, token_count FROM documents ORDER BY content_hash"
    ):
        corpus_digest.update(content_digest)
        encoded_source = source.encode("utf-8", errors="replace")
        corpus_digest.update(struct.pack("<Q", len(encoded_source)))
        corpus_digest.update(encoded_source)
        corpus_digest.update(struct.pack("<q", int(token_count)))
        source_documents[source] += 1
        source_tokens[source] += int(token_count)
    unique_documents = int(
        connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    )
    return {
        **stats,
        "unique_documents": unique_documents,
        "corpus_digest": corpus_digest.hexdigest(),
        "source_documents": dict(sorted(source_documents.items())),
        "source_corpus_tokens": dict(sorted(source_tokens.items())),
    }


def _catalog_prepared_documents_parallel(
    connection: sqlite3.Connection,
    document_root: str | Path,
    tokenizer_path: str | Path,
    workspace: Path,
    *,
    excluded_sources: set[str],
    sample_seed: int,
    workers: int,
    batch_size: int,
    batch_chars: int,
    tokenizer_threads: int,
    tokenizer_hash: str,
) -> dict:
    document_root = Path(document_root)
    with (document_root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("complete"):
        raise ValueError(f"Prepared document corpus is incomplete: {document_root}")
    deduplication = manifest.get("deduplication") or {}
    if deduplication.get("scope") != "global" or not manifest.get(
        "deduplicated_globally"
    ):
        raise ValueError(
            "Parallel catalog requires a globally content-deduplicated prepared corpus"
        )
    entries = manifest.get("ranks")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Prepared document manifest has no ranks: {document_root}")
    raw_source_text_bytes = manifest.get("raw_source_text_bytes")
    raw_source_weight_scope = manifest.get("raw_source_weight_scope")
    ordered = sorted(entries, key=lambda item: str(item["path"]))
    workspace.mkdir(parents=True, exist_ok=True)
    resolved_workers = max(1, min(int(workers), len(ordered)))
    results: dict[int, dict] = {}
    pending: list[tuple[int, Path, Path, dict]] = []
    for stream_id, item in enumerate(ordered):
        path = document_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared document shard: {path}")
        output = workspace / f"catalog-rank{stream_id:03d}.sqlite3"
        identity = {
            "format": CATALOG_WORKER_FORMAT,
            "stream_id": int(stream_id),
            "document_path": str(path.resolve()),
            "document_bytes": path.stat().st_size,
            "document_mtime_ns": path.stat().st_mtime_ns,
            "tokenizer_hash": tokenizer_hash,
            "excluded_sources": sorted(excluded_sources),
            "sample_seed": int(sample_seed),
        }
        reusable = _load_reusable_catalog_worker(output, identity)
        if reusable is not None:
            results[stream_id] = reusable
            print(
                "[sample-plan] reusable catalog "
                f"{len(results)}/{len(ordered)}: stream={stream_id} "
                f"rows={reusable['rows']}",
                flush=True,
            )
        else:
            pending.append((stream_id, path, output, identity))

    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(resolved_workers, len(pending)), mp_context=context
        ) as executor:
            futures = {}
            for stream_id, path, output, identity in pending:
                future = executor.submit(
                    _catalog_worker,
                    document_path=str(path),
                    stream_id=stream_id,
                    output_path=str(output),
                    tokenizer_path=str(tokenizer_path),
                    excluded_sources=tuple(sorted(excluded_sources)),
                    sample_seed=sample_seed,
                    batch_size=batch_size,
                    batch_chars=batch_chars,
                    tokenizer_threads=tokenizer_threads,
                    catalog_identity=identity,
                )
                futures[future] = stream_id
            for future in as_completed(futures):
                result = future.result()
                results[int(result["stream_id"])] = result
                print(
                    "[sample-plan] parallel catalog "
                    f"{len(results)}/{len(ordered)} complete: "
                    f"stream={result['stream_id']} rows={result['rows']}",
                    flush=True,
                )

    expected_documents = sum(int(item.get("documents", 0)) for item in ordered)
    scanned = sum(int(item["documents_scanned"]) for item in results.values())
    excluded = sum(int(item["excluded_records"]) for item in results.values())
    rows = sum(int(item["rows"]) for item in results.values())
    if expected_documents and scanned != expected_documents:
        raise ValueError(
            f"Parallel catalog scanned {scanned} documents, expected {expected_documents}"
        )
    if rows + excluded != scanned:
        raise RuntimeError("Parallel catalog row accounting is inconsistent")

    for stream_id in range(len(ordered)):
        result = results[stream_id]
        connection.execute("ATTACH DATABASE ? AS worker_catalog", (result["database_path"],))
        try:
            connection.execute(
                """
                INSERT INTO documents
                (content_hash, canonical_key, doc_hash, doc_id, source, stream_id,
                 ordinal, token_count, sample_priority)
                SELECT content_hash, canonical_key, doc_hash, doc_id, source, stream_id,
                       ordinal, token_count, sample_priority
                FROM worker_catalog.catalog_rows
                """
            )
            connection.commit()
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise ValueError(
                "Prepared corpus violated global content-hash uniqueness during catalog merge"
            ) from error
        finally:
            connection.execute("DETACH DATABASE worker_catalog")

    return _catalog_summary(
        connection,
        {
            "documents_scanned": scanned,
            "excluded_records": excluded,
            "duplicate_content_records": 0,
            "canonical_replacements": 0,
            "raw_source_text_bytes": (
                {
                    str(source): int(value)
                    for source, value in raw_source_text_bytes.items()
                }
                if isinstance(raw_source_text_bytes, dict)
                else None
            ),
            "raw_source_weight_scope": raw_source_weight_scope,
            "parallel_prepared_catalog": {
                "enabled": True,
                "workers": resolved_workers,
                "batch_size": int(batch_size),
                "batch_chars": int(batch_chars),
                "tokenizer_threads_per_worker": int(tokenizer_threads),
                "input_streams": len(ordered),
                "reused_worker_catalogs": sum(
                    bool(item.get("reused")) for item in results.values()
                ),
            },
        },
    )


def _catalog_documents(
    connection: sqlite3.Connection,
    documents_factory: Callable[[], Iterable[PreparedDocument]],
    tokenizer,
    *,
    excluded_sources: set[str],
    sample_seed: int,
) -> dict:
    scanned = 0
    excluded = 0
    duplicates = 0
    canonical_replacements = 0
    for document in documents_factory():
        scanned += 1
        if document.source in excluded_sources:
            excluded += 1
            continue
        content_digest = _hex_or_sha256(document.content_hash)
        doc_digest = hashlib.sha256(
            document.doc_id.encode("utf-8", errors="replace")
        ).digest()
        # Prefer a stable semantic occurrence over physical rank/shard location.
        # Prepared document streams already use content IDs after global dedup,
        # so this keeps the catalog invariant to prepared world size.
        canonical_key = hashlib.sha256(
            document.source.encode("utf-8", errors="replace")
            + b"\0"
            + document.doc_id.encode("utf-8", errors="replace")
        ).digest()
        sample_priority = _keyed_digest(
            DOCUMENT_PRIORITY_VERSION,
            sample_seed,
            content_digest,
            digest_size=16,
        )
        existing = connection.execute(
            "SELECT canonical_key FROM documents WHERE content_hash=?",
            (content_digest,),
        ).fetchone()
        if existing is not None:
            duplicates += 1
            if canonical_key >= existing[0]:
                continue
            replacement_tokens = _tokenize(tokenizer, document.text)
            token_count = len(replacement_tokens)
            connection.execute(
                """
                UPDATE documents
                SET canonical_key=?, doc_hash=?, doc_id=?, source=?, stream_id=?,
                    ordinal=?, token_count=?, sample_priority=?
                WHERE content_hash=?
                """,
                (
                    canonical_key,
                    doc_digest,
                    document.doc_id,
                    document.source,
                    document.stream_id,
                    document.ordinal,
                    token_count,
                    sample_priority,
                    content_digest,
                ),
            )
            canonical_replacements += 1
        else:
            token_count = len(_tokenize(tokenizer, document.text))
            connection.execute(
                """
                INSERT INTO documents
                (content_hash, canonical_key, doc_hash, doc_id, source, stream_id,
                 ordinal, token_count, sample_priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_digest,
                    canonical_key,
                    doc_digest,
                    document.doc_id,
                    document.source,
                    document.stream_id,
                    document.ordinal,
                    token_count,
                    sample_priority,
                ),
            )
        if scanned % 10_000 == 0:
            connection.commit()
    connection.commit()
    return _catalog_summary(
        connection,
        {
            "documents_scanned": scanned,
            "excluded_records": excluded,
            "duplicate_content_records": duplicates,
            "canonical_replacements": canonical_replacements,
            "parallel_prepared_catalog": {"enabled": False},
        },
    )


def _select_documents(
    connection: sqlite3.Connection,
    counts: dict[tuple[str, int, int], int],
    *,
    seed: int,
    max_document_reuses: int,
) -> None:
    """Assign exact cell quotas to seeded-random documents without token reuse.

    Reuse layers are global: every eligible document receives at most one pair
    before any document receives a second. Token spans are assigned separately
    after all cells are known, so the total allocated length is a hard capacity
    constraint rather than an optimistic per-cell check.
    """

    if max_document_reuses <= 0:
        raise ValueError("max_document_reuses must be positive")
    sources = sorted({source for source, _length_a, _length_b in counts})
    for source in sources:
        quotas = {
            (length_a, length_b): quota
            for (cell_source, length_a, length_b), quota in counts.items()
            if cell_source == source and quota > 0
        }
        remaining = dict(quotas)
        assigned = {cell: 0 for cell in quotas}
        pending = sum(remaining.values())
        for reuse_index in range(max_document_reuses):
            if pending == 0:
                break
            inserts: list[tuple] = []
            updates: list[tuple[int, int, bytes]] = []
            for (
                stream_id,
                ordinal,
                content_digest,
                doc_digest,
                token_count,
                allocated_tokens,
                selection_count,
                sample_priority,
            ) in connection.execute(
                """
                SELECT stream_id, ordinal, content_hash, doc_hash, token_count,
                       allocated_tokens, selection_count, sample_priority
                FROM documents
                WHERE source=? AND selection_count<=?
                ORDER BY sample_priority, content_hash
                """,
                (source, reuse_index),
            ):
                available = int(token_count) - int(allocated_tokens)
                eligible = [
                    cell
                    for cell, count in remaining.items()
                    if count > 0 and sum(cell) <= available
                ]
                if not eligible:
                    continue

                def cell_key(cell: tuple[int, int]) -> tuple:
                    quota = quotas[cell]
                    # Long spans are hardest to place. Within a span size, keep
                    # normalized cell progress even and use a counter hash tie.
                    tie = _keyed_digest(
                        "cell-choice-v2",
                        seed,
                        content_digest,
                        reuse_index,
                        cell[0],
                        cell[1],
                        digest_size=8,
                    )
                    return (
                        sum(cell),
                        remaining[cell] / quota,
                        tie,
                    )

                length_a, length_b = max(eligible, key=cell_key)
                span = length_a + length_b
                priority = bytes(sample_priority) + struct.pack(">I", reuse_index)
                order_key = _keyed_digest(
                    "pair-order-v2",
                    seed,
                    content_digest,
                    source,
                    length_a,
                    length_b,
                    reuse_index,
                    digest_size=16,
                )
                inserts.append(
                    (
                        stream_id,
                        ordinal,
                        source,
                        content_digest,
                        doc_digest,
                        length_a,
                        length_b,
                        reuse_index,
                        priority,
                        order_key,
                    )
                )
                updates.append(
                    (int(allocated_tokens) + span, int(selection_count) + 1, content_digest)
                )
                remaining[(length_a, length_b)] -= 1
                assigned[(length_a, length_b)] += 1
                pending -= 1
                if pending == 0:
                    break
                if len(inserts) >= 10_000:
                    connection.executemany(
                        """
                        INSERT INTO selections
                        (stream_id, ordinal, source, content_hash, doc_hash, length_a,
                         length_b, reuse_index, priority, order_key)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        inserts,
                    )
                    connection.executemany(
                        """
                        UPDATE documents
                        SET allocated_tokens=?, selection_count=?
                        WHERE content_hash=?
                        """,
                        updates,
                    )
                    inserts.clear()
                    updates.clear()
            if inserts:
                connection.executemany(
                    """
                    INSERT INTO selections
                    (stream_id, ordinal, source, content_hash, doc_hash, length_a,
                     length_b, reuse_index, priority, order_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    inserts,
                )
                connection.executemany(
                    """
                    UPDATE documents
                    SET allocated_tokens=?, selection_count=?
                    WHERE content_hash=?
                    """,
                    updates,
                )
            connection.commit()

        if pending:
            unsatisfied = {
                f"{a},{b}": count
                for (a, b), count in sorted(remaining.items())
                if count
            }
            raise ValueError(
                f"Insufficient non-overlapping document capacity for source={source!r} "
                f"with max_document_reuses={max_document_reuses}; "
                f"unfilled_cells={unsatisfied}"
            )
        if assigned != quotas:
            raise RuntimeError(f"Exact cell assignment failed for source={source!r}")


def _assign_nonoverlapping_boundaries(
    connection: sqlite3.Connection,
    *,
    seed: int,
) -> dict:
    """Place selected spans in each document and verify a disjoint interval set."""

    selected_documents = 0
    selected_pairs = 0
    selected_tokens = 0
    pair_histogram: Counter[int] = Counter()
    token_histogram: Counter[int] = Counter()
    for content_digest, document_tokens in connection.execute(
        """
        SELECT content_hash, token_count FROM documents
        WHERE selection_count > 0
        ORDER BY content_hash
        """
    ):
        selections = list(
            connection.execute(
                """
                SELECT selection_id, length_a, length_b, reuse_index
                FROM selections WHERE content_hash=?
                """,
                (content_digest,),
            )
        )
        selections.sort(
            key=lambda row: _keyed_digest(
                "span-order-v2",
                seed,
                content_digest,
                row[0],
                row[1],
                row[2],
                row[3],
                digest_size=16,
            )
        )
        total = sum(int(length_a) + int(length_b) for _, length_a, length_b, _ in selections)
        slack = int(document_tokens) - total
        if slack < 0:
            raise RuntimeError("Selected non-overlap spans exceed document token capacity")

        cuts = sorted(
            int.from_bytes(
                _keyed_digest(
                    BOUNDARY_VERSION,
                    seed,
                    content_digest,
                    gap_index,
                    digest_size=16,
                ),
                "big",
            )
            % (slack + 1)
            for gap_index in range(len(selections))
        )
        gaps: list[int] = []
        previous = 0
        for cut in cuts:
            gaps.append(cut - previous)
            previous = cut
        gaps.append(slack - previous)

        cursor = gaps[0]
        updates = []
        for index, (selection_id, length_a, length_b, _reuse_index) in enumerate(selections):
            updates.append((cursor, selection_id))
            cursor += int(length_a) + int(length_b) + gaps[index + 1]
        if cursor != int(document_tokens):
            raise RuntimeError("Non-overlap boundary allocation did not consume document layout")
        connection.executemany(
            "UPDATE selections SET planned_start_a=? WHERE selection_id=?",
            updates,
        )
        selected_documents += 1
        selected_pairs += len(selections)
        selected_tokens += total
        pair_histogram[len(selections)] += 1
        token_histogram[total] += 1
        if selected_documents % 10_000 == 0:
            connection.commit()
    connection.commit()

    previous_hash: bytes | None = None
    previous_stop = -1
    verified_pairs = 0
    for content_digest, start, length_a, length_b in connection.execute(
        """
        SELECT content_hash, planned_start_a, length_a, length_b
        FROM selections
        ORDER BY content_hash, planned_start_a, selection_id
        """
    ):
        if start is None:
            raise RuntimeError("Selection is missing a planned token boundary")
        start = int(start)
        stop = start + int(length_a) + int(length_b)
        if content_digest == previous_hash and start < previous_stop:
            raise RuntimeError(
                "Planner produced overlapping corpus token spans for one content hash"
            )
        previous_hash = content_digest
        previous_stop = stop
        verified_pairs += 1
    if verified_pairs != selected_pairs:
        raise RuntimeError("Non-overlap verification did not inspect every pair")

    def percentile(histogram: Counter[int], fraction: float) -> int:
        total_count = sum(histogram.values())
        if not total_count:
            return 0
        target = int(fraction * (total_count - 1))
        cursor = 0
        for value, count in sorted(histogram.items()):
            cursor += count
            if cursor > target:
                return value
        return max(histogram)

    return {
        "selected_documents": selected_documents,
        "selected_pairs": selected_pairs,
        "unique_corpus_token_positions": selected_tokens,
        "max_pairs_per_document": max(pair_histogram, default=0),
        "max_tokens_per_document": max(token_histogram, default=0),
        "pairs_per_document_p50": percentile(pair_histogram, 0.50),
        "pairs_per_document_p95": percentile(pair_histogram, 0.95),
        "pairs_per_document_p99": percentile(pair_histogram, 0.99),
        "tokens_per_document_p50": percentile(token_histogram, 0.50),
        "tokens_per_document_p95": percentile(token_histogram, 0.95),
        "tokens_per_document_p99": percentile(token_histogram, 0.99),
        "overlap_verified": True,
    }


def _solve_rank_weight_counts(
    weight_counts: dict[int, int],
    *,
    world_size: int,
    target_tokens: int,
) -> dict[int, dict[int, int]]:
    if world_size <= 0:
        raise ValueError("execution_world_size must be positive")
    if target_tokens % world_size:
        raise ValueError(
            f"target_tokens={target_tokens} is not divisible by execution_world_size={world_size}"
        )
    if world_size == 1:
        return {0: dict(weight_counts)}

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    weights = sorted(weight_counts)
    variable_count = world_size * len(weights)
    constraints = []
    lower = []
    upper = []
    for weight_index, weight in enumerate(weights):
        row = np.zeros(variable_count)
        for rank in range(world_size):
            row[rank * len(weights) + weight_index] = 1
        constraints.append(row)
        lower.append(weight_counts[weight])
        upper.append(weight_counts[weight])
    per_rank_tokens = target_tokens // world_size
    for rank in range(world_size):
        row = np.zeros(variable_count)
        for weight_index, weight in enumerate(weights):
            row[rank * len(weights) + weight_index] = weight
        constraints.append(row)
        lower.append(per_rank_tokens)
        upper.append(per_rank_tokens)

    matrix = np.stack(constraints)
    average = np.array(
        [weight_counts[weight] / world_size for _rank in range(world_size) for weight in weights]
    )
    objective = np.array(
        [
            1e-9
            * (1 + rank * len(weights) + weight_index)
            for rank in range(world_size)
            for weight_index in range(len(weights))
        ]
    )
    result = None
    for slack in (1, 2, 4, 8, 16, 64, None):
        if slack is None:
            variable_lower = np.zeros(variable_count)
            variable_upper = np.array(
                [weight_counts[weight] for _rank in range(world_size) for weight in weights],
                dtype=float,
            )
        else:
            variable_lower = np.maximum(0, np.floor(average) - slack)
            variable_upper = np.ceil(average) + slack
        result = milp(
            c=objective,
            integrality=np.ones(variable_count),
            bounds=Bounds(variable_lower, variable_upper),
            constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
            options={"time_limit": 300},
        )
        if result.success:
            break
    if result is None or not result.success or result.x is None:
        raise ValueError(
            "Unable to assign pair weights so every execution rank receives an exact "
            f"{per_rank_tokens} token occurrences: {getattr(result, 'message', 'no result')}"
        )
    rounded = np.rint(result.x).astype(np.int64)
    assignment = {
        rank: {
            weight: int(rounded[rank * len(weights) + weight_index])
            for weight_index, weight in enumerate(weights)
        }
        for rank in range(world_size)
    }
    for weight in weights:
        if sum(assignment[rank][weight] for rank in range(world_size)) != weight_counts[weight]:
            raise RuntimeError("MILP rank assignment violated a weight-count constraint")
    for rank in range(world_size):
        realized = sum(weight * count for weight, count in assignment[rank].items())
        if realized != per_rank_tokens:
            raise RuntimeError("MILP rank assignment violated an exact token constraint")
    return assignment


def _assign_execution_ranks(
    connection: sqlite3.Connection,
    *,
    world_size: int,
    target_tokens: int,
) -> dict:
    weight_counts = {
        int(weight): int(count)
        for weight, count in connection.execute(
            """
            SELECT length_a + length_b AS weight, COUNT(*)
            FROM selections GROUP BY weight ORDER BY weight
            """
        )
    }
    assignment = _solve_rank_weight_counts(
        weight_counts,
        world_size=world_size,
        target_tokens=target_tokens,
    )
    for weight in sorted(weight_counts):
        remaining = {rank: assignment[rank][weight] for rank in range(world_size)}
        rank_order = sorted(
            range(world_size),
            key=lambda rank: _keyed_digest(
                "rank-weight-order-v2", target_tokens, weight, rank, digest_size=8
            ),
        )
        cursor = 0
        updates = []
        for (selection_id,) in connection.execute(
            """
            SELECT selection_id FROM selections
            WHERE length_a + length_b=?
            ORDER BY order_key, content_hash, selection_id
            """,
            (weight,),
        ):
            checked = 0
            while remaining[rank_order[cursor]] == 0:
                cursor = (cursor + 1) % world_size
                checked += 1
                if checked > world_size:
                    raise RuntimeError("Rank weight quotas were exhausted too early")
            rank = rank_order[cursor]
            remaining[rank] -= 1
            updates.append((rank, selection_id))
            cursor = (cursor + 1) % world_size
            if len(updates) >= 10_000:
                connection.executemany(
                    "UPDATE selections SET execution_rank=? WHERE selection_id=?",
                    updates,
                )
                updates.clear()
        if updates:
            connection.executemany(
                "UPDATE selections SET execution_rank=? WHERE selection_id=?",
                updates,
            )
        if any(remaining.values()):
            raise RuntimeError("Rank weight quotas were not completely assigned")
    connection.commit()
    rank_pairs = [0 for _ in range(world_size)]
    rank_tokens = [0 for _ in range(world_size)]
    for rank, pairs, tokens in connection.execute(
        """
        SELECT execution_rank, COUNT(*), SUM(length_a + length_b)
        FROM selections GROUP BY execution_rank ORDER BY execution_rank
        """
    ):
        rank_pairs[int(rank)] = int(pairs)
        rank_tokens[int(rank)] = int(tokens)
    expected = target_tokens // world_size
    if rank_tokens != [expected] * world_size:
        raise RuntimeError(
            f"Execution-rank token totals are not exact: {rank_tokens}, expected {expected}"
        )
    return {
        "world_size": world_size,
        "tokens_per_rank": rank_tokens,
        "pairs_per_rank": rank_pairs,
        "weight_counts_per_rank": {
            str(rank): {str(weight): count for weight, count in sorted(values.items())}
            for rank, values in sorted(assignment.items())
        },
    }


def _assign_pair_ids(connection: sqlite3.Connection) -> tuple[int, int]:
    pair_id = 0
    occurrence_start = 0
    updates = []
    for selection_id, length_a, length_b in connection.execute(
        """
        SELECT selection_id, length_a, length_b
        FROM selections
        ORDER BY order_key, source, length_a, length_b, content_hash, reuse_index
        """
    ):
        updates.append((pair_id, occurrence_start, selection_id))
        occurrence_start += int(length_a) + int(length_b)
        pair_id += 1
        if len(updates) >= 10_000:
            connection.executemany(
                "UPDATE selections SET pair_id=?, occurrence_start=? WHERE selection_id=?",
                updates,
            )
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE selections SET pair_id=?, occurrence_start=? WHERE selection_id=?",
            updates,
        )
    connection.commit()
    return pair_id, occurrence_start


def _materialize_plan(
    connection: sqlite3.Connection,
    documents_factory: Callable[[], Iterable[PreparedDocument]],
    tokenizer,
    writer: SamplePlanWriter,
    *,
    seed: int,
) -> int:
    # Physical prepared streams may be interleaved by a global document
    # priority, so a locator cursor cannot assume monotonically increasing
    # ``(stream_id, ordinal)``. Querying the indexed locator trades a small
    # SQLite lookup for bounded memory even for multi-million-pair plans.
    processed = 0
    pending_rows: list[tuple] = []
    pending_hashes: list[tuple[int, bytes]] = []

    def flush_pending() -> None:
        if not pending_rows:
            return
        connection.executemany(
            """
            INSERT INTO materialized_rows
            (pair_id, occurrence_start, doc_hash, content_hash, source,
             input_ids_a, input_ids_b, start_a, start_b, document_token_count,
             document_reuse_index, execution_rank, row_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pending_rows,
        )
        connection.executemany(
            "INSERT INTO row_hashes(pair_id, row_hash) VALUES (?, ?)",
            pending_hashes,
        )
        _commit_with_retry(connection)
        pending_rows.clear()
        pending_hashes.clear()

    for document in documents_factory():
        locator = (document.stream_id, document.ordinal)
        cursor = connection.execute(
            """
            SELECT selection_id, stream_id, ordinal, source, content_hash,
                   doc_hash, length_a, length_b, reuse_index, planned_start_a,
                   execution_rank, pair_id, occurrence_start
            FROM selections
            WHERE stream_id=? AND ordinal=?
            ORDER BY selection_id
            """,
            locator,
        )
        try:
            selected = cursor.fetchall()
        finally:
            cursor.close()
        if not selected:
            continue
        connection.execute(
            "INSERT INTO materialized_locators(stream_id, ordinal) VALUES (?, ?)",
            locator,
        )

        token_ids = _tokenize(tokenizer, document.text)
        digest = _hex_or_sha256(document.content_hash)
        used_exact_spans: set[tuple[int, int, int]] = set()
        for (
            selection_id,
            _stream_id,
            _ordinal,
            source,
            expected_content_digest,
            doc_digest,
            length_a,
            length_b,
            reuse_index,
            planned_start_a,
            execution_rank,
            pair_id,
            occurrence_start,
        ) in selected:
            if digest != expected_content_digest:
                raise ValueError(
                    f"Content hash changed for selected document at locator={locator}"
                )
            if source != document.source:
                raise ValueError(
                    f"Source changed for selected document at locator={locator}"
                )
            span = int(length_a) + int(length_b)
            if len(token_ids) < span:
                raise ValueError(
                    f"Selected document became too short: tokens={len(token_ids)}, span={span}"
                )
            if planned_start_a is None:
                raise ValueError(f"Selected pair {pair_id} has no planned boundary")
            if execution_rank is None:
                raise ValueError(f"Selected pair {pair_id} has no execution rank")
            start_a = int(planned_start_a)
            exact_key = (start_a, start_a + span, int(selection_id))
            if any(
                start_a < existing_stop and existing_start < start_a + span
                for existing_start, existing_stop, _ in used_exact_spans
            ):
                raise ValueError(
                    f"Overlapping planned spans detected while materializing {locator}"
                )
            used_exact_spans.add(exact_key)
            start_b = start_a + int(length_a)
            row = PlanRow(
                pair_id=int(pair_id),
                occurrence_start=int(occurrence_start),
                doc_hash=bytes(doc_digest),
                content_hash=bytes(expected_content_digest),
                source=source,
                input_ids_a=tuple(token_ids[start_a:start_b]),
                input_ids_b=tuple(token_ids[start_b : start_b + int(length_b)]),
                start_a=start_a,
                start_b=start_b,
                document_token_count=len(token_ids),
                document_reuse_index=int(reuse_index),
                execution_rank=int(execution_rank),
            )
            digest_bytes = plan_row_hash(row)
            pending_rows.append(
                (
                    row.pair_id,
                    row.occurrence_start,
                    row.doc_hash,
                    row.content_hash,
                    row.source,
                    _int32_bytes(row.input_ids_a),
                    _int32_bytes(row.input_ids_b),
                    row.start_a,
                    row.start_b,
                    row.document_token_count,
                    row.document_reuse_index,
                    row.execution_rank,
                    digest_bytes,
                )
            )
            pending_hashes.append((row.pair_id, digest_bytes))
            processed += 1
            if len(pending_rows) >= 2_000:
                flush_pending()
    flush_pending()
    expected_count = int(
        connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0]
    )
    missing = connection.execute(
        """
        SELECT s.stream_id, s.ordinal
        FROM selections AS s
        LEFT JOIN materialized_locators AS m
          ON m.stream_id=s.stream_id AND m.ordinal=s.ordinal
        WHERE m.stream_id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if processed != expected_count or missing is not None:
        raise RuntimeError(
            "One or more selected document locators were never observed during "
            f"materialization (processed={processed}, expected={expected_count}, "
            f"first_missing={missing})"
        )
    next_pair_id = 0
    for values in connection.execute(
        """
        SELECT pair_id, occurrence_start, doc_hash, content_hash, source,
               input_ids_a, input_ids_b, start_a, start_b,
               document_token_count, document_reuse_index, execution_rank,
               row_hash
        FROM materialized_rows ORDER BY pair_id
        """
    ):
        (
            pair_id,
            occurrence_start,
            doc_hash,
            content_hash_value,
            source,
            input_ids_a,
            input_ids_b,
            start_a,
            start_b,
            document_token_count,
            document_reuse_index,
            execution_rank,
            expected_hash,
        ) = values
        if int(pair_id) != next_pair_id:
            raise RuntimeError(
                f"materialized plan pair IDs are not contiguous: "
                f"expected={next_pair_id}, found={pair_id}"
            )
        row = PlanRow(
            pair_id=int(pair_id),
            occurrence_start=int(occurrence_start),
            doc_hash=bytes(doc_hash),
            content_hash=bytes(content_hash_value),
            source=str(source),
            input_ids_a=_int32_tuple(bytes(input_ids_a)),
            input_ids_b=_int32_tuple(bytes(input_ids_b)),
            start_a=int(start_a),
            start_b=int(start_b),
            document_token_count=int(document_token_count),
            document_reuse_index=int(document_reuse_index),
            execution_rank=int(execution_rank),
        )
        if writer.add(row) != bytes(expected_hash):
            raise RuntimeError(f"staged plan row changed for pair_id={pair_id}")
        next_pair_id += 1
    if next_pair_id != expected_count:
        raise RuntimeError(
            f"published {next_pair_id} materialized rows, expected {expected_count}"
        )
    return processed


def _logical_rows_digest(connection: sqlite3.Connection, expected_pairs: int) -> str:
    digest = hashlib.sha256()
    next_pair_id = 0
    for pair_id, row_hash in connection.execute(
        "SELECT pair_id, row_hash FROM row_hashes ORDER BY pair_id"
    ):
        if int(pair_id) != next_pair_id:
            raise ValueError(
                f"Non-contiguous pair IDs: expected {next_pair_id}, found {pair_id}"
            )
        if row_hash is None or len(row_hash) != 32:
            raise ValueError(f"Missing row hash for pair_id={pair_id}")
        digest.update(row_hash)
        next_pair_id += 1
    if next_pair_id != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} logical rows, found {next_pair_id}")
    return digest.hexdigest()


def build_sample_plan(
    documents_factory: Callable[[], Iterable[PreparedDocument]],
    tokenizer,
    output_dir: str | Path,
    *,
    tokenizer_hash: str,
    target_tokens: int,
    lengths: Sequence[int],
    sample_seed: int,
    sources: Sequence[str] | None = None,
    excluded_sources: Iterable[str] = (),
    shard_token_limit: int = 1_000_000,
    max_document_reuses: int = 64,
    execution_world_size: int = 1,
    workspace_dir: str | Path | None = None,
    overwrite: bool = False,
    keep_workspace: bool = False,
    input_provenance: dict | None = None,
    prepared_document_root: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    catalog_workers: int = 1,
    catalog_batch_size: int = 256,
    catalog_batch_chars: int = 2_000_000,
    tokenizer_threads_per_worker: int = 4,
    source_weighting: str = "balanced",
) -> dict:
    """Build a deterministic, exact-token, rank-independent immutable plan.

    The input factory is traversed twice: once to build a disk-backed canonical
    content catalog and once to materialize only selected token spans. This
    avoids retaining 1B Python token objects or the full corpus in memory.
    """

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    canonical_lengths = sorted(set(int(length) for length in lengths))
    requested_sources = sorted(set(sources)) if sources is not None else None
    raw_weight_basis = "raw_utf8_text_bytes_before_dedup_split_filter"
    prepared_source_weights: dict[str, int] | None = None
    if source_weighting == "corpus_proportional":
        if prepared_document_root is None:
            raise ValueError(
                "corpus_proportional weighting requires a prepared corpus manifest "
                "with raw pre-dedup source weights"
            )
        prepared_manifest_path = Path(prepared_document_root) / "manifest.json"
        with prepared_manifest_path.open(encoding="utf-8") as handle:
            prepared_manifest = json.load(handle)
        raw_weights = prepared_manifest.get("raw_source_text_bytes")
        if not isinstance(raw_weights, dict) or not raw_weights:
            raise ValueError(
                "corpus_proportional weighting requires raw_source_text_bytes in "
                f"{prepared_manifest_path}"
            )
        if prepared_manifest.get("raw_source_weight_scope") != (
            "all_input_records_before_dedup_split_filter"
        ):
            raise ValueError(
                "corpus_proportional weighting requires source weights counted over "
                "all input records before dedup, split, and filtering"
            )
        prepared_source_weights = {
            str(source): int(value) for source, value in raw_weights.items()
        }
        if any(value <= 0 for value in prepared_source_weights.values()):
            raise ValueError("raw_source_text_bytes values must all be positive")
    if execution_world_size <= 0:
        raise ValueError("execution_world_size must be positive")
    if catalog_workers <= 0:
        raise ValueError("catalog_workers must be positive")
    if catalog_batch_size <= 0 or catalog_batch_chars <= 0:
        raise ValueError("catalog batch limits must be positive")
    if (prepared_document_root is None) != (tokenizer_path is None):
        raise ValueError(
            "prepared_document_root and tokenizer_path must be provided together"
        )
    if manifest_path.exists() and not overwrite:
        manifest = load_sample_plan_manifest(output_dir)
        identity = manifest["identity"]
        expected = {
            "target_tokens": int(target_tokens),
            "chunk_lengths": canonical_lengths,
            "sample_seed": int(sample_seed),
            "tokenizer_hash": tokenizer_hash,
            "execution_world_size": int(execution_world_size),
            "source_weighting": source_weighting,
        }
        if prepared_source_weights is not None:
            expected["source_weight_basis"] = raw_weight_basis
            expected_sources = requested_sources or sorted(prepared_source_weights)
            expected["source_token_weights"] = {
                source: int(prepared_source_weights.get(source, 0))
                for source in expected_sources
            }
        mismatches = {
            key: {"expected": value, "actual": identity.get(key)}
            for key, value in expected.items()
            if identity.get(key) != value
        }
        if requested_sources is not None and identity.get("sources") != requested_sources:
            mismatches["sources"] = {
                "expected": requested_sources,
                "actual": identity.get("sources"),
            }
        if mismatches:
            raise ValueError(
                "Existing sample plan does not match requested configuration: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return manifest
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    persistent_workspace = workspace_dir is not None
    parent = Path(workspace_dir) if persistent_workspace else output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        parent / "workspace-v1"
        if persistent_workspace
        else Path(tempfile.mkdtemp(prefix=".sample-plan-work-", dir=parent))
    )
    temporary.mkdir(parents=True, exist_ok=True)
    for pattern in ("plan-*.safetensors", ".plan-*.partial"):
        for stale in output_dir.glob(pattern):
            stale.unlink()
    connection: sqlite3.Connection | None = None
    succeeded = False
    try:
        planner_path = temporary / "planner.sqlite3"
        planner_path.unlink(missing_ok=True)
        connection = _initialize_workspace(planner_path)
        if prepared_document_root is not None:
            catalog = _catalog_prepared_documents_parallel(
                connection,
                prepared_document_root,
                tokenizer_path,
                temporary / "parallel-catalog",
                excluded_sources=set(excluded_sources),
                sample_seed=sample_seed,
                workers=catalog_workers,
                batch_size=catalog_batch_size,
                batch_chars=catalog_batch_chars,
                tokenizer_threads=tokenizer_threads_per_worker,
                tokenizer_hash=tokenizer_hash,
            )
        else:
            catalog = _catalog_documents(
                connection,
                documents_factory,
                tokenizer,
                excluded_sources=set(excluded_sources),
                sample_seed=sample_seed,
            )
        _finalize_catalog_indexes(connection)
        discovered_sources = sorted(catalog["source_documents"])
        selected_sources = requested_sources or discovered_sources
        missing = [source for source in selected_sources if source not in catalog["source_documents"]]
        if missing:
            raise ValueError(f"Requested sources have no canonical documents: {missing}")
        if source_weighting not in {"balanced", "corpus_proportional"}:
            raise ValueError(f"unsupported source_weighting={source_weighting!r}")
        if source_weighting == "corpus_proportional":
            source_weights = prepared_source_weights or {}
            missing_raw_weights = [
                source for source in selected_sources if source not in source_weights
            ]
            if missing_raw_weights:
                raise ValueError(
                    "raw Pile source weights are missing requested sources: "
                    f"{missing_raw_weights}"
                )
            counts = solve_proportional_cell_counts(
                target_tokens,
                selected_sources,
                canonical_lengths,
                source_weights,
                seed=sample_seed,
            )
        else:
            source_weights = catalog.get("source_corpus_tokens", {})
            counts = solve_exact_cell_counts(
                target_tokens,
                selected_sources,
                canonical_lengths,
                seed=sample_seed,
            )
        _select_documents(
            connection,
            counts,
            seed=sample_seed,
            max_document_reuses=max_document_reuses,
        )
        overlap_stats = _assign_nonoverlapping_boundaries(
            connection,
            seed=sample_seed,
        )
        rank_stats = _assign_execution_ranks(
            connection,
            world_size=int(execution_world_size),
            target_tokens=int(target_tokens),
        )
        expected_pairs = sum(counts.values())
        pairs, realized_tokens = _assign_pair_ids(connection)
        if pairs != expected_pairs:
            raise RuntimeError(f"Selected {pairs} pairs, expected {expected_pairs}")
        if realized_tokens != target_tokens:
            raise RuntimeError(
                f"Pair-ID assignment realized {realized_tokens} tokens, "
                f"expected exact target {target_tokens}"
            )

        source_to_id = {
            source: index for index, source in enumerate(selected_sources)
        }
        writer = SamplePlanWriter(
            output_dir,
            source_to_id=source_to_id,
            target_tokens=target_tokens,
            shard_token_limit=shard_token_limit,
        )
        processed = _materialize_plan(
            connection,
            documents_factory,
            tokenizer,
            writer,
            seed=sample_seed,
        )
        if processed != expected_pairs:
            raise RuntimeError(f"Materialized {processed} pairs, expected {expected_pairs}")
        rows_digest = _logical_rows_digest(connection, expected_pairs)
        identity = {
            "format_version": SAMPLE_PLAN_FORMAT,
            "target_tokens": int(target_tokens),
            "sample_seed": int(sample_seed),
            "tokenizer_hash": tokenizer_hash,
            "corpus_digest": catalog["corpus_digest"],
            "sources": selected_sources,
            "chunk_lengths": canonical_lengths,
            "document_priority": DOCUMENT_PRIORITY_VERSION,
            "boundary_sampling": BOUNDARY_VERSION,
            "row_hash": PLAN_ROW_HASH_VERSION,
            "source_balanced": source_weighting == "balanced",
            "source_weighting": source_weighting,
            "source_weight_basis": (
                raw_weight_basis
                if source_weighting == "corpus_proportional"
                else "canonical_qwen_token_count_after_dedup_split_filter"
            ),
            "source_token_weights": {
                source: int(source_weights[source]) for source in selected_sources
            },
            "length_pair_balanced": True,
            "rank_assignment_in_plan": True,
            "execution_world_size": int(execution_world_size),
        }
        manifest = writer.finish(
            identity=identity,
            rows_digest=rows_digest,
            extra={
                "catalog": catalog,
                "requested_cell_counts": {
                    f"{source}\t{length_a}\t{length_b}": count
                    for (source, length_a, length_b), count in sorted(counts.items())
                },
                "source_weighting": source_weighting,
                "source_weight_basis": (
                    raw_weight_basis
                    if source_weighting == "corpus_proportional"
                    else "canonical_qwen_token_count_after_dedup_split_filter"
                ),
                "source_token_weights": {
                    source: int(source_weights[source]) for source in selected_sources
                },
                "max_document_reuses": int(max_document_reuses),
                "corpus_position_overlap_policy": "forbidden",
                "corpus_position_overlap_verified": True,
                "unique_corpus_token_positions": int(
                    overlap_stats["unique_corpus_token_positions"]
                ),
                "document_sampling": {
                    "algorithm": "seeded_document_order_reuse_layers",
                    "priority_version": DOCUMENT_PRIORITY_VERSION,
                    "max_document_reuses": int(max_document_reuses),
                    **overlap_stats,
                },
                "execution_rank_assignment": rank_stats,
                "input_provenance": input_provenance or {},
                "physical_shard_order": "pair_id_contiguous",
            },
        )
        # Re-read metadata and logical coverage before publishing success.
        load_sample_plan_manifest(output_dir, verify_logical_coverage=True)
        succeeded = True
        return manifest
    except Exception:
        # A missing manifest means the directory is not a publishable plan.
        manifest_path.unlink(missing_ok=True)
        if persistent_workspace:
            print(
                f"[sample-plan] preserving failed workspace for retry: {temporary}",
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        if connection is not None:
            connection.close()
        if not keep_workspace and (succeeded or not persistent_workspace):
            shutil.rmtree(temporary, ignore_errors=True)
