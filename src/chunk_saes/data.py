from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import zstandard as zstd

from .corpus import iter_dataset_shards
from .utils import (
    clean_document,
    content_hash,
    document_split,
    looks_like_boilerplate,
    pile_source,
    stable_u64,
)

CONTENT_NORMALIZATION = "clean_document+unicode_nfkc+whitespace_casefold-v2"
SPLIT_KEY = "normalized_content_sha256"
RANK_KEY = "normalized_content_sha256"
DEFAULT_DOCUMENT_ORDER_SEED_OFFSET = 29


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate a single JSONL shard.

    Dataset-directory discovery is intentionally handled by
    :func:`iter_dataset_jsonl` so callers that read prepared rank shards retain
    their previous single-file semantics.
    """
    path = Path(path)
    if path.name.lower().endswith(".jsonl.zst"):
        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as stream:
                with io.TextIOWrapper(stream, encoding="utf-8", errors="replace") as text:
                    for line in text:
                        if line.strip():
                            yield json.loads(line)
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_dataset_jsonl(
    dataset: str | Path,
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Yield ``(relative_shard_path, record_ordinal, record)`` in stable order."""
    for shard in iter_dataset_shards(dataset):
        for ordinal, record in enumerate(iter_jsonl(shard.path)):
            yield shard.relative_path, ordinal, record


def normalized_content(text: str) -> str:
    """Return the versioned representation used for exact-content identity."""
    cleaned = clean_document(text)
    return " ".join(unicodedata.normalize("NFKC", cleaned).casefold().split())


def normalized_content_hash(text: str) -> str:
    """Hash cleaned, Unicode-normalized content before split/rank assignment."""
    # Keep the project's public content_hash helper as the SHA-256 primitive,
    # but feed it the stronger v2 normalization defined above.
    return content_hash(normalized_content(text))


def document_order_priority(text_hash: str, order_seed: int) -> int:
    """Return the deterministic seeded priority used by all document streams."""
    return stable_u64(text_hash, order_seed)


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    split: str
    text: str
    text_hash: str


def _candidate_key(
    *,
    source: str,
    shard_path: str,
    ordinal: int,
    text: str,
) -> str:
    """Choose one canonical occurrence without depending on dataset scan order."""
    raw_hash = content_hash(text)
    return f"{source}\0{raw_hash}\0{shard_path}\0{ordinal:020d}"


def _iter_canonical_documents(
    path: str | Path,
    *,
    split_seed: int,
    wanted_split: str,
    excluded_sources: set[str],
    min_chars: int,
    order_seed: int,
    generic_filtering: bool = True,
) -> Iterator[Document]:
    """Globally deduplicate the full dataset, then yield hash-selected documents.

    SQLite provides an external-memory unique index, avoiding one Python set per
    rank and making duplicate resolution global across every input shard.
    """
    if wanted_split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {wanted_split}")

    with tempfile.TemporaryDirectory(prefix="chunk-saes-corpus-") as temp_dir:
        database = sqlite3.connect(str(Path(temp_dir) / "canonical.sqlite3"))
        try:
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                """
                CREATE TABLE canonical_documents (
                    text_hash TEXT PRIMARY KEY,
                    priority_hex TEXT NOT NULL,
                    split TEXT NOT NULL,
                    canonical_key TEXT NOT NULL,
                    source TEXT NOT NULL,
                    excluded INTEGER NOT NULL,
                    text TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            upsert = """
                INSERT INTO canonical_documents(
                    text_hash, priority_hex, split, canonical_key, source, excluded, text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(text_hash) DO UPDATE SET
                    canonical_key=CASE
                        WHEN excluded.canonical_key < canonical_documents.canonical_key
                        THEN excluded.canonical_key
                        ELSE canonical_documents.canonical_key
                    END,
                    source=CASE
                        WHEN excluded.canonical_key < canonical_documents.canonical_key
                        THEN excluded.source
                        ELSE canonical_documents.source
                    END,
                    text=CASE
                        WHEN excluded.canonical_key < canonical_documents.canonical_key
                        THEN excluded.text
                        ELSE canonical_documents.text
                    END,
                    excluded=MAX(canonical_documents.excluded, excluded.excluded)
            """
            pending = 0
            for shard_path, ordinal, record in iter_dataset_jsonl(path):
                raw = record.get("text")
                if not isinstance(raw, str):
                    continue
                source = pile_source(record)
                identity = normalized_content(raw)
                if not identity:
                    continue
                digest = content_hash(identity)
                split_name = document_split(digest, split_seed)
                text = clean_document(raw)
                excluded = source in excluded_sources
                if generic_filtering and not excluded and (
                    len(text) < min_chars or looks_like_boilerplate(text)
                ):
                    continue
                database.execute(
                    upsert,
                    (
                        digest,
                        f"{document_order_priority(digest, order_seed):016x}",
                        split_name,
                        _candidate_key(
                            source=source,
                            shard_path=shard_path,
                            ordinal=ordinal,
                            text=text,
                        ),
                        source,
                        int(excluded),
                        text,
                    ),
                )
                pending += 1
                if pending >= 10_000:
                    database.commit()
                    pending = 0
            database.commit()

            cursor = database.execute(
                """
                SELECT text_hash, source, text
                FROM canonical_documents
                WHERE split = ? AND excluded = 0
                ORDER BY priority_hex, text_hash
                """,
                (wanted_split,),
            )
            for digest, source, text in cursor:
                yield Document(
                    doc_id=f"content:{digest}",
                    source=source,
                    split=wanted_split,
                    text=text,
                    text_hash=digest,
                )
        finally:
            database.close()


def iter_documents(
    path: str | Path,
    *,
    split_seed: int,
    wanted_split: str,
    rank: int = 0,
    world_size: int = 1,
    excluded_sources: set[str] | None = None,
    min_chars: int = 128,
    order_seed: int | None = None,
    generic_filtering: bool = True,
) -> Iterator[Document]:
    for assigned_rank, document in iter_partitioned_documents(
        path,
        split_seed=split_seed,
        wanted_split=wanted_split,
        world_size=world_size,
        excluded_sources=excluded_sources,
        min_chars=min_chars,
        selected_rank=rank,
        order_seed=order_seed,
        generic_filtering=generic_filtering,
    ):
        if assigned_rank != rank:
            raise RuntimeError(f"partitioner returned rank {assigned_rank}, expected {rank}")
        yield document


def iter_partitioned_documents(
    path: str | Path,
    *,
    split_seed: int,
    wanted_split: str,
    world_size: int,
    excluded_sources: set[str] | None = None,
    min_chars: int = 128,
    selected_rank: int | None = None,
    order_seed: int | None = None,
    generic_filtering: bool = True,
) -> Iterator[tuple[int, Document]]:
    """Yield globally deduplicated documents partitioned by content hash.

    Split and rank are both functions of the normalized content hash, so the
    global document set is invariant to metadata IDs, shard boundaries, input
    order, and physical world size. Rank assignment changes only the placement
    of that fixed set.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if selected_rank is not None and not 0 <= selected_rank < world_size:
        raise ValueError(f"selected_rank={selected_rank} outside [0, {world_size})")
    excluded_sources = excluded_sources or set()
    resolved_order_seed = int(
        split_seed + DEFAULT_DOCUMENT_ORDER_SEED_OFFSET
        if order_seed is None
        else order_seed
    )
    for document in _iter_canonical_documents(
        path,
        split_seed=split_seed,
        wanted_split=wanted_split,
        excluded_sources=excluded_sources,
        min_chars=min_chars,
        order_seed=resolved_order_seed,
        generic_filtering=generic_filtering,
    ):
        rank = stable_u64(document.text_hash, split_seed + 17) % world_size
        if selected_rank is not None and rank != selected_rank:
            continue
        yield rank, document


@dataclass
class ChunkPair:
    doc_id: str
    source: str
    input_ids_a: list[int]
    input_ids_b: list[int]
    text_a: str
    text_b: str
    sample_index: int | None = None
    token_seed: int | None = None

    @property
    def token_count(self) -> int:
        return len(self.input_ids_a) + len(self.input_ids_b)


def assign_token_sample(
    pair: ChunkPair,
    *,
    sample_index: int,
    sample_seed: int,
    rank: int,
) -> ChunkPair:
    """Attach a deterministic token-sampling seed to one logical sample.

    The token location is generated immediately before activation indexing.
    This keeps it independent of queue flush order and ``forward_batch_size``.
    """
    pair.sample_index = int(sample_index)
    pair.token_seed = stable_u64(
        f"{rank}\0{sample_index}\0{pair.doc_id}\0{len(pair.input_ids_a)}\0{len(pair.input_ids_b)}",
        sample_seed,
    )
    return pair


def deterministic_token_locations(
    pairs: list[ChunkPair],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reproducible chunk choices and token indices for a pair batch."""
    if any(pair.token_seed is None for pair in pairs):
        raise ValueError("All ChunkPairs must have token_seed assigned")
    choose_b = torch.empty(len(pairs), dtype=torch.bool)
    indices = torch.empty(len(pairs), dtype=torch.long)
    for row, pair in enumerate(pairs):
        seed = int(pair.token_seed)
        from_b = bool(seed & 1)
        length = len(pair.input_ids_b) if from_b else len(pair.input_ids_a)
        choose_b[row] = from_b
        indices[row] = (seed >> 1) % length
    return choose_b, indices


def sample_adjacent_pair(
    token_ids: list[int],
    tokenizer,
    *,
    lengths: list[int],
    generator: torch.Generator,
    doc_id: str,
    source: str,
    len_a: int | None = None,
    len_b: int | None = None,
    decode_text: bool = True,
) -> ChunkPair | None:
    if len_a is None:
        len_a = lengths[int(torch.randint(len(lengths), (), generator=generator))]
    if len_b is None:
        len_b = lengths[int(torch.randint(len(lengths), (), generator=generator))]
    if len(token_ids) < len_a + len_b:
        return None
    low, high = len_a, len(token_ids) - len_b
    boundary = int(torch.randint(low, high + 1, (), generator=generator))
    ids_a = token_ids[boundary - len_a : boundary]
    ids_b = token_ids[boundary : boundary + len_b]
    return ChunkPair(
        doc_id=doc_id,
        source=source,
        input_ids_a=ids_a,
        input_ids_b=ids_b,
        text_a=tokenizer.decode(ids_a, skip_special_tokens=True) if decode_text else "",
        text_b=tokenizer.decode(ids_b, skip_special_tokens=True) if decode_text else "",
    )


def tokenize_document(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
