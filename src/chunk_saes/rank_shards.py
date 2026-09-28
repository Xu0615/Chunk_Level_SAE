from __future__ import annotations

import fcntl
import io
import json
import os
import shutil
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import zstandard as zstd

from .corpus import dataset_fingerprint
from .data import (
    CONTENT_NORMALIZATION,
    DEFAULT_DOCUMENT_ORDER_SEED_OFFSET,
    RANK_KEY,
    SPLIT_KEY,
    Document,
    iter_jsonl,
    iter_partitioned_documents,
)
from .utils import atomic_json_dump


RANK_SHARD_FORMAT = "chunk-saes-document-rank-shards-v2"
DOCUMENT_ORDER_ALGORITHM = "stable_u64(content_hash, order_seed)"


def _dataset_fingerprint(path: str | Path) -> dict:
    """Backward-compatible private alias used by older callers/tests."""
    return dataset_fingerprint(path)


def _expected_configuration(
    dataset: str | Path,
    *,
    split_seed: int,
    split: str,
    world_size: int,
    excluded_sources: set[str],
    min_chars: int,
    tokenizer_hash: str | None,
    order_seed: int | None,
    generic_filtering: bool = True,
) -> dict:
    resolved_order_seed = int(
        split_seed + DEFAULT_DOCUMENT_ORDER_SEED_OFFSET
        if order_seed is None
        else order_seed
    )
    return {
        "format": RANK_SHARD_FORMAT,
        "dataset": _dataset_fingerprint(dataset),
        "split_seed": int(split_seed),
        "split": split,
        "world_size": int(world_size),
        "rank_assignment_seed": int(split_seed + 17),
        "rank_assignment_key": RANK_KEY,
        "excluded_sources": sorted(excluded_sources),
        "min_chars": int(min_chars),
        "generic_filtering": bool(generic_filtering),
        "cleaned_documents": True,
        "content_normalization": CONTENT_NORMALIZATION,
        "split_key": SPLIT_KEY,
        "split_before_rank": True,
        "deduplication": {
            "scope": "global",
            "key": SPLIT_KEY,
            "across_dataset_shards": True,
            "before_split": True,
            "before_rank": True,
            "excluded_source_policy": "exclude_content_hash_if_any_occurrence_excluded",
        },
        "deduplicated_globally": True,
        "deduplicated_per_rank": False,
        "rank_local_order_preserved": False,
        "document_order": {
            "algorithm": DOCUMENT_ORDER_ALGORITHM,
            "seed": resolved_order_seed,
            "tie_breaker": "content_hash",
            "randomized": True,
        },
        "tokenizer_hash": tokenizer_hash,
    }


def validate_rank_shard_manifest(
    manifest: dict,
    dataset: str | Path,
    *,
    split_seed: int,
    split: str,
    world_size: int,
    excluded_sources: set[str],
    min_chars: int = 128,
    tokenizer_hash: str | None = None,
    order_seed: int | None = None,
    generic_filtering: bool = True,
) -> None:
    expected = _expected_configuration(
        dataset,
        split_seed=split_seed,
        split=split,
        world_size=world_size,
        excluded_sources=excluded_sources,
        min_chars=min_chars,
        tokenizer_hash=tokenizer_hash,
        order_seed=order_seed,
        generic_filtering=generic_filtering,
    )
    mismatches = {}
    for key, value in expected.items():
        if key == "tokenizer_hash" and value is None:
            continue
        if manifest.get(key) != value:
            mismatches[key] = {"expected": value, "actual": manifest.get(key)}
    if not manifest.get("complete"):
        mismatches["complete"] = {"expected": True, "actual": manifest.get("complete")}
    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        mismatches["ranks"] = {"expected": world_size, "actual": None if not isinstance(ranks, list) else len(ranks)}
    if mismatches:
        raise ValueError(f"Rank-shard manifest does not match this run: {json.dumps(mismatches, sort_keys=True)}")


def load_rank_shard_manifest(
    root: str | Path,
    dataset: str | Path,
    *,
    split_seed: int,
    split: str,
    world_size: int,
    excluded_sources: set[str],
    min_chars: int = 128,
    tokenizer_hash: str | None = None,
    order_seed: int | None = None,
    generic_filtering: bool = True,
) -> dict:
    root = Path(root)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_rank_shard_manifest(
        manifest,
        dataset,
        split_seed=split_seed,
        split=split,
        world_size=world_size,
        excluded_sources=excluded_sources,
        min_chars=min_chars,
        tokenizer_hash=tokenizer_hash,
        order_seed=order_seed,
        generic_filtering=generic_filtering,
    )
    for item in manifest["ranks"]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing rank-local document shard: {path}")
    return manifest


def copy_rank_document_shards(source_root: str | Path, destination_root: str | Path) -> dict:
    """Copy a complete shard set, using a lock and manifest-last publication."""
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    with (source_root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("complete"):
        raise ValueError(f"Source rank-shard set is incomplete: {source_root}")
    destination_root.mkdir(parents=True, exist_ok=True)
    lock_path = destination_root / ".copy.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        destination_manifest = destination_root / "manifest.json"
        if destination_manifest.exists():
            with destination_manifest.open(encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing == manifest and all(
                (destination_root / item["path"]).is_file() for item in manifest["ranks"]
            ):
                return existing
        for item in manifest["ranks"]:
            source = source_root / item["path"]
            destination = destination_root / item["path"]
            partial = destination.with_name(f".{destination.name}.partial")
            partial.unlink(missing_ok=True)
            shutil.copyfile(source, partial)
            os.replace(partial, destination)
        atomic_json_dump(manifest, destination_manifest)
    return manifest


def iter_rank_document_shard(root: str | Path, rank: int, manifest: dict) -> Iterator[Document]:
    if not 0 <= rank < int(manifest["world_size"]):
        raise ValueError(f"rank={rank} outside [0, {manifest['world_size']})")
    item = manifest["ranks"][rank]
    if int(item["rank"]) != rank:
        raise ValueError(f"Manifest rank ordering mismatch at index {rank}: {item}")
    path = Path(root) / item["path"]
    for record in iter_jsonl(path):
        yield Document(
            doc_id=record["doc_id"],
            source=record["source"],
            split=record["split"],
            text=record["text"],
            text_hash=record["text_hash"],
        )


def prepare_rank_document_shards(
    dataset: str | Path,
    output_dir: str | Path,
    *,
    split_seed: int,
    split: str,
    world_size: int,
    excluded_sources: set[str],
    min_chars: int = 128,
    compression_level: int = 1,
    overwrite: bool = False,
    tokenizer_hash: str | None = None,
    order_seed: int | None = None,
    generic_filtering: bool = True,
) -> dict:
    """Materialize seeded, globally deduplicated rank-local document streams."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".prepare.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists() and not overwrite:
            return load_rank_shard_manifest(
                output_dir,
                dataset,
                split_seed=split_seed,
                split=split,
                world_size=world_size,
                excluded_sources=excluded_sources,
                min_chars=min_chars,
                tokenizer_hash=tokenizer_hash,
                order_seed=order_seed,
                generic_filtering=generic_filtering,
            )

        resolved_order_seed = int(
            split_seed + DEFAULT_DOCUMENT_ORDER_SEED_OFFSET
            if order_seed is None
            else order_seed
        )
        suffix = ".jsonl.zst" if compression_level > 0 else ".jsonl"
        partial_paths = [output_dir / f".rank{rank:03d}{suffix}.partial" for rank in range(world_size)]
        final_paths = [output_dir / f"rank{rank:03d}{suffix}" for rank in range(world_size)]
        for path in partial_paths:
            path.unlink(missing_ok=True)

        counts = [0] * world_size
        text_bytes = [0] * world_size
        try:
            with ExitStack() as stack:
                writers = []
                for path in partial_paths:
                    raw = stack.enter_context(path.open("wb"))
                    if compression_level > 0:
                        compressed = zstd.ZstdCompressor(
                            level=compression_level
                        ).stream_writer(raw, closefd=False)
                        writers.append(
                            stack.enter_context(
                                io.TextIOWrapper(compressed, encoding="utf-8")
                            )
                        )
                    else:
                        writers.append(
                            stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8"))
                        )

                for rank, document in iter_partitioned_documents(
                    dataset,
                    split_seed=split_seed,
                    wanted_split=split,
                    world_size=world_size,
                    excluded_sources=excluded_sources,
                    min_chars=min_chars,
                    order_seed=resolved_order_seed,
                    generic_filtering=generic_filtering,
                ):
                    writers[rank].write(
                        json.dumps(
                            asdict(document),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    counts[rank] += 1
                    text_bytes[rank] += len(
                        document.text.encode("utf-8", errors="replace")
                    )

            for partial, final in zip(partial_paths, final_paths, strict=True):
                os.replace(partial, final)
        finally:
            for path in partial_paths:
                path.unlink(missing_ok=True)

        manifest = {
            **_expected_configuration(
                dataset,
                split_seed=split_seed,
                split=split,
                world_size=world_size,
                excluded_sources=excluded_sources,
                min_chars=min_chars,
                tokenizer_hash=tokenizer_hash,
                order_seed=resolved_order_seed,
                generic_filtering=generic_filtering,
            ),
            "complete": True,
            "documents": sum(counts),
            "text_bytes": sum(text_bytes),
            "compression": {
                "type": "zstd" if compression_level > 0 else "none",
                "level": compression_level,
            },
            "ranks": [
                {
                    "rank": rank,
                    "path": final_paths[rank].name,
                    "documents": counts[rank],
                    "text_bytes": text_bytes[rank],
                    "compressed_bytes": final_paths[rank].stat().st_size,
                }
                for rank in range(world_size)
            ],
        }
        atomic_json_dump(manifest, manifest_path)
        return manifest
