from __future__ import annotations

import fcntl
import hashlib
import io
import json
import multiprocessing as mp
import os
import pickle
import shutil
import struct
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO

import zstandard as zstd

from .corpus import DatasetShard, dataset_fingerprint, discover_dataset_shards
from .data import _candidate_key, normalized_content
from .rank_shards import _expected_configuration, load_rank_shard_manifest
from .utils import (
    atomic_json_dump,
    clean_document,
    content_hash,
    document_split,
    looks_like_boilerplate,
    pile_source,
    stable_u64,
)


PARALLEL_CORPUS_FORMAT = "chunk-saes-parallel-corpus-workspace-v1"
PARALLEL_PREPARATION_ALGORITHM = (
    "parallel-input-shards+seeded-priority-buckets+global-content-dedup-v1"
)
_LENGTH = struct.Struct("<Q")


def _bucket_for_priority(priority: int, bucket_count: int) -> int:
    return (priority * bucket_count) >> 64


def _scan_directory_name(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"{digest}-{Path(relative_path).name}"


def _write_record(handle: BinaryIO, record: tuple) -> None:
    payload = pickle.dumps(record, protocol=5)
    handle.write(_LENGTH.pack(len(payload)))
    handle.write(payload)


def _iter_records(path: Path):
    with path.open("rb") as handle:
        while True:
            header = handle.read(_LENGTH.size)
            if not header:
                return
            if len(header) != _LENGTH.size:
                raise ValueError(f"Truncated corpus spool header: {path}")
            (length,) = _LENGTH.unpack(header)
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(
                    f"Truncated corpus spool record in {path}: expected {length}, "
                    f"found {len(payload)}"
                )
            yield pickle.loads(payload)


def _iter_jsonl_records(path: Path):
    from .data import iter_jsonl

    yield from iter_jsonl(path)


def _scan_one_shard(
    *,
    shard_path: str,
    relative_path: str,
    shard_size: int,
    shard_sha256: str,
    workspace: str,
    excluded_sources: tuple[str, ...],
    min_chars: int,
    order_seed: int,
    bucket_count: int,
    selected_splits: tuple[str, ...],
    split_seed: int,
    generic_filtering: bool,
) -> dict:
    root = Path(workspace) / "scan" / _scan_directory_name(relative_path)
    complete_path = root / "complete.json"
    expected = {
        "format": PARALLEL_CORPUS_FORMAT,
        "relative_path": relative_path,
        "shard_size": int(shard_size),
        "shard_sha256": shard_sha256,
        "excluded_sources": list(excluded_sources),
        "min_chars": int(min_chars),
        "order_seed": int(order_seed),
        "bucket_count": int(bucket_count),
        "selected_splits": list(selected_splits),
        "split_seed": int(split_seed),
        "generic_filtering": bool(generic_filtering),
    }
    try:
        with complete_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("complete") is True and all(
            existing.get(key) == value for key, value in expected.items()
        ):
            files = existing.get("files", [])
            if (
                isinstance(existing.get("raw_source_documents"), dict)
                and isinstance(existing.get("raw_source_text_bytes"), dict)
                and all(
                (root / item["path"]).is_file()
                and (root / item["path"]).stat().st_size == item["bytes"]
                for item in files
                )
            ):
                return existing
    except (FileNotFoundError, json.JSONDecodeError, OSError, KeyError):
        pass

    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    excluded_set = set(excluded_sources)
    handles: dict[int, BinaryIO] = {}
    bucket_records: dict[int, int] = {}
    records_seen = 0
    records_spooled = 0
    excluded_records = 0
    raw_source_documents: Counter[str] = Counter()
    raw_source_text_bytes: Counter[str] = Counter()
    try:
        for ordinal, record in enumerate(_iter_jsonl_records(Path(shard_path))):
            records_seen += 1
            raw = record.get("text")
            if not isinstance(raw, str):
                continue
            source = pile_source(record)
            raw_source_documents[source] += 1
            raw_source_text_bytes[source] += len(
                raw.encode("utf-8", errors="replace")
            )
            identity = normalized_content(raw)
            if not identity:
                continue
            digest = content_hash(identity)
            if document_split(digest, split_seed) not in selected_splits:
                continue
            excluded = source in excluded_set
            text = clean_document(raw)
            if generic_filtering and not excluded and (
                len(text) < min_chars or looks_like_boilerplate(text)
            ):
                continue
            priority = stable_u64(digest, order_seed)
            bucket = _bucket_for_priority(priority, bucket_count)
            handle = handles.get(bucket)
            if handle is None:
                handle = (root / f"bucket{bucket:04d}.bin").open(
                    "wb", buffering=64 * 1024
                )
                handles[bucket] = handle
            if excluded:
                candidate_key = ""
                source_value = ""
                text_value = ""
                excluded_records += 1
            else:
                candidate_key = _candidate_key(
                    source=source,
                    shard_path=relative_path,
                    ordinal=ordinal,
                    text=text,
                )
                source_value = source
                text_value = text
            _write_record(
                handle,
                (
                    bytes.fromhex(digest),
                    excluded,
                    candidate_key,
                    source_value,
                    text_value,
                ),
            )
            records_spooled += 1
            bucket_records[bucket] = bucket_records.get(bucket, 0) + 1
    finally:
        for handle in handles.values():
            handle.close()

    files = [
        {
            "bucket": bucket,
            "path": f"bucket{bucket:04d}.bin",
            "records": bucket_records[bucket],
            "bytes": (root / f"bucket{bucket:04d}.bin").stat().st_size,
        }
        for bucket in sorted(bucket_records)
    ]
    result = {
        **expected,
        "complete": True,
        "records_seen": records_seen,
        "records_spooled": records_spooled,
        "excluded_records": excluded_records,
        "raw_source_documents": dict(sorted(raw_source_documents.items())),
        "raw_source_text_bytes": dict(sorted(raw_source_text_bytes.items())),
        "files": files,
    }
    atomic_json_dump(result, complete_path)
    return result


def _open_document_writer(stack: ExitStack, path: Path, compression_level: int):
    raw = stack.enter_context(path.open("wb"))
    if compression_level <= 0:
        return stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8"))
    compressed = zstd.ZstdCompressor(level=compression_level).stream_writer(
        raw, closefd=False
    )
    return stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8"))


def _reduce_one_bucket(
    *,
    bucket: int,
    scan_roots: tuple[str, ...],
    workspace: str,
    splits: tuple[str, ...],
    split_seed: int,
    order_seed: int,
    world_size: int,
    compression_level: int,
    bucket_count: int,
) -> dict:
    root = Path(workspace) / "reduce" / f"bucket{bucket:04d}"
    complete_path = root / "complete.json"
    expected = {
        "format": PARALLEL_CORPUS_FORMAT,
        "bucket": int(bucket),
        "splits": list(splits),
        "split_seed": int(split_seed),
        "order_seed": int(order_seed),
        "world_size": int(world_size),
        "compression_level": int(compression_level),
        "bucket_count": int(bucket_count),
    }
    try:
        with complete_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("complete") is True and all(
            existing.get(key) == value for key, value in expected.items()
        ):
            if all(
                (root / item["path"]).is_file()
                and (root / item["path"]).stat().st_size == item["compressed_bytes"]
                for split_rows in existing.get("outputs", {}).values()
                for item in split_rows
            ):
                return existing
    except (FileNotFoundError, json.JSONDecodeError, OSError, KeyError):
        pass

    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    # digest -> (tainted, canonical_key, source, cleaned_text)
    canonical: dict[bytes, tuple[bool, str, str, str]] = {}
    input_records = 0
    for scan_root in scan_roots:
        path = Path(scan_root) / f"bucket{bucket:04d}.bin"
        if not path.exists():
            continue
        for digest, excluded, candidate_key, source, text in _iter_records(path):
            input_records += 1
            previous = canonical.get(digest)
            if excluded:
                if previous is None or not previous[0]:
                    canonical[digest] = (True, "", "", "")
                continue
            if previous is None:
                canonical[digest] = (False, candidate_key, source, text)
            elif not previous[0] and candidate_key < previous[1]:
                canonical[digest] = (False, candidate_key, source, text)

    unique_content_hashes = len(canonical)
    rows: dict[str, list[list[tuple[int, str, str, str]]]] = {
        split: [[] for _ in range(world_size)] for split in splits
    }
    tainted = 0
    for digest_bytes, (excluded, _candidate, source, text) in canonical.items():
        if excluded:
            tainted += 1
            continue
        digest = digest_bytes.hex()
        split = document_split(digest, split_seed)
        if split not in rows:
            continue
        priority = stable_u64(digest, order_seed)
        if _bucket_for_priority(priority, bucket_count) != bucket:
            raise RuntimeError(f"Priority bucket mismatch for {digest}")
        rank = stable_u64(digest, split_seed + 17) % world_size
        rows[split][rank].append((priority, digest, source, text))
    del canonical

    suffix = ".jsonl.zst" if compression_level > 0 else ".jsonl"
    outputs: dict[str, list[dict]] = {}
    with ExitStack() as stack:
        for split in splits:
            outputs[split] = []
            for rank in range(world_size):
                path = root / f"{split}.rank{rank:03d}{suffix}"
                writer = _open_document_writer(stack, path, compression_level)
                selected = rows[split][rank]
                selected.sort(key=lambda item: (item[0], item[1]))
                text_bytes = 0
                for _priority, digest, source, text_value in selected:
                    writer.write(
                        json.dumps(
                            {
                                "doc_id": f"content:{digest}",
                                "source": source,
                                "split": split,
                                "text": text_value,
                                "text_hash": digest,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    text_bytes += len(text_value.encode("utf-8", errors="replace"))
                writer.flush()
                outputs[split].append(
                    {
                        "rank": rank,
                        "path": path.name,
                        "documents": len(selected),
                        "text_bytes": text_bytes,
                    }
                )

    for split_rows in outputs.values():
        for item in split_rows:
            item["compressed_bytes"] = (root / item["path"]).stat().st_size
    result = {
        **expected,
        "complete": True,
        "input_records": input_records,
        "unique_content_hashes": unique_content_hashes,
        "tainted_content_hashes": tainted,
        "outputs": outputs,
    }
    atomic_json_dump(result, complete_path)
    return result


def _validate_workspace(path: Path, expected: dict, *, overwrite: bool) -> None:
    config_path = path / "config.json"
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            actual = json.load(handle)
        if actual != expected:
            raise ValueError(
                f"Parallel corpus workspace configuration mismatch: {path}. "
                "Use --overwrite or choose another workspace."
            )
        return
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"Parallel corpus workspace has no valid config: {path}. "
            "Use --overwrite or choose another workspace."
        )
    path.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(expected, config_path)


def _scan_roots(workspace: Path, shards: tuple[DatasetShard, ...]) -> tuple[str, ...]:
    return tuple(
        str(workspace / "scan" / _scan_directory_name(shard.relative_path))
        for shard in shards
    )


def prepare_rank_document_shards_bundle(
    dataset: str | Path,
    outputs: dict[str, str | Path],
    *,
    split_seed: int,
    world_size: int,
    excluded_sources: set[str],
    min_chars: int = 128,
    compression_level: int = 1,
    overwrite: bool = False,
    tokenizer_hash: str | None = None,
    order_seed: int | None = None,
    scan_workers: int | None = None,
    reduce_workers: int | None = None,
    bucket_count: int = 256,
    workspace_dir: str | Path | None = None,
    keep_workspace: bool = False,
    generic_filtering: bool = True,
) -> dict[str, dict]:
    """Prepare multiple content-hash splits in one parallel dataset scan."""
    if not outputs:
        raise ValueError("At least one split output is required")
    if any(split not in {"train", "validation", "test"} for split in outputs):
        raise ValueError(f"Unsupported splits: {sorted(outputs)}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if bucket_count <= 0 or bucket_count > 4096:
        raise ValueError(f"bucket_count must be in [1, 4096], got {bucket_count}")

    resolved_order_seed = int(split_seed + 29 if order_seed is None else order_seed)
    output_paths = {split: Path(path) for split, path in outputs.items()}
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    locks = ExitStack()
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        for path in sorted(output_paths.values(), key=lambda item: str(item.resolve())):
            lock = locks.enter_context((path / ".prepare.lock").open("a+b"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        existing: dict[str, dict] = {}
        if not overwrite:
            for split, path in output_paths.items():
                try:
                    existing[split] = load_rank_shard_manifest(
                        path,
                        dataset,
                        split_seed=split_seed,
                        split=split,
                        world_size=world_size,
                        excluded_sources=excluded_sources,
                        min_chars=min_chars,
                        tokenizer_hash=tokenizer_hash,
                        order_seed=resolved_order_seed,
                        generic_filtering=generic_filtering,
                    )
                except FileNotFoundError:
                    pass
            if len(existing) == len(outputs):
                return existing

        dataset_manifest = dataset_fingerprint(dataset)
        _root, shards = discover_dataset_shards(dataset)
        shard_entries = {
            item["path"]: item for item in dataset_manifest["shards"]
        }
        splits = tuple(sorted(outputs))
        workspace_config = {
            "format": PARALLEL_CORPUS_FORMAT,
            "algorithm": PARALLEL_PREPARATION_ALGORITHM,
            "dataset": dataset_manifest,
            "splits": list(splits),
            "split_seed": int(split_seed),
            "world_size": int(world_size),
            "excluded_sources": sorted(excluded_sources),
            "min_chars": int(min_chars),
            "generic_filtering": bool(generic_filtering),
            "compression_level": int(compression_level),
            "order_seed": resolved_order_seed,
            "bucket_count": int(bucket_count),
        }
        if workspace_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="chunk-saes-parallel-corpus-")
            workspace = Path(temporary.name)
        else:
            workspace = Path(workspace_dir)
        _validate_workspace(workspace, workspace_config, overwrite=overwrite)

        resolved_scan_workers = max(
            1,
            min(
                len(shards),
                int(scan_workers or min(32, os.cpu_count() or 1)),
            ),
        )
        scan_manifests: dict[str, dict] = {}
        process_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=resolved_scan_workers, mp_context=process_context
        ) as executor:
            futures = {}
            for shard in shards:
                entry = shard_entries[shard.relative_path]
                future = executor.submit(
                    _scan_one_shard,
                    shard_path=str(shard.path),
                    relative_path=shard.relative_path,
                    shard_size=shard.size,
                    shard_sha256=str(entry["sha256"]),
                    workspace=str(workspace),
                    excluded_sources=tuple(sorted(excluded_sources)),
                    min_chars=min_chars,
                    order_seed=resolved_order_seed,
                    bucket_count=bucket_count,
                    selected_splits=splits,
                    split_seed=split_seed,
                    generic_filtering=generic_filtering,
                )
                futures[future] = shard.relative_path
            for future in as_completed(futures):
                relative_path = futures[future]
                result = future.result()
                scan_manifests[relative_path] = result
                print(
                    "[rank-shards-parallel] scan "
                    f"{len(scan_manifests)}/{len(shards)} complete: {relative_path} "
                    f"records={result['records_spooled']}",
                    flush=True,
                )

        scan_roots = _scan_roots(workspace, shards)
        resolved_reduce_workers = max(
            1,
            min(
                bucket_count,
                int(reduce_workers or min(8, os.cpu_count() or 1)),
            ),
        )
        reductions: dict[int, dict] = {}
        with ProcessPoolExecutor(
            max_workers=resolved_reduce_workers, mp_context=process_context
        ) as executor:
            futures = {
                executor.submit(
                    _reduce_one_bucket,
                    bucket=bucket,
                    scan_roots=scan_roots,
                    workspace=str(workspace),
                    splits=splits,
                    split_seed=split_seed,
                    order_seed=resolved_order_seed,
                    world_size=world_size,
                    compression_level=compression_level,
                    bucket_count=bucket_count,
                ): bucket
                for bucket in range(bucket_count)
            }
            for future in as_completed(futures):
                bucket = futures[future]
                reductions[bucket] = future.result()
                completed = len(reductions)
                if completed == bucket_count or completed % max(1, bucket_count // 20) == 0:
                    print(
                        "[rank-shards-parallel] reduce "
                        f"{completed}/{bucket_count} buckets complete",
                        flush=True,
                    )

        raw_source_documents: Counter[str] = Counter()
        raw_source_text_bytes: Counter[str] = Counter()
        for item in scan_manifests.values():
            raw_source_documents.update(
                {key: int(value) for key, value in item["raw_source_documents"].items()}
            )
            raw_source_text_bytes.update(
                {key: int(value) for key, value in item["raw_source_text_bytes"].items()}
            )

        manifests = dict(existing)
        suffix = ".jsonl.zst" if compression_level > 0 else ".jsonl"
        for split, output_dir in output_paths.items():
            if split in existing and not overwrite:
                continue
            counts = [0] * world_size
            text_bytes = [0] * world_size
            final_paths = [
                output_dir / f"rank{rank:03d}{suffix}" for rank in range(world_size)
            ]
            for rank, final_path in enumerate(final_paths):
                partial = final_path.with_name(f".{final_path.name}.partial")
                partial.unlink(missing_ok=True)
                with partial.open("wb") as destination:
                    for bucket in range(bucket_count):
                        item = reductions[bucket]["outputs"][split][rank]
                        fragment = (
                            workspace
                            / "reduce"
                            / f"bucket{bucket:04d}"
                            / item["path"]
                        )
                        with fragment.open("rb") as source:
                            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                        counts[rank] += int(item["documents"])
                        text_bytes[rank] += int(item["text_bytes"])
                os.replace(partial, final_path)

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
                "raw_source_documents": dict(sorted(raw_source_documents.items())),
                "raw_source_text_bytes": dict(sorted(raw_source_text_bytes.items())),
                "raw_source_weight_scope": "all_input_records_before_dedup_split_filter",
                "compression": {
                    "type": "zstd" if compression_level > 0 else "none",
                    "level": compression_level,
                    "concatenated_frames": compression_level > 0,
                },
                "preparation": {
                    "algorithm": PARALLEL_PREPARATION_ALGORITHM,
                    "dataset_scan_passes": 1,
                    "bundled_splits": list(splits),
                    "scan_workers": resolved_scan_workers,
                    "reduce_workers": resolved_reduce_workers,
                    "priority_buckets": bucket_count,
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
            atomic_json_dump(manifest, output_dir / "manifest.json")
            manifests[split] = manifest

        if workspace_dir is not None and not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        return manifests
    finally:
        locks.close()
        if temporary is not None:
            temporary.cleanup()
