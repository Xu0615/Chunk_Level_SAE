from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


DATASET_FORMAT = "chunk-saes-jsonl-dataset-v1"
_JSONL_SUFFIXES = (".jsonl", ".jsonl.zst")
_HASH_BLOCK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class DatasetShard:
    path: Path
    relative_path: str
    size: int
    mtime_ns: int
    ctime_ns: int

    def manifest_entry(
        self,
        *,
        include_mtime: bool = True,
        sha256: str | None = None,
    ) -> dict[str, int | str]:
        entry: dict[str, int | str] = {
            "path": self.relative_path,
            "size": self.size,
        }
        if include_mtime:
            entry["mtime_ns"] = self.mtime_ns
            entry["ctime_ns"] = self.ctime_ns
        if sha256 is not None:
            entry["sha256"] = sha256
        return entry


def is_jsonl_shard(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in _JSONL_SUFFIXES)


def discover_dataset_shards(dataset: str | Path) -> tuple[Path, tuple[DatasetShard, ...]]:
    """Resolve one JSONL shard or recursively discover a JSONL dataset directory."""
    requested = Path(dataset).expanduser()
    if not requested.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {requested}")

    resolved = requested.resolve()
    if resolved.is_file():
        if not is_jsonl_shard(resolved):
            raise ValueError(
                f"Dataset file must end in .jsonl or .jsonl.zst, got: {resolved}"
            )
        stat = resolved.stat()
        return (
            resolved.parent,
            (
                DatasetShard(
                    path=resolved,
                    relative_path=resolved.name,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    ctime_ns=stat.st_ctime_ns,
                ),
            ),
        )

    if not resolved.is_dir():
        raise ValueError(f"Dataset path is neither a regular file nor directory: {resolved}")

    paths = sorted(
        (path for path in resolved.rglob("*") if path.is_file() and is_jsonl_shard(path)),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )
    if not paths:
        raise FileNotFoundError(
            f"No .jsonl or .jsonl.zst shards found recursively under: {resolved}"
        )
    shards = []
    for path in paths:
        stat = path.stat()
        shards.append(
            DatasetShard(
                path=path,
                relative_path=path.relative_to(resolved).as_posix(),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,
            )
        )
    return resolved, tuple(shards)


def _metadata_key(shard: DatasetShard) -> str:
    payload = (
        f"{shard.path}\0{shard.size}\0{shard.mtime_ns}\0{shard.ctime_ns}"
    ).encode(
        "utf-8", errors="replace"
    )
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checksum_cache(root: Path) -> dict[str, dict[str, int | str]]:
    cache_path = root / ".chunk_saes_dataset_checksums.json"
    try:
        with cache_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    entries = payload.get("entries")
    return entries if isinstance(entries, dict) else {}


def _write_checksum_cache(
    root: Path,
    entries: dict[str, dict[str, int | str]],
) -> None:
    cache_path = root / ".chunk_saes_dataset_checksums.json"
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{cache_path.name}.", dir=root)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {"format": DATASET_FORMAT, "entries": entries},
                handle,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
        os.replace(temporary, cache_path)
    except OSError:
        # A read-only dataset is valid; only the optional hash cache is lost.
        try:
            os.unlink(temporary)
        except (OSError, UnboundLocalError):
            pass


def _shard_checksums(
    root: Path,
    shards: tuple[DatasetShard, ...],
) -> dict[str, str]:
    cache = _load_checksum_cache(root)
    updated = dict(cache)
    checksums: dict[str, str] = {}
    changed = False
    for shard in shards:
        key = _metadata_key(shard)
        cached = cache.get(shard.relative_path)
        if isinstance(cached, dict) and cached.get("metadata_key") == key:
            checksum = cached.get("sha256")
        else:
            checksum = None
        if not isinstance(checksum, str) or len(checksum) != 64:
            checksum = _sha256_file(shard.path)
            updated[shard.relative_path] = {
                "metadata_key": key,
                "sha256": checksum,
            }
            changed = True
        checksums[shard.relative_path] = checksum
    if changed:
        _write_checksum_cache(root, updated)
    return checksums


def _aggregate_fingerprint(
    shards: tuple[DatasetShard, ...],
    checksums: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{DATASET_FORMAT}\n".encode("ascii"))
    for shard in shards:
        entry = shard.manifest_entry(
            include_mtime=False,
            sha256=checksums[shard.relative_path],
        )
        digest.update(
            json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_fingerprint(dataset: str | Path) -> dict:
    root, shards = discover_dataset_shards(dataset)
    requested = Path(dataset).expanduser().resolve()
    kind = "file" if requested.is_file() else "directory"
    checksums = _shard_checksums(root, shards)
    return {
        "format": DATASET_FORMAT,
        "path": str(requested),
        "root": str(root),
        "kind": kind,
        "recursive": kind == "directory",
        "shard_order": "relative_path_lexicographic",
        "shard_count": len(shards),
        "total_bytes": sum(shard.size for shard in shards),
        "fingerprint_algorithm": "sha256(relative_path,size,content_sha256)",
        "fingerprint": _aggregate_fingerprint(shards, checksums),
        "shards": [
            shard.manifest_entry(sha256=checksums[shard.relative_path])
            for shard in shards
        ],
    }


def iter_dataset_shards(dataset: str | Path) -> Iterator[DatasetShard]:
    _root, shards = discover_dataset_shards(dataset)
    yield from shards
