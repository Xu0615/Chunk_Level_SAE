#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

from chunk_saes.utils import atomic_json_dump


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mirror a complete local activation cache with manifest-last publication."
    )
    p.add_argument("--source", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--block-size-mib", type=int, default=16)
    p.add_argument("--verify-destination", action="store_true")
    return p


def _file_sha256(path: Path, block_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_if_present(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _same_cache_identity(candidate: dict | None, manifest: dict) -> bool:
    return bool(
        isinstance(candidate, dict)
        and candidate.get("activation_digest") == manifest.get("activation_digest")
        and candidate.get("plan_rows_digest") == manifest.get("plan_rows_digest")
    )


def _copy_one(
    source_root: Path,
    destination_root: Path,
    item: dict,
    *,
    block_size: int,
    verify_destination: bool,
    reuse_existing: bool,
) -> dict:
    source = source_root / item["path"]
    destination = destination_root / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        reuse_existing
        and destination.is_file()
        and destination.stat().st_size == int(item["bytes"])
    ):
        return {
            "path": item["path"],
            "bytes": int(item["bytes"]),
            "status": "reused_by_size",
        }
    partial = destination.with_name(f".{destination.name}.partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as source_handle, partial.open("wb") as output:
        for block in iter(lambda: source_handle.read(block_size), b""):
            output.write(block)
            digest.update(block)
            copied += len(block)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != int(item["bytes"]):
        partial.unlink(missing_ok=True)
        raise IOError(f"cache mirror size mismatch: {source} -> {destination}")
    source_file_sha256 = digest.hexdigest()
    if verify_destination:
        destination_digest = _file_sha256(partial, block_size)
        if destination_digest != source_file_sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(f"cache mirror byte checksum mismatch: {source}")
    os.replace(partial, destination)
    return {
        "path": item["path"],
        "bytes": copied,
        "status": "copied",
        "source_file_sha256": source_file_sha256,
        "destination_rehashed": bool(verify_destination),
    }


def main() -> None:
    args = parser().parse_args()
    source_root = Path(args.source)
    destination_root = Path(args.destination)
    manifest = json.load((source_root / "manifest.json").open(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"source activation cache is incomplete: {source_root}")
    destination_root.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = destination_root / "manifest.json"
    progress_path = destination_root / ".mirror_in_progress.json"
    existing_manifest = _load_json_if_present(existing_manifest_path)
    existing_progress = _load_json_if_present(progress_path)
    reuse_existing = bool(
        (
            isinstance(existing_manifest, dict)
            and existing_manifest.get("complete") is True
            and _same_cache_identity(existing_manifest, manifest)
        )
        or _same_cache_identity(existing_progress, manifest)
    )
    (destination_root / "manifest.json").unlink(missing_ok=True)
    items = [item for rank in manifest["ranks"] for item in rank["shards"]]
    streaming_universe = source_root / "streaming_universe.json"
    if streaming_universe.is_file():
        items.append(
            {
                "path": "streaming_universe.json",
                "bytes": streaming_universe.stat().st_size,
            }
        )
    atomic_json_dump(
        {
            "format": "chunk-saes-activation-cache-mirror-progress-v1",
            "source": str(source_root.resolve()),
            "destination": str(destination_root.resolve()),
            "activation_digest": manifest.get("activation_digest"),
            "plan_rows_digest": manifest.get("plan_rows_digest"),
            "files_expected": len(items),
        },
        progress_path,
    )
    block_size = max(1, int(args.block_size_mib)) * 1024 * 1024
    copied_records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                _copy_one,
                source_root,
                destination_root,
                item,
                block_size=block_size,
                verify_destination=args.verify_destination,
                reuse_existing=reuse_existing,
            )
            for item in items
        ]
        for index, future in enumerate(futures, start=1):
            copied_records.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"[cache-mirror] {index}/{len(futures)} shards verified", flush=True)
    for rank in manifest["ranks"]:
        source_rank_manifest = source_root / f"rank{int(rank['rank']):03d}" / "manifest.json"
        destination_rank_manifest = (
            destination_root / f"rank{int(rank['rank']):03d}" / "manifest.json"
        )
        destination_rank_manifest.parent.mkdir(parents=True, exist_ok=True)
        destination_rank_manifest.write_bytes(source_rank_manifest.read_bytes())
    atomic_json_dump(
        {
            "format": "chunk-saes-activation-cache-mirror-v1",
            "complete": True,
            "source": str(source_root.resolve()),
            "destination": str(destination_root.resolve()),
            "files": copied_records,
            "verification": (
                "source-stream-sha256-and-destination-reread"
                if args.verify_destination
                else "source-stream-sha256-size-fsync-atomic-rename"
            ),
        },
        destination_root / "mirror_manifest.json",
    )
    atomic_json_dump(manifest, destination_root / "manifest.json")
    progress_path.unlink(missing_ok=True)
    print(f"[cache-mirror] complete: {destination_root}", flush=True)


if __name__ == "__main__":
    main()
