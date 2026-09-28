#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from chunk_saes.rank_shards import copy_rank_document_shards, prepare_rank_document_shards
from chunk_saes.utils import parse_csv


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Precompute globally deduplicated, content-hash split rank-local document streams "
            "from one JSONL shard or a recursively discovered dataset directory."
        )
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="A .jsonl/.jsonl.zst file or a directory recursively containing such shards.",
    )
    p.add_argument("--output-dir")
    p.add_argument(
        "--splits",
        help="Comma-separated splits to prepare together in one parallel dataset scan.",
    )
    p.add_argument(
        "--output-dirs",
        help="Comma-separated output directories corresponding to --splits.",
    )
    p.add_argument("--copy-from")
    p.add_argument("--model")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--exclude-sources", default="")
    p.add_argument("--min-chars", type=int, default=128)
    p.add_argument(
        "--order-seed",
        type=int,
        help="Seed for content-hash priority ordering (default: split_seed + 29).",
    )
    p.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="Zstd level; set 0 for uncompressed JSONL on fast local storage.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--scan-workers", type=int)
    p.add_argument("--reduce-workers", type=int)
    p.add_argument("--bucket-count", type=int, default=256)
    p.add_argument("--workspace-dir")
    p.add_argument("--keep-workspace", action="store_true")
    p.add_argument(
        "--generic-filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the minimum-character and boilerplate document filters.",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.splits or args.output_dirs:
        if not args.splits or not args.output_dirs:
            raise ValueError("--splits and --output-dirs must be provided together")
        if args.copy_from:
            raise ValueError("--copy-from is only supported for one --output-dir")
        splits = parse_csv(args.splits)
        output_dirs = parse_csv(args.output_dirs)
        if len(splits) != len(output_dirs):
            raise ValueError("--splits and --output-dirs must have the same length")
        if len(set(splits)) != len(splits):
            raise ValueError("--splits must not contain duplicates")
        tokenizer_hash = None
        if args.model:
            from chunk_saes.utils import tokenizer_fingerprint

            tokenizer_hash = tokenizer_fingerprint(args.model)
        from chunk_saes.parallel_corpus import prepare_rank_document_shards_bundle

        manifests = prepare_rank_document_shards_bundle(
            args.dataset,
            dict(zip(splits, output_dirs, strict=True)),
            split_seed=args.split_seed,
            world_size=args.world_size,
            excluded_sources=set(parse_csv(args.exclude_sources)),
            min_chars=args.min_chars,
            compression_level=args.compression_level,
            overwrite=args.overwrite,
            tokenizer_hash=tokenizer_hash,
            order_seed=args.order_seed,
            scan_workers=args.scan_workers,
            reduce_workers=args.reduce_workers,
            bucket_count=args.bucket_count,
            workspace_dir=args.workspace_dir,
            keep_workspace=args.keep_workspace,
            generic_filtering=args.generic_filtering,
        )
        print(
            "[rank-shards] "
            + json.dumps(
                {
                    "outputs": {
                        split: {
                            "output_dir": output_dirs[index],
                            "documents": manifests[split]["documents"],
                            "compressed_bytes": sum(
                                item["compressed_bytes"]
                                for item in manifests[split]["ranks"]
                            ),
                        }
                        for index, split in enumerate(splits)
                    },
                    "dataset_shards": next(iter(manifests.values()))["dataset"][
                        "shard_count"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if not args.output_dir:
        raise ValueError("--output-dir is required unless --splits is used")
    if args.copy_from:
        copied = copy_rank_document_shards(args.copy_from, args.output_dir)
        if args.model:
            from chunk_saes.utils import atomic_json_dump, tokenizer_fingerprint

            expected_hash = tokenizer_fingerprint(args.model)
            existing_hash = copied.get("tokenizer_hash")
            if existing_hash not in (None, expected_hash):
                raise ValueError(
                    f"Copied shard tokenizer hash {existing_hash} does not match {args.model}"
                )
            if existing_hash is None:
                copied["tokenizer_hash"] = expected_hash
                atomic_json_dump(copied, f"{args.output_dir}/manifest.json")
        print(
            "[rank-shards] "
            + json.dumps(
                {
                    "output_dir": args.output_dir,
                    "copied_from": args.copy_from,
                    "documents": copied["documents"],
                    "world_size": copied["world_size"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    tokenizer_hash = None
    if args.model:
        from chunk_saes.utils import tokenizer_fingerprint

        tokenizer_hash = tokenizer_fingerprint(args.model)
    manifest = prepare_rank_document_shards(
        args.dataset,
        args.output_dir,
        split_seed=args.split_seed,
        split=args.split,
        world_size=args.world_size,
        excluded_sources=set(parse_csv(args.exclude_sources)),
        min_chars=args.min_chars,
        compression_level=args.compression_level,
        overwrite=args.overwrite,
        tokenizer_hash=tokenizer_hash,
        order_seed=args.order_seed,
        generic_filtering=args.generic_filtering,
    )
    print(
        "[rank-shards] "
        + json.dumps(
            {
                "output_dir": args.output_dir,
                "documents": manifest["documents"],
                "world_size": manifest["world_size"],
                "dataset_shards": manifest["dataset"]["shard_count"],
                "dataset_fingerprint": manifest["dataset"]["fingerprint"],
                "compressed_bytes": sum(item["compressed_bytes"] for item in manifest["ranks"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
