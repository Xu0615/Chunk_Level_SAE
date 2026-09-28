#!/usr/bin/env python
from __future__ import annotations

import argparse
import heapq
import json
import re
import shutil
from pathlib import Path

from datasets import load_dataset

from chunk_saes.artifacts import (
    ensure_reusable_artifact,
    file_record,
    write_artifact_manifest,
)
from chunk_saes.data import normalized_content_hash
from chunk_saes.utils import content_hash


BENCHMARK_FORMAT = "chunk-saes-arxiv-benchmark-v2"
BENCHMARK_SPLIT_ALGORITHM = (
    "global-normalized-content-dedup+content-hash-pre-ood-split+first-submission-year-ood-v2"
)


COARSE = {
    "cs": "CS",
    "math": "Mathematics",
    "stat": "Statistics",
    "physics": "Physics",
    "astro-ph": "Physics",
    "cond-mat": "Physics",
    "gr-qc": "Physics",
    "hep-ex": "Physics",
    "hep-lat": "Physics",
    "hep-ph": "Physics",
    "hep-th": "Physics",
    "math-ph": "Physics",
    "nlin": "Physics",
    "nucl-ex": "Physics",
    "nucl-th": "Physics",
    "quant-ph": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "econ": "Economics",
    "eess": "Electrical Engineering",
}


def coarse_label(categories: str) -> str | None:
    for category in str(categories).split():
        prefix = category.split(".", 1)[0]
        if category in COARSE:
            return COARSE[category]
        if prefix in COARSE:
            return COARSE[prefix]
    return None


def year_of(value: str) -> int:
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else 0


def publication_year(record: dict) -> int:
    versions = record.get("versions") or []
    if versions and isinstance(versions[0], dict):
        year = year_of(versions[0].get("created", ""))
        if year:
            return year
    arxiv_id = str(record.get("id", ""))
    match = re.match(r"(\d{2})(?:0[1-9]|1[0-2])\.", arxiv_id)
    if match:
        short = int(match.group(1))
        return 1900 + short if short >= 91 else 2000 + short
    return year_of(record.get("update_date", ""))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a bounded, decontaminated ArXiv probe split.")
    p.add_argument("--dataset", default="librarian-bots/arxiv-metadata-snapshot")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-per-class", type=int, default=1024)
    p.add_argument("--validation-per-class", type=int, default=256)
    p.add_argument("--test-per-class", type=int, default=256)
    p.add_argument("--ood-per-class", type=int, default=256)
    p.add_argument("--ood-year", type=int, default=2023)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--reuse-existing-splits", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def add_reservoir(heaps: dict[tuple[str, str], list], key: tuple[str, str], row: dict, quota: int, score: int) -> None:
    heap = heaps.setdefault(key, [])
    item = (-score, row)
    if len(heap) < quota:
        heapq.heappush(heap, item)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _read_and_validate_existing(output_dir: Path, quotas: dict[str, int]) -> tuple[dict, int]:
    counts: dict[str, dict[str, int]] = {}
    ids_by_split: dict[str, set[str]] = {}
    total = 0
    labels = set(COARSE.values())
    for split, quota in quotas.items():
        path = output_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        row_ids = [str(row["id"]) for row in rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError(f"ArXiv split {split} contains duplicate IDs")
        ids_by_split[split] = set(row_ids)
        content_ids = [normalized_content_hash(str(row["text"])) for row in rows]
        if len(content_ids) != len(set(content_ids)):
            raise ValueError(f"ArXiv split {split} contains duplicate normalized content")
        ids_by_split[f"{split}:content"] = set(content_ids)
        counts[split] = {
            label: sum(row.get("label") == label for row in rows)
            for label in sorted(labels)
        }
        if any(counts[split][label] != quota for label in labels):
            raise ValueError(
                f"ArXiv split {split} does not have exactly {quota} rows per class: "
                f"{counts[split]}"
            )
        total += len(rows)
    for index, left in enumerate(quotas):
        for right in list(quotas)[index + 1 :]:
            if not ids_by_split[left].isdisjoint(ids_by_split[right]):
                raise ValueError(f"ArXiv IDs overlap between {left} and {right}")
            if not ids_by_split[f"{left}:content"].isdisjoint(
                ids_by_split[f"{right}:content"]
            ):
                raise ValueError(
                    f"ArXiv normalized content overlaps between {left} and {right}"
                )
    return counts, total


def _publish_manifest(
    args,
    output_dir: Path,
    quotas: dict[str, int],
    counts: dict,
    scanned_records: int | None,
) -> None:
    write_artifact_manifest(
        {
            "format": BENCHMARK_FORMAT,
            "complete": True,
            "identity": {
                "dataset": args.dataset,
                "ood_year": args.ood_year,
                "quotas": quotas,
                "seed": args.seed,
                "class_mapping": COARSE,
                "split_algorithm": BENCHMARK_SPLIT_ALGORITHM,
            },
            "dataset": args.dataset,
            "ood_year": args.ood_year,
            "class_mapping": COARSE,
            "quotas": quotas,
            "counts": counts,
            "scanned_records": scanned_records,
            "decontamination": (
                "Benchmark splits are content-deduplicated from each other, but the "
                "v3 SAE corpus includes the Pile ArXiv subset; downstream probe "
                "results may therefore reflect domain pre-exposure."
            ),
            "sae_training_source_overlap": "Pile ArXiv included in v3 training corpus",
            "split_identity": BENCHMARK_SPLIT_ALGORITHM,
            "deduplication": {
                "scope": "global",
                "key": "normalized_content_sha256",
                "before_split": True,
            },
            "files": {
                split: file_record(
                    output_dir / f"{split}.jsonl", relative_to=output_dir
                )
                for split in quotas
            },
        },
        output_dir / "manifest.json",
    )


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir)
    quotas = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
        "ood": args.ood_per_class,
    }
    expected_identity = {
        "dataset": args.dataset,
        "ood_year": args.ood_year,
        "quotas": quotas,
        "seed": args.seed,
        "class_mapping": COARSE,
        "split_algorithm": BENCHMARK_SPLIT_ALGORITHM,
    }
    marker = output_dir / "manifest.json"
    if marker.exists() and not args.overwrite:
        try:
            existing = ensure_reusable_artifact(
                marker,
                expected_format=BENCHMARK_FORMAT,
                expected_identity=expected_identity,
            )
        except ValueError:
            if not args.reuse_existing_splits:
                raise
        else:
            print(json.dumps(existing["counts"], indent=2, sort_keys=True), flush=True)
            return
    if args.reuse_existing_splits and all(
        (output_dir / f"{split}.jsonl").is_file() for split in quotas
    ):
        counts, total = _read_and_validate_existing(output_dir, quotas)
        _publish_manifest(args, output_dir, quotas, counts, scanned_records=None)
        print(
            json.dumps({"counts": counts, "validated_existing_rows": total}, indent=2, sort_keys=True),
            flush=True,
        )
        return
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heaps: dict[tuple[str, str], list] = {}
    seen_content: set[str] = set()
    # Downloaded Parquet is memory-mapped; avoiding the streaming worker also
    # avoids pyarrow background-thread finalization crashes after a full scan.
    stream = load_dataset(args.dataset, split="train", streaming=False)
    seen = 0
    for record in stream:
        label = coarse_label(record.get("categories", ""))
        if label is None:
            continue
        year = publication_year(record)
        text = f"Title: {str(record.get('title', '')).strip()}\n\nAbstract: {str(record.get('abstract', '')).strip()}"
        text_hash = normalized_content_hash(text)
        if text_hash in seen_content:
            continue
        seen_content.add(text_hash)
        if year >= args.ood_year:
            split = "ood"
        else:
            # Content-hash split keeps different ArXiv IDs with identical text together.
            bucket = int(text_hash[:8], 16) % 10
            split = "validation" if bucket == 8 else "test" if bucket == 9 else "train"
        quota = quotas[split]
        row = {
            "id": str(record.get("id", seen)),
            "text": text,
            "label": label,
            "year": year,
            "categories": str(record.get("categories", "")),
            "content_hash": text_hash,
        }
        score = int(content_hash(text_hash + f"\0{args.seed}")[:16], 16)
        add_reservoir(heaps, (split, label), row, quota, score)
        seen += 1
        if seen % 100_000 == 0:
            print(f"scanned {seen} records", flush=True)

    counts = {}
    for split in quotas:
        rows = []
        for (row_split, _label), heap in heaps.items():
            if row_split == split:
                rows.extend(item[1] for item in heap)
        rows.sort(key=lambda row: (row["label"], row["id"]))
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = {
            label: sum(row["label"] == label for row in rows)
            for label in sorted({row["label"] for row in rows})
        }
        missing = {
            label: quotas[split] - counts[split].get(label, 0)
            for label in sorted(set(COARSE.values()))
            if counts[split].get(label, 0) < quotas[split]
        }
        if missing:
            raise RuntimeError(f"ArXiv split {split} does not meet per-class quotas: {missing}")
    _publish_manifest(args, output_dir, quotas, counts, scanned_records=seen)
    print(json.dumps(counts, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
