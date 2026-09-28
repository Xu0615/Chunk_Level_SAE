from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard as zstd

from chunk_saes.corpus import dataset_fingerprint, discover_dataset_shards
from chunk_saes.data import (
    document_order_priority,
    iter_documents,
    iter_partitioned_documents,
    normalized_content_hash,
)
from chunk_saes.rank_shards import (
    RANK_SHARD_FORMAT,
    iter_rank_document_shard,
    load_rank_shard_manifest,
    prepare_rank_document_shards,
)
from chunk_saes.parallel_corpus import prepare_rank_document_shards_bundle
from chunk_saes.utils import document_split


def _record(text: str, identifier: str, source: str = "Pile-CC") -> dict:
    return {
        "text": text,
        "meta": {"id": identifier, "pile_set_name": source},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl_zst(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_bytes(zstd.ZstdCompressor(level=1).compress(payload.encode("utf-8")))


def _long_text(label: str) -> str:
    return (
        f"{label} contains enough alphabetic content to pass filtering and exercise "
        "the canonical corpus protocol. "
    ) * 3


def _find_text_for_split(split: str, seed: int, prefix: str) -> str:
    for index in range(10_000):
        text = _long_text(f"{prefix}-{index}")
        if document_split(normalized_content_hash(text), seed) == split:
            return text
    raise AssertionError(f"could not synthesize text for split={split}")


def _all_prepared(root: Path, manifest: dict) -> list:
    documents = []
    for rank in range(manifest["world_size"]):
        documents.extend(iter_rank_document_shard(root, rank, manifest))
    return documents


def test_dataset_file_or_directory_discovers_all_shards_in_stable_order(tmp_path: Path):
    single = tmp_path / "single.jsonl"
    _write_jsonl(single, [_record(_long_text("single"), "single")])
    root, shards = discover_dataset_shards(single)
    assert root == tmp_path.resolve()
    assert [item.relative_path for item in shards] == ["single.jsonl"]

    dataset = tmp_path / "dataset"
    _write_jsonl(dataset / "z-last.jsonl", [_record(_long_text("z"), "z")])
    _write_jsonl_zst(dataset / "nested" / "a-first.jsonl.zst", [_record(_long_text("a"), "a")])
    (dataset / "README.md").write_text("not data", encoding="utf-8")

    _root, shards = discover_dataset_shards(dataset)
    assert [item.relative_path for item in shards] == [
        "nested/a-first.jsonl.zst",
        "z-last.jsonl",
    ]
    fingerprint = dataset_fingerprint(dataset)
    assert fingerprint["kind"] == "directory"
    assert fingerprint["recursive"] is True
    assert fingerprint["shard_count"] == 2
    assert [item["path"] for item in fingerprint["shards"]] == [
        "nested/a-first.jsonl.zst",
        "z-last.jsonl",
    ]
    assert all(len(item["sha256"]) == 64 for item in fingerprint["shards"])
    assert len(fingerprint["fingerprint"]) == 64

    before = fingerprint["fingerprint"]
    path = dataset / "z-last.jsonl"
    original_stat = path.stat()
    original_bytes = path.read_bytes()
    replacement = original_bytes.replace(b'"z"', b'"y"', 1)
    assert len(replacement) == len(original_bytes)
    path.write_bytes(replacement)
    path.touch()
    fingerprint_after = dataset_fingerprint(dataset)
    assert fingerprint_after["fingerprint"] != before
    assert path.stat().st_size == original_stat.st_size


def test_content_hash_controls_split_and_global_dedup_across_shards_and_ranks(
    tmp_path: Path,
):
    seed = 42
    duplicate = _find_text_for_split("train", seed, "duplicate")
    validation = _find_text_for_split("validation", seed, "validation")
    dataset = tmp_path / "dataset"
    _write_jsonl(
        dataset / "part-1.jsonl",
        [
            _record(duplicate, "metadata-a"),
            _record(validation, "validation-id"),
        ],
    )
    _write_jsonl_zst(
        dataset / "nested" / "part-2.jsonl.zst",
        [
            _record(
                "  " + duplicate.upper().replace(" ", "   ") + "\n",
                "different-metadata-b",
            ),
            _record(duplicate, "different-metadata-c"),
        ],
    )

    train = list(
        iter_partitioned_documents(
            dataset,
            split_seed=seed,
            wanted_split="train",
            world_size=4,
            min_chars=1,
        )
    )
    assert len(train) == 1
    rank, document = train[0]
    assert rank in range(4)
    assert document.text_hash == normalized_content_hash(duplicate)
    assert document.doc_id == f"content:{document.text_hash}"
    assert document.split == "train"

    validation_documents = [
        document
        for _rank, document in iter_partitioned_documents(
            dataset,
            split_seed=seed,
            wanted_split="validation",
            world_size=4,
            min_chars=1,
        )
    ]
    assert {item.text_hash for _rank, item in train}.isdisjoint(
        {item.text_hash for item in validation_documents}
    )
    assert {item.text_hash for item in validation_documents} == {
        normalized_content_hash(validation)
    }


def test_document_union_is_world_size_invariant_and_rank_is_content_hash_based(
    tmp_path: Path,
):
    seed = 7
    dataset = tmp_path / "dataset"
    records = [
        _record(_find_text_for_split("train", seed, f"doc-{index}"), f"id-{index}")
        for index in range(30)
    ]
    _write_jsonl(dataset / "b.jsonl", list(reversed(records[::2])))
    _write_jsonl_zst(dataset / "a.jsonl.zst", list(reversed(records[1::2])))

    by_world_size = {}
    for world_size in (1, 2, 5):
        rows = list(
            iter_partitioned_documents(
                dataset,
                split_seed=seed,
                wanted_split="train",
                world_size=world_size,
                min_chars=1,
            )
        )
        by_world_size[world_size] = {document.text_hash for _rank, document in rows}
        assert len(rows) == len(by_world_size[world_size])
        for rank, document in rows:
            selected = list(
                iter_documents(
                    dataset,
                    split_seed=seed,
                    wanted_split="train",
                    rank=rank,
                    world_size=world_size,
                    min_chars=1,
                )
            )
            assert document in selected
    assert by_world_size[1] == by_world_size[2] == by_world_size[5]


def test_excluded_source_taints_duplicate_content_globally(tmp_path: Path):
    seed = 31
    shared = _find_text_for_split("train", seed, "tainted")
    safe = _find_text_for_split("train", seed, "safe")
    dataset = tmp_path / "dataset"
    _write_jsonl(
        dataset / "part-0.jsonl",
        [
            _record(shared, "included-copy", source="Pile-CC"),
            _record(safe, "safe-copy", source="Pile-CC"),
        ],
    )
    _write_jsonl_zst(
        dataset / "nested" / "part-1.jsonl.zst",
        [_record(shared, "excluded-copy", source="ArXiv")],
    )
    documents = list(
        iter_documents(
            dataset,
            split_seed=seed,
            wanted_split="train",
            world_size=1,
            excluded_sources={"ArXiv"},
            min_chars=1,
        )
    )
    assert {document.text_hash for document in documents} == {
        normalized_content_hash(safe)
    }


def test_prepared_shards_v2_are_seeded_random_reproducible_and_manifested(
    tmp_path: Path,
):
    seed = 19
    dataset = tmp_path / "dataset"
    records = [
        _record(_find_text_for_split("train", seed, f"prepared-{index}"), f"id-{index}")
        for index in range(40)
    ]
    _write_jsonl(dataset / "part-b.jsonl", records[::2])
    _write_jsonl_zst(dataset / "nested" / "part-a.jsonl.zst", records[1::2])

    root_a = tmp_path / "prepared-a"
    root_b = tmp_path / "prepared-b"
    manifest_a = prepare_rank_document_shards(
        dataset,
        root_a,
        split_seed=seed,
        split="train",
        world_size=3,
        excluded_sources=set(),
        min_chars=1,
        compression_level=0,
        order_seed=123,
    )
    manifest_b = prepare_rank_document_shards(
        dataset,
        root_b,
        split_seed=seed,
        split="train",
        world_size=3,
        excluded_sources=set(),
        min_chars=1,
        compression_level=0,
        order_seed=123,
    )

    assert manifest_a["format"] == RANK_SHARD_FORMAT
    assert manifest_a["dataset"]["shard_count"] == 2
    assert manifest_a["deduplication"] == {
        "scope": "global",
        "key": "normalized_content_sha256",
        "across_dataset_shards": True,
        "before_split": True,
        "before_rank": True,
        "excluded_source_policy": "exclude_content_hash_if_any_occurrence_excluded",
    }
    assert manifest_a["split_key"] == "normalized_content_sha256"
    assert manifest_a["rank_assignment_key"] == "normalized_content_sha256"
    assert manifest_a["document_order"]["seed"] == 123
    assert manifest_a["document_order"]["randomized"] is True
    assert manifest_a["rank_local_order_preserved"] is False

    loaded = load_rank_shard_manifest(
        root_a,
        dataset,
        split_seed=seed,
        split="train",
        world_size=3,
        excluded_sources=set(),
        min_chars=1,
        order_seed=123,
    )
    assert loaded == manifest_a
    assert [
        document.text_hash for document in _all_prepared(root_a, manifest_a)
    ] == [
        document.text_hash for document in _all_prepared(root_b, manifest_b)
    ]

    direct_hash_order = sorted(
        document.text_hash
        for _rank, document in iter_partitioned_documents(
            dataset,
            split_seed=seed,
            wanted_split="train",
            world_size=3,
            min_chars=1,
        )
    )
    prepared_by_rank = [
        list(iter_rank_document_shard(root_a, rank, manifest_a))
        for rank in range(3)
    ]
    prepared_hashes = sorted(document.text_hash for rows in prepared_by_rank for document in rows)
    assert prepared_hashes == direct_hash_order
    assert any(
        [document.text_hash for document in rows]
        != sorted(document.text_hash for document in rows)
        for rows in prepared_by_rank
        if len(rows) > 2
    )


def test_prepared_global_order_is_world_size_invariant(tmp_path: Path):
    seed = 23
    dataset = tmp_path / "dataset"
    records = [
        _record(_find_text_for_split("train", seed, f"ws-{index}"), f"id-{index}")
        for index in range(25)
    ]
    _write_jsonl(dataset / "part-0.jsonl", records[::2])
    _write_jsonl_zst(dataset / "nested" / "part-1.jsonl.zst", records[1::2])

    direct_orders = []
    for world_size in (1, 4):
        root = tmp_path / f"prepared-ws{world_size}"
        manifest = prepare_rank_document_shards(
            dataset,
            root,
            split_seed=seed,
            split="train",
            world_size=world_size,
            excluded_sources=set(),
            min_chars=1,
            compression_level=0,
            order_seed=456,
        )
        prepared = _all_prepared(root, manifest)
        direct = list(
            iter_partitioned_documents(
                dataset,
                split_seed=seed,
                wanted_split="train",
                world_size=world_size,
                min_chars=1,
                order_seed=456,
            )
        )
        assert {document.text_hash for document in prepared} == {
            document.text_hash for _rank, document in direct
        }
        direct_orders.append([document.text_hash for _rank, document in direct])

    assert direct_orders[0] == direct_orders[1]
    assert direct_orders[0] == sorted(
        direct_orders[0],
        key=lambda digest: (document_order_priority(digest, 456), digest),
    )


def test_parallel_bundle_matches_canonical_train_and_validation(tmp_path: Path):
    seed = 37
    train_rows = [
        _record(_find_text_for_split("train", seed, f"parallel-train-{index}"), f"t-{index}")
        for index in range(20)
    ]
    validation_rows = [
        _record(
            _find_text_for_split("validation", seed, f"parallel-validation-{index}"),
            f"v-{index}",
        )
        for index in range(8)
    ]
    tainted = train_rows[3]
    dataset = tmp_path / "dataset"
    _write_jsonl(dataset / "part-0.jsonl", train_rows[::2] + validation_rows[::2])
    _write_jsonl_zst(
        dataset / "part-1.jsonl.zst",
        train_rows[1::2]
        + validation_rows[1::2]
        + [_record(tainted["text"], "excluded-copy", source="ArXiv")],
    )
    outputs = {
        "train": tmp_path / "parallel-train",
        "validation": tmp_path / "parallel-validation",
    }
    manifests = prepare_rank_document_shards_bundle(
        dataset,
        outputs,
        split_seed=seed,
        world_size=3,
        excluded_sources={"ArXiv"},
        min_chars=1,
        compression_level=1,
        order_seed=987,
        scan_workers=2,
        reduce_workers=2,
        bucket_count=4,
        workspace_dir=tmp_path / "workspace",
    )

    for split, root in outputs.items():
        prepared_by_rank = [
            list(iter_rank_document_shard(root, rank, manifests[split]))
            for rank in range(3)
        ]
        direct = list(
            iter_partitioned_documents(
                dataset,
                split_seed=seed,
                wanted_split=split,
                world_size=3,
                excluded_sources={"ArXiv"},
                min_chars=1,
                order_seed=987,
            )
        )
        assert {
            document.text_hash for rows in prepared_by_rank for document in rows
        } == {document.text_hash for _rank, document in direct}
        for rank, rows in enumerate(prepared_by_rank):
            assert rows == [document for assigned, document in direct if assigned == rank]
        assert manifests[split]["preparation"]["dataset_scan_passes"] == 1
        assert manifests[split]["preparation"]["bundled_splits"] == [
            "train",
            "validation",
        ]
    assert normalized_content_hash(tainted["text"]) not in {
        document.text_hash
        for rows in [
            list(iter_rank_document_shard(outputs["train"], rank, manifests["train"]))
            for rank in range(3)
        ]
        for document in rows
    }
    expected_documents = {"Pile-CC": 28, "ArXiv": 1}
    expected_bytes = {
        "Pile-CC": sum(
            len(record["text"].encode("utf-8"))
            for record in train_rows + validation_rows
        ),
        "ArXiv": len(tainted["text"].encode("utf-8")),
    }
    for manifest in manifests.values():
        assert manifest["raw_source_documents"] == expected_documents
        assert manifest["raw_source_text_bytes"] == expected_bytes
        assert manifest["raw_source_weight_scope"] == (
            "all_input_records_before_dedup_split_filter"
        )


def test_parallel_bundle_can_disable_generic_filtering(tmp_path: Path):
    seed = 41
    short = _find_text_for_split("train", seed, "short")[:20]
    while document_split(normalized_content_hash(short), seed) != "train":
        short += "x"
    dataset = tmp_path / "dataset"
    _write_jsonl(dataset / "part.jsonl", [_record(short, "short", source="Pile-CC")])

    output = tmp_path / "prepared"
    manifests = prepare_rank_document_shards_bundle(
        dataset,
        {"train": output},
        split_seed=seed,
        world_size=1,
        excluded_sources=set(),
        min_chars=128,
        compression_level=0,
        order_seed=71,
        scan_workers=1,
        reduce_workers=1,
        bucket_count=2,
        workspace_dir=tmp_path / "workspace",
        generic_filtering=False,
    )
    documents = list(iter_rank_document_shard(output, 0, manifests["train"]))
    assert len(documents) == 1
    assert documents[0].text == short
    assert manifests["train"]["generic_filtering"] is False
    assert not (tmp_path / "workspace").exists()


def test_manifest_validation_rejects_dataset_shard_set_changes(tmp_path: Path):
    seed = 5
    dataset = tmp_path / "dataset"
    _write_jsonl(
        dataset / "part-0.jsonl",
        [_record(_find_text_for_split("train", seed, "first"), "first")],
    )
    output = tmp_path / "prepared"
    prepare_rank_document_shards(
        dataset,
        output,
        split_seed=seed,
        split="train",
        world_size=1,
        excluded_sources=set(),
        min_chars=1,
        compression_level=0,
    )
    _write_jsonl(
        dataset / "part-1.jsonl",
        [_record(_find_text_for_split("train", seed, "second"), "second")],
    )
    with pytest.raises(ValueError, match="Rank-shard manifest does not match"):
        load_rank_shard_manifest(
            output,
            dataset,
            split_seed=seed,
            split="train",
            world_size=1,
            excluded_sources=set(),
            min_chars=1,
        )
