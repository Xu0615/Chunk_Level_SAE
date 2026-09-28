from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from chunk_saes.cache import (
    ACTIVATION_CACHE_V2_FORMAT,
    ActivationCacheV2Writer,
    load_activation_cache_v2_manifest,
    merge_v2_rank_manifests,
)
from chunk_saes.sample_plan import (
    PreparedDocument,
    build_sample_plan,
    iter_sample_plan_rows,
)


class WhitespaceTokenizer:
    def __call__(self, text: str, **_kwargs):
        return {"input_ids": [int(token) for token in text.split()]}


def tiny_documents() -> list[PreparedDocument]:
    return [
        PreparedDocument(
            stream_id=index % 2,
            ordinal=index // 2,
            doc_id=f"doc-{index}",
            source="source-a" if index < 3 else "source-b",
            text=" ".join(map(str, range(index * 100, index * 100 + 32))),
            content_hash=f"{index + 1:064x}",
        )
        for index in range(6)
    ]


def create_plan(root: Path) -> dict:
    return build_sample_plan(
        lambda: iter(tiny_documents()),
        WhitespaceTokenizer(),
        root,
        tokenizer_hash="tokenizer-test",
        target_tokens=48,
        lengths=[4],
        sample_seed=11,
        sources=["source-a", "source-b"],
        shard_token_limit=17,
        max_document_reuses=4,
    )


def activation_for_row(row, hidden_size: int = 3):
    token_ids = torch.tensor(row.input_ids, dtype=torch.float32)
    hidden = torch.stack(
        (token_ids, token_ids + row.pair_id, token_ids * 0.5), dim=-1
    )
    mean_a = hidden[: row.length_a].mean(dim=0)
    mean_b = hidden[row.length_a :].mean(dim=0)
    return hidden, mean_a, mean_b


def test_activation_cache_v2_roundtrip_preserves_all_token_rows_and_pair_metadata(
    tmp_path: Path,
):
    plan_root = tmp_path / "plan"
    plan = create_plan(plan_root)
    cache_root = tmp_path / "cache"
    writer = ActivationCacheV2Writer(
        cache_root,
        rank=0,
        world_size=1,
        hidden_size=3,
        plan_digest=plan["plan_digest"],
        target_token_occurrences=plan["target_token_occurrences"],
        shard_token_limit=19,
        source_to_id={"source-a": 0, "source-b": 1},
    )
    expected = {}
    for row in iter_sample_plan_rows(plan_root):
        hidden, mean_a, mean_b = activation_for_row(row)
        expected[row.pair_id] = (row, hidden.to(torch.bfloat16), mean_a, mean_b)
        writer.add(row, hidden, mean_a, mean_b)
    writer.finish()
    merged = merge_v2_rank_manifests(
        cache_root,
        world_size=1,
        plan_manifest=plan,
        verify_payload_checksums=True,
    )
    assert merged["format"] == ACTIVATION_CACHE_V2_FORMAT
    assert merged["token_occurrences"] == merged["target_token_occurrences"] == 48
    assert merged["coverage"]["token_hidden_rows_equal_occurrences"] is True
    assert load_activation_cache_v2_manifest(cache_root)["plan_digest"] == plan["plan_digest"]

    observed = {}
    for item in merged["ranks"][0]["shards"]:
        with safe_open(
            str(cache_root / item["path"]), framework="pt", device="cpu"
        ) as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        offsets = tensors["chunk_offsets"]
        for index, pair_id in enumerate(tensors["pair_id"].tolist()):
            start = int(offsets[2 * index])
            split = int(offsets[2 * index + 1])
            stop = int(offsets[2 * index + 2])
            observed[pair_id] = {
                "hidden": tensors["token_hidden"][start:stop],
                "mean_a": tensors["mean_a"][index],
                "mean_b": tensors["mean_b"][index],
                "occurrence_start": int(tensors["occurrence_start"][index]),
                "length_a": split - start,
                "length_b": stop - split,
                "content_hash": bytes(tensors["content_hash"][index].tolist()),
            }
    assert set(observed) == set(expected)
    for pair_id, (row, hidden, mean_a, mean_b) in expected.items():
        torch.testing.assert_close(observed[pair_id]["hidden"], hidden)
        torch.testing.assert_close(
            observed[pair_id]["mean_a"], mean_a.to(torch.bfloat16)
        )
        torch.testing.assert_close(
            observed[pair_id]["mean_b"], mean_b.to(torch.bfloat16)
        )
        assert observed[pair_id]["occurrence_start"] == row.occurrence_start
        assert observed[pair_id]["length_a"] == row.length_a
        assert observed[pair_id]["length_b"] == row.length_b
        assert observed[pair_id]["content_hash"] == row.content_hash


def test_async_batched_writer_preserves_legacy_shards_and_digests(tmp_path: Path):
    plan_root = tmp_path / "plan"
    plan = create_plan(plan_root)
    rows = list(iter_sample_plan_rows(plan_root))

    reference_root = tmp_path / "reference"
    reference = ActivationCacheV2Writer(
        reference_root,
        rank=0,
        world_size=1,
        hidden_size=3,
        plan_digest=plan["plan_digest"],
        target_token_occurrences=plan["target_token_occurrences"],
        shard_token_limit=19,
        source_to_id={"source-a": 0, "source-b": 1},
    )
    activations = {}
    for row in rows:
        hidden, mean_a, mean_b = activation_for_row(row)
        activations[row.pair_id] = (hidden, mean_a, mean_b)
        reference.add(row, hidden, mean_a, mean_b)
    reference_manifest = reference.finish()

    async_root = tmp_path / "async"
    batched = ActivationCacheV2Writer(
        async_root,
        rank=0,
        world_size=1,
        hidden_size=3,
        plan_digest=plan["plan_digest"],
        target_token_occurrences=plan["target_token_occurrences"],
        shard_token_limit=19,
        source_to_id={"source-a": 0, "source-b": 1},
        async_write_batches=3,
    )
    # Submit forward batches out of legacy order. Explicit order indices must
    # restore the exact pair stream before shard boundaries and hashing.
    for indices in ([2, 3], [0, 1], [5], [4]):
        batch_rows = [rows[index] for index in indices]
        hidden_parts = [activations[row.pair_id][0] for row in batch_rows]
        means = torch.stack(
            [
                torch.stack(
                    (
                        activations[row.pair_id][1],
                        activations[row.pair_id][2],
                    )
                )
                for row in batch_rows
            ]
        )
        batched.add_batch(
            batch_rows,
            torch.cat(hidden_parts),
            means,
            order_indices=indices,
        )
    async_manifest = batched.finish()

    assert async_manifest["activation_digest"] == reference_manifest["activation_digest"]
    assert async_manifest["target_sufficient_statistics"] == (
        reference_manifest["target_sufficient_statistics"]
    )
    assert async_manifest["occurrence_range_digest"] == (
        reference_manifest["occurrence_range_digest"]
    )
    assert [item["tokens"] for item in async_manifest["shards"]] == [
        item["tokens"] for item in reference_manifest["shards"]
    ]
    assert [item["payload_sha256"] for item in async_manifest["shards"]] == [
        item["payload_sha256"] for item in reference_manifest["shards"]
    ]


def test_cache_merge_is_world_size_independent_and_rejects_missing_pair(tmp_path: Path):
    plan_root = tmp_path / "plan"
    plan = create_plan(plan_root)
    cache_root = tmp_path / "cache"
    rows = list(iter_sample_plan_rows(plan_root))
    for rank in range(2):
        writer = ActivationCacheV2Writer(
            cache_root,
            rank=rank,
            world_size=2,
            hidden_size=3,
            plan_digest=plan["plan_digest"],
            target_token_occurrences=plan["target_token_occurrences"],
            shard_token_limit=16,
            source_to_id={"source-a": 0, "source-b": 1},
        )
        for row in rows:
            if row.pair_id % 2 != rank:
                continue
            hidden, mean_a, mean_b = activation_for_row(row)
            writer.add(row, hidden, mean_a, mean_b)
        writer.finish()
    merged = merge_v2_rank_manifests(
        cache_root,
        world_size=2,
        plan_manifest=plan,
    )
    assert merged["pairs"] == plan["pairs"]
    assert merged["token_occurrences"] == 48

    broken_root = tmp_path / "broken-cache"
    for rank in range(2):
        writer = ActivationCacheV2Writer(
            broken_root,
            rank=rank,
            world_size=2,
            hidden_size=3,
            plan_digest=plan["plan_digest"],
            target_token_occurrences=plan["target_token_occurrences"],
            shard_token_limit=16,
            source_to_id={"source-a": 0, "source-b": 1},
        )
        for row in rows:
            if row.pair_id == rows[-1].pair_id or row.pair_id % 2 != rank:
                continue
            hidden, mean_a, mean_b = activation_for_row(row)
            writer.add(row, hidden, mean_a, mean_b)
        writer.finish()
    with pytest.raises(ValueError, match="missing pair IDs"):
        merge_v2_rank_manifests(
            broken_root,
            world_size=2,
            plan_manifest=plan,
        )


def test_cache_v2_loader_rejects_legacy_manifest(tmp_path: Path):
    root = tmp_path / "legacy"
    root.mkdir()
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "chunk-saes-activation-cache-v1",
                "complete": True,
                "tokens": 100,
            },
            handle,
        )
    with pytest.raises(ValueError, match="v1 activation caches"):
        load_activation_cache_v2_manifest(root)
