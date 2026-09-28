from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import zstandard as zstd

from chunk_saes.cache import ActivationCacheV2Writer, merge_v2_rank_manifests
from chunk_saes.sample_plan import (
    SAMPLE_PLAN_FORMAT,
    iter_prepared_document_records,
    iter_sample_plan_rows,
    plan_row_hash,
    solve_proportional_cell_counts,
)
from chunk_saes.streaming_plan import (
    STREAMING_SAMPLING_VERSION,
    StreamingExactPlanner,
    StreamingRankAllocator,
    build_streaming_sample_plan,
    iter_tokenized_prepared_documents,
    prepared_corpus_digest,
)
from chunk_saes.utils import stable_u64
from train_chunk_saes import validate_exact_training_coverage


class BatchWhitespaceTokenizer:
    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            return {"input_ids": [int(token) for token in texts.split()]}
        return {
            "input_ids": [
                [int(token) for token in text.split()] for text in texts
            ]
        }


def _documents() -> list[dict]:
    rows = []
    for index in range(24):
        source = "source-a" if index % 2 == 0 else "source-b"
        tokens = list(range(index * 100, index * 100 + 32))
        digest = f"{index + 1:064x}"
        rows.append(
            {
                "doc_id": f"content:{digest}",
                "source": source,
                "split": "train",
                "text": " ".join(map(str, tokens)),
                "text_hash": digest,
            }
        )
    return rows


def _write_prepared(root: Path, world_size: int) -> None:
    root.mkdir()
    order_seed = 71
    rows = sorted(
        _documents(),
        key=lambda item: (
            stable_u64(item["text_hash"], order_seed),
            item["text_hash"],
        ),
    )
    ranks = [[] for _ in range(world_size)]
    for row in rows:
        rank = stable_u64(row["text_hash"], 59) % world_size
        ranks[rank].append(row)
    entries = []
    for rank, rank_rows in enumerate(ranks):
        path = root / f"rank{rank:03d}.jsonl.zst"
        payload = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rank_rows
        ).encode()
        path.write_bytes(zstd.ZstdCompressor(level=1).compress(payload))
        entries.append(
            {
                "rank": rank,
                "path": path.name,
                "documents": len(rank_rows),
                "text_bytes": sum(len(row["text"].encode()) for row in rank_rows),
                "compressed_bytes": path.stat().st_size,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "chunk-saes-document-rank-shards-v2",
                "complete": True,
                "dataset": {"fingerprint": "tiny-canonical-corpus"},
                "split_seed": 42,
                "split": "train",
                "world_size": world_size,
                "excluded_sources": [],
                "min_chars": 0,
                "generic_filtering": False,
                "cleaned_documents": True,
                "content_normalization": "test",
                "split_key": "normalized_content_sha256",
                "split_before_rank": True,
                "deduplicated_globally": True,
                "deduplication": {"scope": "global"},
                "document_order": {
                    "algorithm": "stable_u64(content_hash, order_seed)",
                    "seed": order_seed,
                },
                "documents": len(rows),
                "raw_source_documents": {
                    "source-a": 12,
                    "source-b": 12,
                },
                "raw_source_text_bytes": {
                    "source-a": 1000,
                    "source-b": 1000,
                },
                "raw_source_weight_scope": (
                    "all_input_records_before_dedup_split_filter"
                ),
                "ranks": entries,
            }
        ),
        encoding="utf-8",
    )


def _logical_rows(root: Path):
    return [
        (
            row.pair_id,
            row.occurrence_start,
            row.content_hash,
            row.source,
            row.input_ids_a,
            row.input_ids_b,
            row.start_a,
            row.start_b,
            row.document_reuse_index,
            row.execution_rank,
        )
        for row in iter_sample_plan_rows(root)
    ]


def test_streaming_plan_is_exact_nonoverlapping_and_physical_shard_independent(
    tmp_path: Path,
):
    prepared_one = tmp_path / "prepared-one"
    prepared_three = tmp_path / "prepared-three"
    _write_prepared(prepared_one, 1)
    _write_prepared(prepared_three, 3)
    common = {
        "tokenizer_hash": "batch-whitespace",
        "target_tokens": 128,
        "lengths": [2, 4],
        "sample_seed": 43,
        "sources": ["source-a", "source-b"],
        "shard_token_limit": 32,
        "max_document_reuses": 4,
        "execution_world_size": 2,
        "tokenizer_batch_size": 4,
        "tokenizer_batch_chars": 10_000,
        "source_weighting": "corpus_proportional",
        "progress_every_documents": 0,
    }
    root_one = tmp_path / "plan-one"
    root_three = tmp_path / "plan-three"
    manifest_one = build_streaming_sample_plan(
        prepared_one,
        BatchWhitespaceTokenizer(),
        root_one,
        **common,
    )
    manifest_three = build_streaming_sample_plan(
        prepared_three,
        BatchWhitespaceTokenizer(),
        root_three,
        **common,
    )

    assert manifest_one["plan_digest"] == manifest_three["plan_digest"]
    assert _logical_rows(root_one) == _logical_rows(root_three)
    assert manifest_one["token_occurrences"] == 128
    assert manifest_one["unique_corpus_token_positions"] == 128
    assert manifest_one["source_tokens"] == {
        "source-a": 64,
        "source-b": 64,
    }
    assert manifest_one["execution_rank_assignment"]["tokens_per_rank"] == [
        64,
        64,
    ]
    assert manifest_one["catalog"]["enabled"] is False

    intervals: dict[bytes, list[tuple[int, int]]] = {}
    rank_tokens = [0, 0]
    for row in iter_sample_plan_rows(root_one):
        intervals.setdefault(row.content_hash, []).append(
            (row.start_a, row.start_b + row.length_b)
        )
        rank_tokens[row.execution_rank] += row.token_count
    for values in intervals.values():
        values.sort()
        for left, right in zip(values, values[1:]):
            assert right[0] >= left[1]
    assert rank_tokens == [64, 64]


def test_streaming_rows_can_publish_a_complete_cache_without_plan_shards(
    tmp_path: Path,
):
    prepared = tmp_path / "prepared"
    _write_prepared(prepared, 2)
    prepared_manifest = json.loads(
        (prepared / "manifest.json").read_text(encoding="utf-8")
    )
    counts = solve_proportional_cell_counts(
        128,
        ["source-a", "source-b"],
        [2, 4],
        {"source-a": 1000, "source-b": 1000},
        seed=43,
    )
    allocator = StreamingRankAllocator(
        counts,
        world_size=2,
        target_tokens=128,
        seed=43,
    )
    planner = StreamingExactPlanner(
        counts,
        seed=43,
        max_document_reuses=4,
        rank_allocator=allocator,
    )
    rows = []
    digest = hashlib.sha256()
    for document, token_ids in iter_tokenized_prepared_documents(
        iter_prepared_document_records(prepared),
        BatchWhitespaceTokenizer(),
        batch_size=4,
        batch_chars=10_000,
    ):
        for row in planner.add_document(document, token_ids):
            rows.append(row)
            digest.update(plan_row_hash(row))
        if planner.complete:
            break
    stats = planner.finish(target_tokens=128, expected_pairs=sum(counts.values()))
    identity = {
        "format_version": "chunk-saes-streaming-universe",
        "target_tokens": 128,
        "sample_seed": 43,
        "tokenizer_hash": "batch-whitespace",
        "corpus_digest": prepared_corpus_digest(prepared_manifest),
        "sources": ["source-a", "source-b"],
        "chunk_lengths": [2, 4],
        "execution_world_size": 2,
        "streaming_sampling": STREAMING_SAMPLING_VERSION,
        "forward_fused": True,
    }
    rows_digest = digest.hexdigest()
    plan_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    cache_root = tmp_path / "cache"
    source_to_id = {"source-a": 0, "source-b": 1}
    for rank in range(2):
        writer = ActivationCacheV2Writer(
            cache_root,
            rank=rank,
            world_size=2,
            hidden_size=4,
            plan_digest=plan_digest,
            target_token_occurrences=128,
            shard_token_limit=32,
            source_to_id=source_to_id,
            activation_dtype=torch.float32,
        )
        for row in rows:
            if row.execution_rank != rank:
                continue
            hidden = torch.arange(
                row.token_count * 4,
                dtype=torch.float32,
            ).reshape(row.token_count, 4)
            mean_a = hidden[: row.length_a].mean(dim=0)
            mean_b = hidden[row.length_a :].mean(dim=0)
            writer.add(row, hidden, mean_a, mean_b)
        rank_manifest = writer.finish()
        assert rank_manifest["token_occurrences"] == 64
    source_tokens = {
        source: sum(
            (length_a + length_b) * count
            for (cell_source, length_a, length_b), count in counts.items()
            if cell_source == source
        )
        for source in ("source-a", "source-b")
    }
    cell_counts = {
        f"{source}\t{length_a}\t{length_b}": count
        for (source, length_a, length_b), count in sorted(counts.items())
    }
    universe = {
        "format": SAMPLE_PLAN_FORMAT,
        "complete": True,
        "identity": identity,
        "plan_digest": plan_digest,
        "rows_digest": rows_digest,
        "target_token_occurrences": 128,
        "token_occurrences": 128,
        "pairs": len(rows),
        "source_tokens": source_tokens,
        "cell_counts": cell_counts,
        "corpus_position_overlap_policy": "forbidden",
        "corpus_position_overlap_verified": True,
        "unique_corpus_token_positions": 128,
        "execution_rank_assignment": stats["execution_rank_assignment"],
        "shards": [],
    }
    merged = merge_v2_rank_manifests(
        cache_root,
        world_size=2,
        plan_manifest=universe,
        extra={
            "streaming_sampling": STREAMING_SAMPLING_VERSION,
            "streaming_forward_fused": True,
        },
    )
    assert merged["complete"] is True
    assert merged["token_occurrences"] == 128
    assert merged["source_tokens"] == source_tokens
    assert merged["cell_counts"] == cell_counts
    assert validate_exact_training_coverage(
        merged,
        world_size=2,
        global_batch_size=32,
        steps=4,
    ) == 128
