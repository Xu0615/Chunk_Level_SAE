from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import pytest
import zstandard as zstd

from chunk_saes.sample_plan import (
    PreparedDocument,
    build_sample_plan,
    iter_prepared_document_records,
    iter_sample_plan_rows,
    solve_exact_cell_counts,
    solve_proportional_cell_counts,
)
from chunk_saes.utils import stable_u64


class WhitespaceTokenizer:
    def __call__(self, text: str, **_kwargs):
        return {"input_ids": [int(token) for token in text.split()]}


class _MappingEncoding(Mapping):
    def __init__(self, values: dict):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


def test_tokenizer_mapping_result_uses_input_ids():
    from chunk_saes.sample_plan import _batch_token_lengths, _tokenize

    class MappingTokenizer:
        def __call__(self, _text, **_kwargs):
            return _MappingEncoding(
                {"input_ids": [7, 11, 13], "attention_mask": [1, 1, 1]}
            )

    assert _tokenize(MappingTokenizer(), "ignored") == [7, 11, 13]

    class BatchTokenizer:
        def __call__(self, texts, **_kwargs):
            return _MappingEncoding(
                {
                    "input_ids": [[index] * (index + 1) for index in range(len(texts))],
                    "length": [index + 1 for index in range(len(texts))],
                }
            )

    assert _batch_token_lengths(BatchTokenizer(), ["a", "b", "c"]) == [1, 2, 3]


def test_proportional_solver_matches_raw_source_mass_exactly():
    counts = solve_proportional_cell_counts(
        1_000_000_000,
        ["large", "medium", "small"],
        [32, 64, 128, 256, 512],
        {"large": 80, "medium": 15, "small": 5},
        seed=43,
    )
    source_tokens = {
        source: sum(
            (length_a + length_b) * count
            for (cell_source, length_a, length_b), count in counts.items()
            if cell_source == source
        )
        for source in ("large", "medium", "small")
    }
    assert source_tokens == {
        "large": 800_000_000,
        "medium": 150_000_000,
        "small": 50_000_000,
    }


def test_proportional_solver_uses_integer_largest_remainders():
    weights = {
        "a": 334_664_485_567,
        "b": 111_554_828_523,
        "c": 17_000_000_003,
    }
    counts = solve_proportional_cell_counts(
        1_000_000_000,
        list(weights),
        [32, 64, 128, 256, 512],
        weights,
        seed=99,
    )
    source_tokens = {
        source: sum(
            (length_a + length_b) * count
            for (cell_source, length_a, length_b), count in counts.items()
            if cell_source == source
        )
        for source in weights
    }
    assert sum(source_tokens.values()) == 1_000_000_000
    unit = 32
    total_units = 1_000_000_000 // unit
    expected_units = {
        source: (total_units * weight) // sum(weights.values())
        for source, weight in weights.items()
    }
    remainder_units = total_units - sum(expected_units.values())
    remainder_order = sorted(
        weights,
        key=lambda source: -((total_units * weights[source]) % sum(weights.values())),
    )
    for source in remainder_order[:remainder_units]:
        expected_units[source] += 1
    assert source_tokens == {
        source: units * unit for source, units in expected_units.items()
    }


def test_boundary_lookup_uses_content_hash_index(tmp_path: Path):
    from chunk_saes.sample_plan import _initialize_workspace

    connection = _initialize_workspace(tmp_path / "planner.sqlite3")
    try:
        query_plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT selection_id, length_a, length_b, reuse_index
            FROM selections WHERE content_hash=?
            """,
            (bytes(32),),
        ).fetchall()
    finally:
        connection.close()

    details = "\n".join(str(row[-1]) for row in query_plan)
    assert "SEARCH selections USING INDEX selections_content_hash" in details
    assert "SCAN selections" not in details


def test_materialization_commit_retries_transient_sqlite_lock(monkeypatch):
    import sqlite3

    from chunk_saes import sample_plan

    class Connection:
        attempts = 0

        def commit(self):
            self.attempts += 1
            if self.attempts < 3:
                raise sqlite3.OperationalError("database is locked")

    connection = Connection()
    delays = []
    monkeypatch.setattr(sample_plan.time, "sleep", delays.append)
    sample_plan._commit_with_retry(connection)
    assert connection.attempts == 3
    assert delays == [0.25, 0.5]


def test_materialization_commit_does_not_hide_other_sqlite_errors(monkeypatch):
    import sqlite3

    from chunk_saes import sample_plan

    class Connection:
        def commit(self):
            raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sample_plan.time, "sleep", lambda _delay: None)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        sample_plan._commit_with_retry(Connection())


def test_parallel_prepared_catalog_matches_generic_plan(tmp_path: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer_model = Tokenizer(
        WordLevel(
            {"[UNK]": 0, **{str(value): value + 1 for value in range(1_000)}},
            unk_token="[UNK]",
        )
    )
    tokenizer_model.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="[UNK]",
    )
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_root)

    documents = documents_for_order(list(range(8)))
    order_seed = 777
    ordered = sorted(
        documents,
        key=lambda item: (
            stable_u64(item.content_hash, order_seed),
            item.content_hash,
        ),
    )
    ranks = [[], []]
    for document in ordered:
        rank = stable_u64(document.content_hash, 59) % 2
        ranks[rank].append(document)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    entries = []
    for rank, rows in enumerate(ranks):
        path = prepared / f"rank{rank:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for document in rows:
                handle.write(
                    json.dumps(
                        {
                            "doc_id": document.doc_id,
                            "source": document.source,
                            "split": "train",
                            "text": document.text,
                            "text_hash": document.content_hash,
                        }
                    )
                    + "\n"
                )
        entries.append({"rank": rank, "path": path.name, "documents": len(rows)})
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format": "chunk-saes-document-rank-shards-v2",
                "complete": True,
                "documents": len(documents),
                "world_size": 2,
                "deduplicated_globally": True,
                "deduplication": {"scope": "global"},
                "document_order": {"seed": order_seed},
                "ranks": entries,
            }
        ),
        encoding="utf-8",
    )

    common = {
        "tokenizer_hash": "tiny-fast-tokenizer-v1",
        "target_tokens": 80,
        "lengths": [4, 8],
        "sample_seed": 23,
        "shard_token_limit": 18,
        "max_document_reuses": 4,
        "execution_world_size": 1,
    }
    generic_root = tmp_path / "generic-plan"
    parallel_root = tmp_path / "parallel-plan"
    generic = build_sample_plan(
        lambda: iter_prepared_document_records(prepared),
        tokenizer,
        generic_root,
        workspace_dir=tmp_path / "generic-workspace",
        **common,
    )
    parallel = build_sample_plan(
        lambda: iter_prepared_document_records(prepared),
        tokenizer,
        parallel_root,
        workspace_dir=tmp_path / "parallel-workspace",
        prepared_document_root=prepared,
        tokenizer_path=tokenizer_root,
        catalog_workers=2,
        catalog_batch_size=2,
        catalog_batch_chars=1_000,
        tokenizer_threads_per_worker=1,
        **common,
    )
    assert parallel["plan_digest"] == generic["plan_digest"]
    assert logical_rows(parallel_root) == logical_rows(generic_root)
    assert parallel["catalog"]["parallel_prepared_catalog"]["enabled"] is True

    # A failed plan attempt must retain complete worker catalogs, and a retry
    # with a feasible reuse cap must not tokenize the prepared corpus again.
    retry_workspace = tmp_path / "retry-workspace"
    retry_common = {
        **common,
        "target_tokens": 160,
        "max_document_reuses": 1,
    }
    with pytest.raises(ValueError, match="Insufficient non-overlapping document capacity"):
        build_sample_plan(
            lambda: iter_prepared_document_records(prepared),
            tokenizer,
            tmp_path / "infeasible-plan",
            workspace_dir=retry_workspace,
            prepared_document_root=prepared,
            tokenizer_path=tokenizer_root,
            catalog_workers=2,
            catalog_batch_size=2,
            catalog_batch_chars=1_000,
            tokenizer_threads_per_worker=1,
            **retry_common,
        )
    catalog_root = retry_workspace / "workspace-v1" / "parallel-catalog"
    catalog_paths = sorted(catalog_root.glob("catalog-rank*.sqlite3"))
    metadata_paths = sorted(catalog_root.glob("catalog-rank*.sqlite3.manifest.json"))
    assert len(catalog_paths) == len(metadata_paths) == 2
    catalog_mtimes = {path.name: path.stat().st_mtime_ns for path in catalog_paths}

    retry_common["max_document_reuses"] = 4
    recovered = build_sample_plan(
        lambda: iter_prepared_document_records(prepared),
        tokenizer,
        tmp_path / "recovered-plan",
        workspace_dir=retry_workspace,
        keep_workspace=True,
        prepared_document_root=prepared,
        tokenizer_path=tokenizer_root,
        catalog_workers=2,
        catalog_batch_size=2,
        catalog_batch_chars=1_000,
        tokenizer_threads_per_worker=1,
        **retry_common,
    )
    assert recovered["token_occurrences"] == 160
    assert recovered["unique_corpus_token_positions"] == 160
    assert recovered["catalog"]["parallel_prepared_catalog"][
        "reused_worker_catalogs"
    ] == 2
    assert {
        path.name: path.stat().st_mtime_ns for path in catalog_paths
    } == catalog_mtimes


def test_prepared_plan_uses_raw_pre_dedup_source_weights(tmp_path: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer_model = Tokenizer(
        WordLevel(
            {"[UNK]": 0, **{str(value): value + 1 for value in range(256)}},
            unk_token="[UNK]",
        )
    )
    tokenizer_model.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="[UNK]",
    )
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_root)

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = []
    for index, source in enumerate(("source-a", "source-b")):
        text = " ".join(str(value) for value in range(index * 64, (index + 1) * 64))
        rows.append(
            {
                "doc_id": f"doc-{index}",
                "source": source,
                "split": "train",
                "text": text,
                "text_hash": f"{index + 1:064x}",
            }
        )
    shard = prepared / "rank000.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format": "chunk-saes-document-rank-shards-v2",
                "complete": True,
                "documents": 2,
                "world_size": 1,
                "deduplicated_globally": True,
                "deduplication": {"scope": "global"},
                "document_order": {"seed": 71},
                "raw_source_text_bytes": {"source-a": 300, "source-b": 100},
                "raw_source_weight_scope": (
                    "all_input_records_before_dedup_split_filter"
                ),
                "ranks": [
                    {"rank": 0, "path": shard.name, "documents": len(rows)}
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_sample_plan(
        lambda: iter_prepared_document_records(prepared),
        tokenizer,
        tmp_path / "plan",
        tokenizer_hash="tiny-fast-tokenizer-v1",
        target_tokens=64,
        lengths=[4],
        sample_seed=43,
        sources=["source-a", "source-b"],
        shard_token_limit=64,
        max_document_reuses=6,
        prepared_document_root=prepared,
        tokenizer_path=tokenizer_root,
        source_weighting="corpus_proportional",
    )
    assert manifest["source_tokens"] == {"source-a": 48, "source-b": 16}
    assert manifest["source_token_weights"] == {"source-a": 300, "source-b": 100}
    assert manifest["source_weight_basis"] == (
        "raw_utf8_text_bytes_before_dedup_split_filter"
    )


def documents_for_order(order: list[int]) -> list[PreparedDocument]:
    documents = []
    for index in order:
        source = "source-a" if index < 4 else "source-b"
        tokens = list(range(index * 100, index * 100 + 40 + index))
        documents.append(
            PreparedDocument(
                stream_id=index % 3,
                ordinal=index // 3,
                doc_id=f"metadata-{index}",
                source=source,
                text=" ".join(map(str, tokens)),
                content_hash=f"{index + 1:064x}",
            )
        )
    return documents


def logical_rows(root: Path):
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
        )
        for row in iter_sample_plan_rows(root)
    ]


def test_formal_one_billion_token_plan_is_exact_and_source_balanced():
    sources = [f"source-{index:02d}" for index in range(16)]
    counts = solve_exact_cell_counts(
        1_000_000_000,
        sources,
        [32, 64, 128, 256, 512],
        seed=43,
    )
    assert (
        sum(
            (length_a + length_b) * count
            for (_source, length_a, length_b), count in counts.items()
        )
        == 1_000_000_000
    )
    source_tokens = {
        source: sum(
            (length_a + length_b) * count
            for (cell_source, length_a, length_b), count in counts.items()
            if cell_source == source
        )
        for source in sources
    }
    assert set(source_tokens.values()) == {62_500_000}
    cell_counts = list(counts.values())
    assert max(cell_counts) - min(cell_counts) <= 1


def test_unrepresentable_token_budget_fails_loudly():
    with pytest.raises(ValueError, match="not representable"):
        solve_exact_cell_counts(65, ["source"], [32], seed=1)


def test_plan_is_reproducible_input_order_independent_and_rank_independent(tmp_path: Path):
    tokenizer = WhitespaceTokenizer()
    root_a = tmp_path / "plan-a"
    root_b = tmp_path / "plan-b"
    documents_a = documents_for_order(list(range(8)))
    documents_b = documents_for_order([7, 2, 5, 0, 6, 3, 1, 4])
    manifest_a = build_sample_plan(
        lambda: iter(documents_a),
        tokenizer,
        root_a,
        tokenizer_hash="tokenizer-test",
        target_tokens=96,
        lengths=[4, 8],
        sample_seed=17,
        sources=["source-a", "source-b"],
        shard_token_limit=24,
        max_document_reuses=4,
        execution_world_size=3,
    )
    manifest_b = build_sample_plan(
        lambda: iter(documents_b),
        tokenizer,
        root_b,
        tokenizer_hash="tokenizer-test",
        target_tokens=96,
        lengths=[4, 8],
        sample_seed=17,
        sources=["source-a", "source-b"],
        shard_token_limit=37,
        max_document_reuses=4,
        execution_world_size=3,
    )
    assert manifest_a["plan_digest"] == manifest_b["plan_digest"]
    assert manifest_a["rows_digest"] == manifest_b["rows_digest"]
    assert logical_rows(root_a) == logical_rows(root_b)
    for manifest in (manifest_a, manifest_b):
        assert manifest["physical_shard_order"] == "pair_id_contiguous"
        previous_max = -1
        for item in manifest["shards"]:
            assert item["pair_id_min"] == previous_max + 1
            previous_max = item["pair_id_max"]

    all_rows = logical_rows(root_a)
    rank_rows = []
    for rank in range(3):
        current_rank_rows = list(
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
            )
            for row in iter_sample_plan_rows(root_a, rank=rank, world_size=3)
        )
        assert sum(len(row[4]) + len(row[5]) for row in current_rank_rows) == 32
        rank_rows.extend(current_rank_rows)
    assert sorted(rank_rows) == all_rows


def test_plan_rows_contain_reconstructable_adjacent_spans_and_exact_occurrences(tmp_path: Path):
    root = tmp_path / "plan"
    manifest = build_sample_plan(
        lambda: iter(documents_for_order(list(range(8)))),
        WhitespaceTokenizer(),
        root,
        tokenizer_hash="tokenizer-test",
        target_tokens=80,
        lengths=[4, 8],
        sample_seed=23,
        sources=["source-a", "source-b"],
        shard_token_limit=18,
        max_document_reuses=4,
    )
    rows = list(iter_sample_plan_rows(root))
    assert sum(row.token_count for row in rows) == 80
    assert [row.pair_id for row in rows] == list(range(len(rows)))
    cursor = 0
    for row in rows:
        assert row.occurrence_start == cursor
        assert row.start_b == row.start_a + row.length_a
        assert row.occurrence_stop == cursor + row.token_count
        assert len(row.doc_hash) == 32
        assert len(row.content_hash) == 32
        cursor = row.occurrence_stop
    assert cursor == manifest["target_token_occurrences"]
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["token_occurrences"] == saved["target_token_occurrences"] == 80
    assert saved["rank_independent"] is True


def test_prepared_rank_shards_are_merged_in_rank_independent_document_order(
    tmp_path: Path,
):
    documents = [
        {
            "doc_id": f"content:{index:064x}",
            "source": "source",
            "split": "train",
            "text": f"{index} {index + 1} {index + 2} {index + 3}",
            "text_hash": f"{index:064x}",
        }
        for index in range(1, 9)
    ]
    seed = 123

    def write_prepared(root: Path, world_size: int) -> None:
        root.mkdir()
        ranks = [[] for _ in range(world_size)]
        ordered = sorted(
            documents,
            key=lambda item: (
                stable_u64(item["text_hash"], seed),
                item["text_hash"],
            ),
        )
        for item in ordered:
            rank = stable_u64(item["text_hash"], 59) % world_size
            ranks[rank].append(item)
        entries = []
        for rank, rows in enumerate(ranks):
            path = root / f"rank{rank:03d}.jsonl.zst"
            payload = "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in rows
            ).encode()
            path.write_bytes(zstd.ZstdCompressor(level=1).compress(payload))
            entries.append({"rank": rank, "path": path.name})
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "chunk-saes-document-rank-shards-v2",
                    "complete": True,
                    "world_size": world_size,
                    "document_order": {"seed": seed},
                    "ranks": entries,
                }
            ),
            encoding="utf-8",
        )

    root_one = tmp_path / "ws1"
    root_three = tmp_path / "ws3"
    write_prepared(root_one, 1)
    write_prepared(root_three, 3)
    observed_one = [
        row.content_hash for row in iter_prepared_document_records(root_one)
    ]
    observed_three = [
        row.content_hash for row in iter_prepared_document_records(root_three)
    ]
    assert observed_one == observed_three
