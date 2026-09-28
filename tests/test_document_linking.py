from __future__ import annotations

import numpy as np
from scipy import sparse

from evaluate_lexical_controlled_document_linking import (
    RETRIEVAL_BENCHMARK_KEYS,
    RETRIEVAL_BENCHMARK_NAMES,
    build_pairs,
    filter_pairs_for_gallery_support,
    lexical_jaccard,
    retrieval_ranks,
)


def test_build_pairs_selects_lowest_overlap_and_one_pair_per_document():
    chunks = [
        {
            "doc_id": "a",
            "global_chunk_id": 0,
            "length": 32,
            "text": "alpha beta gamma delta",
        },
        {
            "doc_id": "a",
            "global_chunk_id": 1,
            "length": 64,
            "text": "alpha beta epsilon",
        },
        {
            "doc_id": "a",
            "global_chunk_id": 2,
            "length": 32,
            "text": "zeta eta theta",
        },
        {
            "doc_id": "b",
            "global_chunk_id": 3,
            "length": 32,
            "text": "one two three",
        },
        {
            "doc_id": "b",
            "global_chunk_id": 4,
            "length": 32,
            "text": "four five six",
        },
    ]
    pairs = build_pairs(
        chunks,
        lexical_jaccard_max=0.1,
        min_word_length=3,
    )
    assert len(pairs) == 2
    by_doc = {pair.doc_id: pair for pair in pairs}
    assert {
        by_doc["a"].left_index,
        by_doc["a"].right_index,
    } == {1, 2}
    assert by_doc["a"].lexical_jaccard == 0.0
    assert by_doc["b"].lexical_jaccard == 0.0
    assert lexical_jaccard("same words here", "same other words") > 0


def test_retrieval_ranks_are_exact_length_controlled_and_sparse_safe():
    # Rows 0/1 are queries; rows 2/3 are their correct galleries.  Row 4 is a
    # distractor with a different target length and a perfect similarity to
    # query 0; it must be excluded by the exact-length protocol.
    dense = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.asarray([0, 1])
    gallery = np.asarray([2, 3])
    lengths = np.asarray([32, 32])
    ranks_dense = retrieval_ranks(
        dense,
        query_indices=query,
        gallery_indices=gallery,
        gallery_lengths=lengths,
    )
    ranks_sparse = retrieval_ranks(
        sparse.csr_matrix(dense),
        query_indices=query,
        gallery_indices=gallery,
        gallery_lengths=lengths,
    )
    np.testing.assert_allclose(ranks_dense, [1.0, 1.0])
    np.testing.assert_allclose(ranks_sparse, ranks_dense)


def test_gallery_support_filter_removes_singleton_retokenized_lengths():
    chunks = [
        {"doc_id": "a", "global_chunk_id": 0, "length": 32, "text": "a"},
        {"doc_id": "a", "global_chunk_id": 1, "length": 32, "text": "b"},
        {"doc_id": "b", "global_chunk_id": 2, "length": 32, "text": "c"},
        {"doc_id": "b", "global_chunk_id": 3, "length": 32, "text": "d"},
        {"doc_id": "c", "global_chunk_id": 4, "length": 32, "text": "e"},
        {"doc_id": "c", "global_chunk_id": 5, "length": 32, "text": "f"},
    ]
    pairs = [
        # Pair c has a singleton left target length and should be removed.
        type("P", (), {"doc_id": "a", "left_index": 0, "right_index": 1})(),
        type("P", (), {"doc_id": "b", "left_index": 2, "right_index": 3})(),
        type("P", (), {"doc_id": "c", "left_index": 4, "right_index": 5})(),
    ]
    kept, audit = filter_pairs_for_gallery_support(
        pairs,
        chunks,
        retokenized_length_by_chunk={
            0: 32,
            1: 32,
            2: 32,
            3: 32,
            4: 31,
            5: 32,
        },
    )
    assert [pair.doc_id for pair in kept] == ["a", "b"]
    assert audit["pairs_removed"] == 1


def test_retrieval_plot_orders_all_four_sae_methods():
    assert RETRIEVAL_BENCHMARK_KEYS == (
        "word_jaccard",
        "raw_mean_hidden",
        "token",
        "temporal",
        "mean",
        "cross",
    )
    assert RETRIEVAL_BENCHMARK_NAMES[2:] == (
        "BatchTopK SAE",
        "Temporal SAE",
        "Mean-Chunk SAE",
        "Cross-Chunk SAE",
    )
