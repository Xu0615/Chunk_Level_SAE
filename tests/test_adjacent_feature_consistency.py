from __future__ import annotations

import numpy as np

from analyze_adjacent_feature_consistency import (
    _analyze_codes,
    _matched_derangement,
)


def test_matched_derangement_preserves_lengths_and_changes_documents() -> None:
    lengths_a = np.asarray([32] * 6 + [64] * 6)
    lengths_b = np.asarray([64] * 6 + [32] * 6)
    document_hashes = np.asarray(
        [f"doc-{index}".encode() for index in range(12)],
        dtype="S32",
    )
    permutation = _matched_derangement(
        lengths_a,
        lengths_b,
        document_hashes,
        seed=17,
    )
    assert np.all(permutation != np.arange(permutation.size))
    assert np.array_equal(lengths_a[permutation], lengths_a)
    assert np.array_equal(lengths_b[permutation], lengths_b)
    assert np.all(document_hashes[permutation] != document_hashes)


def test_adjacent_analysis_detects_feature_persistence_and_utilization() -> None:
    rng = np.random.default_rng(5)
    pairs, features = 128, 64
    codes_a = np.zeros((pairs, features), dtype=np.float32)
    codes_b = np.zeros_like(codes_a)
    for row in range(pairs):
        persistent = rng.choice(features, size=8, replace=False)
        codes_a[row, persistent] = rng.uniform(0.5, 1.5, size=8)
        codes_b[row, persistent] = rng.uniform(0.5, 1.5, size=8)
    shuffled = np.roll(np.arange(pairs), 1)
    result = _analyze_codes(
        codes_a,
        codes_b,
        shuffled,
        min_feature_support=4,
        utilization_k=8,
        bootstrap_samples=100,
        seed=9,
    )
    assert result["pair_metrics"]["cosine_separation_mean"] > 0.5
    assert result["pair_metrics"]["adjacent_retrieval_auc"] > 0.9
    assert result["feature_metrics"]["mean_persistence_lift"] > 0.5
    utilization = result["dictionary_utilization"]
    assert utilization["common_activity_budget"] == 8
    assert 0 < utilization["effective_feature_fraction"] <= 1
    assert 0 <= utilization["gini"] <= 1
