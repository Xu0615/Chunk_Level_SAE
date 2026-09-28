from __future__ import annotations

import torch

from chunk_saes.sae import BatchTopKSAE
from chunk_saes.temporal import (
    symmetric_temporal_contrastive_loss,
    temporal_high_feature_count,
)


def test_temporal_high_feature_count_is_strict_prefix() -> None:
    assert temporal_high_feature_count(100, 0.2) == 20
    assert temporal_high_feature_count(3, 0.99) == 2


def test_temporal_contrastive_prefers_matched_pairs() -> None:
    current = torch.eye(8)
    matched = current.clone()
    shuffled = torch.roll(current, shifts=1, dims=0)
    mask = torch.ones(8, dtype=torch.bool)
    good = symmetric_temporal_contrastive_loss(
        current,
        matched,
        mask,
        temperature=0.1,
        block_size=8,
    )
    bad = symmetric_temporal_contrastive_loss(
        current,
        shuffled,
        mask,
        temperature=0.1,
        block_size=8,
    )
    assert good.loss < bad.loss
    assert good.accuracy > bad.accuracy
    assert good.pairs == 8


def test_temporal_sae_forward_returns_previous_codes_and_high_reconstruction() -> None:
    model = BatchTopKSAE(activation_dim=4, dict_size=10, k=2)
    current = torch.randn(6, 4)
    previous = torch.randn(6, 4)
    output = model(
        current,
        batch_topk=True,
        return_activity_counts=True,
        return_auxiliary=True,
        temporal_previous=previous,
        temporal_high_features=2,
        temporal_sample_mask=torch.ones(6, dtype=torch.bool),
        temporal_global_sample_count=6,
    )
    assert len(output) == 9
    (
        reconstructed,
        features,
        _threshold,
        active,
        counts,
        auxiliary,
        temporal_current,
        old,
        high,
    ) = output
    assert reconstructed.shape == current.shape
    assert features.shape == (6, 10)
    assert active.shape == (6,)
    assert counts.shape == (10,)
    assert auxiliary.shape == current.shape
    assert temporal_current.shape == features.shape
    assert old.shape == features.shape
    assert high.shape == current.shape
