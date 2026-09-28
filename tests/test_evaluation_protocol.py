from __future__ import annotations

import pytest
import torch

from chunk_saes.evaluation_protocol import (
    LEGACY_VARIABLE_LENGTH_REPRESENTATION_PROTOCOL,
    fixed_chunk_protocol_metadata,
    full_dictionary_feature_widths,
    legacy_variable_length_protocol_metadata,
    mean_after_threshold,
    validate_full_dictionary_width,
)
from extract_probe_features import fixed_chunk_ids


def test_mean_after_threshold_uses_only_valid_tokens() -> None:
    codes = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 4.0], [100.0, 100.0]],
            [[3.0, 6.0], [9.0, 0.0], [0.0, 3.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])

    observed = mean_after_threshold(codes, mask)

    torch.testing.assert_close(
        observed,
        torch.tensor([[1.0, 2.0], [4.0, 3.0]]),
    )


def test_mean_after_threshold_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="empty chunk"):
        mean_after_threshold(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.bool),
        )


def test_full_dictionary_protocol_requires_equal_widths() -> None:
    sae_set = {
        "common": {"dict_size": 65_536},
        "modes": {
            mode: {"mode": mode}
            for mode in ("token", "temporal", "mean", "cross")
        },
    }
    widths = full_dictionary_feature_widths(sae_set)

    assert widths == {
        "token": 65_536,
        "temporal": 65_536,
        "mean": 65_536,
        "cross": 65_536,
    }
    metadata = fixed_chunk_protocol_metadata(
        feature_widths=widths,
        chunk_lengths=[128, 32, 128],
    )
    assert metadata["fixed_chunk_lengths"] == [32, 128]
    assert metadata["token_temporal_aggregation"] == "mean_after_threshold"
    assert metadata["feature_scope"] == "complete_dictionary"

    with pytest.raises(ValueError, match="same dictionary width"):
        fixed_chunk_protocol_metadata(
            feature_widths={"token": 65_536, "temporal": 13_107},
        )
    with pytest.raises(ValueError, match="full dictionary width"):
        validate_full_dictionary_width("temporal", 13_107, 65_536)


def test_probe_chunk_selection_is_exact_length() -> None:
    class Tokenizer:
        @staticmethod
        def encode(text, **_kwargs):
            return list(range(len(text)))

    assert fixed_chunk_ids(Tokenizer(), "abcdef", 4) == [0, 1, 2, 3]
    assert fixed_chunk_ids(Tokenizer(), "abc", 4) is None


def test_legacy_variable_length_protocol_is_explicit() -> None:
    protocol = legacy_variable_length_protocol_metadata(
        feature_widths={"token": 65_536, "mean": 65_536},
        max_length=512,
    )
    assert protocol["name"] == LEGACY_VARIABLE_LENGTH_REPRESENTATION_PROTOCOL
    assert protocol["legacy"] is True
    assert protocol["length_policy"] == "native_tokenized_length_with_attention_mask"
    assert protocol["max_length"] == 512
    assert protocol["feature_scope"] == "complete_dictionary"
