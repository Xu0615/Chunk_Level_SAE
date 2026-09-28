from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


FIXED_CHUNK_REPRESENTATION_PROTOCOL = "fixed-independent-chunk-v1"
# The ArXiv probe archive shipped with the first task-layout revision was
# extracted before the fixed-length sampler was introduced.  It is still a
# valid, reproducible artifact, but its rows represent the native tokenized
# length (up to the extractor's truncation limit), rather than one exact
# chunk length.  Keeping a separate protocol name makes that distinction
# explicit in manifests and prevents accidental apples-to-oranges claims.
LEGACY_VARIABLE_LENGTH_REPRESENTATION_PROTOCOL = "legacy-variable-length-v1"
TOKEN_AGGREGATION = "mean_after_threshold"


def mean_after_threshold(
    thresholded_codes: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average thresholded token codes into one code per independent chunk."""

    if thresholded_codes.ndim == 2:
        if mask is not None:
            if mask.ndim != 1 or mask.shape[0] != thresholded_codes.shape[0]:
                raise ValueError("1-D mask must match the token dimension")
            valid = mask.to(
                device=thresholded_codes.device,
                dtype=torch.bool,
            )
            if not bool(valid.any()):
                raise ValueError("cannot aggregate an empty chunk")
            return thresholded_codes[valid].float().mean(dim=0)
        if thresholded_codes.shape[0] == 0:
            raise ValueError("cannot aggregate an empty chunk")
        return thresholded_codes.float().mean(dim=0)

    if thresholded_codes.ndim != 3:
        raise ValueError("thresholded codes must have shape [T,F] or [B,T,F]")
    if mask is None:
        if thresholded_codes.shape[1] == 0:
            raise ValueError("cannot aggregate empty chunks")
        return thresholded_codes.float().mean(dim=1)
    if mask.ndim != 2 or mask.shape != thresholded_codes.shape[:2]:
        raise ValueError("2-D mask must match the batch and token dimensions")
    valid = mask.to(
        device=thresholded_codes.device,
        dtype=torch.bool,
    )
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("cannot aggregate an empty chunk")
    weights = valid.to(torch.float32).unsqueeze(-1)
    return (
        (thresholded_codes.float() * weights).sum(dim=1)
        / counts.to(torch.float32).unsqueeze(-1)
    )


def full_dictionary_feature_widths(
    sae_set: Mapping[str, Any],
    *,
    modes: Sequence[str] | None = None,
) -> dict[str, int]:
    """Return the common full dictionary width for every evaluated SAE mode."""

    common = sae_set.get("common")
    entries = sae_set.get("modes")
    if not isinstance(common, Mapping) or not isinstance(entries, Mapping):
        raise ValueError("invalid SAE artifact set")
    width = int(common.get("dict_size", 0))
    if width <= 0:
        raise ValueError("SAE artifact set has no positive dictionary width")
    selected = tuple(entries) if modes is None else tuple(modes)
    missing = [mode for mode in selected if mode not in entries]
    if missing:
        raise ValueError(f"SAE artifact set lacks modes: {missing}")
    return {str(mode): width for mode in selected}


def validate_full_dictionary_width(
    mode: str,
    actual_width: int,
    expected_width: int,
) -> None:
    if int(actual_width) != int(expected_width):
        raise ValueError(
            f"{mode} encoder width {actual_width} does not match the common "
            f"full dictionary width {expected_width}"
        )


def fixed_chunk_protocol_metadata(
    *,
    feature_widths: Mapping[str, int],
    chunk_lengths: Sequence[int] | None = None,
) -> dict[str, Any]:
    widths = {str(mode): int(width) for mode, width in feature_widths.items()}
    if not widths or any(width <= 0 for width in widths.values()):
        raise ValueError("feature widths must be positive")
    if len(set(widths.values())) != 1:
        raise ValueError("all evaluated SAE modes must use the same dictionary width")
    metadata: dict[str, Any] = {
        "name": FIXED_CHUNK_REPRESENTATION_PROTOCOL,
        "independent_forward": True,
        "shared_hidden_states_across_methods": True,
        "shared_chunk_or_pair_ids_across_methods": True,
        "token_temporal_aggregation": TOKEN_AGGREGATION,
        "chunk_sae_input": "mean_hidden_state",
        "threshold_order": {
            "token_temporal": "threshold_each_token_then_mean",
            "mean_cross": "mean_hidden_then_threshold",
        },
        "formal_codes": {
            "token": "mean_t(threshold(Token(H[t])))",
            "temporal": "mean_t(threshold(Temporal(H[t])))",
            "mean": "threshold(Mean(mean_t(H[t])))",
            "cross": "threshold(Cross(mean_t(H[t])))",
        },
        "feature_scope": "complete_dictionary",
        "feature_widths": widths,
    }
    if chunk_lengths is not None:
        lengths = [int(length) for length in chunk_lengths]
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("chunk lengths must be positive")
        metadata["fixed_chunk_lengths"] = sorted(set(lengths))
    return metadata


def legacy_variable_length_protocol_metadata(
    *,
    feature_widths: Mapping[str, int],
    max_length: int | None = None,
) -> dict[str, Any]:
    """Describe the preserved pre-fixed-length probe representation.

    This helper is intentionally as explicit as :func:`fixed_chunk_protocol_metadata`.
    Legacy rows were produced by independent forwards and use the same
    threshold-then-mean aggregation, but each example keeps its native number
    of valid tokenizer positions.  The protocol is accepted only for existing
    published probe artifacts; new extraction continues to use the fixed
    protocol by default.
    """

    widths = {str(mode): int(width) for mode, width in feature_widths.items()}
    if not widths or any(width <= 0 for width in widths.values()):
        raise ValueError("feature widths must be positive")
    if len(set(widths.values())) != 1:
        raise ValueError("all evaluated SAE modes must use the same dictionary width")
    metadata: dict[str, Any] = {
        "name": LEGACY_VARIABLE_LENGTH_REPRESENTATION_PROTOCOL,
        "legacy": True,
        "independent_forward": True,
        "shared_hidden_states_across_methods": True,
        "shared_chunk_or_pair_ids_across_methods": True,
        "token_temporal_aggregation": TOKEN_AGGREGATION,
        "chunk_sae_input": "mean_hidden_state",
        "threshold_order": {
            "token_temporal": "threshold_each_token_then_mean",
            "mean_cross": "mean_hidden_then_threshold",
        },
        "formal_codes": {
            "token": "mean_t(threshold(Token(H[t])))",
            "temporal": "mean_t(threshold(Temporal(H[t])))",
            "mean": "threshold(Mean(mean_t(H[t])))",
            "cross": "threshold(Cross(mean_t(H[t])))",
        },
        "feature_scope": "complete_dictionary",
        "feature_widths": widths,
        "length_policy": "native_tokenized_length_with_attention_mask",
    }
    if max_length is not None:
        if int(max_length) <= 0:
            raise ValueError("max_length must be positive")
        metadata["max_length"] = int(max_length)
    return metadata
