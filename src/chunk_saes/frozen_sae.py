"""Small encoder-only views of frozen SAE checkpoints for evaluation."""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .sae import DecoderHead, SAE_PARAMETER_SCHEMA_VERSION


class SelectedSAEEncoder:
    """Thresholded SAE encoder retaining only requested dictionary rows."""

    def __init__(
        self,
        checkpoint: str | Path,
        feature_ids: Sequence[int],
        device: torch.device,
    ) -> None:
        if not feature_ids:
            raise ValueError("feature_ids must not be empty")
        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        ids = torch.as_tensor(feature_ids, dtype=torch.long)
        with safe_open(
            str(checkpoint / "sae.safetensors"), framework="pt", device="cpu"
        ) as handle:
            names = set(handle.keys())
            counts = handle.get_tensor("feature_counts")
            self.width = int(counts.numel())
            if int(ids.min()) < 0 or int(ids.max()) >= self.width:
                raise ValueError(f"feature id outside encoder width for {checkpoint}")
            self.weight = handle.get_tensor("encoder_weight")[ids].to(device)
            self.bias = handle.get_tensor("encoder_bias")[ids].to(device)
            if "pre_bias" in names:
                self.pre_bias = handle.get_tensor("pre_bias").to(device)
            elif config.get("sae_parameter_schema_version") == SAE_PARAMETER_SCHEMA_VERSION:
                raise ValueError(f"{checkpoint} declares pre_bias but does not store it")
            else:
                self.pre_bias = handle.get_tensor("decoder_bias").to(device)
            self.threshold = handle.get_tensor("threshold").to(device)
            self.scale = handle.get_tensor("activation_scale").to(device)
        self.device = device

    @torch.inference_mode()
    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        pre = self.raw_scores(hidden)
        return pre * (pre > self.threshold.to(pre.dtype))

    @torch.inference_mode()
    def raw_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return positive pre-threshold activations for top-k inference views."""
        hidden = hidden.to(self.device, dtype=self.weight.dtype)
        return F.relu(
            F.linear(
                hidden * self.scale.to(self.weight.dtype) - self.pre_bias,
                self.weight,
                self.bias,
            )
        )

    def close(self) -> None:
        del self.weight, self.bias, self.pre_bias, self.threshold, self.scale
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def selected_decoder_target_view(
    checkpoint: str | Path,
    feature_ids: Sequence[int],
    *,
    head: DecoderHead | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Load selected decoder atoms and target centering terms.

    A legacy (single-head) checkpoint has one unambiguous decoder and accepts
    ``head=None``.  A Joint Chunk checkpoint has two readouts for every shared
    feature coordinate; callers must name ``head='mean'`` or ``head='cross'``
    so an interpretation cannot silently use the wrong target space.
    """

    if not feature_ids:
        raise ValueError("feature_ids must not be empty")
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    ids = torch.as_tensor(feature_ids, dtype=torch.long)
    with safe_open(
        str(checkpoint / "sae.safetensors"), framework="pt", device="cpu"
    ) as handle:
        names = set(handle.keys())
        has_cross_weight = "decoder_cross_weight" in names
        has_cross_bias = "decoder_cross_bias" in names
        nested_prefix = int(config.get("joint_cross_prefix") or 0)
        is_nested = (
            config.get("joint_chunk_layout") == "nested_prefix"
            or nested_prefix > 0
        )
        if is_nested:
            if nested_prefix <= 0 or nested_prefix >= int(
                config.get("dict_size", nested_prefix)
            ):
                raise ValueError(
                    f"{checkpoint} has an invalid nested Cross prefix "
                    f"{nested_prefix}"
                )
            if has_cross_weight or not has_cross_bias:
                raise ValueError(
                    f"{checkpoint} has invalid nested Joint decoder tensors"
                )
        elif has_cross_weight != has_cross_bias:
            raise ValueError(
                f"{checkpoint} stores an incomplete Cross decoder head"
            )
        has_cross_head = has_cross_weight and has_cross_bias
        declared_heads = int(
            config.get("decoder_heads", 2 if has_cross_head else 1)
        )
        if declared_heads == 2 and not has_cross_head:
            raise ValueError(
                f"{checkpoint} declares two decoder heads but lacks "
                "decoder_cross_weight"
            )
        if has_cross_head and declared_heads != 2:
            raise ValueError(
                f"{checkpoint} stores a Cross decoder head but config declares "
                f"decoder_heads={declared_heads}"
            )
        if is_nested:
            if head not in (None, "mean", "cross"):
                raise ValueError(
                    f"unknown decoder head {head!r}; expected 'mean' or 'cross'"
                )
            if head == "cross" and bool((ids >= nested_prefix).any()):
                raise ValueError(
                    f"nested Cross decoder uses only feature ids "
                    f"[0, {nested_prefix})"
                )
        elif has_cross_head:
            if head is None:
                raise ValueError(
                    "Joint Chunk checkpoint has two decoder heads; pass "
                    "head='mean' or head='cross' explicitly"
                )
            if head not in ("mean", "cross"):
                raise ValueError(
                    f"unknown decoder head {head!r}; expected 'mean' or 'cross'"
                )
        elif head not in (None, "mean"):
            raise ValueError(
                f"single-head checkpoint {checkpoint} has no decoder head {head!r}"
            )
        width = int(handle.get_tensor("feature_counts").numel())
        if int(ids.min()) < 0 or int(ids.max()) >= width:
            raise ValueError(f"feature id outside decoder width for {checkpoint}")
        weight_name = (
            "decoder_cross_weight"
            if head == "cross" and not is_nested
            else "decoder_weight"
        )
        bias_name = "decoder_cross_bias" if head == "cross" else "decoder_bias"
        if weight_name not in names or bias_name not in names:
            raise ValueError(
                f"{checkpoint} is missing the requested decoder head tensors: "
                f"{weight_name}, {bias_name}"
            )
        vectors = handle.get_tensor(weight_name)[:, ids].T.float()
        bias = handle.get_tensor(bias_name).float()
        scale = float(handle.get_tensor("activation_scale"))
    return vectors, bias, scale


def selected_joint_decoder_vectors(
    checkpoint: str | Path,
    feature_ids: Sequence[int],
    *,
    loss_normalized: bool = True,
) -> torch.Tensor:
    """Return one canonical direct-sum atom per Joint feature coordinate.

    The raw representation is ``[d_mean; d_cross]`` in ``R^(2H)``.  With
    ``loss_normalized=True`` (the default), each block is scaled so Euclidean
    squared error in this direct-sum space is exactly the configured Joint
    task loss.  This is a canonical task-complete single vector; either
    H-dimensional block alone describes only one readout head.
    """

    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    if int(config.get("decoder_heads", 1)) != 2:
        raise ValueError(
            f"selected_joint_decoder_vectors requires a two-head checkpoint: "
            f"{checkpoint}"
        )
    mean, _mean_bias, scale = selected_decoder_target_view(
        checkpoint,
        feature_ids,
        head="mean",
    )
    cross, _cross_bias, cross_scale = selected_decoder_target_view(
        checkpoint,
        feature_ids,
        head="cross",
    )
    if not math.isclose(cross_scale, scale, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"decoder heads disagree on activation scale in {checkpoint}")
    if not loss_normalized:
        return torch.cat((mean, cross), dim=1)

    alpha = float(config.get("joint_chunk_alpha", 0.0))
    mean_baseline = float(config.get("joint_chunk_mean_baseline", 0.0))
    cross_baseline = float(config.get("joint_chunk_cross_baseline", 0.0))
    if alpha <= 0 or mean_baseline <= 0 or cross_baseline <= 0 or scale <= 0:
        raise ValueError(
            f"{checkpoint} lacks positive alpha/baselines/activation scale "
            "required for the normalized Joint decoder representation"
        )
    # Training config stores the raw (unscaled) baseline SSE; decoder atoms
    # are in the scaled target coordinates, hence the extra ``scale**2``.
    mean_factor = 1.0 / math.sqrt(
        (1.0 + alpha) * mean_baseline * scale * scale
    )
    cross_factor = math.sqrt(alpha) / math.sqrt(
        (1.0 + alpha) * cross_baseline * scale * scale
    )
    return torch.cat((mean * mean_factor, cross * cross_factor), dim=1)
