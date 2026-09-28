from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .artifacts import (
    file_record,
    file_sha256,
    json_digest,
    load_artifact_manifest,
    write_artifact_manifest,
)
from .utils import atomic_json_dump

SAE_TRAINING_IMPLEMENTATION_VERSION = (
    "batchtopk-separate-centers-preemptive-auxk-lr-floor-v3"
)
SAE_PARAMETER_SCHEMA_VERSION = "batchtopk-pre-bias-v2"

# A Joint Chunk SAE has one shared latent coordinate system and two
# task-specific readout dictionaries.  Keep the names deliberately explicit:
# they are part of the checkpoint/interpretation API, not just implementation
# details.
DecoderHead = Literal["mean", "cross"]


class BatchTopKSAE(nn.Module):
    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        *,
        batch_topk_candidate_multiplier: float = 2.0,
        batch_topk_bf16_histogram: bool = False,
        histogram_chunk_elements: int = 16_777_216,
        decoder_backend: str = "dense",
    ) -> None:
        super().__init__()
        if not 0 < k <= dict_size:
            raise ValueError(f"k must be in [1, {dict_size}], got {k}")
        if batch_topk_candidate_multiplier < 1.0:
            raise ValueError("batch_topk_candidate_multiplier must be at least 1")
        self.activation_dim = int(activation_dim)
        self.dict_size = int(dict_size)
        self.k = int(k)
        self.batch_topk_candidate_multiplier = float(
            batch_topk_candidate_multiplier
        )
        self.batch_topk_bf16_histogram = bool(batch_topk_bf16_histogram)
        if decoder_backend not in {"dense", "sparse", "auto"}:
            raise ValueError(
                "decoder_backend must be one of dense, sparse, or auto"
            )
        self.decoder_backend = decoder_backend
        self._sparse_decoder_failed = False
        self.histogram_chunk_elements = max(1, int(histogram_chunk_elements))
        self._batch_topk_gather_buffer: torch.Tensor | None = None
        decoder = torch.randn(activation_dim, dict_size, dtype=torch.float32)
        decoder /= decoder.norm(dim=0, keepdim=True).clamp_min(1e-12)
        self.encoder_weight = nn.Parameter(decoder.T.clone())
        self.encoder_bias = nn.Parameter(torch.zeros(dict_size))
        self.decoder_weight = nn.Parameter(decoder)
        self.decoder_bias = nn.Parameter(torch.zeros(activation_dim))
        # Input and target distributions are identical for Token/Mean, but
        # differ for Cross (owning mean -> partner mean). Keep their centers
        # independent so Cross does not force one bias to serve two roles.
        self.register_buffer("pre_bias", torch.zeros(activation_dim))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer("activation_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("feature_counts", torch.zeros(dict_size, dtype=torch.int64))
        self.register_buffer(
            "num_occurrences_since_fired",
            torch.zeros(dict_size, dtype=torch.int64),
            persistent=False,
        )

    def pre_activations(
        self,
        x: torch.Tensor,
        feature_ids: torch.Tensor | None = None,
        *,
        apply_relu: bool = True,
    ) -> torch.Tensor:
        centered = x - self.pre_bias
        if feature_ids is None:
            values = F.linear(centered, self.encoder_weight, self.encoder_bias)
        else:
            values = F.linear(
                centered,
                self.encoder_weight[feature_ids],
                self.encoder_bias[feature_ids],
            )
        return F.relu(values) if apply_relu else values

    def _auxiliary_reconstruction(
        self,
        scaled: torch.Tensor,
        feature_ids: torch.Tensor | None,
        auxiliary_k: int,
        unique_rows: torch.Tensor | None = None,
        dedup_inverse: torch.Tensor | None = None,
        decoder_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct the residual with a small set of underused features.

        The sparse COO path avoids allocating a second dense ``[batch,
        dict_size]`` feature matrix. The auxiliary branch is training-only;
        the exact primary BatchTopK reconstruction remains unchanged.
        """
        if feature_ids is None or feature_ids.numel() == 0 or auxiliary_k <= 0:
            return scaled.new_zeros((scaled.shape[0], self.activation_dim))
        feature_ids = feature_ids.to(device=scaled.device, dtype=torch.long)
        auxiliary_inputs = (
            scaled
            if unique_rows is None
            else scaled.index_select(0, unique_rows.to(scaled.device, dtype=torch.long))
        )
        aux_pre = self.pre_activations(
            auxiliary_inputs,
            feature_ids,
            apply_relu=False,
        )
        take = min(int(auxiliary_k), int(feature_ids.numel()))
        values, local_indices = aux_pre.topk(take, dim=1, sorted=False)
        # A small leaky branch gives features whose raw pre-activation is
        # currently negative a nonzero rescue gradient.
        values = F.leaky_relu(values, negative_slope=0.01)
        selected_ids = feature_ids.index_select(0, local_indices.reshape(-1)).reshape(
            local_indices.shape
        )
        rows = torch.arange(
            auxiliary_inputs.shape[0], device=scaled.device, dtype=torch.long
        ).repeat_interleave(take)
        cols = selected_ids.reshape(-1)
        indices = torch.stack((rows, cols), dim=0)
        sparse_features = torch.sparse_coo_tensor(
            indices,
            values.reshape(-1),
            size=(auxiliary_inputs.shape[0], self.dict_size),
            device=scaled.device,
            dtype=values.dtype,
        ).coalesce()
        # CUDA does not implement sparse matmul for BF16. AuxK is a rescue
        # branch rather than the primary path, so compute this small product
        # in FP32 while retaining autograd paths to both encoder values and
        # decoder columns.
        if decoder_weight is None:
            decoder_weight = self.decoder_weight
        with torch.autocast(device_type=scaled.device.type, enabled=False):
            reconstruction = torch.sparse.mm(
                sparse_features.float(),
                decoder_weight.transpose(0, 1).float(),
            )
        if dedup_inverse is not None:
            reconstruction = reconstruction.index_select(
                0,
                dedup_inverse.to(reconstruction.device, dtype=torch.long),
            )
        return reconstruction

    @torch.no_grad()
    def auxiliary_feature_ids(
        self,
        dead_feature_threshold: int,
        max_features: int,
    ) -> torch.Tensor:
        if dead_feature_threshold <= 0 or max_features <= 0:
            return torch.empty(
                0,
                dtype=torch.long,
                device=self.num_occurrences_since_fired.device,
            )
        dead = torch.nonzero(
            self.num_occurrences_since_fired >= dead_feature_threshold,
            as_tuple=False,
        ).flatten()
        if dead.numel() <= max_features:
            return dead
        ages = self.num_occurrences_since_fired.index_select(0, dead)
        # Stable ordering makes equal-age candidate selection deterministic.
        order = torch.argsort(ages, descending=True, stable=True)
        return dead.index_select(0, order[:max_features])

    def _gather_buffer(
        self,
        elements: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        current = self._batch_topk_gather_buffer
        if (
            current is None
            or current.numel() < elements
            or current.dtype != dtype
            or current.device != device
        ):
            current = torch.empty(elements, dtype=dtype, device=device)
            self._batch_topk_gather_buffer = current
        return current.narrow(0, 0, elements)

    def batch_topk(
        self,
        pre: torch.Tensor,
        distributed: bool,
        *,
        return_activity_counts: bool = False,
        sample_mask: torch.Tensor | None = None,
        global_sample_count: int | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        world_size = dist.get_world_size() if distributed and dist.is_initialized() else 1
        if sample_mask is not None:
            sample_mask = sample_mask.to(device=pre.device, dtype=torch.bool).reshape(-1)
            if sample_mask.numel() != pre.shape[0]:
                raise ValueError("sample_mask does not match BatchTopK rows")
            # Keep the autograd version of ReLU's output untouched. Invalid
            # rows are removed from selection through a derived tensor.
            selection_pre = pre.masked_fill(~sample_mask.unsqueeze(1), 0)
        else:
            selection_pre = pre
        if global_sample_count is None:
            global_sample_count = pre.shape[0] * world_size
        global_budget = self.k * int(global_sample_count)
        if global_budget <= 0:
            raise ValueError("BatchTopK requires at least one accepted sample")
        flat = selection_pre.flatten()
        maximum_candidates = min(global_budget, flat.numel())
        use_histogram = self.batch_topk_bf16_histogram and pre.dtype == torch.bfloat16
        if use_histogram:
            histogram = torch.zeros(65_536, dtype=torch.int64, device=pre.device)
            bit_values = flat.contiguous().view(torch.uint16)
            for start in range(0, bit_values.numel(), self.histogram_chunk_elements):
                values = bit_values.narrow(
                    0,
                    start,
                    min(self.histogram_chunk_elements, bit_values.numel() - start),
                ).to(torch.int64)
                histogram.add_(torch.bincount(values, minlength=65_536))
            if distributed and dist.is_initialized():
                dist.all_reduce(histogram, op=dist.ReduceOp.SUM)
            descending = histogram.flip(0).cumsum(0)
            budget_tensor = torch.tensor(
                global_budget,
                dtype=torch.int64,
                device=pre.device,
            )
            offset = torch.searchsorted(descending, budget_tensor, right=False)
            threshold_bits = (65_535 - offset).to(torch.uint16)
            threshold = threshold_bits.view(torch.bfloat16)
        elif distributed and dist.is_initialized() and dist.get_world_size() > 1:
            expected_local = math.ceil(global_budget / world_size)
            candidate_count = min(
                maximum_candidates,
                max(
                    expected_local,
                    math.ceil(
                        expected_local * self.batch_topk_candidate_multiplier
                    ),
                ),
            )
            while True:
                local_values = (
                    flat.topk(candidate_count, sorted=False).values.detach().contiguous()
                )
                gathered = self._gather_buffer(
                    world_size * candidate_count,
                    dtype=local_values.dtype,
                    device=local_values.device,
                )
                dist.all_gather_into_tensor(gathered, local_values)
                threshold = (
                    gathered.topk(global_budget, sorted=False).values.min()
                )
                # If every omitted local value is <= the tentative threshold,
                # the gathered set contains every value that can change the
                # exact global threshold. Equal omitted ties are handled below
                # from the complete pre-activation tensor.
                incomplete = (local_values.min() > threshold).to(torch.int32)
                if candidate_count >= maximum_candidates:
                    incomplete.zero_()
                dist.all_reduce(incomplete, op=dist.ReduceOp.MAX)
                if not int(incomplete.item()):
                    break
                candidate_count = min(
                    maximum_candidates,
                    max(candidate_count + 1, candidate_count * 2),
                )
        else:
            candidate_count = maximum_candidates
            local_values = flat.topk(candidate_count, sorted=False).values.detach()
            threshold = local_values.min()

        threshold_typed = threshold.to(pre.dtype)
        mask = pre > threshold_typed
        if sample_mask is not None:
            mask &= sample_mask.unsqueeze(1)
        local_greater = mask.sum(dtype=torch.int64)
        global_greater = local_greater.clone()
        if distributed and dist.is_initialized():
            dist.all_reduce(global_greater, op=dist.ReduceOp.SUM)
        ties_needed = max(0, global_budget - int(global_greater.item()))
        if ties_needed:
            ties = selection_pre == threshold_typed
            if sample_mask is not None:
                ties &= sample_mask.unsqueeze(1)
            tie_indices = torch.nonzero(ties.flatten(), as_tuple=False).flatten()
            local_ties = torch.tensor([tie_indices.numel()], dtype=torch.int64, device=pre.device)
            if distributed and dist.is_initialized() and world_size > 1:
                tie_counts = torch.empty(
                    world_size, dtype=torch.int64, device=pre.device
                )
                dist.all_gather_into_tensor(tie_counts, local_ties)
                rank = dist.get_rank()
                tie_offset = int(tie_counts[:rank].sum().item())
            else:
                tie_offset = 0
            local_take = min(tie_indices.numel(), max(0, ties_needed - tie_offset))
            if local_take:
                mask.flatten()[tie_indices[:local_take]] = True
        encoded = pre * mask
        if return_activity_counts:
            active_mask = encoded.detach() != 0
            active_per_sample = active_mask.sum(dim=1, dtype=torch.int64)
            active_per_feature = active_mask.sum(dim=0, dtype=torch.int64)
            return encoded, threshold, active_per_sample, active_per_feature
        return encoded, threshold

    def encode(
        self,
        x: torch.Tensor,
        *,
        use_threshold: bool = True,
        feature_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x * self.activation_scale.to(x.dtype)
        pre = self.pre_activations(x, feature_ids)
        if use_threshold:
            return pre * (pre > self.threshold.to(pre.dtype))
        if feature_ids is not None:
            raise ValueError("feature_ids is only valid with threshold inference")
        return self.batch_topk(pre, distributed=False)[0]

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        use_sparse = self.decoder_backend == "sparse"
        if self.decoder_backend == "auto":
            nonzero = int((features.detach() != 0).sum().item())
            density = nonzero / max(1, features.numel())
            use_sparse = features.device.type == "cuda" and density <= 0.01
        if use_sparse and not self._sparse_decoder_failed:
            try:
                sparse = features.to_sparse_coo().coalesce()
                return torch.sparse.mm(
                    sparse,
                    self.decoder_weight.transpose(0, 1),
                ) + self.decoder_bias
            except RuntimeError as error:
                self._sparse_decoder_failed = True
                print(
                    "[chunk-saes] sparse decoder unavailable for this "
                    f"dtype/device; falling back to dense: {error}",
                    flush=True,
                )
        return F.linear(features, self.decoder_weight) + self.decoder_bias

    def decode_prefix(
        self,
        features: torch.Tensor,
        feature_count: int,
    ) -> torch.Tensor:
        """Decode the leading Matryoshka feature group plus the shared bias."""

        feature_count = int(feature_count)
        if not 0 < feature_count <= self.dict_size:
            raise ValueError(
                f"feature_count must be in [1, {self.dict_size}], got {feature_count}"
            )
        return F.linear(
            features[:, :feature_count],
            self.decoder_weight[:, :feature_count],
        ) + self.decoder_bias

    def forward(
        self,
        x: torch.Tensor,
        *,
        batch_topk: bool = False,
        distributed: bool = False,
        return_activity_counts: bool = False,
        sample_mask: torch.Tensor | None = None,
        global_sample_count: int | None = None,
        unique_rows: torch.Tensor | None = None,
        dedup_inverse: torch.Tensor | None = None,
        auxiliary_feature_ids: torch.Tensor | None = None,
        auxiliary_k: int = 0,
        return_auxiliary: bool = False,
        temporal_previous: torch.Tensor | None = None,
        temporal_high_features: int = 0,
        temporal_sample_mask: torch.Tensor | None = None,
        temporal_global_sample_count: int | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        scaled = x * self.activation_scale.to(x.dtype)
        if (unique_rows is None) != (dedup_inverse is None):
            raise ValueError("unique_rows and dedup_inverse must be provided together")
        if unique_rows is None:
            pre = self.pre_activations(scaled)
        else:
            unique_rows = unique_rows.to(device=x.device, dtype=torch.long)
            dedup_inverse = dedup_inverse.to(device=x.device, dtype=torch.long)
            if dedup_inverse.shape != (x.shape[0],):
                raise ValueError("dedup_inverse does not match the input rows")
            unique_pre = self.pre_activations(
                scaled.index_select(0, unique_rows)
            )
            pre = unique_pre.index_select(0, dedup_inverse)
        if batch_topk:
            if temporal_previous is not None:
                if temporal_sample_mask is None:
                    raise ValueError(
                        "temporal_sample_mask is required with temporal_previous"
                    )
                pair_mask = temporal_sample_mask.to(
                    device=pre.device,
                    dtype=torch.bool,
                ).reshape(-1)
                if sample_mask is not None:
                    pair_mask = pair_mask & sample_mask.to(
                        device=pre.device,
                        dtype=torch.bool,
                    ).reshape(-1)
                temporal_current_features, threshold = self.batch_topk(
                    pre,
                    distributed=distributed,
                    sample_mask=pair_mask,
                    global_sample_count=temporal_global_sample_count,
                )
                # T-SAE's current and previous codes are selected over the exact
                # adjacent-pair batch. To preserve the four-way one-billion-row
                # reconstruction universe, independently encode the comparatively
                # rare chunk-boundary rows with row-wise K and merge them into the
                # disjoint zero rows left by pair BatchTopK.
                accepted_mask = (
                    torch.ones_like(pair_mask)
                    if sample_mask is None
                    else sample_mask.to(
                        device=pre.device,
                        dtype=torch.bool,
                    ).reshape(-1)
                )
                boundary_mask = accepted_mask & ~pair_mask
                if bool(boundary_mask.any()):
                    boundary_indices = torch.nonzero(
                        boundary_mask,
                        as_tuple=False,
                    ).flatten()
                    boundary_pre = pre.index_select(0, boundary_indices)
                    boundary_values, boundary_columns = boundary_pre.topk(
                        min(self.k, self.dict_size),
                        dim=1,
                        sorted=False,
                    )
                    boundary_rows = torch.zeros_like(boundary_pre).scatter(
                        1,
                        boundary_columns,
                        boundary_values,
                    )
                    boundary_dense = torch.zeros_like(pre).index_copy(
                        0,
                        boundary_indices,
                        boundary_rows,
                    )
                    temporal_current_features = (
                        temporal_current_features + boundary_dense
                    )
                features = temporal_current_features
                if return_activity_counts:
                    active_mask = features.detach() != 0
                    active_per_sample = active_mask.sum(
                        dim=1,
                        dtype=torch.int64,
                    )
                    active_per_feature = active_mask.sum(
                        dim=0,
                        dtype=torch.int64,
                    )
            elif return_activity_counts:
                (
                    features,
                    threshold,
                    active_per_sample,
                    active_per_feature,
                ) = self.batch_topk(
                    pre,
                    distributed=distributed,
                    return_activity_counts=True,
                    sample_mask=sample_mask,
                    global_sample_count=global_sample_count,
                )
            else:
                features, threshold = self.batch_topk(
                    pre,
                    distributed=distributed,
                    sample_mask=sample_mask,
                    global_sample_count=global_sample_count,
                )
        else:
            threshold = self.threshold
            features = pre * (pre > threshold.to(pre.dtype))
            if return_activity_counts:
                active_mask = features.detach() != 0
                active_per_sample = active_mask.sum(dim=1, dtype=torch.int64)
                active_per_feature = active_mask.sum(dim=0, dtype=torch.int64)
        reconstructed = self.decode(features)
        temporal_output = None
        if temporal_previous is not None:
            if unique_rows is not None:
                raise ValueError(
                    "temporal_previous is incompatible with chunk-input deduplication"
                )
            if temporal_previous.shape != x.shape:
                raise ValueError("temporal_previous must match the current input shape")
            if not 0 < int(temporal_high_features) < self.dict_size:
                raise ValueError(
                    "temporal_high_features must define a non-empty strict prefix"
                )
            previous_scaled = temporal_previous * self.activation_scale.to(
                temporal_previous.dtype
            )
            previous_pre = self.pre_activations(previous_scaled)
            if batch_topk:
                previous_features, _previous_threshold = self.batch_topk(
                    previous_pre,
                    distributed=distributed,
                    sample_mask=temporal_sample_mask,
                    global_sample_count=temporal_global_sample_count,
                )
            else:
                if temporal_current_features is None:
                    temporal_current_features = pre * (
                        pre > self.threshold.to(pre.dtype)
                    )
                    if temporal_sample_mask is not None:
                        temporal_current_features = (
                            temporal_current_features
                            * temporal_sample_mask.to(
                                temporal_current_features.dtype
                            ).unsqueeze(1)
                        )
                previous_features = previous_pre * (
                    previous_pre > self.threshold.to(previous_pre.dtype)
                )
            temporal_output = (
                features,
                previous_features,
                self.decode_prefix(
                    features,
                    int(temporal_high_features),
                ),
            )
        output = (reconstructed, features, threshold)
        if return_activity_counts:
            result = (*output, active_per_sample, active_per_feature)
            if return_auxiliary:
                result = (
                    *result,
                    self._auxiliary_reconstruction(
                        scaled,
                        auxiliary_feature_ids,
                        auxiliary_k,
                        unique_rows,
                        dedup_inverse,
                    ),
                )
            if temporal_output is not None:
                result = (*result, *temporal_output)
            return result
        if return_auxiliary:
            result = (
                *output,
                self._auxiliary_reconstruction(
                    scaled,
                    auxiliary_feature_ids,
                    auxiliary_k,
                    unique_rows,
                    dedup_inverse,
                ),
            )
            if temporal_output is not None:
                result = (*result, *temporal_output)
            return result
        if temporal_output is not None:
            return (*output, *temporal_output)
        return output

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Old checkpoints used decoder_bias for both centers. Preserve their
        # inference behavior while allowing the new schema to load strictly.
        pre_key = prefix + "pre_bias"
        decoder_key = prefix + "decoder_bias"
        if pre_key not in state_dict and decoder_key in state_dict:
            state_dict[pre_key] = state_dict[decoder_key].clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def update_dead_feature_stats_(
        self,
        active_per_feature: torch.Tensor,
        *,
        global_samples: int,
        distributed: bool,
    ) -> None:
        did_fire = (active_per_feature > 0).to(torch.uint8)
        if distributed and dist.is_initialized():
            dist.all_reduce(did_fire, op=dist.ReduceOp.MAX)
        self.num_occurrences_since_fired.add_(int(global_samples))
        self.num_occurrences_since_fired.masked_fill_(did_fire.bool(), 0)

    @torch.no_grad()
    def dead_feature_count(self, dead_feature_threshold: int) -> int:
        if dead_feature_threshold <= 0:
            return 0
        return int(
            (self.num_occurrences_since_fired >= dead_feature_threshold).sum().item()
        )

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            norms = self.decoder_weight.norm(dim=0).clamp_min(1e-12)
            self.decoder_weight.div_(norms)

    def remove_parallel_decoder_gradient_(self) -> None:
        if self.decoder_weight.grad is None:
            return
        directions = F.normalize(self.decoder_weight.detach(), dim=0)
        parallel = (self.decoder_weight.grad * directions).sum(dim=0, keepdim=True)
        self.decoder_weight.grad.sub_(parallel * directions)

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().contiguous().clone()
            for key, value in self.state_dict().items()
        }


class JointChunkSAE(BatchTopKSAE):
    """Shared BatchTopK encoder for Joint Mean/Cross chunk objectives.

    The legacy/default layout uses independent Mean and Cross decoder matrices.
    When ``cross_prefix`` is set, the nested layout instead uses one decoder
    matrix: Mean reads the full dictionary and Cross reads only its leading
    ``cross_prefix`` columns. Both layouts retain independent output biases.
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        *,
        batch_topk_candidate_multiplier: float = 2.0,
        batch_topk_bf16_histogram: bool = False,
        histogram_chunk_elements: int = 16_777_216,
        decoder_backend: str = "dense",
        cross_prefix: int | None = None,
    ) -> None:
        super().__init__(
            activation_dim,
            dict_size,
            k,
            batch_topk_candidate_multiplier=batch_topk_candidate_multiplier,
            batch_topk_bf16_histogram=batch_topk_bf16_histogram,
            histogram_chunk_elements=histogram_chunk_elements,
            decoder_backend=decoder_backend,
        )
        if cross_prefix is not None:
            cross_prefix = int(cross_prefix)
            if not 0 < cross_prefix < self.dict_size:
                raise ValueError(
                    "cross_prefix must define a non-empty strict dictionary prefix; "
                    f"got {cross_prefix} for dict_size={self.dict_size}"
                )
        self._cross_prefix_size = cross_prefix or 0
        self.register_buffer(
            "cross_prefix",
            torch.tensor(self._cross_prefix_size, dtype=torch.int64),
            persistent=False,
        )
        if cross_prefix is None:
            self.decoder_cross_weight = nn.Parameter(
                self.decoder_weight.detach().clone()
            )
        else:
            self.register_parameter("decoder_cross_weight", None)
        self.decoder_cross_bias = nn.Parameter(self.decoder_bias.detach().clone())

    @property
    def nested(self) -> bool:
        return self._cross_prefix_size > 0

    @property
    def cross_prefix_size(self) -> int:
        return self._cross_prefix_size

    @property
    def decoder_head_names(self) -> tuple[str, str]:
        """Names of the two readout heads in stable checkpoint order."""

        return ("mean", "cross")

    @property
    def decoder_mean_weight(self) -> torch.Tensor:
        """Explicit alias for the legacy-named Mean decoder matrix."""

        return self.decoder_weight

    @property
    def decoder_mean_bias(self) -> torch.Tensor:
        """Explicit alias for the legacy-named Mean decoder bias."""

        return self.decoder_bias

    @staticmethod
    def _validate_decoder_head(head: str) -> DecoderHead:
        if head not in ("mean", "cross"):
            raise ValueError(
                f"unknown Joint decoder head {head!r}; expected 'mean' or 'cross'"
            )
        return cast(DecoderHead, head)

    def decoder_parameters(
        self,
        head: DecoderHead | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(weight, bias)`` for an explicitly named readout head.

        Feature coordinates live in the shared encoder, while decoder columns
        are head-specific.  This accessor prevents callers from guessing which
        matrix a Joint feature should use.
        """

        if head is None:
            raise ValueError(
                "Joint decoder_parameters requires head='mean' or head='cross'"
            )
        head = self._validate_decoder_head(head)
        if head == "mean":
            return self.decoder_weight, self.decoder_bias
        if self.nested:
            return (
                self.decoder_weight[:, : self.cross_prefix_size],
                self.decoder_cross_bias,
            )
        return self.decoder_cross_weight, self.decoder_cross_bias

    def decoder_vector(
        self,
        feature_id: int,
        *,
        head: DecoderHead | None = None,
    ) -> torch.Tensor:
        """Return one feature's decoder atom from a named head.

        The returned vector is a view into the model parameter.  ``head`` is
        intentionally required for interpretation: use ``head='mean'`` for
        owning-chunk reconstruction and ``head='cross'`` for partner-chunk
        prediction.
        """

        if head is None and not self.nested:
            raise ValueError(
                "Joint decoder_vector requires head='mean' or head='cross'"
            )
        feature_id = int(feature_id)
        if not 0 <= feature_id < self.dict_size:
            raise ValueError(
                f"feature_id must be in [0, {self.dict_size}), got {feature_id}"
            )
        if head == "cross" and self.nested and feature_id >= self.cross_prefix_size:
            raise ValueError(
                f"feature {feature_id} is self_only in the nested Joint Chunk SAE; "
                f"Cross uses only feature ids [0, {self.cross_prefix_size})"
            )
        if head is None:
            return self.decoder_weight[:, feature_id]
        weight, _ = self.decoder_parameters(head)
        return weight[:, feature_id]

    def feature_role(self, feature_id: int) -> Literal["shared", "self_only"]:
        """Return a nested feature's role in the Mean/Cross decomposition."""

        feature_id = int(feature_id)
        if not 0 <= feature_id < self.dict_size:
            raise ValueError(
                f"feature_id must be in [0, {self.dict_size}), got {feature_id}"
            )
        if not self.nested:
            raise ValueError("feature roles are defined only for nested Joint Chunk SAEs")
        return "shared" if feature_id < self.cross_prefix_size else "self_only"

    def _decode_head(
        self,
        features: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        use_sparse = self.decoder_backend == "sparse"
        if self.decoder_backend == "auto":
            nonzero = int((features.detach() != 0).sum().item())
            density = nonzero / max(1, features.numel())
            use_sparse = features.device.type == "cuda" and density <= 0.01
        if use_sparse and not self._sparse_decoder_failed:
            try:
                sparse = features.to_sparse_coo().coalesce()
                return torch.sparse.mm(sparse, weight.transpose(0, 1)) + bias
            except RuntimeError as error:
                self._sparse_decoder_failed = True
                print(
                    "[chunk-saes] sparse decoder unavailable for this "
                    f"dtype/device; falling back to dense: {error}",
                    flush=True,
                )
        return F.linear(features, weight) + bias

    def decode_head(
        self,
        features: torch.Tensor,
        *,
        head: DecoderHead,
    ) -> torch.Tensor:
        """Decode shared features through the selected task head."""

        if head == "cross" and self.nested:
            return self.decode_cross_prefix(features)
        weight, bias = self.decoder_parameters(head)
        return self._decode_head(features, weight, bias)

    def decode(
        self,
        features: torch.Tensor,
        *,
        head: DecoderHead = "mean",
    ) -> torch.Tensor:
        """Decode through ``head``; the default preserves legacy Mean behavior."""

        return self.decode_head(features, head=head)

    def decode_cross(self, features: torch.Tensor) -> torch.Tensor:
        if self.nested:
            return self.decode_cross_prefix(features)
        return self.decode_head(features, head="cross")

    def decode_cross_prefix(self, features: torch.Tensor) -> torch.Tensor:
        """Decode the Cross target from the nested shared-feature prefix."""

        if not self.nested:
            raise ValueError(
                "decode_cross_prefix requires a nested Joint Chunk SAE"
            )
        prefix = self.cross_prefix_size
        return self._decode_head(
            features[:, :prefix],
            self.decoder_weight[:, :prefix],
            self.decoder_cross_bias,
        )

    def decode_mean(self, features: torch.Tensor) -> torch.Tensor:
        """Decode shared features through the owning-chunk Mean head."""

        return self.decode_head(features, head="mean")

    def forward(self, x: torch.Tensor, *, joint: bool = False, **kwargs: Any):
        """Keep the DDP call on ``forward`` while exposing the two-head path."""

        if joint:
            return self.forward_joint(x, **kwargs)
        return super().forward(x, **kwargs)

    def forward_joint(
        self,
        x: torch.Tensor,
        *,
        batch_topk: bool = False,
        distributed: bool = False,
        return_activity_counts: bool = False,
        sample_mask: torch.Tensor | None = None,
        global_sample_count: int | None = None,
        unique_rows: torch.Tensor | None = None,
        dedup_inverse: torch.Tensor | None = None,
        auxiliary_feature_ids: torch.Tensor | None = None,
        auxiliary_k: int = 0,
        return_auxiliary: bool = False,
    ) -> tuple:
        """Encode once, then decode the shared code with both heads."""

        primary = super().forward(
            x,
            batch_topk=batch_topk,
            distributed=distributed,
            return_activity_counts=return_activity_counts,
            sample_mask=sample_mask,
            global_sample_count=global_sample_count,
            unique_rows=unique_rows,
            dedup_inverse=dedup_inverse,
            return_auxiliary=False,
        )
        if return_activity_counts:
            reconstructed_mean, features, threshold, active_counts, active_per_feature = (
                primary[:5]
            )
        else:
            reconstructed_mean, features, threshold = primary[:3]
            active_counts = None
            active_per_feature = None
        reconstructed_cross = self.decode_cross(features)
        result = (
            reconstructed_mean,
            reconstructed_cross,
            features,
            threshold,
            active_counts,
            active_per_feature,
        )
        if return_auxiliary:
            scaled = x * self.activation_scale.to(x.dtype)
            auxiliary_mean = self._auxiliary_reconstruction(
                scaled,
                auxiliary_feature_ids,
                auxiliary_k,
                unique_rows,
                dedup_inverse,
                decoder_weight=self.decoder_weight,
            )
            if not self.nested:
                assert self.decoder_cross_weight is not None
                auxiliary_cross = self._auxiliary_reconstruction(
                    scaled,
                    auxiliary_feature_ids,
                    auxiliary_k,
                    unique_rows,
                    dedup_inverse,
                    decoder_weight=self.decoder_cross_weight,
                )
                result = (*result, auxiliary_mean, auxiliary_cross)
            else:
                # In the nested variant AuxK only serves the full Mean decoder.
                # This is sufficient for dead-feature rescue and avoids an
                # ambiguous Cross auxiliary branch when candidates are suffix
                # (self-only) features.
                result = (*result, auxiliary_mean)
        return result

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        super().normalize_decoder_()
        if self.decoder_cross_weight is None:
            return
        norms = self.decoder_cross_weight.norm(dim=0).clamp_min(1e-12)
        self.decoder_cross_weight.div_(norms)

    def remove_parallel_decoder_gradient_(self) -> None:
        super().remove_parallel_decoder_gradient_()
        if (
            self.decoder_cross_weight is None
            or self.decoder_cross_weight.grad is None
        ):
            return
        directions = F.normalize(self.decoder_cross_weight.detach(), dim=0)
        parallel = (
            self.decoder_cross_weight.grad * directions
        ).sum(dim=0, keepdim=True)
        self.decoder_cross_weight.grad.sub_(parallel * directions)


@dataclass
class SAECheckpoint:
    model: BatchTopKSAE
    config: dict


def save_sae(model: BatchTopKSAE, output_dir: str | Path, config: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.checkpoint_tensors(), str(output_dir / "sae.safetensors"))
    atomic_json_dump(config, output_dir / "config.json")


def save_inference_checkpoint(
    tensors: dict[str, torch.Tensor] | None,
    output_dir: str | Path,
    state: dict[str, Any],
    *,
    source_weights: str | Path | None = None,
) -> None:
    """Atomically publish a weights-only validation-best checkpoint.

    Best checkpoints are evaluation artifacts, not resume points. Keeping Adam
    state in every improving validation checkpoint writes roughly three times
    more data than evaluation requires. Full resumability remains provided by
    the periodic ``latest`` checkpoint.
    """

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        weights_path = tmp_dir / "sae.safetensors"
        if source_weights is not None:
            source_weights = Path(source_weights)
            try:
                os.link(source_weights, weights_path)
            except OSError:
                shutil.copyfile(source_weights, weights_path)
        else:
            if tensors is None:
                raise ValueError(
                    "weights-only checkpoint requires tensors or source_weights"
                )
            safe_tensors = {
                key: value.detach().cpu().contiguous()
                for key, value in tensors.items()
            }
            save_file(safe_tensors, str(weights_path))
        config = state.get("config")
        if not isinstance(config, dict):
            raise ValueError("weights-only checkpoint requires a config dictionary")
        atomic_json_dump(config, tmp_dir / "config.json")
        write_artifact_manifest(
            {
                "format": "chunk-saes-sae-checkpoint-v2",
                "complete": True,
                "resumable": False,
                "step": int(state.get("step", -1)),
                "samples_seen": int(state.get("samples_seen", -1)),
                "world_size": int(state.get("world_size", -1)),
                "best_metric_value": state.get("best_metric_value"),
                "best_step": state.get("best_step"),
                "config_digest": json_digest(config),
                "files": {
                    name: file_record(
                        tmp_dir / name,
                        relative_to=tmp_dir,
                        hash_content=name == "config.json",
                    )
                    for name in ("sae.safetensors", "config.json")
                },
            },
            tmp_dir / "checkpoint_manifest.json",
        )
        previous = output_dir.with_name(f".{output_dir.name}.previous")
        if previous.exists():
            shutil.rmtree(previous)
        if output_dir.exists():
            os.replace(output_dir, previous)
        os.replace(tmp_dir, output_dir)
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def save_training_checkpoint(
    model: BatchTopKSAE,
    optimizer: torch.optim.Optimizer,
    output_dir: str | Path,
    state: dict[str, Any],
) -> None:
    """Atomically publish a complete, resumable training checkpoint directory."""

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        save_file(model.checkpoint_tensors(), str(tmp_dir / "sae.safetensors"))
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        torch.save(state, tmp_dir / "training_state.pt")
        config = state.get("config")
        if isinstance(config, dict):
            atomic_json_dump(config, tmp_dir / "config.json")
        file_names = ["sae.safetensors", "optimizer.pt", "training_state.pt"]
        if isinstance(config, dict):
            file_names.append("config.json")
        write_artifact_manifest(
            {
                "format": "chunk-saes-sae-checkpoint-v2",
                "complete": True,
                "resumable": True,
                "step": int(state.get("step", -1)),
                "samples_seen": int(state.get("samples_seen", -1)),
                "world_size": int(state.get("world_size", -1)),
                "best_metric_value": state.get("best_metric_value"),
                "best_step": state.get("best_step"),
                "config_digest": json_digest(config) if isinstance(config, dict) else None,
                "files": {
                    name: file_record(
                        tmp_dir / name,
                        relative_to=tmp_dir,
                        hash_content=name == "config.json",
                    )
                    for name in file_names
                },
            },
            tmp_dir / "checkpoint_manifest.json",
        )
        previous = output_dir.with_name(f".{output_dir.name}.previous")
        if previous.exists():
            shutil.rmtree(previous)
        if output_dir.exists():
            os.replace(output_dir, previous)
        os.replace(tmp_dir, output_dir)
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def snapshot_training_checkpoint(
    model: BatchTopKSAE,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Detach a complete checkpoint payload before handing it to a writer."""

    tensors = model.checkpoint_tensors()

    def detach_to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: detach_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [detach_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(detach_to_cpu(item) for item in value)
        return copy.deepcopy(value)

    # Optimizer.state_dict() shares its nested per-parameter dictionaries with
    # the live optimizer. Build a new tree instead of replacing those tensors
    # in place, which would move rank 0's live Adam state to CPU.
    optimizer_state = detach_to_cpu(optimizer.state_dict())
    detached_state = detach_to_cpu(state)
    return tensors, optimizer_state, detached_state


def _save_training_checkpoint_snapshot_directory(
    tensors: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    output_dir: str | Path,
    state: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
            str(tmp_dir / "sae.safetensors"),
        )
        torch.save(optimizer_state, tmp_dir / "optimizer.pt")
        torch.save(state, tmp_dir / "training_state.pt")
        config = state.get("config")
        if isinstance(config, dict):
            atomic_json_dump(config, tmp_dir / "config.json")
        file_names = ["sae.safetensors", "optimizer.pt", "training_state.pt"]
        if isinstance(config, dict):
            file_names.append("config.json")
        write_artifact_manifest(
            {
                "format": "chunk-saes-sae-checkpoint-v2",
                "complete": True,
                "resumable": True,
                "step": int(state.get("step", -1)),
                "samples_seen": int(state.get("samples_seen", -1)),
                "world_size": int(state.get("world_size", -1)),
                "best_metric_value": state.get("best_metric_value"),
                "best_step": state.get("best_step"),
                "config_digest": json_digest(config) if isinstance(config, dict) else None,
                "files": {
                    name: file_record(
                        tmp_dir / name,
                        relative_to=tmp_dir,
                        hash_content=name == "config.json",
                    )
                    for name in file_names
                },
            },
            tmp_dir / "checkpoint_manifest.json",
        )
        previous = output_dir.with_name(f".{output_dir.name}.previous")
        if previous.exists():
            shutil.rmtree(previous)
        if output_dir.exists():
            os.replace(output_dir, previous)
        os.replace(tmp_dir, output_dir)
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def _mirror_checkpoint_directory(source_dir: Path, output_dir: Path) -> None:
    """Copy a complete checkpoint and publish it only after byte verification."""

    manifest = load_artifact_manifest(
        source_dir / "checkpoint_manifest.json",
        expected_format="chunk-saes-sae-checkpoint-v2",
        verify_files=True,
    )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"checkpoint manifest has no file map: {source_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for record in files.values():
            if not isinstance(record, dict):
                raise ValueError(f"invalid checkpoint file record: {record!r}")
            relative = Path(str(record["path"]))
            source = source_dir / relative
            destination = tmp_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            expected_size = int(record["bytes"])
            if destination.stat().st_size != expected_size:
                raise IOError(
                    f"checkpoint mirror size mismatch: {source} -> {destination}"
                )
            if file_sha256(source) != file_sha256(destination):
                raise IOError(
                    f"checkpoint mirror digest mismatch: {source} -> {destination}"
                )

        # The manifest is the publication marker and must be copied last.
        shutil.copyfile(
            source_dir / "checkpoint_manifest.json",
            tmp_dir / "checkpoint_manifest.json",
        )
        load_artifact_manifest(
            tmp_dir / "checkpoint_manifest.json",
            expected_format="chunk-saes-sae-checkpoint-v2",
            verify_files=True,
        )
        previous = output_dir.with_name(f".{output_dir.name}.previous")
        if previous.exists():
            shutil.rmtree(previous)
        if output_dir.exists():
            os.replace(output_dir, previous)
        os.replace(tmp_dir, output_dir)
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def save_training_checkpoint_snapshot(
    tensors: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    output_dir: str | Path,
    state: dict[str, Any],
    staging_dir: str | Path | None = None,
) -> None:
    """Write a detached snapshot locally, then publish a verified remote copy."""

    output_dir = Path(output_dir)
    if staging_dir is None:
        _save_training_checkpoint_snapshot_directory(
            tensors, optimizer_state, output_dir, state
        )
        return
    staging_dir = Path(staging_dir)
    if staging_dir.resolve() == output_dir.resolve():
        _save_training_checkpoint_snapshot_directory(
            tensors, optimizer_state, output_dir, state
        )
        return
    _save_training_checkpoint_snapshot_directory(
        tensors, optimizer_state, staging_dir, state
    )
    _mirror_checkpoint_directory(staging_dir, output_dir)


def load_training_checkpoint(
    path: str | Path,
    model: BatchTopKSAE,
    optimizer: torch.optim.Optimizer,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"training checkpoint directory not found: {path}")
    manifest_path = path / "checkpoint_manifest.json"
    if manifest_path.is_file():
        from .artifacts import load_artifact_manifest

        manifest = load_artifact_manifest(
            manifest_path,
            expected_format="chunk-saes-sae-checkpoint-v2",
            verify_files=True,
        )
        if manifest.get("complete") is not True:
            raise ValueError(f"training checkpoint is incomplete: {path}")
        if manifest.get("resumable", True) is not True:
            raise ValueError(f"checkpoint is weights-only and not resumable: {path}")
    model.load_state_dict(load_file(str(path / "sae.safetensors"), device="cpu"))
    model.to(map_location)
    optimizer.load_state_dict(
        torch.load(path / "optimizer.pt", map_location=map_location, weights_only=False)
    )
    state = torch.load(
        path / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state, dict):
        raise ValueError(f"invalid training state in {path}")
    return state


def load_sae(path: str | Path, device: str = "cpu") -> SAECheckpoint:
    path = Path(path)
    if path.is_file():
        weights_path, config_path = path, path.with_name("config.json")
    else:
        weights_path, config_path = path / "sae.safetensors", path / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    is_joint = (
        int(config.get("decoder_heads", 1)) == 2
        or config.get("joint_chunk_layout") == "nested_prefix"
        or config.get("joint_cross_prefix") is not None
    )
    common_kwargs = {
        "batch_topk_candidate_multiplier": float(
            config.get("batch_topk_candidate_multiplier", 2.0)
        ),
        "batch_topk_bf16_histogram": bool(
            config.get("batch_topk_bf16_histogram", False)
        ),
        "decoder_backend": str(config.get("decoder_backend", "dense")),
    }
    if is_joint:
        model = JointChunkSAE(
            config["activation_dim"],
            config["dict_size"],
            config["k"],
            cross_prefix=(
                int(config["joint_cross_prefix"])
                if config.get("joint_cross_prefix") is not None
                else None
            ),
            **common_kwargs,
        )
    else:
        model = BatchTopKSAE(
            config["activation_dim"],
            config["dict_size"],
            config["k"],
            **common_kwargs,
        )
    model.load_state_dict(load_file(str(weights_path), device=device))
    model.to(device).eval()
    return SAECheckpoint(model, config)


def learning_rate_multiplier(
    step: int,
    steps: int,
    warmup_steps: int,
    min_ratio: float = 0.0,
) -> float:
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, steps - warmup_steps)
    cosine = 0.5 * (
        1.0 + math.cos(math.pi * min(1.0, max(0.0, progress)))
    )
    return max(min_ratio, cosine)
