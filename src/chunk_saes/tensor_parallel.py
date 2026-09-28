from __future__ import annotations

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .sae import BatchTopKSAE
from .sae import SAE_PARAMETER_SCHEMA_VERSION, SAE_TRAINING_IMPLEMENTATION_VERSION
from .utils import atomic_json_dump


class _AllReduceSumIdentityBackward(torch.autograd.Function):
    """Sum tensor-parallel decoder contributions without duplicating gradients."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group) -> torch.Tensor:
        output = value.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Every rank evaluates the same global loss. The local partial output
        # therefore receives that loss gradient directly; another all-reduce
        # would multiply the parameter gradient by the tensor-parallel size.
        return grad_output, None


def all_gather_rows(value: torch.Tensor, group=None) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return value
    world_size = dist.get_world_size(group)
    output = torch.empty(
        (world_size * value.shape[0], *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_gather_into_tensor(output, value.contiguous(), group=group)
    return output


class FeatureTensorParallelSAE(nn.Module):
    """Dictionary-dimension sharded BatchTopK SAE.

    Each rank owns a contiguous feature range. Inputs are gathered across data
    ranks, the exact global BatchTopK threshold is selected across feature
    shards, and partial reconstructions are summed through NVLink.
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        *,
        group=None,
        batch_topk_candidate_multiplier: float = 1.25,
        decoder_backend: str = "dense",
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            world_size = 1
            rank = 0
        else:
            world_size = dist.get_world_size(group)
            rank = dist.get_rank(group)
        if dict_size % world_size:
            raise ValueError(
                f"dict_size={dict_size} must be divisible by TP size={world_size}"
            )
        self.group = group
        self.world_size = world_size
        self.rank = rank
        self.activation_dim = int(activation_dim)
        self.dict_size = int(dict_size)
        self.local_dict_size = dict_size // world_size
        self.feature_start = rank * self.local_dict_size
        self.feature_stop = self.feature_start + self.local_dict_size
        self.k = int(k)
        self.local = BatchTopKSAE(
            activation_dim,
            self.local_dict_size,
            k,
            batch_topk_candidate_multiplier=batch_topk_candidate_multiplier,
            batch_topk_bf16_histogram=False,
            decoder_backend=decoder_backend,
        )

    @torch.no_grad()
    def initialize_from_global_seed(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        decoder = torch.randn(
            self.activation_dim,
            self.dict_size,
            dtype=torch.float32,
            generator=generator,
        )
        decoder /= decoder.norm(dim=0, keepdim=True).clamp_min(1e-12)
        local_decoder = decoder[
            :, self.feature_start : self.feature_stop
        ].to(self.local.decoder_weight.device)
        self.local.decoder_weight.copy_(local_decoder)
        self.local.encoder_weight.copy_(local_decoder.transpose(0, 1))
        self.local.encoder_bias.zero_()
        self.local.decoder_bias.zero_()
        self.local.pre_bias.zero_()

    @property
    def activation_scale(self) -> torch.Tensor:
        return self.local.activation_scale

    @property
    def decoder_bias(self) -> nn.Parameter:
        return self.local.decoder_bias

    @property
    def pre_bias(self) -> torch.Tensor:
        return self.local.pre_bias

    @torch.no_grad()
    def auxiliary_feature_ids(
        self,
        dead_feature_threshold: int,
        max_features: int,
    ) -> torch.Tensor:
        return self.local.auxiliary_feature_ids(
            dead_feature_threshold,
            max_features,
        )

    @property
    def threshold(self) -> torch.Tensor:
        return self.local.threshold

    @property
    def feature_counts(self) -> torch.Tensor:
        return self.local.feature_counts

    @property
    def num_occurrences_since_fired(self) -> torch.Tensor:
        return self.local.num_occurrences_since_fired

    def forward(
        self,
        local_inputs: torch.Tensor,
        *,
        batch_topk: bool = True,
        return_activity_counts: bool = True,
        sample_mask: torch.Tensor | None = None,
        **_: Any,
    ):
        global_inputs = all_gather_rows(local_inputs, group=self.group)
        global_mask = (
            all_gather_rows(sample_mask.to(torch.uint8), group=self.group).bool()
            if sample_mask is not None
            else None
        )
        scaled = global_inputs * self.local.activation_scale.to(
            global_inputs.dtype
        )
        pre = self.local.pre_activations(scaled)
        if batch_topk:
            (
                features,
                threshold,
                active_per_sample,
                active_per_feature,
            ) = self.local.batch_topk(
                pre,
                distributed=self.world_size > 1,
                return_activity_counts=True,
                sample_mask=global_mask,
                global_sample_count=(
                    int(global_mask.sum())
                    if global_mask is not None
                    else global_inputs.shape[0]
                ),
            )
        else:
            threshold = self.local.threshold
            features = pre * (pre > threshold.to(pre.dtype))
            active = features.detach() != 0
            active_per_sample = active.sum(dim=1, dtype=torch.int64)
            active_per_feature = active.sum(dim=0, dtype=torch.int64)
        if self.world_size > 1:
            active_per_sample = active_per_sample.clone()
            dist.all_reduce(
                active_per_sample,
                op=dist.ReduceOp.SUM,
                group=self.group,
            )
        partial = self.local.decode(features) - self.local.decoder_bias
        if self.world_size > 1:
            reconstructed = _AllReduceSumIdentityBackward.apply(
                partial,
                self.group,
            )
        else:
            reconstructed = partial
        reconstructed = reconstructed + self.local.decoder_bias
        output = (reconstructed, features, threshold)
        if return_activity_counts:
            return (*output, active_per_sample, active_per_feature)
        return output

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.local.normalize_decoder_()

    def remove_parallel_decoder_gradient_(self) -> None:
        self.local.remove_parallel_decoder_gradient_()

    @torch.no_grad()
    def update_dead_feature_stats_(
        self,
        active_per_feature: torch.Tensor,
        *,
        global_samples: int,
        distributed: bool,
    ) -> None:
        # Features are sharded, so firing status is already complete locally.
        self.local.num_occurrences_since_fired.add_(int(global_samples))
        self.local.num_occurrences_since_fired.masked_fill_(
            active_per_feature > 0,
            0,
        )

    @torch.no_grad()
    def dead_feature_count(self, dead_feature_threshold: int) -> int:
        local = torch.tensor(
            self.local.dead_feature_count(dead_feature_threshold),
            dtype=torch.int64,
            device=self.local.decoder_weight.device,
        )
        if self.world_size > 1:
            dist.all_reduce(local, op=dist.ReduceOp.SUM, group=self.group)
        return int(local.item())

    @torch.no_grad()
    def gather_checkpoint_tensors(self) -> dict[str, torch.Tensor] | None:
        """Collect a full standard SAE checkpoint on tensor-parallel rank zero."""

        local_state = self.local.state_dict()
        sharded_names = {
            "encoder_weight": 0,
            "encoder_bias": 0,
            "decoder_weight": 1,
            "feature_counts": 0,
        }
        output: dict[str, torch.Tensor] = {}
        for name, tensor in local_state.items():
            cpu = tensor.detach().cpu().contiguous()
            if name not in sharded_names or self.world_size == 1:
                if self.rank == 0:
                    output[name] = cpu
                continue
            gathered: list[torch.Tensor] | None = (
                [torch.empty_like(cpu) for _ in range(self.world_size)]
                if self.rank == 0
                else None
            )
            dist.gather_object(cpu, gathered, dst=0, group=self.group)
            if self.rank == 0:
                assert gathered is not None
                output[name] = torch.cat(
                    gathered,
                    dim=sharded_names[name],
                )
        return output if self.rank == 0 else None

    def load_full_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        local_state = {}
        for name, tensor in state.items():
            if name in {"encoder_weight", "encoder_bias", "feature_counts"}:
                local_state[name] = tensor[
                    self.feature_start : self.feature_stop
                ]
            elif name == "decoder_weight":
                local_state[name] = tensor[
                    :, self.feature_start : self.feature_stop
                ]
            else:
                local_state[name] = tensor
        self.local.load_state_dict(local_state)


def save_sharded_training_checkpoint(
    *,
    model: FeatureTensorParallelSAE,
    optimizer: torch.optim.Optimizer,
    output_dir: str | Path,
    state: dict[str, Any],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rank_dir = root / f"rank{model.rank:03d}"
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{rank_dir.name}.", dir=root)
    )
    try:
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in model.local.state_dict().items()
            },
            str(tmp_dir / "sae-local.safetensors"),
        )
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        torch.save(state, tmp_dir / "training_state.pt")
        previous = root / f".{rank_dir.name}.previous"
        if previous.exists():
            shutil.rmtree(previous)
        if rank_dir.exists():
            os.replace(rank_dir, previous)
        os.replace(tmp_dir, rank_dir)
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
    if dist.is_initialized():
        dist.barrier(group=model.group)
    if model.rank == 0:
        atomic_json_dump(
            {
                "format": "chunk-saes-feature-tensor-checkpoint-v2",
                "sae_training_implementation_version": (
                    (state.get("config") or {}).get(
                        "sae_training_implementation_version",
                        SAE_TRAINING_IMPLEMENTATION_VERSION,
                    )
                ),
                "sae_parameter_schema_version": (
                    (state.get("config") or {}).get(
                        "sae_parameter_schema_version",
                        SAE_PARAMETER_SCHEMA_VERSION,
                    )
                ),
                "complete": True,
                "world_size": model.world_size,
                "dict_size": model.dict_size,
                "activation_dim": model.activation_dim,
                "step": int(state["step"]),
                "samples_seen": int(state["samples_seen"]),
                "ranks": [
                    {
                        "rank": rank,
                        "path": f"rank{rank:03d}",
                    }
                    for rank in range(model.world_size)
                ],
            },
            root / "checkpoint_manifest.json",
        )
    if dist.is_initialized():
        dist.barrier(group=model.group)


def load_sharded_training_checkpoint(
    *,
    model: FeatureTensorParallelSAE,
    optimizer: torch.optim.Optimizer,
    input_dir: str | Path,
) -> dict[str, Any]:
    root = Path(input_dir)
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "chunk-saes-feature-tensor-checkpoint-v2":
        raise ValueError(
            "feature-tensor checkpoint uses an incompatible parameter schema; "
            "start a new run instead of resuming it"
        )
    if manifest.get("complete") is not True:
        raise ValueError(f"feature-tensor checkpoint is incomplete: {root}")
    if (
        manifest.get("sae_training_implementation_version")
        != SAE_TRAINING_IMPLEMENTATION_VERSION
        or manifest.get("sae_parameter_schema_version")
        != SAE_PARAMETER_SCHEMA_VERSION
    ):
        raise ValueError(
            "feature-tensor checkpoint uses an incompatible SAE implementation; "
            "start a new run instead of resuming it"
        )
    if int(manifest.get("world_size", -1)) != model.world_size:
        raise ValueError(
            "feature-tensor checkpoint world size does not match current run"
        )
    rank_dir = root / f"rank{model.rank:03d}"
    model.local.load_state_dict(
        load_file(str(rank_dir / "sae-local.safetensors"), device="cpu")
    )
    model.to(model.local.decoder_weight.device)
    optimizer.load_state_dict(
        torch.load(
            rank_dir / "optimizer.pt",
            map_location=model.local.decoder_weight.device,
            weights_only=False,
        )
    )
    state = torch.load(
        rank_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state, dict):
        raise ValueError("invalid tensor-parallel checkpoint state")
    saved_config = state.get("config") or {}
    if (
        saved_config.get("sae_training_implementation_version")
        != SAE_TRAINING_IMPLEMENTATION_VERSION
        or saved_config.get("sae_parameter_schema_version")
        != SAE_PARAMETER_SCHEMA_VERSION
    ):
        raise ValueError(
            "feature-tensor checkpoint uses an incompatible SAE implementation; "
            "start a new run instead of resuming it"
        )
    if dist.is_initialized():
        dist.barrier(group=model.group)
    return state
