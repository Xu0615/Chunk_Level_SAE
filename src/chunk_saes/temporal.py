from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F


DEFAULT_TEMPORAL_HIGH_FRACTION = 0.2
DEFAULT_TEMPORAL_TEMPERATURE = 0.1
DEFAULT_TEMPORAL_CONTRASTIVE_BLOCK_SIZE = 1024


def temporal_high_feature_count(
    dict_size: int,
    high_fraction: float = DEFAULT_TEMPORAL_HIGH_FRACTION,
) -> int:
    """Return the deterministic high-level prefix size used by Temporal SAE."""

    if dict_size <= 1:
        raise ValueError("Temporal SAE requires dict_size > 1")
    if not 0.0 < high_fraction < 1.0:
        raise ValueError("temporal high fraction must be in (0, 1)")
    high = int(dict_size * high_fraction)
    return min(dict_size - 1, max(1, high))


def temporal_feature_stop(config: Mapping[str, Any], dict_size: int) -> int:
    """Return Temporal SAE's training-only high-level prefix boundary.

    Cross-method evaluation must use the complete ``dict_size`` dictionary.
    This helper remains for inspecting the Temporal training partition only.
    """

    if str(config.get("mode", "")).lower() != "temporal":
        return int(dict_size)
    explicit = config.get("temporal_high_level_features")
    if explicit is not None:
        value = int(explicit)
        if not 0 < value <= dict_size:
            raise ValueError(
                f"invalid temporal_high_level_features={value} for dict_size={dict_size}"
            )
        return value
    return temporal_high_feature_count(
        dict_size,
        float(
            config.get(
                "temporal_high_fraction",
                DEFAULT_TEMPORAL_HIGH_FRACTION,
            )
        ),
    )


@dataclass(frozen=True)
class TemporalContrastiveResult:
    loss: torch.Tensor
    accuracy: torch.Tensor
    pairs: int
    blocks: int


class _AllGatherVariableWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_size = torch.tensor(
            [values.shape[0]],
            dtype=torch.long,
            device=values.device,
        )
        sizes = [torch.empty_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        size_values = [int(value.item()) for value in sizes]
        maximum = max(size_values)
        if values.shape[0] < maximum:
            values = torch.cat(
                (
                    values,
                    values.new_zeros(
                        (maximum - values.shape[0], *values.shape[1:])
                    ),
                ),
                dim=0,
            )
        gathered = [torch.empty_like(values) for _ in range(world_size)]
        dist.all_gather(gathered, values.contiguous())
        ctx.rank = rank
        ctx.sizes = size_values
        ctx.maximum = maximum
        ctx.tail_shape = values.shape[1:]
        return torch.cat(
            [part[:size] for part, size in zip(gathered, size_values, strict=True)],
            dim=0,
        )

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        chunks = list(torch.split(gradient, ctx.sizes, dim=0))
        padded = gradient.new_zeros(
            (len(ctx.sizes), ctx.maximum, *ctx.tail_shape)
        )
        for index, (chunk, size) in enumerate(
            zip(chunks, ctx.sizes, strict=True)
        ):
            padded[index, :size].copy_(chunk)
        dist.all_reduce(padded, op=dist.ReduceOp.SUM)
        return (padded[ctx.rank, : ctx.sizes[ctx.rank]],)


def _gather_variable_with_grad(values: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return values
    return _AllGatherVariableWithGrad.apply(values)


def _sparse_rows(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.nonzero(values != 0, as_tuple=False)
    return (
        positions[:, 0].to(torch.long),
        positions[:, 1].to(torch.long),
        values[positions[:, 0], positions[:, 1]].float(),
    )


def _sparse_logits(
    queries: torch.Tensor,
    *,
    global_rows: torch.Tensor,
    global_columns: torch.Tensor,
    global_values: torch.Tensor,
    global_count: int,
) -> torch.Tensor:
    if global_values.numel() == 0:
        return queries.new_zeros(
            (queries.shape[0], global_count),
            dtype=torch.float32,
        )
    sparse_keys = torch.sparse_coo_tensor(
        torch.stack((global_rows, global_columns), dim=0),
        global_values,
        size=(global_count, queries.shape[1]),
        device=queries.device,
        dtype=torch.float32,
    ).coalesce()
    with torch.autocast(device_type=queries.device.type, enabled=False):
        return torch.sparse.mm(
            sparse_keys.float(),
            queries.float().transpose(0, 1),
        ).transpose(0, 1)


def distributed_symmetric_temporal_contrastive_loss(
    current_features: torch.Tensor,
    previous_features: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPORAL_TEMPERATURE,
) -> TemporalContrastiveResult:
    """Exact global symmetric normalized InfoNCE using sparse gathered codes."""

    if temperature <= 0:
        raise ValueError("temporal contrastive temperature must be positive")
    pair_mask = pair_mask.to(
        device=current_features.device,
        dtype=torch.bool,
    ).reshape(-1)
    current = current_features[pair_mask].float()
    previous = previous_features[pair_mask].float()
    local_count = int(current.shape[0])
    if local_count == 0:
        raise ValueError("every rank must contribute at least one temporal pair")

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return symmetric_temporal_contrastive_loss(
            current_features,
            previous_features,
            pair_mask,
            temperature=temperature,
            block_size=max(2, local_count),
        )

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_size = torch.tensor(
        [local_count],
        dtype=torch.long,
        device=current.device,
    )
    sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    counts = [int(value.item()) for value in sizes]
    global_count = sum(counts)
    rank_offset = sum(counts[:rank])
    labels = torch.arange(
        rank_offset,
        rank_offset + local_count,
        device=current.device,
    )

    current_rows, current_columns, current_values = _sparse_rows(current)
    previous_rows, previous_columns, previous_values = _sparse_rows(previous)
    current_rows = current_rows + rank_offset
    previous_rows = previous_rows + rank_offset
    global_current_rows = _gather_variable_with_grad(current_rows)
    global_current_columns = _gather_variable_with_grad(current_columns)
    global_current_values = _gather_variable_with_grad(current_values)
    global_previous_rows = _gather_variable_with_grad(previous_rows)
    global_previous_columns = _gather_variable_with_grad(previous_columns)
    global_previous_values = _gather_variable_with_grad(previous_values)

    current_norm = current.pow(2).sum(dim=-1).clamp_min(1e-16).sqrt()
    previous_norm = previous.pow(2).sum(dim=-1).clamp_min(1e-16).sqrt()
    global_current_norm_sq = torch.zeros(
        global_count,
        dtype=torch.float32,
        device=current.device,
    )
    global_previous_norm_sq = torch.zeros_like(global_current_norm_sq)
    global_current_norm_sq.scatter_add_(
        0,
        global_current_rows,
        global_current_values.pow(2),
    )
    global_previous_norm_sq.scatter_add_(
        0,
        global_previous_rows,
        global_previous_values.pow(2),
    )
    global_current_norm = global_current_norm_sq.clamp_min(1e-16).sqrt()
    global_previous_norm = global_previous_norm_sq.clamp_min(1e-16).sqrt()

    forward_logits = _sparse_logits(
        current,
        global_rows=global_previous_rows,
        global_columns=global_previous_columns,
        global_values=global_previous_values,
        global_count=global_count,
    )
    backward_logits = _sparse_logits(
        previous,
        global_rows=global_current_rows,
        global_columns=global_current_columns,
        global_values=global_current_values,
        global_count=global_count,
    )
    forward_logits = (
        forward_logits
        / current_norm[:, None]
        / global_previous_norm[None, :]
        / float(temperature)
    )
    backward_logits = (
        backward_logits
        / previous_norm[:, None]
        / global_current_norm[None, :]
        / float(temperature)
    )
    loss = 0.5 * (
        F.cross_entropy(forward_logits, labels)
        + F.cross_entropy(backward_logits, labels)
    )
    # DDP averages rank-local gradients. Reweight each local mean so the final
    # averaged gradient is the exact global pair-weighted InfoNCE gradient.
    loss = loss * (world_size * local_count / global_count)
    accuracy = 0.5 * (
        (forward_logits.argmax(dim=1) == labels).float().mean()
        + (backward_logits.argmax(dim=1) == labels).float().mean()
    )
    return TemporalContrastiveResult(
        loss=loss,
        accuracy=accuracy.detach(),
        pairs=local_count,
        blocks=1,
    )


def symmetric_temporal_contrastive_loss(
    current_features: torch.Tensor,
    previous_features: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPORAL_TEMPERATURE,
    block_size: int = DEFAULT_TEMPORAL_CONTRASTIVE_BLOCK_SIZE,
) -> TemporalContrastiveResult:
    """Symmetric normalized InfoNCE on matched adjacent-token feature pairs.

    The optimizer batch remains unchanged. To avoid a quadratic 32k-global
    logits tensor, each rank evaluates the same InfoNCE objective in contiguous
    blocks of its deterministically shuffled local occurrence batch. Every
    eligible temporal pair participates exactly once.
    """

    if current_features.shape != previous_features.shape:
        raise ValueError("current and previous temporal features must have equal shapes")
    if current_features.ndim != 2:
        raise ValueError("temporal features must be rank-2")
    pair_mask = pair_mask.to(
        device=current_features.device,
        dtype=torch.bool,
    ).reshape(-1)
    if pair_mask.numel() != current_features.shape[0]:
        raise ValueError("temporal pair mask does not match feature rows")
    if temperature <= 0:
        raise ValueError("temporal contrastive temperature must be positive")
    if block_size <= 1:
        raise ValueError("temporal contrastive block size must exceed one")

    current = current_features[pair_mask].float()
    previous = previous_features[pair_mask].float()
    pair_count = int(current.shape[0])
    zero = current_features.sum() * 0.0
    if pair_count < 2:
        return TemporalContrastiveResult(
            loss=zero,
            accuracy=zero.detach(),
            pairs=pair_count,
            blocks=0,
        )

    weighted_loss = zero
    weighted_accuracy = zero.detach()
    used_pairs = 0
    blocks = 0
    block_size = int(block_size)
    starts = list(range(0, pair_count, block_size))
    if len(starts) > 1 and pair_count - starts[-1] == 1:
        starts.pop()
    for block_index, start in enumerate(starts):
        stop = (
            starts[block_index + 1]
            if block_index + 1 < len(starts)
            else pair_count
        )
        rows = stop - start
        query = current[start:stop]
        key = previous[start:stop]
        weight = rows

        query = F.normalize(query, p=2, dim=-1, eps=1e-8)
        key = F.normalize(key, p=2, dim=-1, eps=1e-8)
        logits = query @ key.transpose(0, 1)
        logits = logits / float(temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        block_loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )
        block_accuracy = 0.5 * (
            (logits.argmax(dim=1) == labels).float().mean()
            + (logits.argmax(dim=0) == labels).float().mean()
        )
        weighted_loss = weighted_loss + block_loss * weight
        weighted_accuracy = weighted_accuracy + block_accuracy.detach() * weight
        used_pairs += weight
        blocks += 1

    if used_pairs <= 0:
        return TemporalContrastiveResult(
            loss=zero,
            accuracy=zero.detach(),
            pairs=pair_count,
            blocks=0,
        )
    return TemporalContrastiveResult(
        loss=weighted_loss / used_pairs,
        accuracy=weighted_accuracy / used_pairs,
        pairs=pair_count,
        blocks=blocks,
    )
