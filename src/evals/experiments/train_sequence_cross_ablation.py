#!/usr/bin/env python
"""Train the masked sequence Cross ablations (E3/E4).

The base model has already been run independently on A and B.  This trainer
therefore consumes the pair-major activation-cache-v2 shards directly.  A
direction is represented by ``source tokens + norm-matched mask tokens`` and
the target is the other chunk.  No base-model concatenation or causal
attention is performed here.

The sequence context is deliberately small and explicit: a learned projection
of the owning-chunk mean plus a learned target-position table conditions the
mask positions.  The shared dictionary is the same BatchTopK SAE used by the
mean Cross trainer.  Keeping the context module separate makes the objective
and the self/partner loss masks auditable in the run manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel

from chunk_saes.runtime import (
    bind_local_rank_cpu_affinity,
    configure_cpu_threads,
    initialize_distributed,
)
from chunk_saes.sae import BatchTopKSAE
from chunk_saes.utils import atomic_json_dump, distributed_info, seed_everything


POLICIES = ("both",)
LOSS_MASKS = ("partner-only", "all")
MASK_REPRESENTATIONS = ("norm-matched-mean",)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activation-cache-dir", required=True)
    p.add_argument("--validation-cache-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--dict-size", type=int, default=65536)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--global-batch-size", type=int, default=32000)
    p.add_argument(
        "--target-occurrences",
        type=int,
        required=True,
        help="Global target-token budget for this run (P/C/F use 10016000/100000000/1000000000).",
    )
    p.add_argument("--validation-occurrences", type=int, default=0)
    p.add_argument("--periodic-validation-occurrences", type=int, default=262144)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--examples-per-batch", type=int, default=32)
    p.add_argument("--max-chunk-length", type=int, default=512)
    p.add_argument("--context-dim", type=int, default=256)
    p.add_argument("--context-heads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--validate-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--autocast-dtype", default="bfloat16")
    p.add_argument("--decoder-backend", choices=("dense", "sparse", "auto"), default="sparse")
    p.add_argument("--batch-topk-candidate-multiplier", type=float, default=2.0)
    p.add_argument("--batch-topk-bf16-histogram", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--loss-mask", choices=LOSS_MASKS, default="partner-only")
    p.add_argument("--mask-representation", choices=MASK_REPRESENTATIONS, default="norm-matched-mean")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite-output", action="store_true")
    p.add_argument("--final-validation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loader-shards", type=int, default=1)
    return p.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    value = str(name).strip().lower().replace("torch.", "")
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[value]


def _all_reduce_scalar(value: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=op)
    return value


def _manifest(path: str | Path) -> dict[str, Any]:
    with (Path(path) / "manifest.json").open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not value.get("complete"):
        raise ValueError(f"activation cache is incomplete: {path}")
    if value.get("format") != "chunk-saes-activation-cache-v2":
        raise ValueError(f"unsupported activation cache format: {value.get('format')!r}")
    return value


def _rank_manifest(manifest: dict[str, Any], rank: int) -> dict[str, Any]:
    ranks = manifest.get("ranks")
    if not isinstance(ranks, list):
        raise ValueError("cache manifest lacks rank manifests")
    for item in ranks:
        if int(item.get("rank", -1)) == rank:
            return item
    raise ValueError(f"cache manifest lacks rank {rank}")


@dataclass
class DirectionExample:
    source: torch.Tensor
    target: torch.Tensor
    source_len: int
    target_len: int
    pair_id: int
    direction_b: bool
    doc_hash: bytes
    occurrence_start: int


class PairShardStream:
    """Yield both directions for each pair, preserving token provenance."""

    def __init__(
        self,
        cache_dir: str | Path,
        manifest: dict[str, Any],
        *,
        rank: int,
        seed: int,
        max_chunk_length: int,
        shard_shuffle: bool = True,
    ) -> None:
        self.root = Path(cache_dir)
        self.manifest = manifest
        self.rank = int(rank)
        self.seed = int(seed)
        self.max_chunk_length = int(max_chunk_length)
        self.rank_info = _rank_manifest(manifest, rank)
        self.entries = [dict(item) for item in self.rank_info.get("shards", [])]
        if not self.entries:
            raise ValueError(f"rank {rank} has no sequence shards")
        self.pairs = int(self.rank_info.get("pairs", sum(int(x.get("pairs", 0)) for x in self.entries)))
        self.tokens = int(self.rank_info.get("token_occurrences", 0))
        self.shard_shuffle = bool(shard_shuffle)

    def _iter_shard(self, entry: dict[str, Any], rng: random.Random) -> Iterator[DirectionExample]:
        path = self.root / str(entry["path"])
        # A shard is pair-contained.  Keeping one mmap-backed shard alive while
        # its examples are packed avoids materialising a second multi-GB copy.
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            hidden = handle.get_tensor("token_hidden")
            offsets = handle.get_tensor("chunk_offsets").to(torch.long)
            pair_ids = handle.get_tensor("pair_id").to(torch.long)
            lengths_a = handle.get_tensor("length_a").to(torch.long)
            lengths_b = handle.get_tensor("length_b").to(torch.long)
            starts = handle.get_tensor("occurrence_start").to(torch.long)
            doc_hashes = handle.get_tensor("doc_hash").to(torch.uint8)
            order = list(range(int(pair_ids.numel())))
            if self.shard_shuffle:
                rng.shuffle(order)
            for index in order:
                la = int(lengths_a[index])
                lb = int(lengths_b[index])
                if not (0 < la <= self.max_chunk_length and 0 < lb <= self.max_chunk_length):
                    raise ValueError(
                        f"pair {int(pair_ids[index])} has lengths {la},{lb}; "
                        f"max is {self.max_chunk_length}"
                    )
                left = hidden[int(offsets[2 * index]) : int(offsets[2 * index + 1])]
                right = hidden[int(offsets[2 * index + 1]) : int(offsets[2 * index + 2])]
                digest = bytes(doc_hashes[index].tolist())
                pair_id = int(pair_ids[index])
                occurrence_start = int(starts[index])
                yield DirectionExample(left, right, la, lb, pair_id, False, digest, occurrence_start)
                yield DirectionExample(right, left, lb, la, pair_id, True, digest, occurrence_start)

    def iter_examples(self, epoch: int = 0) -> Iterator[DirectionExample]:
        entries = list(self.entries)
        rng = random.Random(self.seed + 1009 * int(epoch))
        if self.shard_shuffle:
            rng.shuffle(entries)
        for entry in entries:
            yield from self._iter_shard(entry, rng)


class PairBatcher:
    """Pack fixed-shape padded batches and stop at an exact target budget."""

    def __init__(
        self,
        stream: PairShardStream,
        *,
        target_occurrences: int,
        examples_per_batch: int,
        max_chunk_length: int,
        hidden_size: int,
        sentinel: torch.Tensor,
    ) -> None:
        self.stream = stream
        self.target_occurrences = int(target_occurrences)
        self.examples_per_batch = int(examples_per_batch)
        self.max_chunk_length = int(max_chunk_length)
        self.hidden_size = int(hidden_size)
        self.sentinel = sentinel.detach().to(dtype=torch.bfloat16, device="cpu")
        if self.target_occurrences <= 0:
            raise ValueError("target_occurrences must be positive")
        if self.examples_per_batch <= 0:
            raise ValueError("examples_per_batch must be positive")

    def _empty(self) -> dict[str, torch.Tensor]:
        b, l, d = self.examples_per_batch, self.max_chunk_length, self.hidden_size
        return {
            "source": torch.zeros((b, l, d), dtype=torch.bfloat16),
            "target": torch.zeros((b, l, d), dtype=torch.bfloat16),
            "source_mask": torch.zeros((b, l), dtype=torch.bool),
            "partner_mask": torch.zeros((b, l), dtype=torch.bool),
            "side_b": torch.zeros((b,), dtype=torch.bool),
            "pair_id": torch.full((b,), -1, dtype=torch.long),
            "source_len": torch.zeros((b,), dtype=torch.int32),
            "target_len": torch.zeros((b,), dtype=torch.int32),
            "doc_hash": torch.zeros((b, 32), dtype=torch.uint8),
            "occurrence_start": torch.full((b,), -1, dtype=torch.long),
        }

    def _pack(self, examples: list[DirectionExample]) -> dict[str, torch.Tensor]:
        batch = self._empty()
        for row, example in enumerate(examples):
            source = example.source
            target = example.target
            sl = int(example.source_len)
            tl = int(example.target_len)
            batch["source"][row, :sl].copy_(source[:sl])
            batch["target"][row, :tl].copy_(target[:tl])
            batch["source_mask"][row, :sl] = True
            batch["partner_mask"][row, :tl] = True
            batch["side_b"][row] = bool(example.direction_b)
            batch["pair_id"][row] = int(example.pair_id)
            batch["source_len"][row] = sl
            batch["target_len"][row] = tl
            batch["doc_hash"][row] = torch.tensor(list(example.doc_hash), dtype=torch.uint8)
            batch["occurrence_start"][row] = int(example.occurrence_start)
        return batch

    def batches(self, epoch: int = 0) -> Iterator[tuple[dict[str, torch.Tensor], bool, int]]:
        examples: list[DirectionExample] = []
        consumed = 0
        for example in self.stream.iter_examples(epoch):
            if consumed >= self.target_occurrences:
                break
            remaining = self.target_occurrences - consumed
            if example.target_len > remaining:
                # Truncate only the final target side.  The source remains the
                # complete independently-forwarded chunk and provenance stays
                # attached to the original pair.
                example = DirectionExample(
                    source=example.source,
                    target=example.target[:remaining],
                    source_len=example.source_len,
                    target_len=remaining,
                    pair_id=example.pair_id,
                    direction_b=example.direction_b,
                    doc_hash=example.doc_hash,
                    occurrence_start=example.occurrence_start,
                )
            examples.append(example)
            consumed += int(example.target_len)
            if len(examples) >= self.examples_per_batch:
                yield self._pack(examples), True, consumed
                examples = []
        if examples:
            yield self._pack(examples), True, consumed
        empty = self._empty()
        while True:
            yield empty, False, consumed


class MaskedSequenceSAE(nn.Module):
    """A shared sequence context plus the standard sparse dictionary."""

    def __init__(
        self,
        hidden_size: int,
        dict_size: int,
        k: int,
        max_chunk_length: int,
        *,
        loss_mask: str,
        sentinel: torch.Tensor,
        self_mean: torch.Tensor,
        context_dim: int,
        context_heads: int,
        decoder_backend: str,
        candidate_multiplier: float,
        bf16_histogram: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_chunk_length = int(max_chunk_length)
        self.loss_mask = str(loss_mask)
        self.context_dim = int(context_dim)
        if self.context_dim <= 0 or self.context_dim % int(context_heads):
            raise ValueError("context-dim must be positive and divisible by context-heads")
        self.source_projection = nn.Linear(hidden_size, self.context_dim, bias=False)
        self.context_attention = nn.MultiheadAttention(
            self.context_dim,
            int(context_heads),
            batch_first=True,
        )
        self.context_output = nn.Linear(self.context_dim, hidden_size, bias=False)
        self.context_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.query_norm = nn.LayerNorm(hidden_size)
        self.position = nn.Parameter(torch.zeros(max_chunk_length, self.context_dim))
        nn.init.normal_(self.position, mean=0.0, std=0.01)
        self.register_buffer("sentinel", sentinel.detach().float().clone())
        self.register_buffer("self_mean", self_mean.detach().float().clone())
        self.sae = BatchTopKSAE(
            hidden_size,
            dict_size,
            k,
            batch_topk_candidate_multiplier=candidate_multiplier,
            batch_topk_bf16_histogram=bf16_histogram,
            decoder_backend=decoder_backend,
        )

    def _selected_inputs(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        source_mask = batch["source_mask"].to(device, non_blocking=True)
        partner_mask = batch["partner_mask"].to(device, non_blocking=True)
        safe_source_mask = source_mask.clone()
        empty_rows = ~safe_source_mask.any(dim=1)
        if bool(empty_rows.any()):
            safe_source_mask[empty_rows, 0] = True
        source_context = self.source_projection(source.float())
        positions = self.position[: target.shape[1]].to(device=device)
        query_context = positions.unsqueeze(0).expand(source.shape[0], -1, -1)
        context, _ = self.context_attention(
            query_context,
            source_context,
            source_context,
            key_padding_mask=~safe_source_mask,
            need_weights=False,
        )
        context = self.context_output(context)
        query = (
            self.sentinel.to(dtype=context.dtype, device=device)[None, None, :]
            + self.query_norm(context) * self.context_gate.to(context.dtype)
        )
        # Keep the large token-row matrix in the cache's bf16 representation;
        # the attention/context block itself remains fp32 for stable logits.
        query = query.to(dtype=source.dtype)
        if self.loss_mask == "partner-only":
            selected_input = query[partner_mask]
            selected_target = target[partner_mask]
            selected_kind = torch.ones(selected_input.shape[0], dtype=torch.bool, device=device)
            selected_mask = partner_mask
        else:
            # The sparse dictionary has one input center (the mask sentinel).
            # Shift owning-chunk self inputs by the corresponding difference
            # between the token mean and sentinel so both branches use the
            # intended centered coordinates while retaining raw source targets.
            self_input = (
                source[source_mask]
                - self.self_mean.to(source.device, dtype=source.dtype)
                + self.sentinel.to(source.device, dtype=source.dtype)
            )
            selected_input = torch.cat((self_input, query[partner_mask]), dim=0)
            selected_target = torch.cat((source[source_mask], target[partner_mask]), dim=0)
            selected_kind = torch.cat(
                (
                    torch.zeros(int(source_mask.sum()), dtype=torch.bool, device=device),
                    torch.ones(int(partner_mask.sum()), dtype=torch.bool, device=device),
                ),
                dim=0,
            )
            selected_mask = source_mask | partner_mask
        return selected_input, selected_target, selected_kind, selected_mask

    def _run_sparse_group(
        self,
        selected_input: torch.Tensor,
        selected_target: torch.Tensor,
        *,
        distributed: bool,
        return_activity_counts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply exact global BatchTopK to one semantically balanced row group."""

        device = selected_input.device
        local_rows = torch.tensor([selected_input.shape[0]], dtype=torch.int64, device=device)
        global_rows = local_rows.clone()
        max_rows = local_rows.clone()
        if distributed and dist.is_initialized():
            dist.all_reduce(global_rows, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_rows, op=dist.ReduceOp.MAX)
        capacity = max(int(max_rows.item()), 1)
        padded_input = torch.zeros(
            (capacity, self.hidden_size), dtype=selected_input.dtype, device=device
        )
        padded_target = torch.zeros_like(padded_input)
        valid = torch.zeros((capacity,), dtype=torch.bool, device=device)
        n = int(selected_input.shape[0])
        if n:
            padded_input[:n] = selected_input
            padded_target[:n] = selected_target
            valid[:n] = True
        else:
            # Keep the attention/context parameters in the autograd graph on
            # ranks that have reached the end of their local stream.  The
            # scalar is exactly zero, but DDP still observes a matching
            # parameter set on every rank for the final dummy batches.
            padded_input = padded_input + selected_input.sum() * 0.0
        result = self.sae(
            padded_input,
            batch_topk=True,
            distributed=distributed,
            return_activity_counts=return_activity_counts,
            sample_mask=valid,
            global_sample_count=max(1, int(global_rows.item())),
        )
        reconstructed, features = result[:2]
        active_per_feature = (
            result[4]
            if return_activity_counts
            else torch.zeros(
                self.sae.dict_size, dtype=torch.int64, device=device
            )
        )
        return (
            reconstructed[:n],
            padded_target[:n],
            features[:n],
            active_per_feature,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        distributed: bool,
        return_activity_counts: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.sae.decoder_weight.device
        selected_input, selected_target, selected_kind, _ = self._selected_inputs(batch, device)
        if self.loss_mask == "partner-only":
            reconstructed, target, features, active_per_feature = self._run_sparse_group(
                selected_input,
                selected_target,
                distributed=distributed,
                return_activity_counts=return_activity_counts,
            )
        else:
            # E4 gives self and partner equal loss weight.  Select BatchTopK
            # independently for those two row groups as well: the total active
            # budget remains K * (N_self + N_partner), but the easier self rows
            # cannot starve every partner row of gradient at initialization.
            group_outputs = [
                self._run_sparse_group(
                    selected_input[selected_kind == partner],
                    selected_target[selected_kind == partner],
                    distributed=distributed,
                    return_activity_counts=return_activity_counts,
                )
                for partner in (False, True)
            ]
            reconstructed = torch.cat([row[0] for row in group_outputs], dim=0)
            target = torch.cat([row[1] for row in group_outputs], dim=0)
            features = torch.cat([row[2] for row in group_outputs], dim=0)
            active_per_feature = group_outputs[0][3] + group_outputs[1][3]
            selected_kind = torch.cat(
                (
                    torch.zeros(group_outputs[0][0].shape[0], dtype=torch.bool, device=device),
                    torch.ones(group_outputs[1][0].shape[0], dtype=torch.bool, device=device),
                )
            )
        return (
            reconstructed,
            target,
            features,
            selected_kind,
            active_per_feature,
        )


@dataclass
class MetricState:
    partner_sse: float = 0.0
    partner_baseline: float = 0.0
    partner_count: int = 0
    self_sse: float = 0.0
    self_baseline: float = 0.0
    self_count: int = 0
    active_sum: float = 0.0
    active_count: int = 0
    partner_active_sum: float = 0.0
    partner_active_count: int = 0
    self_active_sum: float = 0.0
    self_active_count: int = 0
    direction_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.direction_counts is None:
            self.direction_counts = {"a_to_b": 0, "b_to_a": 0}


def _update_metrics(
    state: MetricState,
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor,
    kind_partner: torch.Tensor,
    side_b: torch.Tensor,
    target_mean_a_to_b: torch.Tensor,
    target_mean_b_to_a: torch.Tensor,
    self_mean: torch.Tensor,
) -> None:
    if reconstructed.numel() == 0:
        return
    residual = reconstructed.float() - target.float()
    active = (features.detach() != 0).sum(dim=1).float()
    state.active_sum += float(active.sum().item())
    state.active_count += int(active.numel())
    for is_partner, label in ((True, "partner"), (False, "self")):
        mask = kind_partner if is_partner else ~kind_partner
        if not bool(mask.any()):
            continue
        values = residual[mask].square().sum(dim=-1)
        active_values = active[mask]
        if is_partner:
            base_mean = torch.where(
                side_b[mask].unsqueeze(1),
                target_mean_b_to_a.to(target.device).unsqueeze(0),
                target_mean_a_to_b.to(target.device).unsqueeze(0),
            )
        else:
            base_mean = self_mean.to(target.device).unsqueeze(0).expand(
                int(mask.sum()), -1
            )
        baseline = (target[mask].float() - base_mean.float()).square().sum(dim=-1)
        if is_partner:
            state.partner_sse += float(values.sum().item())
            state.partner_baseline += float(baseline.sum().item())
            state.partner_count += int(mask.sum().item())
            state.partner_active_sum += float(active_values.sum().item())
            state.partner_active_count += int(active_values.numel())
            state.direction_counts["b_to_a"] += int((side_b[mask]).sum().item())
            state.direction_counts["a_to_b"] += int((~side_b[mask]).sum().item())
        else:
            state.self_sse += float(values.sum().item())
            state.self_baseline += float(baseline.sum().item())
            state.self_count += int(mask.sum().item())
            state.self_active_sum += float(active_values.sum().item())
            state.self_active_count += int(active_values.numel())


def _metric_payload(state: MetricState, *, loss_mask: str, target_occurrences: int) -> dict[str, Any]:
    def fve(sse: float, baseline: float) -> float:
        return 1.0 - sse / baseline if baseline > 0 else float("nan")

    payload: dict[str, Any] = {
        "target_occurrences": int(target_occurrences),
        "partner_tokens": state.partner_count,
        "partner_mse": state.partner_sse / max(1, state.partner_count),
        "partner_baseline_mse": state.partner_baseline / max(1, state.partner_count),
        "partner_fve": fve(state.partner_sse, state.partner_baseline),
        "effective_l0": state.active_sum / max(1, state.active_count),
        "partner_effective_l0": state.partner_active_sum / max(1, state.partner_active_count),
        "direction_counts": dict(state.direction_counts or {}),
    }
    if loss_mask == "all":
        balanced_mse = 0.5 * (
            state.self_sse / max(1, state.self_count)
            + state.partner_sse / max(1, state.partner_count)
        )
        balanced_baseline = 0.5 * (
            state.self_baseline / max(1, state.self_count)
            + state.partner_baseline / max(1, state.partner_count)
        )
        payload.update(
            {
                "self_tokens": state.self_count,
                "self_mse": state.self_sse / max(1, state.self_count),
                "self_baseline_mse": state.self_baseline / max(1, state.self_count),
                "self_fve": fve(state.self_sse, state.self_baseline),
                "self_effective_l0": state.self_active_sum / max(1, state.self_active_count),
                "total_tokens": state.self_count + state.partner_count,
                "total_sse": state.self_sse + state.partner_sse,
                "total_baseline": state.self_baseline + state.partner_baseline,
                "total_mse": balanced_mse,
                "total_baseline_mse": balanced_baseline,
                "total_fve": fve(balanced_mse, balanced_baseline),
                "pooled_total_fve": fve(
                    state.self_sse + state.partner_sse,
                    state.self_baseline + state.partner_baseline,
                ),
            }
        )
    return payload


def _merge_metric_states(local: MetricState, device: torch.device) -> MetricState:
    values = torch.tensor(
        [
            local.partner_sse,
            local.partner_baseline,
            local.partner_count,
            local.self_sse,
            local.self_baseline,
            local.self_count,
            local.active_sum,
            local.active_count,
            local.partner_active_sum,
            local.partner_active_count,
            local.self_active_sum,
            local.self_active_count,
            local.direction_counts["a_to_b"],
            local.direction_counts["b_to_a"],
        ],
        dtype=torch.float64,
        device=device,
    )
    _all_reduce_scalar(values)
    return MetricState(
        partner_sse=float(values[0]),
        partner_baseline=float(values[1]),
        partner_count=int(values[2]),
        self_sse=float(values[3]),
        self_baseline=float(values[4]),
        self_count=int(values[5]),
        active_sum=float(values[6]),
        active_count=int(values[7]),
        partner_active_sum=float(values[8]),
        partner_active_count=int(values[9]),
        self_active_sum=float(values[10]),
        self_active_count=int(values[11]),
        direction_counts={"a_to_b": int(values[12]), "b_to_a": int(values[13])},
    )


def _lr_multiplier(step: int, total_steps: int, warmup: int, floor: float) -> float:
    if step < warmup:
        return max(1e-8, (step + 1) / max(1, warmup))
    if total_steps <= warmup:
        return floor
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return floor + (1.0 - floor) * cosine


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _token_mean(manifest: dict[str, Any], hidden_size: int, device: torch.device) -> torch.Tensor:
    values = manifest.get("target_sufficient_statistics", {}).get("mean_by_mode", {}).get("token")
    if not isinstance(values, list) or len(values) != hidden_size:
        raise ValueError("cache manifest lacks token target mean")
    return torch.tensor(values, dtype=torch.float32, device=device)


def _cross_mean(
    manifest: dict[str, Any],
    name: str,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = (
        manifest.get("target_sufficient_statistics", {})
        .get("mean_by_mode", {})
        .get(name)
    )
    if not isinstance(values, list) or len(values) != hidden_size:
        return _token_mean(manifest, hidden_size, device)
    return torch.tensor(values, dtype=torch.float32, device=device)


def _estimate_norm(cache_dir: str | Path, manifest: dict[str, Any], rank: int) -> float:
    info = _rank_manifest(manifest, rank)
    entry = info["shards"][0]
    path = Path(cache_dir) / str(entry["path"])
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        hidden = handle.get_tensor("token_hidden")
        take = min(4096, hidden.shape[0])
        return float(hidden[:take].float().norm(dim=1).mean().item())


def _save_sequence_checkpoint(path: Path, model: nn.Module, config: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in raw.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    save_file(tensors, str(path / "model.safetensors"))
    atomic_json_dump(config, path / "config.json")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batcher: PairBatcher,
    *,
    device: torch.device,
    target_mean_a_to_b: torch.Tensor,
    target_mean_b_to_a: torch.Tensor,
    self_mean: torch.Tensor,
    loss_mask: str,
    max_batches: int = 0,
) -> dict[str, Any]:
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    was_training = raw.training
    raw.eval()
    state = MetricState()
    iterator = batcher.batches(epoch=17)
    batches = 0
    while True:
        batch, local_has, _ = next(iterator)
        flag = torch.tensor([int(local_has)], dtype=torch.int64, device=device)
        _all_reduce_scalar(flag, op=dist.ReduceOp.MAX)
        if int(flag.item()) == 0:
            break
        moved = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reconstructed, target, features, kind, _ = raw(
                moved,
                distributed=dist.is_initialized() and dist.get_world_size() > 1,
            )
        # The selected rows are ordered by source rows then partner rows for E4,
        # and partner rows only for E3.  Reconstruct side labels from the masks.
        source_mask = moved["source_mask"]
        partner_mask = moved["partner_mask"]
        side_rows = moved["side_b"][:, None].expand_as(partner_mask)
        if loss_mask == "partner-only":
            side = side_rows[partner_mask]
        else:
            side = torch.cat((side_rows[source_mask], side_rows[partner_mask]))
        _update_metrics(
            state,
            reconstructed,
            target,
            features,
            kind,
            side,
            target_mean_a_to_b,
            target_mean_b_to_a,
            self_mean,
        )
        batches += 1
        if max_batches and batches >= max_batches:
            break
    merged = _merge_metric_states(state, device)
    if was_training:
        raw.train()
    return _metric_payload(
        merged,
        loss_mask=loss_mask,
        target_occurrences=batcher.target_occurrences
        * (dist.get_world_size() if dist.is_initialized() else 1),
    )


def main() -> None:
    args = _parse_args()
    if args.target_occurrences <= 0 or args.target_occurrences % 1 != 0:
        raise ValueError("target-occurrences must be positive")
    if args.examples_per_batch <= 0 or args.max_chunk_length <= 0:
        raise ValueError("examples-per-batch and max-chunk-length must be positive")
    preliminary_rank, preliminary_world, preliminary_local = distributed_info()
    assigned = bind_local_rank_cpu_affinity(
        local_rank=preliminary_local,
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", preliminary_world)),
    )
    configure_cpu_threads(default=4)
    rank, world_size, local_rank = initialize_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1 and device.type != "cuda":
        raise RuntimeError("sequence DDP requires CUDA")
    seed_everything(args.seed + rank)
    train_manifest = _manifest(args.activation_cache_dir)
    validation_manifest = _manifest(args.validation_cache_dir) if args.validation_cache_dir else None
    hidden_size = int(train_manifest.get("hidden_size", 0))
    if hidden_size <= 0:
        hidden_size = len(train_manifest["target_sufficient_statistics"]["mean_by_mode"]["token"])
    if world_size > 1:
        if len(train_manifest.get("ranks", [])) != world_size:
            raise ValueError("cache world size differs from sequence trainer world size")
        if args.target_occurrences % world_size:
            raise ValueError("target-occurrences must be divisible by world size")
    local_target = args.target_occurrences // world_size
    norm = _estimate_norm(args.activation_cache_dir, train_manifest, 0)
    norm_tensor = torch.tensor([norm], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.broadcast(norm_tensor, src=0)
    norm = float(norm_tensor.item())
    token_mean = _token_mean(train_manifest, hidden_size, device)
    sentinel = token_mean * (norm / token_mean.norm().clamp_min(1e-6))
    output = Path(args.output_dir)
    if rank == 0:
        if output.exists() and args.overwrite_output:
            for child in output.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
        output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    stream = PairShardStream(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        seed=args.seed + 11,
        max_chunk_length=args.max_chunk_length,
    )
    batcher = PairBatcher(
        stream,
        target_occurrences=local_target,
        examples_per_batch=args.examples_per_batch,
        max_chunk_length=args.max_chunk_length,
        hidden_size=hidden_size,
        sentinel=sentinel,
    )
    model = MaskedSequenceSAE(
        hidden_size,
        args.dict_size,
        args.k,
        args.max_chunk_length,
        loss_mask=args.loss_mask,
        sentinel=sentinel,
        self_mean=token_mean,
        context_dim=args.context_dim,
        context_heads=args.context_heads,
        decoder_backend=args.decoder_backend,
        candidate_multiplier=args.batch_topk_candidate_multiplier,
        bf16_histogram=args.batch_topk_bf16_histogram,
    ).to(device)
    model.sae.pre_bias.copy_(sentinel)
    model.sae.decoder_bias.data.copy_(token_mean)
    model.sae.activation_scale.fill_(1.0)
    ddp: nn.Module = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=256,
        )
        if world_size > 1
        else model
    )
    optimizer = torch.optim.Adam(
        ddp.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        fused=bool(device.type == "cuda"),
    )
    run_config: dict[str, Any] = {
        "format": "chunk-saes-sequence-cross-ablation-v1",
        "model": args.model,
        "layer": args.layer,
        "activation_dim": hidden_size,
        "dict_size": args.dict_size,
        "k": args.k,
        "global_batch_size": args.global_batch_size,
        "target_occurrences": args.target_occurrences,
        "validation_occurrences": args.validation_occurrences,
        "periodic_validation_occurrences": args.periodic_validation_occurrences,
        "local_target_occurrences": local_target,
        "loss_mask": args.loss_mask,
        "mask_representation": args.mask_representation,
        "direction_policy": "both",
        "sequence_interface": "independent_HA_HB_with_A_B_position_reset",
        "sequence_context": "source_token_multihead_attention_plus_target_position",
        "context_dim": args.context_dim,
        "context_heads": args.context_heads,
        "padding_policy": "fixed_zeros_excluded_by_boolean_masks",
        "batch_topk_grouping": "partner-only_or_balanced_self_and_partner",
        "self_input_centering": "source_minus_token_mean_plus_mask_sentinel",
        "sentinel_norm": norm,
        "examples_per_batch": args.examples_per_batch,
        "max_chunk_length": args.max_chunk_length,
        "seed": args.seed,
        "train_cache": str(Path(args.activation_cache_dir).resolve()),
        "validation_cache": str(Path(args.validation_cache_dir).resolve()) if args.validation_cache_dir else None,
        "world_size": world_size,
    }
    if rank == 0:
        atomic_json_dump(run_config, output / "run_config.json")
    log_path = output / "metrics.jsonl"
    autocast_dtype = _dtype_from_name(args.autocast_dtype)
    mean_manifest = validation_manifest or train_manifest
    target_mean_a_to_b = _cross_mean(
        mean_manifest, "cross_a_to_b", hidden_size, device
    )
    target_mean_b_to_a = _cross_mean(
        mean_manifest, "cross_b_to_a", hidden_size, device
    )
    self_mean = _token_mean(mean_manifest, hidden_size, device)
    iterator = batcher.batches(epoch=0)
    state = MetricState()
    started = time.time()
    step = 0
    consumed_global = 0
    padding_rows_excluded = 0
    last_validation: dict[str, Any] | None = None
    best_validation: dict[str, Any] | None = None
    best_metric_value = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    while True:
        batch, local_has, consumed_local = next(iterator)
        flag = torch.tensor([int(local_has)], dtype=torch.int64, device=device)
        _all_reduce_scalar(flag, op=dist.ReduceOp.MAX)
        if int(flag.item()) == 0:
            break
        moved = _move_batch(batch, device)
        source_mask = moved["source_mask"]
        partner_mask = moved["partner_mask"]
        side_rows = moved["side_b"][:, None].expand_as(partner_mask)
        if args.loss_mask == "partner-only":
            side = side_rows[partner_mask]
        else:
            side = torch.cat((side_rows[source_mask], side_rows[partner_mask]))
        valid_local = int(side.numel())
        valid_global = torch.tensor([valid_local], dtype=torch.int64, device=device)
        _all_reduce_scalar(valid_global)
        # The model pads each rank to the global maximum selected-row count;
        # keep an explicit audit total proving those rows never entered loss.
        max_valid = torch.tensor([valid_local], dtype=torch.int64, device=device)
        _all_reduce_scalar(max_valid, op=dist.ReduceOp.MAX)
        local_padding = max_valid - valid_local
        _all_reduce_scalar(local_padding)
        padding_rows_excluded += int(local_padding.item())
        # DDP averages gradients across ranks.  Because chunk lengths vary,
        # a plain local ``mean`` would give a short-token rank the same weight
        # as a long-token rank.  Keep the objective an exact global token mean
        # by scaling each local component with its global token share.
        partner_local = int(partner_mask.sum().item())
        self_local = int(source_mask.sum().item()) if args.loss_mask == "all" else 0
        component_counts = torch.tensor(
            [partner_local, self_local], dtype=torch.int64, device=device
        )
        _all_reduce_scalar(component_counts)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            reconstructed, target, features, kind, active_per_feature = ddp(
                moved,
                distributed=world_size > 1,
            )
            residual = reconstructed.float() - target.float()
            partner_mask_rows = kind
            self_mask_rows = ~kind
            partner_loss_mean = (
                residual[partner_mask_rows].square().sum(dim=-1).mean()
                if bool(partner_mask_rows.any())
                else residual.sum() * 0.0
            )
            self_loss_mean = (
                residual[self_mask_rows].square().sum(dim=-1).mean()
                if bool(self_mask_rows.any())
                else residual.sum() * 0.0
            )
            world = float(world_size)
            partner_weight = (
                world * float(partner_local) / max(1.0, float(component_counts[0].item()))
            )
            partner_loss = partner_loss_mean * partner_weight
            if args.loss_mask == "all":
                self_weight = (
                    world * float(self_local) / max(1.0, float(component_counts[1].item()))
                )
                self_loss = self_loss_mean * self_weight
                loss = 0.5 * (self_loss + partner_loss)
            else:
                self_loss = self_loss_mean
                loss = partner_loss
        loss.backward()
        raw_model = model
        raw_model.sae.remove_parallel_decoder_gradient_()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        estimated_steps = max(
            1, math.ceil(args.target_occurrences / max(1, args.global_batch_size))
        )
        multiplier = _lr_multiplier(
            step,
            max(1, args.steps or estimated_steps),
            args.warmup_steps,
            args.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * multiplier
        optimizer.step()
        raw_model.sae.normalize_decoder_()
        with torch.no_grad():
            # Feature activity is a distributed diagnostic buffer.  Keep its
            # value identical on every rank so the published alive/dead counts
            # describe the complete global stream rather than rank 0 only.
            global_active_per_feature = active_per_feature.detach().clone()
            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(global_active_per_feature, op=dist.ReduceOp.SUM)
            raw_model.sae.feature_counts.add_(global_active_per_feature)
            raw_model.sae.update_dead_feature_stats_(
                active_per_feature,
                global_samples=max(1, int(valid_global.item())),
                distributed=world_size > 1,
            )
        _update_metrics(
            state,
            reconstructed.detach(),
            target.detach(),
            features.detach(),
            kind.detach(),
            side.detach(),
            target_mean_a_to_b,
            target_mean_b_to_a,
            self_mean,
        )
        step += 1
        consumed_global = min(args.target_occurrences, consumed_local * world_size)
        if step == 1 or step % args.log_every == 0:
            merged = _merge_metric_states(state, device)
            if rank == 0:
                row = {
                    "split": "train",
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "partner_loss": float(partner_loss.detach().item()),
                    "self_loss": float(self_loss.detach().item()),
                    "progress/target_occurrences_seen": consumed_global,
                    "progress/target_coverage": consumed_global / args.target_occurrences,
                    "system/elapsed_seconds": time.time() - started,
                    **_metric_payload(
                        merged,
                        loss_mask=args.loss_mask,
                        target_occurrences=consumed_global,
                    ),
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        if (
            args.periodic_validation_occurrences
            and args.validate_every > 0
            and step % args.validate_every == 0
        ):
            # Validation uses a fresh deterministic stream and never changes
            # optimizer state.  It is intentionally bounded for periodic logs.
            if validation_manifest is not None:
                val_global = args.periodic_validation_occurrences
                if val_global % world_size:
                    raise ValueError("validation-occurrences must be divisible by world size")
                val_stream = PairShardStream(
                    args.validation_cache_dir,
                    validation_manifest,
                    rank=rank,
                    seed=args.seed + 701,
                    max_chunk_length=args.max_chunk_length,
                )
                val_batcher = PairBatcher(
                    val_stream,
                    target_occurrences=val_global // world_size,
                    examples_per_batch=args.examples_per_batch,
                    max_chunk_length=args.max_chunk_length,
                    hidden_size=hidden_size,
                    sentinel=sentinel,
                )
                last_validation = evaluate(
                    ddp,
                    val_batcher,
                    device=device,
                    target_mean_a_to_b=target_mean_a_to_b,
                    target_mean_b_to_a=target_mean_b_to_a,
                    self_mean=self_mean,
                    loss_mask=args.loss_mask,
                )
                selection_value = (
                    float(last_validation["partner_mse"])
                    if args.loss_mask == "partner-only"
                    else 0.5
                    * (
                        float(last_validation["partner_mse"])
                        + float(last_validation["self_mse"])
                    )
                )
                if selection_value < best_metric_value:
                    best_metric_value = selection_value
                    best_step = step
                    best_validation = dict(last_validation)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                if rank == 0:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "split": "validation",
                                    "step": step,
                                    "is_best": step == best_step,
                                    "selection_loss": selection_value,
                                    **last_validation,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
        if args.steps and step >= args.steps:
            break
        if args.save_every > 0 and step % args.save_every == 0 and rank == 0:
            _save_sequence_checkpoint(output / "checkpoints" / "latest", ddp, {**run_config, "step": step})
    # Ensure all ranks have consumed the same logical target budget before the
    # final artifact is published.  Ranks with fewer examples emitted masked
    # dummy batches above and therefore remain in lockstep.
    if dist.is_initialized():
        dist.barrier()
    cache_tokens = int(train_manifest.get("token_occurrences", 0))
    exact_target_coverage = bool(
        cache_tokens > 0
        and args.target_occurrences == cache_tokens
        and consumed_global == args.target_occurrences
    )
    if best_state is None:
        best_step = step
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    del best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final_validation = best_validation or last_validation
    if args.final_validation and validation_manifest is not None:
        val_global = args.validation_occurrences or int(validation_manifest.get("token_occurrences", 0))
        if val_global % world_size:
            raise ValueError("validation target must be divisible by world size")
        val_stream = PairShardStream(
            args.validation_cache_dir,
            validation_manifest,
            rank=rank,
            seed=args.seed + 701,
            max_chunk_length=args.max_chunk_length,
        )
        val_batcher = PairBatcher(
            val_stream,
            target_occurrences=val_global // world_size,
            examples_per_batch=args.examples_per_batch,
            max_chunk_length=args.max_chunk_length,
            hidden_size=hidden_size,
            sentinel=sentinel,
        )
        final_validation = evaluate(
            ddp,
            val_batcher,
            device=device,
            target_mean_a_to_b=target_mean_a_to_b,
            target_mean_b_to_a=target_mean_b_to_a,
            self_mean=self_mean,
            loss_mask=args.loss_mask,
        )
        final_selection_loss = (
            float(final_validation["partner_mse"])
            if args.loss_mask == "partner-only"
            else 0.5
            * (
                float(final_validation["partner_mse"])
                + float(final_validation["self_mse"])
            )
        )
        if not math.isfinite(best_metric_value):
            best_metric_value = final_selection_loss
    else:
        final_selection_loss = best_metric_value
    merged = _merge_metric_states(state, device)
    if rank == 0:
        complete = {
            **run_config,
            "complete": True,
            "steps_completed": step,
            "best_step": best_step,
            "best_metric": "validation_partner_mse"
            if args.loss_mask == "partner-only"
            else "validation_balanced_self_partner_mse",
            "best_metric_value": best_metric_value,
            "final_validation_selection_loss": final_selection_loss,
            "checkpoint_selection": "minimum_periodic_validation_loss",
            "target_occurrences_seen": consumed_global,
            "target_coverage": consumed_global / args.target_occurrences,
            "exact_target_coverage": exact_target_coverage,
            "padding_rows_excluded": padding_rows_excluded,
            "training_metrics": _metric_payload(merged, loss_mask=args.loss_mask, target_occurrences=consumed_global),
            "validation_metrics": final_validation,
            "alive_features": int((model.sae.feature_counts > 0).sum().item()),
            "dead_features": int((model.sae.feature_counts == 0).sum().item()),
            "elapsed_seconds": time.time() - started,
        }
        _save_sequence_checkpoint(output / "checkpoints" / "best", ddp, complete)
        atomic_json_dump(complete, output / "complete.json")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
