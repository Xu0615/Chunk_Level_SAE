#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
from collections import OrderedDict
import queue
import random
import shutil
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel

from chunk_saes.metrics import (
    DASHBOARD_SCALAR_TAGS,
    JOINT_CHUNK_SCALAR_TAGS,
    ReconstructionMetricAccumulator,
    TensorBoardLogger,
    parse_fidelity_reference_fves,
    prefix_metrics,
)
from chunk_saes.runtime import (
    all_gather_objects,
    bind_local_rank_cpu_affinity,
    broadcast_object,
    configure_cpu_threads,
    initialize_distributed,
)
from chunk_saes.sae import (
    BatchTopKSAE,
    JointChunkSAE,
    SAE_PARAMETER_SCHEMA_VERSION,
    SAE_TRAINING_IMPLEMENTATION_VERSION,
    learning_rate_multiplier,
    load_training_checkpoint,
    save_inference_checkpoint,
    save_sae,
    save_training_checkpoint,
    save_training_checkpoint_snapshot,
    snapshot_training_checkpoint,
)
from chunk_saes.tensor_parallel import (
    FeatureTensorParallelSAE,
    all_gather_rows,
    load_sharded_training_checkpoint,
    save_sharded_training_checkpoint,
)
from chunk_saes.temporal import (
    DEFAULT_TEMPORAL_CONTRASTIVE_BLOCK_SIZE,
    DEFAULT_TEMPORAL_HIGH_FRACTION,
    DEFAULT_TEMPORAL_TEMPERATURE,
    distributed_symmetric_temporal_contrastive_loss,
    symmetric_temporal_contrastive_loss,
    temporal_high_feature_count,
)
from chunk_saes.utils import (
    append_jsonl,
    atomic_json_dump,
    distributed_info,
    dtype_from_name,
    log,
    parse_csv,
    seed_everything,
)


CACHE_FORMAT_V2_MARKERS = (
    "v2",
    "occurrence",
)

# Live training and historical backfill share one non-duplicative dashboard
# surface. metrics.jsonl remains the canonical record for every raw statistic.
# Keep legacy mode event surfaces stable. Joint-only tags are added by the
# dedicated joint writer below.
CORE_TENSORBOARD_SCALARS = DASHBOARD_SCALAR_TAGS - JOINT_CHUNK_SCALAR_TAGS
JOINT_TENSORBOARD_SCALARS = DASHBOARD_SCALAR_TAGS


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train occurrence-matched Token, Temporal, Mean-Chunk, and "
            "Cross-Chunk SAEs."
        )
    )
    p.add_argument("--activation-cache-dir", required=True)
    p.add_argument("--validation-cache-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--modes", default="token,temporal,mean,cross")
    p.add_argument(
        "--joint-chunk-alpha",
        type=float,
        default=0.25,
        help="Positive normalized Cross-task weight for the joint_chunk mode.",
    )
    p.add_argument(
        "--joint-cross-prefix",
        type=int,
        default=0,
        help=(
            "Leading dictionary features used by the nested Joint Cross readout. "
            "Zero preserves the existing independent two-decoder-head layout."
        ),
    )
    p.add_argument(
        "--direction-policy",
        choices=("both", "a-to-b", "b-to-a"),
        default="both",
        help=(
            "Cross occurrence direction policy. `both` keeps the cache's two "
            "directions; a single direction deterministically replaces the "
            "opposite-direction rows with the same pair's selected direction."
        ),
    )
    p.add_argument(
        "--target-granularity",
        choices=("mean", "sequence"),
        default="mean",
        help="Target granularity; sequence objectives use the dedicated sequence trainer.",
    )
    p.add_argument(
        "--loss-mask",
        choices=("partner-only", "all"),
        default="partner-only",
        help="Sequence target loss mask (standard mean Cross always uses partner-only).",
    )
    p.add_argument(
        "--mask-representation",
        choices=("norm-matched-mean",),
        default="norm-matched-mean",
        help="Fixed mask sentinel representation for sequence objectives.",
    )
    p.add_argument("--dict-size", type=int, default=65536)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--global-batch-size", type=int, default=32000)
    p.add_argument(
        "--steps",
        "--train-steps",
        dest="steps",
        type=int,
        default=4000,
        help="Optimizer updates per mode.",
    )
    p.add_argument(
        "--require-exact-coverage",
        action="store_true",
        help="Require one finite pass over exactly the cache occurrence universe.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help=(
            "Minimum post-warmup learning-rate ratio. A nonzero floor keeps "
            "late-stage AuxK rescue updates effective."
        ),
    )
    p.add_argument("--threshold-beta", type=float, default=0.99)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--autocast-dtype", default="bfloat16")
    p.add_argument("--normalize-activations", action="store_true")
    p.add_argument(
        "--normalization-samples",
        type=int,
        default=65536,
        help="Global occurrence count used to estimate activation scale/train target mean.",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument(
        "--validation-samples",
        type=int,
        default=262144,
        help="Global held-out occurrences per validation. Zero consumes one finite cache epoch.",
    )
    p.add_argument(
        "--final-validation-samples",
        type=int,
        default=0,
        help="Final validation rows; zero means the complete held-out cache.",
    )
    p.add_argument("--validation-batch-size", type=int, default=0)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from")
    p.add_argument("--overwrite-output", action="store_true")
    p.add_argument("--run-name")
    p.add_argument("--tensorboard-dir")
    p.add_argument("--tensorboard-flush-secs", type=int, default=30)
    p.add_argument("--tensorboard-max-queue", type=int, default=100)
    p.add_argument(
        "--attainable-reference-fves",
        "--fidelity-reference-fves",
        dest="fidelity_reference_fves",
        default="token=1.0,mean=1.0",
        help=(
            "Comma-separated mode=FVE task ceilings. Token/Mean use 1.0; "
            "Cross should use a held-out FVE from a train-fitted high-capacity "
            "predictor. TensorBoard reports sparse FVE divided by this ceiling."
        ),
    )
    p.add_argument("--best-metric", choices=("nmse", "fve"), default="nmse")
    p.add_argument("--validation-use-threshold", action="store_true")
    p.add_argument(
        "--loader-prefetch-shards",
        type=int,
        default=4,
        help="Number of deterministic cache shards prepared ahead of consumption.",
    )
    p.add_argument(
        "--loader-prefetch-workers",
        type=int,
        default=2,
        help="Background shard readers per distributed rank.",
    )
    p.add_argument(
        "--loader-prefetch-batches",
        type=int,
        default=4,
        help="Background pinned/H2D batch queue depth per distributed rank.",
    )
    p.add_argument(
        "--loader-materialize-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sequentially materialize required safetensors payloads before random "
            "row gathers, avoiding random page faults against shared storage."
        ),
    )
    p.add_argument(
        "--loader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pinned host batches and a dedicated CUDA copy stream.",
    )
    p.add_argument(
        "--loader-gpu-shards",
        type=int,
        default=0,
        help="Keep this many materialized activation shards resident on the training GPU.",
    )
    p.add_argument(
        "--local-cache-dir",
        default="",
        help=(
            "Optional node-local rolling shard staging directory. Files are copied "
            "atomically, consumed in the original order, and removed afterwards."
        ),
    )
    p.add_argument(
        "--cache-periodic-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize the fixed periodic-validation subset once per mode.",
    )
    p.add_argument(
        "--validation-cache-max-bytes",
        type=int,
        default=1_073_741_824,
        help="Maximum estimated per-rank bytes for resident periodic validation.",
    )
    p.add_argument(
        "--batch-topk-candidate-multiplier",
        type=float,
        default=2.0,
        help=(
            "Initial exact distributed BatchTopK candidates relative to the "
            "expected per-rank contribution. The algorithm expands only when a "
            "cutoff proof shows that more candidates are required."
        ),
    )
    p.add_argument(
        "--batch-topk-bf16-histogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the exact 65,536-bin BF16 histogram threshold path.",
    )
    p.add_argument(
        "--decoder-backend",
        choices=("dense", "sparse", "auto"),
        default="sparse",
        help="Dense or sparse BatchTopK decoder matmul.",
    )
    p.add_argument(
        "--fused-adam",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--deduplicate-chunk-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode each unique (pair,side) mean once before exact row expansion.",
    )
    p.add_argument(
        "--joint-modes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train Token/Temporal/Mean/Cross in lockstep from one cache pass.",
    )
    p.add_argument(
        "--parallelism",
        choices=("ddp", "feature-tensor"),
        default="ddp",
        help=(
            "DDP replicates the dictionary; feature-tensor shards dictionary "
            "columns and sums partial reconstructions over NVLink."
        ),
    )
    p.add_argument(
        "--max-activation-norm-multiple",
        type=float,
        default=10.0,
        help=(
            "Reject rows whose input or target norm exceeds this multiple of "
            "the current distributed-batch median; non-positive disables it."
        ),
    )
    p.add_argument(
        "--dead-feature-threshold",
        type=int,
        default=10_000_000,
        help="Occurrences without firing before a feature is reported dead.",
    )
    p.add_argument(
        "--auxk-alpha",
        type=float,
        default=0.0625,
        help="Residual reconstruction loss weight for recently dead features.",
    )
    p.add_argument(
        "--auxk-activation-age",
        type=int,
        default=5_000_000,
        help=(
            "Occurrences without firing before a feature enters AuxK rescue; "
            "this should be lower than dead-feature-threshold."
        ),
    )
    p.add_argument(
        "--auxk",
        type=int,
        default=512,
        help="Dead features selected per sample for the auxiliary reconstruction.",
    )
    p.add_argument(
        "--auxk-candidate-features",
        type=int,
        default=4096,
        help="Oldest dead features considered by the auxiliary branch per update.",
    )
    p.add_argument(
        "--temporal-high-fraction",
        type=float,
        default=DEFAULT_TEMPORAL_HIGH_FRACTION,
        help="Leading dictionary fraction assigned to T-SAE high-level features.",
    )
    p.add_argument(
        "--temporal-high-reconstruction-weight",
        type=float,
        default=0.2,
        help="Weight on reconstruction from the high-level feature prefix.",
    )
    p.add_argument(
        "--temporal-full-reconstruction-weight",
        type=float,
        default=0.8,
        help="Weight on full-dictionary token reconstruction.",
    )
    p.add_argument(
        "--temporal-alpha",
        type=float,
        default=1.0,
        help="Weight on the high-level adjacent-token contrastive objective.",
    )
    p.add_argument(
        "--temporal-temperature",
        type=float,
        default=DEFAULT_TEMPORAL_TEMPERATURE,
        help="Temperature for normalized symmetric temporal InfoNCE.",
    )
    p.add_argument(
        "--temporal-contrastive-block-size",
        type=int,
        default=DEFAULT_TEMPORAL_CONTRASTIVE_BLOCK_SIZE,
        help=(
            "Maximum local matched-pair block used for exact within-block "
            "symmetric InfoNCE."
        ),
    )
    p.add_argument(
        "--defer-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep improving best weights in rank-0 host memory and publish a "
            "weights-only best artifact at resumable latest-checkpoint boundaries."
        ),
    )
    p.add_argument(
        "--async-latest-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach latest checkpoints to CPU and publish them on a background writer.",
    )
    p.add_argument(
        "--checkpoint-staging-dir",
        default="",
        help=(
            "Optional node-local root for async latest checkpoints. A complete "
            "local snapshot is written first, then copied, SHA-256 verified, and "
            "published to the output directory with its manifest last."
        ),
    )
    p.add_argument(
        "--bind-cpu-affinity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bind each local rank to a disjoint CPU set near its physical GPU.",
    )
    return p


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    for key in (
        "manifest_digest",
        "cache_digest",
        "token_manifest_digest",
        "occurrence_manifest_digest",
        "activation_digest",
        "plan_digest",
    ):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


def load_cache_manifest(cache_dir: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(cache_dir) / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("complete"):
        raise ValueError(f"activation cache is not complete: {path}")
    return manifest, manifest_digest(manifest)


def _manifest_value(manifest: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in manifest and manifest[key] is not None:
            return manifest[key]
    return None


def validate_cache_compatibility(
    train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any] | None,
) -> None:
    if validation_manifest is None:
        return
    train_hidden = int(_manifest_value(train_manifest, "hidden_size", "activation_dim"))
    val_hidden = int(_manifest_value(validation_manifest, "hidden_size", "activation_dim"))
    if train_hidden != val_hidden:
        raise ValueError(
            f"train/validation hidden sizes differ: {train_hidden} != {val_hidden}"
        )
    for keys in (
        ("model_hash", "model_digest"),
        ("tokenizer_hash", "tokenizer_digest"),
        ("layer", "layer_index"),
    ):
        left = _manifest_value(train_manifest, *keys)
        right = _manifest_value(validation_manifest, *keys)
        if left is not None and right is not None and left != right:
            raise ValueError(f"train/validation cache mismatch for {keys}: {left} != {right}")


def validate_exact_training_coverage(
    manifest: Mapping[str, Any],
    *,
    world_size: int,
    global_batch_size: int,
    steps: int,
) -> int:
    if not cache_is_v2(manifest):
        raise ValueError("exact coverage requires an activation-cache v2 manifest")
    occurrences = cache_occurrences(manifest)
    if occurrences is None or occurrences <= 0:
        raise ValueError("exact coverage cache lacks a positive occurrence count")
    if manifest.get("token_occurrences") != manifest.get("target_token_occurrences"):
        raise ValueError("training cache does not contain its exact target occurrence count")
    if manifest.get("corpus_position_overlap_policy") != "forbidden":
        raise ValueError("training cache was not built from a no-overlap sample plan")
    if manifest.get("corpus_position_overlap_verified") is not True:
        raise ValueError("training cache lacks verified no-overlap provenance")
    if int(manifest.get("unique_corpus_token_positions", -1)) != occurrences:
        raise ValueError("training cache does not represent unique corpus token positions")
    if global_batch_size <= 0 or global_batch_size % world_size:
        raise ValueError("global batch size must be positive and divisible by world size")
    if steps * global_batch_size != occurrences:
        raise ValueError(
            "exact coverage requires steps * global_batch_size == cache occurrences; "
            f"{steps} * {global_batch_size} != {occurrences}"
        )
    if occurrences % world_size:
        raise ValueError("cache occurrences must be divisible by world size")
    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise ValueError("cache rank manifests do not match training world size")
    expected_per_rank = occurrences // world_size
    observed = sorted(
        (int(item.get("rank", -1)), int(item.get("token_occurrences", -1)))
        for item in ranks
    )
    if observed != [(rank, expected_per_rank) for rank in range(world_size)]:
        raise ValueError(
            f"cache ranks do not provide exact equal token coverage: {observed}"
        )
    statistics = manifest.get("target_sufficient_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("exact coverage cache lacks target sufficient statistics")
    if int(statistics.get("count", -1)) != occurrences:
        raise ValueError("target sufficient statistics do not cover every occurrence")
    means = statistics.get("mean_by_mode")
    if not isinstance(means, Mapping) or not {
        "token",
        "mean",
        "cross",
        "cross_a_to_b",
        "cross_b_to_a",
    }.issubset(means):
        raise ValueError("target sufficient statistics do not cover all SAE modes")
    counts = statistics.get("count_by_mode")
    if not isinstance(counts, Mapping):
        raise ValueError("target sufficient statistics lack per-mode counts")
    if any(
        int(counts.get(mode, -1)) != occurrences
        for mode in ("token", "mean", "cross")
    ):
        raise ValueError("primary target-stat counts differ from cache occurrences")
    if int(counts.get("cross_a_to_b", -1)) + int(
        counts.get("cross_b_to_a", -1)
    ) != occurrences:
        raise ValueError("Cross directional target-stat counts do not cover the cache")
    return occurrences


def cache_target_mean(
    manifest: Mapping[str, Any],
    mode: str,
    *,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    statistics = manifest.get("target_sufficient_statistics")
    if not isinstance(statistics, Mapping):
        return None
    means = statistics.get("mean_by_mode")
    source_mode = "token" if mode == "temporal" else mode
    if not isinstance(means, Mapping) or source_mode not in means:
        return None
    value = torch.tensor(means[source_mode], dtype=torch.float32, device=device)
    if value.shape != (hidden_size,) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"cache target mean for mode={mode} is invalid")
    return value


def cache_input_mean(
    manifest: Mapping[str, Any],
    mode: str,
    *,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    # Token and Mean are self-reconstruction objectives. Cross consumes the
    # same owning chunk mean as Mean while predicting the partner chunk mean.
    source_mode = (
        "mean"
        if mode == "cross"
        else "token"
        if mode == "temporal"
        else mode
    )
    return cache_target_mean(
        manifest,
        source_mode,
        hidden_size=hidden_size,
        device=device,
    )


def cache_hidden_size(manifest: Mapping[str, Any]) -> int:
    value = _manifest_value(manifest, "hidden_size", "activation_dim")
    if value is None:
        raise ValueError("cache manifest lacks hidden_size/activation_dim")
    return int(value)


def cache_occurrences(manifest: Mapping[str, Any]) -> int | None:
    value = _manifest_value(
        manifest,
        "occurrences",
        "token_occurrences",
        "unique_token_occurrences",
        "samples",
        "tokens",
    )
    return int(value) if value is not None else None


def cache_is_v2(manifest: Mapping[str, Any]) -> bool:
    format_name = str(manifest.get("format", "")).lower()
    return any(marker in format_name for marker in CACHE_FORMAT_V2_MARKERS) or any(
        key in manifest
        for key in (
            "occurrences",
            "token_occurrences",
            "occurrence_manifest_digest",
            "token_manifest_digest",
            "chunk_offsets",
        )
    )


def _field(batch: Any, *names: str) -> Any:
    if isinstance(batch, Mapping):
        for name in names:
            if name in batch:
                return batch[name]
    for name in names:
        if hasattr(batch, name):
            return getattr(batch, name)
    return None


def _direction(batch: Any, rows: int, device: torch.device) -> torch.Tensor | None:
    raw = _field(
        batch,
        "side",
        "chunk_side",
        "occurrence_side",
        "direction",
        "is_b",
        "choose_b",
    )
    if raw is None:
        chunk_index = _field(batch, "chunk_index", "chunk_id", "occurrence_chunk_index")
        if chunk_index is not None:
            raw = torch.as_tensor(chunk_index) % 2
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], str):
        values = [1 if str(value).strip().lower() in {"b", "b_to_a", "1"} else 0 for value in raw]
        tensor = torch.tensor(values, device=device)
    elif isinstance(raw, str):
        tensor = torch.full(
            (rows,),
            1 if raw.strip().lower() in {"b", "b_to_a", "1"} else 0,
            device=device,
        )
    else:
        tensor = torch.as_tensor(raw, device=device)
    tensor = tensor.reshape(-1)
    if tensor.numel() != rows:
        raise ValueError(f"direction metadata has {tensor.numel()} rows, expected {rows}")
    if tensor.dtype == torch.bool:
        return tensor
    return tensor.to(torch.int64) != 0


def _direction_policy_batch(batch: Any, policy: str) -> Any:
    """Rewrite occurrence side metadata while retaining the cache row universe.

    The cache stores one row for every token occurrence.  For a one-way Cross
    ablation the opposite-direction slots are intentionally reused as rows of
    the selected direction from the same pair.  Rewriting all side aliases is
    important because transitional cache readers expose different names.
    """

    if policy == "both":
        return batch
    if policy not in {"a-to-b", "b-to-a"}:
        raise ValueError(f"unknown direction policy: {policy}")
    rows = _batch_rows(batch)
    # The loader moves every tensor to the target device before this helper.
    reference = _field(
        batch,
        "token_hidden",
        "occurrence_hidden",
        "hidden",
        "token",
        "chunk_mean",
        "mean",
        "mean_a",
    )
    device = reference.device if isinstance(reference, torch.Tensor) else torch.device("cpu")
    forced = torch.full(
        (rows,),
        policy == "b-to-a",
        dtype=torch.bool,
        device=device,
    )
    aliases = {
        "side",
        "chunk_side",
        "occurrence_side",
        "direction",
        "is_b",
        "choose_b",
    }
    if dataclasses.is_dataclass(batch):
        values = {}
        fields = {field.name for field in dataclasses.fields(batch)}
        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            if field.name in aliases and field.name in fields:
                value = forced.to(dtype=value.dtype) if isinstance(value, torch.Tensor) else forced
            values[field.name] = value
        return type(batch)(**values)
    if isinstance(batch, Mapping):
        result = dict(batch)
        mean_a = result.get("mean_a")
        mean_b = result.get("mean_b")
        if isinstance(mean_a, torch.Tensor) and isinstance(mean_b, torch.Tensor):
            if policy == "a-to-b":
                result["chunk_mean"] = mean_a
                result["partner_mean"] = mean_b
            else:
                result["chunk_mean"] = mean_b
                result["partner_mean"] = mean_a
        for name in aliases:
            if name in result:
                old = result[name]
                result[name] = (
                    forced.to(dtype=old.dtype)
                    if isinstance(old, torch.Tensor)
                    else forced
                )
        # Always add the canonical alias so a packed/transitional batch with no
        # side field still follows the requested policy.
        result["direction"] = forced
        return result
    raise TypeError(f"cannot apply direction policy to batch type {type(batch)!r}")


def _cross_target_mode(policy: str) -> str:
    if policy == "a-to-b":
        return "cross_a_to_b"
    if policy == "b-to-a":
        return "cross_b_to_a"
    return "cross"


def _require_tensor(batch: Any, names: tuple[str, ...], description: str) -> torch.Tensor:
    value = _field(batch, *names)
    if value is None:
        raise ValueError(f"cache batch lacks {description}; tried fields {names}")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def select_occurrence_view(
    mode: str,
    batch: Any,
    *,
    allow_legacy_v1: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Map one occurrence schedule row to the three matched SAE views.

    Cache-v2 loaders should expose either explicit ``token_hidden``,
    ``chunk_mean`` and ``partner_mean`` rows, or packed means plus deterministic
    occurrence-to-chunk indices. The fallback names keep this trainer compatible
    while the cache-v2 implementation is landing.
    """

    token = _field(
        batch,
        "token_hidden",
        "token_activation",
        "occurrence_hidden",
        "hidden",
        "token",
    )
    chunk_mean = _field(
        batch,
        "chunk_mean",
        "source_mean",
        "occurrence_mean",
        "mean",
        "input_mean",
    )
    partner_mean = _field(
        batch,
        "partner_mean",
        "target_mean",
        "other_chunk_mean",
        "cross_target",
    )

    if chunk_mean is None:
        means = _field(batch, "chunk_means", "means")
        chunk_index = _field(batch, "chunk_index", "occurrence_chunk_index")
        if means is not None and chunk_index is not None:
            means = torch.as_tensor(means)
            chunk_index = torch.as_tensor(chunk_index, device=means.device, dtype=torch.long)
            chunk_mean = means.index_select(0, chunk_index)
            partner_index = _field(batch, "partner_chunk_index", "partner_index")
            if partner_index is None:
                partner_index = torch.bitwise_xor(chunk_index, torch.ones_like(chunk_index))
            else:
                partner_index = torch.as_tensor(
                    partner_index, device=means.device, dtype=torch.long
                )
            partner_mean = means.index_select(0, partner_index)

    # Transitional cache-v2 writers may retain aligned mean_a/mean_b rows plus
    # occurrence side. This is deterministic and remains occurrence-matched.
    if chunk_mean is None or (mode == "cross" and partner_mean is None):
        mean_a = _field(batch, "mean_a")
        mean_b = _field(batch, "mean_b")
        if mean_a is not None and mean_b is not None:
            mean_a = torch.as_tensor(mean_a)
            mean_b = torch.as_tensor(mean_b)
            direction = _direction(batch, mean_a.shape[0], mean_a.device)
            if direction is not None:
                choose_b = direction.unsqueeze(-1)
                chunk_mean = torch.where(choose_b, mean_b, mean_a)
                partner_mean = torch.where(choose_b, mean_a, mean_b)
            elif allow_legacy_v1:
                # v1 has one row per pair, not one row per occurrence. This
                # deterministic A-side compatibility path is intentionally not
                # advertised as occurrence matched and is rejected for v2 runs.
                chunk_mean = mean_a
                partner_mean = mean_b

    if mode in {"token", "temporal"}:
        if token is None:
            token = _require_tensor(
                batch,
                ("token_hidden", "occurrence_hidden", "hidden", "token"),
                "per-occurrence token hidden state",
            )
        token = torch.as_tensor(token)
        direction = _direction(batch, token.shape[0], token.device)
        return token, token, direction
    if mode == "mean":
        if chunk_mean is None:
            raise ValueError(
                "cache-v2 batch must provide the owning chunk mean for every occurrence"
            )
        chunk_mean = torch.as_tensor(chunk_mean)
        direction = _direction(batch, chunk_mean.shape[0], chunk_mean.device)
        return chunk_mean, chunk_mean, direction
    if mode == "cross":
        if chunk_mean is None or partner_mean is None:
            raise ValueError(
                "cache-v2 batch must provide owning and partner chunk means for cross mode"
            )
        chunk_mean = torch.as_tensor(chunk_mean)
        partner_mean = torch.as_tensor(partner_mean)
        direction = _direction(batch, chunk_mean.shape[0], chunk_mean.device)
        return chunk_mean, partner_mean, direction
    if mode == "joint_chunk":
        if chunk_mean is None or partner_mean is None:
            raise ValueError(
                "cache-v2 batch must provide owning and partner chunk means for joint_chunk"
            )
        chunk_mean = torch.as_tensor(chunk_mean)
        partner_mean = torch.as_tensor(partner_mean)
        direction = _direction(batch, chunk_mean.shape[0], chunk_mean.device)
        return chunk_mean, partner_mean, direction
    raise ValueError(f"unknown mode: {mode}")


def select_joint_chunk_view(
    batch: Any,
    *,
    allow_legacy_v1: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return owning mean, partner mean, and side for Joint Chunk training."""

    return select_occurrence_view(
        "joint_chunk",
        batch,
        allow_legacy_v1=allow_legacy_v1,
    )


def select_temporal_pair_view(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    previous = _require_tensor(
        batch,
        (
            "previous_token_hidden",
            "temporal_previous",
            "previous_hidden",
        ),
        "per-occurrence previous-token hidden state",
    )
    mask = _require_tensor(
        batch,
        (
            "temporal_pair_mask",
            "has_previous_token",
            "temporal_valid",
        ),
        "within-chunk temporal-pair mask",
    )
    previous = torch.as_tensor(previous)
    mask = torch.as_tensor(mask, device=previous.device, dtype=torch.bool).reshape(-1)
    if previous.shape[0] != mask.numel():
        raise ValueError("temporal previous-token rows and mask are misaligned")
    return previous, mask


def chunk_input_deduplication(
    batch: Any,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return first-row indices and inverse mapping for identical chunk inputs."""

    pair_id = _field(batch, "pair_id")
    if not isinstance(pair_id, torch.Tensor):
        return None
    pair_id = pair_id.reshape(-1).to(dtype=torch.long)
    if pair_id.numel() != rows:
        return None
    side = _direction(batch, rows, pair_id.device)
    if side is None:
        return None
    keys = pair_id * 2 + side.to(torch.long)
    _unique, inverse = torch.unique(
        keys,
        sorted=True,
        return_inverse=True,
    )
    counts = torch.bincount(inverse)
    order = torch.argsort(inverse, stable=True)
    starts = torch.cumsum(counts, dim=0) - counts
    first_rows = order.index_select(0, starts)
    return first_rows, inverse


def activation_norm_ok_mask(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_activation_norm_multiple: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a distributed-batch median norm filter and its two thresholds."""

    rows = inputs.shape[0]
    if max_activation_norm_multiple <= 0:
        ones = torch.ones(rows, dtype=torch.bool, device=inputs.device)
        nan = torch.full((), float("nan"), device=inputs.device)
        return ones, nan, nan
    input_norms = inputs.float().norm(dim=-1)
    target_norms = targets.float().norm(dim=-1)
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        gathered_inputs = torch.empty(
            world_size * rows,
            dtype=input_norms.dtype,
            device=input_norms.device,
        )
        gathered_targets = torch.empty_like(gathered_inputs)
        dist.all_gather_into_tensor(gathered_inputs, input_norms.contiguous())
        dist.all_gather_into_tensor(gathered_targets, target_norms.contiguous())
    else:
        gathered_inputs = input_norms
        gathered_targets = target_norms
    input_threshold = gathered_inputs.median() * max_activation_norm_multiple
    target_threshold = gathered_targets.median() * max_activation_norm_multiple
    valid = (
        torch.isfinite(input_norms)
        & torch.isfinite(target_norms)
        & (input_norms <= input_threshold)
        & (target_norms <= target_threshold)
    )
    return valid, input_threshold, target_threshold


def equalize_valid_rows(mask: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Keep the same accepted row count on every rank for DDP BatchTopK."""

    local_valid = mask.sum(dtype=torch.int64)
    common_valid = local_valid.clone()
    if dist.is_initialized():
        dist.all_reduce(common_valid, op=dist.ReduceOp.MIN)
    keep_count = int(common_valid.item())
    dropped_for_balance = int(local_valid.item()) - keep_count
    if keep_count < int(local_valid.item()):
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        balanced = torch.zeros_like(mask)
        balanced[valid_indices[:keep_count]] = True
        mask = balanced
    return mask, keep_count, dropped_for_balance


def _filter_batch(batch: Any, mask: torch.Tensor) -> Any:
    if dataclasses.is_dataclass(batch):
        values = {}
        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == mask.shape[0]:
                value = value[mask]
            values[field.name] = value
        return type(batch)(**values)
    if isinstance(batch, Mapping):
        return {
            key: value[mask]
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == mask.shape[0]
            else value
            for key, value in batch.items()
        }
    raise TypeError(f"cannot filter cache batch type {type(batch)!r}")


def _slice_batch(batch: Any, count: int) -> Any:
    tensor = next(
        (
            value
            for value in (
                _field(batch, "token_hidden", "occurrence_hidden", "hidden", "token"),
                _field(batch, "chunk_mean", "mean", "mean_a"),
            )
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    if tensor is None:
        return batch
    mask = torch.arange(tensor.shape[0], device=tensor.device) < count
    return _filter_batch(batch, mask)


def _batch_rows(batch: Any) -> int:
    for names in (
        ("token_hidden", "occurrence_hidden", "hidden", "token"),
        ("chunk_mean", "mean", "mean_a"),
    ):
        value = _field(batch, *names)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    raise ValueError("unable to infer cache batch row count")


def _move_batch(batch: Any, device: str | torch.device) -> Any:
    if dataclasses.is_dataclass(batch):
        values = {}
        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
            values[field.name] = value
        return type(batch)(**values)
    if isinstance(batch, Mapping):
        return {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
    return batch


def _pin_batch(batch: Any) -> Any:
    if dataclasses.is_dataclass(batch):
        values = {}
        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                value = value.pin_memory()
            values[field.name] = value
        return type(batch)(**values)
    if isinstance(batch, Mapping):
        return {
            key: value.pin_memory()
            if isinstance(value, torch.Tensor) and value.device.type == "cpu"
            else value
            for key, value in batch.items()
        }
    return batch


def _record_batch_stream(batch: Any, stream: torch.cuda.Stream) -> None:
    values: Iterable[Any]
    if dataclasses.is_dataclass(batch):
        values = (getattr(batch, field.name) for field in dataclasses.fields(batch))
    elif isinstance(batch, Mapping):
        values = batch.values()
    else:
        values = ()
    for value in values:
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            value.record_stream(stream)


@dataclasses.dataclass
class _AsyncBatchValue:
    batch: Any
    event: torch.cuda.Event | None = None
    host_batch: Any | None = None


@dataclasses.dataclass
class _AsyncBatchFailure:
    error: BaseException


_ASYNC_BATCH_END = object()


class _AsyncBatchIterator:
    """Prepare batches and H2D copies without blocking the training thread."""

    def __init__(
        self,
        source: Iterator[Any],
        *,
        device: torch.device,
        queue_depth: int,
        pin_memory: bool,
    ) -> None:
        self.source = source
        self.device = device
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_depth))
        self.stop_event = threading.Event()
        self.closed = False
        self.pin_memory = bool(pin_memory and device.type == "cuda")
        self.inflight_host_batches: list[
            tuple[torch.cuda.Event, Any]
        ] = []
        self.thread = threading.Thread(
            target=self._worker,
            name=f"sae-batch-prefetch-{device}",
            daemon=True,
        )
        self.thread.start()

    def _put(self, value: Any) -> bool:
        while not self.stop_event.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self) -> None:
        copy_stream = None
        try:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    copy_stream = torch.cuda.Stream(device=self.device)
            for batch in self.source:
                if self.stop_event.is_set():
                    break
                host_batch = _pin_batch(batch) if self.pin_memory else batch
                if copy_stream is None:
                    value = _AsyncBatchValue(batch=host_batch)
                else:
                    producer_stream = torch.cuda.current_stream(self.device)
                    with torch.cuda.device(self.device), torch.cuda.stream(copy_stream):
                        copy_stream.wait_stream(producer_stream)
                        device_batch = _move_batch(host_batch, self.device)
                        event = torch.cuda.Event()
                        event.record(copy_stream)
                    # Keep the pinned source alive until the copy event is consumed.
                    value = _AsyncBatchValue(
                        batch=device_batch,
                        event=event,
                        host_batch=host_batch,
                    )
                if not self._put(value):
                    break
        except BaseException as error:  # propagate worker failures to the consumer
            self._put(_AsyncBatchFailure(error))
        finally:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()
            self._put(_ASYNC_BATCH_END)

    def __iter__(self) -> "_AsyncBatchIterator":
        return self

    def __next__(self) -> Any:
        value = self.queue.get()
        if value is _ASYNC_BATCH_END:
            self.close()
            raise StopIteration
        if isinstance(value, _AsyncBatchFailure):
            self.close()
            raise value.error
        assert isinstance(value, _AsyncBatchValue)
        if value.event is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_event(value.event)
            _record_batch_stream(value.batch, current_stream)
            self.inflight_host_batches = [
                (event, host)
                for event, host in self.inflight_host_batches
                if not event.query()
            ]
            self.inflight_host_batches.append((value.event, value.host_batch))
        return value.batch

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=2.0)
        for event, _host_batch in self.inflight_host_batches:
            event.synchronize()
        self.inflight_host_batches.clear()

    def __del__(self) -> None:
        self.close()


@dataclasses.dataclass
class _PreparedV2Shard:
    path: Path
    item: Mapping[str, Any]
    mode: str
    tensors: dict[str, torch.Tensor]
    permutation: torch.Tensor
    pair_index: torch.Tensor
    side: torch.Tensor
    occurrence_id: torch.Tensor
    previous_row: torch.Tensor
    temporal_pair_mask: torch.Tensor
    cleanup_path: Path | None = None
    release_gpu: Callable[[], None] | None = None

    @property
    def rows(self) -> int:
        return int(self.permutation.numel())

    def take(self, start: int, count: int) -> dict[str, torch.Tensor]:
        occurrence_rows = self.permutation.narrow(0, start, count)
        pair_rows = self.pair_index.index_select(0, occurrence_rows)
        side = self.side.index_select(0, occurrence_rows)
        rows: dict[str, torch.Tensor] = {
            "side": side,
            "occurrence_id": self.occurrence_id.index_select(0, occurrence_rows),
            "pair_id": self.tensors["pair_id"]
            .to(torch.long)
            .index_select(0, pair_rows),
        }
        if self.mode in {"token", "temporal", "all"}:
            rows["token_hidden"] = self.tensors["token_hidden"].index_select(
                0, occurrence_rows
            )
            if self.mode == "token":
                return rows
            if self.mode in {"temporal", "all"}:
                rows["previous_token_hidden"] = self.tensors[
                    "token_hidden"
                ].index_select(
                    0,
                    self.previous_row.index_select(0, occurrence_rows),
                )
                rows["temporal_pair_mask"] = self.temporal_pair_mask.index_select(
                    0,
                    occurrence_rows,
                )
                if self.mode == "temporal":
                    return rows

        mean_a = self.tensors["mean_a"].index_select(0, pair_rows)
        mean_b = self.tensors["mean_b"].index_select(0, pair_rows)
        # Retain both pair-side means so the one-way ablation can replace the
        # opposite-direction occurrence slots without changing row count or
        # BatchTopK budget.  The adapter removes these fields for non-Cross
        # modes only when they are not requested.
        if self.mode in {"cross", "joint_chunk", "all"}:
            rows["mean_a"] = mean_a
            rows["mean_b"] = mean_b
        choose_b = side.unsqueeze(-1)
        rows["chunk_mean"] = torch.where(choose_b, mean_b, mean_a)
        if self.mode in {"cross", "joint_chunk", "all"}:
            rows["partner_mean"] = torch.where(choose_b, mean_a, mean_b)
        return rows

    def close(self) -> None:
        if self.release_gpu is not None:
            self.release_gpu()
            self.release_gpu = None
        if self.cleanup_path is not None:
            try:
                self.cleanup_path.unlink(missing_ok=True)
            finally:
                self.cleanup_path = None


class PackedV2OccurrenceLoader:
    """Deterministic cache-v2 stream with lazy gathers and bounded prefetch.

    The previous implementation expanded and randomly permuted complete
    131K-row hidden tensors before producing a 4K-row minibatch. This loader
    preserves the exact shard order, per-shard permutation, resume offset, and
    cross-shard minibatch boundaries while gathering only the rows required by
    the current minibatch. Required file payloads can be staged and
    sequentially materialized in background workers, and complete batches can
    be pinned/copied on a dedicated CUDA stream.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        manifest: Mapping[str, Any],
        *,
        rank: int,
        batch_size: int,
        seed: int,
        device: str,
        mode: str,
        prefetch_shards: int = 4,
        prefetch_workers: int = 2,
        prefetch_batches: int = 4,
        materialize_shards: bool = True,
        pin_memory: bool = True,
        local_cache_dir: str | Path | None = None,
        gpu_shards: int = 0,
    ) -> None:
        self.root = Path(cache_dir)
        self.manifest = manifest
        self.rank = rank
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)
        self.prefetch_shards = max(1, int(prefetch_shards))
        self.prefetch_workers = max(1, int(prefetch_workers))
        self.prefetch_batches = max(0, int(prefetch_batches))
        self.materialize_shards = bool(materialize_shards)
        self.pin_memory = bool(pin_memory)
        self.local_cache_dir = (
            Path(local_cache_dir).expanduser()
            if local_cache_dir is not None and str(local_cache_dir)
            else None
        )
        if self.local_cache_dir is not None:
            self.local_cache_dir = self.local_cache_dir / f"rank{rank:03d}"
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        if mode not in {"token", "temporal", "mean", "cross", "joint_chunk", "all"}:
            raise ValueError(f"unsupported cache-v2 occurrence mode: {mode}")
        self.mode = mode
        ranks = manifest.get("ranks")
        if not isinstance(ranks, list):
            raise ValueError("cache-v2 manifest lacks rank manifests")
        rank_manifest = next(
            (item for item in ranks if int(item.get("rank", -1)) == rank),
            None,
        )
        if rank_manifest is None:
            raise ValueError(f"cache-v2 manifest lacks rank {rank}")
        self.shard_entries = [dict(item) for item in rank_manifest["shards"]]
        self.shards = [self.root / item["path"] for item in self.shard_entries]
        if not self.shards:
            raise ValueError(f"cache-v2 rank {rank} has no shards")
        self.occurrences = int(rank_manifest["token_occurrences"])
        self.gpu_shards = max(0, int(gpu_shards))
        self._gpu_cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._gpu_cache_lock = threading.Lock()
        self._gpu_copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.gpu_shards > 0 and self.device.type == "cuda"
            else None
        )

    def _required_names(self) -> set[str]:
        required = {
            "pair_id",
            "occurrence_start",
            "length_a",
            "length_b",
        }
        if self.mode in {"token", "temporal"}:
            required.add("token_hidden")
        elif self.mode in {"mean", "cross", "joint_chunk"}:
            required.update(("mean_a", "mean_b"))
        else:
            required.update(("token_hidden", "mean_a", "mean_b"))
        return required

    def _stage(self, source: Path) -> Path:
        # Mean/Cross read only the compact per-pair means from a 1 GiB token
        # shard. Copying the complete file would be counterproductive; local
        # rolling staging is reserved for Token's full hidden payload.
        if self.local_cache_dir is None or self.mode not in {
            "token",
            "temporal",
            "all",
        }:
            return source
        try:
            if source.stat().st_dev == self.local_cache_dir.stat().st_dev:
                return source
        except OSError:
            pass
        identity = hashlib.blake2b(
            str(source).encode("utf-8"), digest_size=8
        ).hexdigest()
        target = self.local_cache_dir / f"{identity}-{source.name}"
        if target.is_file() and target.stat().st_size == source.stat().st_size:
            return target
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.unlink(missing_ok=True)
        try:
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != source.stat().st_size:
                raise IOError(
                    f"incomplete local cache staging copy: {source} -> {target}"
                )
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def _load(
        self, path: Path
    ) -> tuple[dict[str, torch.Tensor], Path | None]:
        staged = self._stage(path)
        cleanup_path = staged if staged != path else None
        required = self._required_names()
        try:
            with safe_open(str(staged), framework="pt", device="cpu") as handle:
                missing = required - set(handle.keys())
                if missing:
                    raise ValueError(
                        f"cache-v2 shard {path} lacks tensors {sorted(missing)}"
                    )
                tensors = {}
                for name in required:
                    value = handle.get_tensor(name)
                    # A clone forces a sequential file read before random row
                    # gathers and detaches the tensor from the staged mmap.
                    tensors[name] = (
                        value.clone(memory_format=torch.contiguous_format)
                        if self.materialize_shards
                        else value
                    )
            if self.materialize_shards and cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
                cleanup_path = None
            return tensors, cleanup_path
        except BaseException:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
            raise

    def _to_gpu_cache(self, path: Path, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.gpu_shards <= 0 or self.device.type != "cuda":
            return tensors
        key = str(path)
        with self._gpu_cache_lock:
            cached = self._gpu_cache.get(key)
            if cached is not None:
                self._gpu_cache.move_to_end(key)
                return cached
            assert self._gpu_copy_stream is not None
            pinned = {
                name: value
                if value.is_pinned()
                else value.pin_memory()
                for name, value in tensors.items()
            }
            with torch.cuda.device(self.device), torch.cuda.stream(
                self._gpu_copy_stream
            ):
                moved = {
                    name: value.to(self.device, non_blocking=True)
                    for name, value in pinned.items()
                }
                ready = torch.cuda.Event()
                ready.record(self._gpu_copy_stream)
            # The worker may return only after every destination tensor is ready;
            # training continues concurrently on its own stream during this wait.
            ready.synchronize()
            self._gpu_cache[key] = moved
            while len(self._gpu_cache) > self.gpu_shards:
                self._gpu_cache.popitem(last=False)
            return moved

    def _release_gpu_cache(self, path: Path) -> None:
        if self.gpu_shards <= 0:
            return
        with self._gpu_cache_lock:
            self._gpu_cache.pop(str(path), None)

    @staticmethod
    def _row_metadata(
        tensors: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths_a = tensors["length_a"].to(torch.long)
        lengths_b = tensors["length_b"].to(torch.long)
        pair_lengths = lengths_a + lengths_b
        pair_numbers = torch.arange(
            pair_lengths.shape[0], dtype=torch.long, device=pair_lengths.device
        )
        pair_index = torch.repeat_interleave(pair_numbers, pair_lengths)
        pair_starts = torch.cumsum(pair_lengths, dim=0) - pair_lengths
        row_numbers = torch.arange(
            pair_index.shape[0], dtype=torch.long, device=pair_index.device
        )
        within_pair = row_numbers - pair_starts.index_select(0, pair_index)
        side = within_pair >= lengths_a.index_select(0, pair_index)
        occurrence_id = (
            tensors["occurrence_start"].to(torch.long).index_select(0, pair_index)
            + within_pair
        )
        return pair_index, side, occurrence_id

    @staticmethod
    def _temporal_metadata(
        tensors: Mapping[str, torch.Tensor],
        pair_index: torch.Tensor,
        side: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths_a = tensors["length_a"].to(
            device=pair_index.device,
            dtype=torch.long,
        )
        lengths_b = tensors["length_b"].to(
            device=pair_index.device,
            dtype=torch.long,
        )
        pair_lengths = lengths_a + lengths_b
        pair_starts = torch.cumsum(pair_lengths, dim=0) - pair_lengths
        row_numbers = torch.arange(
            pair_index.shape[0],
            dtype=torch.long,
            device=pair_index.device,
        )
        within_pair = row_numbers - pair_starts.index_select(0, pair_index)
        within_chunk = torch.where(
            side,
            within_pair - lengths_a.index_select(0, pair_index),
            within_pair,
        )
        pair_mask = within_chunk > 0
        previous_row = row_numbers - pair_mask.to(torch.long)
        return previous_row, pair_mask

    @staticmethod
    def _expand(
        tensors: Mapping[str, torch.Tensor],
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reference full expansion retained for compatibility and tests."""

        if mode is None:
            mode = "all"
        pair_index, side, occurrence_id = PackedV2OccurrenceLoader._row_metadata(
            tensors
        )
        row_count = int(pair_index.numel())
        if mode in {"token", "temporal", "all"} and row_count != tensors[
            "token_hidden"
        ].shape[0]:
            raise ValueError("cache-v2 pair lengths do not cover token_hidden")
        rows = {
            "side": side,
            "occurrence_id": occurrence_id,
            "pair_id": tensors["pair_id"].to(torch.long).index_select(0, pair_index),
        }
        if mode in {"token", "temporal", "all"}:
            rows["token_hidden"] = tensors["token_hidden"]
            if mode == "token":
                return rows
            if mode in {"temporal", "all"}:
                previous_row, temporal_pair_mask = (
                    PackedV2OccurrenceLoader._temporal_metadata(
                        tensors,
                        pair_index,
                        side,
                    )
                )
                rows["previous_token_hidden"] = tensors["token_hidden"].index_select(
                    0,
                    previous_row,
                )
                rows["temporal_pair_mask"] = temporal_pair_mask
                if mode == "temporal":
                    return rows
        mean_a = tensors["mean_a"].index_select(0, pair_index)
        mean_b = tensors["mean_b"].index_select(0, pair_index)
        choose_b = side.unsqueeze(-1)
        rows["chunk_mean"] = torch.where(choose_b, mean_b, mean_a)
        if mode in {"cross", "all"}:
            rows["partner_mean"] = torch.where(choose_b, mean_a, mean_b)
        return rows

    def _prepare_shard(
        self,
        item: Mapping[str, Any],
        *,
        epoch: int,
        skip_rows: int,
    ) -> _PreparedV2Shard:
        path = self.root / str(item["path"])
        declared_tokens = int(item.get("tokens", 0))
        if declared_tokens <= 0:
            raise ValueError(f"cache-v2 shard has invalid token count: {path}")
        tensors, cleanup_path = self._load(path)
        tensors = self._to_gpu_cache(path, tensors)
        pair_index, side, occurrence_id = self._row_metadata(tensors)
        previous_row, temporal_pair_mask = self._temporal_metadata(
            tensors,
            pair_index,
            side,
        )
        row_count = int(pair_index.numel())
        if row_count != declared_tokens:
            raise ValueError(
                f"cache-v2 shard {path} declares {declared_tokens} rows but "
                f"metadata contains {row_count}"
            )
        if self.mode in {"token", "temporal", "all"} and tensors[
            "token_hidden"
        ].shape[0] != row_count:
            raise ValueError("cache-v2 pair lengths do not cover token_hidden")
        seed_material = hashlib.blake2b(
            f"{self.seed}:{epoch}:{item['path']}".encode("utf-8"),
            digest_size=8,
        ).digest()
        generator = torch.Generator().manual_seed(
            int.from_bytes(seed_material, "little")
        )
        permutation = torch.randperm(row_count, generator=generator).to(
            tensors["pair_id"].device
        )
        if skip_rows:
            permutation = permutation[skip_rows:]
        return _PreparedV2Shard(
            path=path,
            item=item,
            mode=self.mode,
            tensors=tensors,
            permutation=permutation,
            pair_index=pair_index,
            side=side,
            occurrence_id=occurrence_id,
            previous_row=previous_row,
            temporal_pair_mask=temporal_pair_mask,
            cleanup_path=cleanup_path,
            release_gpu=(
                (lambda path=path: self._release_gpu_cache(path))
                if self.gpu_shards > 0
                else None
            ),
        )

    def _epoch_tasks(
        self,
        epoch: int,
        start_offset: int,
    ) -> list[tuple[Mapping[str, Any], int]]:
        order = list(self.shard_entries)
        random.Random(self.seed + epoch).shuffle(order)
        tasks: list[tuple[Mapping[str, Any], int]] = []
        remaining_offset = int(start_offset)
        for item in order:
            declared_tokens = int(item.get("tokens", 0))
            if declared_tokens <= 0:
                raise ValueError(
                    f"cache-v2 shard has invalid token count: {self.root / item['path']}"
                )
            if remaining_offset >= declared_tokens:
                remaining_offset -= declared_tokens
                continue
            tasks.append((item, remaining_offset))
            remaining_offset = 0
        return tasks

    def _iter_prepared_shards(
        self,
        epoch: int,
        start_offset: int,
    ) -> Iterator[_PreparedV2Shard]:
        tasks = self._epoch_tasks(epoch, start_offset)
        if not tasks:
            return
        depth = min(self.prefetch_shards, len(tasks))
        workers = min(self.prefetch_workers, depth)
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"sae-shard-r{self.rank:03d}",
        )
        futures: dict[int, concurrent.futures.Future[_PreparedV2Shard]] = {}

        def submit(index: int) -> None:
            item, skip_rows = tasks[index]
            futures[index] = executor.submit(
                self._prepare_shard,
                item,
                epoch=epoch,
                skip_rows=skip_rows,
            )

        for index in range(depth):
            submit(index)
        next_to_submit = depth
        try:
            for index in range(len(tasks)):
                shard = futures.pop(index).result()
                try:
                    yield shard
                finally:
                    shard.close()
                if next_to_submit < len(tasks):
                    submit(next_to_submit)
                    next_to_submit += 1
        finally:
            for future in futures.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _merge_parts(
        parts: Mapping[str, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {
            name: values[0] if len(values) == 1 else torch.cat(values, dim=0)
            for name, values in parts.items()
        }

    def _iter_epoch_cpu(
        self,
        epoch: int,
        *,
        start_offset: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        carry: dict[str, list[torch.Tensor]] = {}
        carry_count = 0
        yielded = 0
        for shard in self._iter_prepared_shards(epoch, start_offset):
            position = 0
            while position < shard.rows:
                take = min(self.batch_size - carry_count, shard.rows - position)
                piece = shard.take(position, take)
                for name, value in piece.items():
                    carry.setdefault(name, []).append(value)
                position += take
                carry_count += take
                if carry_count == self.batch_size:
                    batch = self._merge_parts(carry)
                    carry = {}
                    carry_count = 0
                    yielded += self.batch_size
                    yield batch
        if carry_count:
            batch = self._merge_parts(carry)
            yielded += carry_count
            yield batch
        consumed = int(start_offset) + yielded
        if consumed != self.occurrences:
            raise ValueError(
                f"cache-v2 rank {self.rank} consumed {consumed} occurrences; "
                f"manifest declares {self.occurrences}"
            )

    def iter_epoch(
        self,
        epoch: int = 0,
        *,
        start_offset: int = 0,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if start_offset < 0 or start_offset > self.occurrences:
            raise ValueError(
                f"start_offset={start_offset} outside cache epoch [0, {self.occurrences}]"
            )
        source = self._iter_epoch_cpu(epoch, start_offset=start_offset)
        if self.prefetch_batches <= 0:
            for batch in source:
                yield _move_batch(batch, self.device)
            return
        prefetcher = _AsyncBatchIterator(
            source,
            device=self.device,
            queue_depth=self.prefetch_batches,
            pin_memory=self.pin_memory,
        )
        try:
            yield from prefetcher
        finally:
            prefetcher.close()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        epoch = 0
        while True:
            yield from self.iter_epoch(epoch)
            epoch += 1


class OccurrenceLoaderAdapter:
    """Thin exact-occurrence adapter around the sole supported cache-v2 loader."""

    def __init__(
        self,
        cache_dir: str | Path,
        manifest: Mapping[str, Any],
        *,
        rank: int,
        world_size: int,
        batch_size: int,
        seed: int,
        device: str,
        mode: str,
        finite: bool = False,
        prefetch_shards: int = 4,
        prefetch_workers: int = 2,
        prefetch_batches: int = 4,
        materialize_shards: bool = True,
        pin_memory: bool = True,
        local_cache_dir: str | Path | None = None,
        gpu_shards: int = 0,
        direction_policy: str = "both",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.mode = mode
        self.finite = finite
        self.prefetch_shards = prefetch_shards
        self.prefetch_workers = prefetch_workers
        self.prefetch_batches = prefetch_batches
        self.materialize_shards = materialize_shards
        self.pin_memory = pin_memory
        self.local_cache_dir = local_cache_dir
        self.gpu_shards = max(0, int(gpu_shards))
        if direction_policy not in {"both", "a-to-b", "b-to-a"}:
            raise ValueError(f"unknown direction policy: {direction_policy}")
        self.direction_policy = direction_policy
        self.is_v2 = cache_is_v2(manifest)
        if not self.is_v2:
            raise ValueError("Only exact activation-cache v2 is supported")
        self.loader = PackedV2OccurrenceLoader(
            self.cache_dir,
            manifest,
            rank=rank,
            batch_size=batch_size,
            seed=seed,
            device=device,
            mode=mode,
            prefetch_shards=prefetch_shards,
            prefetch_workers=prefetch_workers,
            prefetch_batches=prefetch_batches,
            materialize_shards=materialize_shards,
            pin_memory=pin_memory,
            local_cache_dir=local_cache_dir,
            gpu_shards=gpu_shards,
        )

    def _iter_epoch(
        self,
        epoch: int,
        *,
        start_offset: int = 0,
    ) -> Iterator[Any]:
        yield from self.loader.iter_epoch(
            epoch=epoch,
            start_offset=start_offset,
        )

    def iter_epoch(self, epoch: int = 0, *, start_offset: int = 0) -> Iterator[Any]:
        per_rank_expected = getattr(self.loader, "occurrences", None)
        if per_rank_expected is None:
            ranks = self.manifest.get("ranks")
            if isinstance(ranks, list):
                rank_manifest = next(
                    (
                        item
                        for item in ranks
                        if int(item.get("rank", -1)) == self.rank
                    ),
                    None,
                )
                if rank_manifest is not None:
                    per_rank_expected = _manifest_value(
                        rank_manifest,
                        "token_occurrences",
                        "occurrences",
                        "samples",
                        "tokens",
                    )
        if per_rank_expected is not None:
            per_rank_expected = int(per_rank_expected)
        if per_rank_expected is None:
            expected = cache_occurrences(self.manifest)
            if expected is not None and expected % self.world_size == 0:
                per_rank_expected = expected // self.world_size
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        expected_remaining = (
            per_rank_expected - start_offset if per_rank_expected is not None else None
        )
        if expected_remaining is not None and expected_remaining < 0:
            raise ValueError(
                f"start_offset={start_offset} exceeds expected rank occurrences {per_rank_expected}"
            )
        seen = 0
        for batch in self._iter_epoch(epoch, start_offset=start_offset):
            batch = _move_batch(batch, self.device)
            if per_rank_expected is not None:
                remaining = expected_remaining - seen
                if remaining <= 0:
                    break
                rows = _batch_rows(batch)
                if rows > remaining:
                    batch = _slice_batch(batch, remaining)
                    rows = remaining
                seen += rows
            yield _direction_policy_batch(batch, self.direction_policy)
            if per_rank_expected is not None and seen >= expected_remaining:
                break
        if per_rank_expected is not None and seen != expected_remaining:
            raise ValueError(
                f"rank {self.rank} cache epoch yielded {seen} occurrences after offset; "
                f"expected {expected_remaining}"
            )

    def __iter__(self) -> Iterator[Any]:
        if not self.finite:
            for batch in iter(self.loader):
                yield _direction_policy_batch(batch, self.direction_policy)
            return
        yield from self.iter_epoch(0)


def _global_limit_to_local(global_samples: int, world_size: int) -> int | None:
    if global_samples <= 0:
        return None
    if global_samples % world_size:
        raise ValueError(
            f"global sample limit {global_samples} must be divisible by world size {world_size}"
        )
    return global_samples // world_size


def _iter_limited(loader: Iterable[Any], local_limit: int | None) -> Iterator[Any]:
    seen = 0
    for batch in loader:
        if local_limit is not None:
            remaining = local_limit - seen
            if remaining <= 0:
                return
            rows = _batch_rows(batch)
            if rows > remaining:
                batch = _slice_batch(batch, remaining)
                rows = remaining
            seen += rows
        yield batch
        if local_limit is not None and seen >= local_limit:
            return


def _batch_nbytes(batch: Any) -> int:
    if dataclasses.is_dataclass(batch):
        values = (getattr(batch, field.name) for field in dataclasses.fields(batch))
    elif isinstance(batch, Mapping):
        values = batch.values()
    else:
        values = ()
    return sum(
        value.numel() * value.element_size()
        for value in values
        if isinstance(value, torch.Tensor)
    )


def _estimated_validation_cache_bytes(
    *,
    mode: str,
    local_samples: int,
    hidden_size: int,
    manifest: Mapping[str, Any],
) -> int:
    dtype_name = str(
        _manifest_value(manifest, "activation_dtype", "dtype") or "bfloat16"
    )
    element_size = torch.empty((), dtype=dtype_from_name(dtype_name)).element_size()
    hidden_views = (
        4
        if mode == "all"
        else 2
        if mode in {"cross", "temporal", "joint_chunk"}
        else 1
    )
    # side(bool), occurrence_id(int64), pair_id(int64)
    metadata_bytes = local_samples * (1 + 8 + 8)
    return local_samples * hidden_size * element_size * hidden_views + metadata_bytes


def _device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device.type, dtype=dtype, enabled=enabled)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _gather_rank_objects(value: Any, world_size: int) -> list[Any]:
    return all_gather_objects(value, world_size=world_size)


def _broadcast_object(value: Any, rank: int) -> Any:
    return broadcast_object(value, rank=rank)


@torch.no_grad()
def estimate_scale_and_train_mean(
    mode: str,
    loader: Iterable[Any],
    hidden_size: int,
    device: torch.device,
    *,
    allow_legacy_v1: bool,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    norm_sum = torch.zeros((), dtype=torch.float64, device=device)
    input_sum = torch.zeros(hidden_size, dtype=torch.float64, device=device)
    target_sum = torch.zeros(hidden_size, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for batch in loader:
        inputs, targets, _ = select_occurrence_view(
            mode, batch, allow_legacy_v1=allow_legacy_v1
        )
        norm_sum += inputs.float().norm(dim=-1).double().sum()
        input_sum += inputs.double().sum(dim=0)
        target_sum += targets.double().sum(dim=0)
        count += targets.shape[0]
    if dist.is_initialized():
        dist.all_reduce(norm_sum)
        dist.all_reduce(input_sum)
        dist.all_reduce(target_sum)
        dist.all_reduce(count)
    if count.item() <= 0:
        raise ValueError("normalization/train-mean loader produced no samples")
    mean_norm = (norm_sum / count).item()
    scale = hidden_size**0.5 / max(mean_norm, 1e-12)
    train_input_mean = (input_sum / count).float()
    train_target_mean = (target_sum / count).float()
    return scale, train_input_mean, train_target_mean


@torch.no_grad()
def estimate_joint_chunk_stats(
    loader: Iterable[Any],
    hidden_size: int,
    device: torch.device,
    *,
    allow_legacy_v1: bool,
) -> tuple[float, torch.Tensor, torch.Tensor, float, float]:
    """Estimate input scale, target means, and fixed normalized-loss baselines."""

    norm_sum = torch.zeros((), dtype=torch.float64, device=device)
    input_sum = torch.zeros(hidden_size, dtype=torch.float64, device=device)
    cross_sum = torch.zeros(hidden_size, dtype=torch.float64, device=device)
    input_sq_sum = torch.zeros((), dtype=torch.float64, device=device)
    cross_sq_sum = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for batch in loader:
        inputs, targets, _ = select_joint_chunk_view(
            batch,
            allow_legacy_v1=allow_legacy_v1,
        )
        inputs_f = inputs.float()
        targets_f = targets.float()
        norm_sum += inputs_f.norm(dim=-1).double().sum()
        input_sum += inputs_f.double().sum(dim=0)
        cross_sum += targets_f.double().sum(dim=0)
        input_sq_sum += inputs_f.double().square().sum()
        cross_sq_sum += targets_f.double().square().sum()
        count += inputs.shape[0]
    if dist.is_initialized():
        for value in (
            norm_sum,
            input_sum,
            cross_sum,
            input_sq_sum,
            cross_sq_sum,
            count,
        ):
            dist.all_reduce(value)
    n = float(count.item())
    if n <= 0:
        raise ValueError("joint chunk normalization loader produced no samples")
    input_mean = (input_sum / n).float()
    cross_mean = (cross_sum / n).float()
    scale = hidden_size**0.5 / max(float((norm_sum / n).item()), 1e-12)
    # These are raw-space SSE baselines; scaling cancels in the normalized ratio.
    input_baseline = max(
        float((input_sq_sum - input_sum.square().sum() / n).item()) / n,
        1e-12,
    )
    cross_baseline = max(
        float((cross_sq_sum - cross_sum.square().sum() / n).item()) / n,
        1e-12,
    )
    return scale, input_mean, cross_mean, input_baseline, cross_baseline


@torch.no_grad()
def estimate_joint_scales(
    modes: Iterable[str],
    loader: Iterable[Any],
    hidden_size: int,
    device: torch.device,
    *,
    allow_legacy_v1: bool,
) -> dict[str, tuple[float, torch.Tensor, torch.Tensor]]:
    modes = tuple(modes)
    norm_sums = {
        mode: torch.zeros((), dtype=torch.float64, device=device)
        for mode in modes
    }
    target_sums = {
        mode: torch.zeros(hidden_size, dtype=torch.float64, device=device)
        for mode in modes
    }
    input_sums = {
        mode: torch.zeros(hidden_size, dtype=torch.float64, device=device)
        for mode in modes
    }
    counts = {
        mode: torch.zeros((), dtype=torch.float64, device=device)
        for mode in modes
    }
    for batch in loader:
        for mode in modes:
            inputs, targets, _ = select_occurrence_view(
                mode,
                batch,
                allow_legacy_v1=allow_legacy_v1,
            )
            norm_sums[mode] += inputs.float().norm(dim=-1).double().sum()
            input_sums[mode] += inputs.double().sum(dim=0)
            target_sums[mode] += targets.double().sum(dim=0)
            counts[mode] += targets.shape[0]
    if dist.is_initialized():
        for mode in modes:
            dist.all_reduce(norm_sums[mode])
            dist.all_reduce(input_sums[mode])
            dist.all_reduce(target_sums[mode])
            dist.all_reduce(counts[mode])
    result = {}
    for mode in modes:
        if counts[mode].item() <= 0:
            raise ValueError(f"joint normalization produced no rows for {mode}")
        mean_norm = (norm_sums[mode] / counts[mode]).item()
        result[mode] = (
            hidden_size**0.5 / max(mean_norm, 1e-12),
            (input_sums[mode] / counts[mode]).float(),
            (target_sums[mode] / counts[mode]).float(),
        )
    return result


@torch.no_grad()
def _evaluate_views(
    *,
    model: BatchTopKSAE,
    mode: str,
    loader: Iterable[Any],
    train_target_mean: torch.Tensor,
    device: torch.device,
    use_threshold: bool,
    allow_legacy_v1: bool,
    directional_train_target_means: Mapping[str, torch.Tensor] | None = None,
) -> tuple[
    dict[str, float | int | bool],
    dict[str, dict[str, float | int | bool]],
]:
    accumulators = {"all": ReconstructionMetricAccumulator(device)}
    if mode == "cross":
        accumulators["a_to_b"] = ReconstructionMetricAccumulator(device)
        accumulators["b_to_a"] = ReconstructionMetricAccumulator(device)
    for batch in loader:
        inputs, targets, is_b = select_occurrence_view(
            mode, batch, allow_legacy_v1=allow_legacy_v1
        )
        (
            reconstructed,
            features,
            _,
            active_counts,
            _active_per_feature,
        ) = model(
            inputs,
            batch_topk=not use_threshold,
            distributed=(
                not use_threshold and dist.is_initialized() and dist.get_world_size() > 1
            ),
            return_activity_counts=True,
        )
        accumulators["all"].update(
            reconstructed,
            targets,
            activation_scale=model.activation_scale,
            mean_predictor_raw=train_target_mean,
            active_counts=active_counts,
        )
        if mode == "cross":
            if is_b is None:
                raise ValueError("cross direction metrics require occurrence side metadata")
            for label, mask in (("a_to_b", ~is_b), ("b_to_a", is_b)):
                if not bool(mask.any()):
                    continue
                accumulators[label].update(
                    reconstructed[mask],
                    targets[mask],
                    activation_scale=model.activation_scale,
                    mean_predictor_raw=(
                        directional_train_target_means[label]
                        if directional_train_target_means is not None
                        else train_target_mean
                    ),
                    active_counts=active_counts[mask],
                )
    for accumulator in accumulators.values():
        accumulator.all_reduce_()
    macro = accumulators["all"].summary().as_dict()
    directional = {
        label: accumulator.summary().as_dict()
        for label, accumulator in accumulators.items()
        if label != "all"
    }
    return macro, directional


@torch.no_grad()
def validate_model(
    *,
    model: BatchTopKSAE,
    mode: str,
    loader_factory,
    train_target_mean: torch.Tensor,
    device: torch.device,
    use_threshold: bool,
    allow_legacy_v1: bool,
    directional_train_target_means: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, float | int | bool]:
    was_training = model.training
    model.eval()
    try:
        metrics: dict[str, float | int | bool] = {}
        inference_name = "threshold" if use_threshold else "batchtopk"
        macro, directional_metrics = _evaluate_views(
            model=model,
            mode=mode,
            loader=loader_factory(),
            train_target_mean=train_target_mean,
            device=device,
            use_threshold=use_threshold,
            allow_legacy_v1=allow_legacy_v1,
            directional_train_target_means=directional_train_target_means,
        )
        metrics.update(prefix_metrics(f"validation/{inference_name}", macro))
        # Stable short aliases for dashboards and checkpoint selection.
        for key in (
            "samples",
            "elements",
            "raw_mse",
            "mean_predictor_mse",
            "normalized_mse",
            "nmse",
            "fve",
            "scaled_objective",
            "effective_l0",
            "zero_code_fraction",
        ):
            metrics[f"validation/{key}"] = macro[key]
        if mode == "cross":
            for label, directional in directional_metrics.items():
                metrics.update(prefix_metrics(f"validation/{label}", directional))
        return metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def validate_joint_chunk_model(
    *,
    model: JointChunkSAE,
    loader_factory,
    train_mean: torch.Tensor,
    train_cross_mean: torch.Tensor,
    device: torch.device,
    use_threshold: bool,
    allow_legacy_v1: bool,
    directional_train_cross_means: Mapping[str, torch.Tensor] | None = None,
    deduplicate_chunk_inputs: bool = True,
) -> dict[str, float | int | bool]:
    """Validate both Joint outputs on the same paired occurrence rows."""

    was_training = model.training
    model.eval()
    mean_accumulator = ReconstructionMetricAccumulator(device)
    cross_accumulators = {
        "all": ReconstructionMetricAccumulator(device),
        "a_to_b": ReconstructionMetricAccumulator(device),
        "b_to_a": ReconstructionMetricAccumulator(device),
    }
    prefix_active = torch.zeros((), dtype=torch.float64, device=device)
    prefix_samples = torch.zeros((), dtype=torch.float64, device=device)
    try:
        for batch in loader_factory():
            inputs, cross_targets, is_b = select_joint_chunk_view(
                batch,
                allow_legacy_v1=allow_legacy_v1,
            )
            deduplication = (
                chunk_input_deduplication(batch, inputs.shape[0])
                if deduplicate_chunk_inputs
                else None
            )
            output = model(
                inputs,
                joint=True,
                batch_topk=not use_threshold,
                distributed=(
                    not use_threshold
                    and dist.is_initialized()
                    and dist.get_world_size() > 1
                ),
                return_activity_counts=True,
                unique_rows=None if deduplication is None else deduplication[0],
                dedup_inverse=None if deduplication is None else deduplication[1],
            )
            (
                reconstructed_mean,
                reconstructed_cross,
                _features,
                _threshold,
                active_counts,
                active_per_feature,
            ) = output[:6]
            mean_accumulator.update(
                reconstructed_mean,
                inputs,
                activation_scale=model.activation_scale,
                mean_predictor_raw=train_mean,
                active_counts=active_counts,
            )
            cross_accumulators["all"].update(
                reconstructed_cross,
                cross_targets,
                activation_scale=model.activation_scale,
                mean_predictor_raw=train_cross_mean,
                active_counts=active_counts,
            )
            if is_b is None:
                raise ValueError(
                    "Joint Cross direction metrics require occurrence side metadata"
                )
            for label, mask in (("a_to_b", ~is_b), ("b_to_a", is_b)):
                if not bool(mask.any()):
                    continue
                cross_accumulators[label].update(
                    reconstructed_cross[mask],
                    cross_targets[mask],
                    activation_scale=model.activation_scale,
                    mean_predictor_raw=(
                        directional_train_cross_means[label]
                        if directional_train_cross_means is not None
                        else train_cross_mean
                    ),
                    active_counts=active_counts[mask],
                )
            if model.nested:
                prefix_active.add_(
                    active_per_feature[: model.cross_prefix_size].sum().double()
                )
                prefix_samples.add_(inputs.shape[0])

        mean_accumulator.all_reduce_()
        for accumulator in cross_accumulators.values():
            accumulator.all_reduce_()
        if model.nested and dist.is_initialized():
            dist.all_reduce(prefix_active, op=dist.ReduceOp.SUM)
            dist.all_reduce(prefix_samples, op=dist.ReduceOp.SUM)

        mean_metrics = mean_accumulator.summary().as_dict()
        cross_metrics = cross_accumulators["all"].summary().as_dict()
        metrics: dict[str, float | int | bool] = {}
        metrics.update(prefix_metrics("validation/joint_mean", mean_metrics))
        metrics.update(prefix_metrics("validation/joint_cross", cross_metrics))
        for label in ("a_to_b", "b_to_a"):
            metrics.update(
                prefix_metrics(
                    f"validation/joint_cross/{label}",
                    cross_accumulators[label].summary().as_dict(),
                )
            )
        for key in (
            "scaled_objective",
            "raw_mse",
            "mean_predictor_mse",
            "normalized_mse",
            "nmse",
            "fve",
            "effective_l0",
            "zero_code_fraction",
        ):
            metrics[f"validation/joint_mean_{key}"] = mean_metrics[key]
            metrics[f"validation/joint_cross_{key}"] = cross_metrics[key]
        if model.nested:
            metrics["validation/joint_k_prefix"] = float(
                (prefix_active / prefix_samples.clamp_min(1.0)).item()
            )
        return metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def validate_models_joint(
    *,
    models: Mapping[str, BatchTopKSAE],
    loader: Iterable[Any],
    train_target_means: Mapping[str, torch.Tensor],
    directional_train_target_means: Mapping[
        str, Mapping[str, torch.Tensor] | None
    ],
    use_threshold: bool,
    allow_legacy_v1: bool,
    deduplicate_chunk_inputs: bool,
) -> dict[str, dict[str, float | int | bool]]:
    training_states = {mode: model.training for mode, model in models.items()}
    for model in models.values():
        model.eval()
    accumulators: dict[str, dict[str, ReconstructionMetricAccumulator]] = {}
    device = next(iter(models.values())).decoder_weight.device
    for mode in models:
        accumulators[mode] = {"all": ReconstructionMetricAccumulator(device)}
        if mode == "cross":
            accumulators[mode]["a_to_b"] = ReconstructionMetricAccumulator(device)
            accumulators[mode]["b_to_a"] = ReconstructionMetricAccumulator(device)
    try:
        for batch in loader:
            for mode, model in models.items():
                inputs, targets, is_b = select_occurrence_view(
                    mode,
                    batch,
                    allow_legacy_v1=allow_legacy_v1,
                )
                deduplication = (
                    chunk_input_deduplication(batch, inputs.shape[0])
                    if deduplicate_chunk_inputs and mode in {"mean", "cross"}
                    else None
                )
                reconstructed, _features, _, active_counts, _active_per_feature = (
                    model(
                        inputs,
                        batch_topk=not use_threshold,
                        distributed=(
                            not use_threshold
                            and dist.is_initialized()
                            and dist.get_world_size() > 1
                        ),
                        return_activity_counts=True,
                        unique_rows=(
                            None if deduplication is None else deduplication[0]
                        ),
                        dedup_inverse=(
                            None if deduplication is None else deduplication[1]
                        ),
                    )
                )
                accumulators[mode]["all"].update(
                    reconstructed,
                    targets,
                    activation_scale=model.activation_scale,
                    mean_predictor_raw=train_target_means[mode],
                    active_counts=active_counts,
                )
                if mode == "cross":
                    if is_b is None:
                        raise ValueError(
                            "cross direction metrics require occurrence side metadata"
                        )
                    directional = directional_train_target_means.get(mode)
                    for label, mask in (("a_to_b", ~is_b), ("b_to_a", is_b)):
                        if not bool(mask.any()):
                            continue
                        accumulators[mode][label].update(
                            reconstructed[mask],
                            targets[mask],
                            activation_scale=model.activation_scale,
                            mean_predictor_raw=(
                                directional[label]
                                if directional is not None
                                else train_target_means[mode]
                            ),
                            active_counts=active_counts[mask],
                        )
        result: dict[str, dict[str, float | int | bool]] = {}
        inference_name = "threshold" if use_threshold else "batchtopk"
        for mode, mode_accumulators in accumulators.items():
            for accumulator in mode_accumulators.values():
                accumulator.all_reduce_()
            macro = mode_accumulators["all"].summary().as_dict()
            metrics: dict[str, float | int | bool] = {}
            metrics.update(prefix_metrics(f"validation/{inference_name}", macro))
            for key in (
                "samples",
                "elements",
                "raw_mse",
                "mean_predictor_mse",
                "normalized_mse",
                "nmse",
                "fve",
                "scaled_objective",
                "effective_l0",
                "zero_code_fraction",
            ):
                metrics[f"validation/{key}"] = macro[key]
            if mode == "cross":
                for label in ("a_to_b", "b_to_a"):
                    metrics.update(
                        prefix_metrics(
                            f"validation/{label}",
                            mode_accumulators[label].summary().as_dict(),
                        )
                    )
            result[mode] = metrics
        return result
    finally:
        for mode, model in models.items():
            model.train(training_states[mode])


def _best_value(metrics: Mapping[str, Any], best_metric: str) -> float:
    value = float(metrics[f"validation/{best_metric}"])
    return value


def _is_better(value: float, best: float | None, metric: str) -> bool:
    if not math.isfinite(value):
        return False
    if best is None or not math.isfinite(best):
        return True
    return value < best if metric == "nmse" else value > best


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    append_jsonl(path, [dict(row)])


def _new_run_id(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _mode_config(
    args: argparse.Namespace,
    *,
    mode: str,
    hidden_size: int,
    scale: float,
    train_cache_digest: str,
    train_cache_format: str,
    validation_cache_digest: str | None,
    run_id: str,
) -> dict[str, Any]:
    return {
        "project": "chunk-saes",
        "sae_training_implementation_version": SAE_TRAINING_IMPLEMENTATION_VERSION,
        "sae_parameter_schema_version": SAE_PARAMETER_SCHEMA_VERSION,
        "input_center_policy": "fixed_training_input_mean",
        "output_bias_policy": "learned_initialized_from_training_target_mean",
        "training_format": train_cache_format,
        "mode": mode,
        "direction_policy": args.direction_policy,
        "target_granularity": args.target_granularity,
        "loss_mask": args.loss_mask,
        "mask_representation": args.mask_representation,
        "model": args.model,
        "layer": args.layer,
        "activation_dim": hidden_size,
        "dict_size": args.dict_size,
        "k": args.k,
        "global_batch_size": args.global_batch_size,
        "steps": args.steps,
        "exact_coverage_required": bool(args.require_exact_coverage),
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "threshold_beta": args.threshold_beta,
        "batch_topk_candidate_multiplier": args.batch_topk_candidate_multiplier,
        "batch_topk_bf16_histogram": args.batch_topk_bf16_histogram,
        "decoder_backend": args.decoder_backend,
        "fused_adam": args.fused_adam,
        "deduplicate_chunk_inputs": args.deduplicate_chunk_inputs,
        "joint_modes": args.joint_modes,
        "parallelism": args.parallelism,
        "max_activation_norm_multiple": args.max_activation_norm_multiple,
        "dead_feature_threshold": args.dead_feature_threshold,
        "auxk_alpha": args.auxk_alpha,
        "auxk_activation_age": args.auxk_activation_age,
        "auxk": args.auxk,
        "auxk_candidate_features": args.auxk_candidate_features,
        "attainable_reference_fve": args.fidelity_reference_fves.get(mode),
        # Compatibility key for older readers/checkpoints.
        "fidelity_reference_fve": args.fidelity_reference_fves.get(mode),
        "activation_scale": scale,
        "normalization": args.normalize_activations,
        "objective": (
            "temporal_matryoshka_self_reconstruction_plus_adjacent_infonce"
            if mode == "temporal"
            else "self_reconstruction_scaled_sse_per_sample"
            if mode != "cross"
            else "cross_reconstruction_scaled_sse_per_sample"
        ),
        "seed": args.seed,
        "run_id": run_id,
        "train_cache_digest": train_cache_digest,
        "validation_cache_digest": validation_cache_digest,
        "best_metric": args.best_metric,
        "final_validation_samples": args.final_validation_samples,
        "loader_prefetch_shards": args.loader_prefetch_shards,
        "loader_prefetch_workers": args.loader_prefetch_workers,
        "loader_prefetch_batches": args.loader_prefetch_batches,
        "loader_materialize_shards": args.loader_materialize_shards,
        "loader_pin_memory": args.loader_pin_memory,
        "loader_gpu_shards": args.loader_gpu_shards,
        "local_cache_dir": args.local_cache_dir,
        "periodic_validation_cached": args.cache_periodic_validation,
        "validation_cache_max_bytes": args.validation_cache_max_bytes,
        "best_checkpoint_policy": (
            "deferred_weights_only_at_latest_boundaries"
            if args.defer_best_checkpoint
            else "full_resumable_on_every_improvement"
        ),
        "latest_checkpoint_policy": (
            "cpu_snapshot_async_local_stage_verified_publish"
            if args.async_latest_checkpoint and args.checkpoint_staging_dir
            else "cpu_snapshot_async_atomic_publish"
            if args.async_latest_checkpoint
            else "synchronous_atomic_publish"
        ),
        "checkpoint_staging_dir": args.checkpoint_staging_dir,
        **(
            {
                "temporal_pairing": (
                    "immediate_previous_token_within_each_independently_forwarded_chunk"
                ),
                "temporal_boundary_policy": (
                    "chunk_first_tokens_reconstruct_but_are_excluded_from_contrastive_loss"
                ),
                "temporal_high_fraction": args.temporal_high_fraction,
                "temporal_high_level_features": temporal_high_feature_count(
                    args.dict_size,
                    args.temporal_high_fraction,
                ),
                "temporal_high_reconstruction_weight": (
                    args.temporal_high_reconstruction_weight
                ),
                "temporal_full_reconstruction_weight": (
                    args.temporal_full_reconstruction_weight
                ),
                "temporal_reconstruction_reduction": (
                    "mean_of_weighted_high_and_full_terms"
                ),
                "temporal_alpha": args.temporal_alpha,
                "temporal_temperature": args.temporal_temperature,
                "temporal_contrastive_block_size": (
                    args.temporal_contrastive_block_size
                ),
                "temporal_contrastive_scope": (
                    "exact_global_sparse_symmetric_infonce_across_all_ddp_ranks"
                ),
                "temporal_dead_feature_activity": (
                    "union_of_full_current_and_pair_current_previous_codes"
                ),
            }
            if mode == "temporal"
            else {}
        ),
    }


def _validate_resume_state(
    state: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    rank: int,
    world_size: int,
) -> None:
    saved_config = state.get("config") or {}
    for key in (
        "sae_training_implementation_version",
        "sae_parameter_schema_version",
        "input_center_policy",
        "output_bias_policy",
        "mode",
        "direction_policy",
        "target_granularity",
        "loss_mask",
        "mask_representation",
        "activation_dim",
        "dict_size",
        "k",
        "global_batch_size",
        "steps",
        "exact_coverage_required",
        "lr",
        "warmup_steps",
        "min_lr_ratio",
        "dead_feature_threshold",
        "auxk_alpha",
        "auxk_activation_age",
        "auxk",
        "auxk_candidate_features",
        "joint_chunk_alpha",
        "joint_chunk_layout",
        "joint_cross_prefix",
        "joint_chunk_auxk_policy",
        "temporal_high_fraction",
        "temporal_high_reconstruction_weight",
        "temporal_full_reconstruction_weight",
        "temporal_alpha",
        "temporal_temperature",
        "temporal_contrastive_block_size",
        "train_cache_digest",
        "validation_cache_digest",
        "seed",
    ):
        if saved_config.get(key) != config.get(key):
            raise ValueError(
                f"resume checkpoint mismatch for {key}: "
                f"{saved_config.get(key)!r} != {config.get(key)!r}"
            )
    saved_world_size = int(state.get("world_size", world_size))
    if saved_world_size != world_size:
        raise ValueError(
            f"resume world size {saved_world_size} != current world size {world_size}"
        )
    rng_by_rank = state.get("rng_by_rank")
    if not isinstance(rng_by_rank, list) or rank >= len(rng_by_rank):
        raise ValueError("resume checkpoint lacks per-rank RNG state")


def _save_checkpoint_collective(
    *,
    path: Path,
    model: BatchTopKSAE,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    samples_seen: int,
    best_metric_value: float | None,
    best_step: int | None,
    rank: int,
    world_size: int,
    accepted_samples_seen: int = 0,
    activation_norm_filtered_rows: int = 0,
    activation_norm_balance_dropped_rows: int = 0,
) -> None:
    rng_by_rank = _gather_rank_objects(_capture_rng_state(), world_size)
    feature_counts_by_rank = _gather_rank_objects(
        model.feature_counts.detach().cpu().clone(), world_size
    )
    dead_ages_by_rank = _gather_rank_objects(
        model.num_occurrences_since_fired.detach().cpu().clone(), world_size
    )
    if rank == 0:
        global_feature_counts = torch.stack(feature_counts_by_rank).sum(dim=0)
        state = {
            "format": "chunk-saes-training-checkpoint-v2",
            "step": step,
            "samples_seen": samples_seen,
            "accepted_training_rows": accepted_samples_seen,
            "activation_norm_filtered_rows": activation_norm_filtered_rows,
            "activation_norm_balance_dropped_rows": (
                activation_norm_balance_dropped_rows
            ),
            "world_size": world_size,
            "best_metric_value": best_metric_value,
            "best_step": best_step,
            "rng_by_rank": rng_by_rank,
            "feature_counts_by_rank": feature_counts_by_rank,
            "num_occurrences_since_fired_by_rank": dead_ages_by_rank,
            "config": config,
        }
        local_feature_counts = model.feature_counts.detach().clone()
        model.feature_counts.copy_(global_feature_counts.to(model.feature_counts.device))
        try:
            save_training_checkpoint(model, optimizer, path, state)
        finally:
            model.feature_counts.copy_(local_feature_counts)
    if dist.is_initialized():
        dist.barrier()


def _snapshot_checkpoint_collective(
    *,
    model: BatchTopKSAE,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    samples_seen: int,
    best_metric_value: float | None,
    best_step: int | None,
    rank: int,
    world_size: int,
    accepted_samples_seen: int,
    activation_norm_filtered_rows: int,
    activation_norm_balance_dropped_rows: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]] | None:
    rng_by_rank = _gather_rank_objects(_capture_rng_state(), world_size)
    feature_counts_by_rank = _gather_rank_objects(
        model.feature_counts.detach().cpu().clone(), world_size
    )
    dead_ages_by_rank = _gather_rank_objects(
        model.num_occurrences_since_fired.detach().cpu().clone(), world_size
    )
    if rank != 0:
        return None
    global_feature_counts = torch.stack(feature_counts_by_rank).sum(dim=0)
    state = {
        "format": "chunk-saes-training-checkpoint-v2",
        "step": step,
        "samples_seen": samples_seen,
        "accepted_training_rows": accepted_samples_seen,
        "activation_norm_filtered_rows": activation_norm_filtered_rows,
        "activation_norm_balance_dropped_rows": activation_norm_balance_dropped_rows,
        "world_size": world_size,
        "best_metric_value": best_metric_value,
        "best_step": best_step,
        "rng_by_rank": rng_by_rank,
        "feature_counts_by_rank": feature_counts_by_rank,
        "num_occurrences_since_fired_by_rank": dead_ages_by_rank,
        "config": config,
    }
    local_feature_counts = model.feature_counts.detach().clone()
    model.feature_counts.copy_(global_feature_counts.to(model.feature_counts.device))
    try:
        return snapshot_training_checkpoint(model, optimizer, state)
    finally:
        model.feature_counts.copy_(local_feature_counts)


def _snapshot_best_model_collective(
    *,
    model: BatchTopKSAE,
    rank: int,
    device_snapshot: bool = True,
) -> dict[str, torch.Tensor] | None:
    global_feature_counts = model.feature_counts.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(global_feature_counts, op=dist.ReduceOp.SUM)
    if rank != 0:
        return None
    if device_snapshot:
        tensors = {
            key: value.detach().contiguous().clone()
            for key, value in model.state_dict().items()
        }
        tensors["feature_counts"] = global_feature_counts.contiguous()
    else:
        tensors = model.checkpoint_tensors()
        tensors["feature_counts"] = global_feature_counts.cpu().contiguous()
    return tensors


def _save_best_checkpoint_collective(
    *,
    path: Path,
    tensors: dict[str, torch.Tensor] | None,
    config: dict[str, Any],
    step: int,
    samples_seen: int,
    best_metric_value: float | None,
    best_step: int | None,
    rank: int,
    world_size: int,
    source_weights: Path | None = None,
) -> None:
    if rank == 0:
        if tensors is None and source_weights is None:
            raise ValueError("rank zero lacks validation-best model tensors")
        save_inference_checkpoint(
            tensors,
            path,
            {
                "step": step,
                "samples_seen": samples_seen,
                "world_size": world_size,
                "best_metric_value": best_metric_value,
                "best_step": best_step,
                "config": config,
            },
            source_weights=source_weights,
        )
    if dist.is_initialized():
        dist.barrier()


def _load_resume_state(
    *,
    path: Path,
    model: BatchTopKSAE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    # Every rank reads the shared checkpoint directly. Broadcasting a multi-GB
    # Adam state through broadcast_object_list requires pickle-sized buffers and
    # is substantially less reliable than parallel safetensors/torch.load I/O.
    state = load_training_checkpoint(path, model, optimizer, map_location=device)
    if dist.is_initialized():
        dist.barrier()
    if not isinstance(state, dict):
        raise ValueError(f"invalid training checkpoint state in {path}")
    counts_by_rank = state.get("feature_counts_by_rank")
    if not isinstance(counts_by_rank, list) or rank >= len(counts_by_rank):
        raise ValueError("training checkpoint lacks per-rank feature counts")
    model.feature_counts.copy_(
        torch.as_tensor(
            counts_by_rank[rank],
            dtype=model.feature_counts.dtype,
            device=model.feature_counts.device,
        )
    )
    return state


def train_mode(
    args: argparse.Namespace,
    mode: str,
    rank: int,
    world_size: int,
    local_rank: int,
    hidden_size: int,
    train_manifest: Mapping[str, Any],
    train_cache_digest: str,
    validation_manifest: Mapping[str, Any] | None,
    validation_cache_digest: str | None,
    run_id: str,
) -> None:
    if args.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    exact_occurrences = None
    if args.require_exact_coverage:
        exact_occurrences = validate_exact_training_coverage(
            train_manifest,
            world_size=world_size,
            global_batch_size=args.global_batch_size,
            steps=args.steps,
        )
    per_rank_batch = args.global_batch_size // world_size
    validation_global_batch = args.validation_batch_size or args.global_batch_size
    if validation_global_batch % world_size:
        raise ValueError("validation batch size must be divisible by world size")
    per_rank_validation_batch = validation_global_batch // world_size
    device = _device(local_rank)
    seed_everything(args.seed + rank)
    allow_legacy_v1 = not cache_is_v2(train_manifest)

    mode_dir = Path(args.output_dir) / mode
    checkpoint_root = mode_dir / "checkpoints"
    latest_path = checkpoint_root / "latest"
    best_path = checkpoint_root / "best"
    latest_staging_path = (
        Path(args.checkpoint_staging_dir)
        / Path(args.output_dir).name
        / mode
        / "latest"
        if args.checkpoint_staging_dir
        else None
    )
    resume_path = Path(args.resume_from) / mode if args.resume_from else latest_path
    resume_mode = bool(args.resume_from) or (args.resume and resume_path.is_dir())
    if rank == 0:
        if mode_dir.exists() and not resume_mode:
            if args.overwrite_output:
                shutil.rmtree(mode_dir)
            elif any(mode_dir.iterdir()):
                raise FileExistsError(
                    f"output directory is not empty: {mode_dir}; use --resume or --overwrite-output"
                )
        mode_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    init_loader = OccurrenceLoaderAdapter(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        world_size=world_size,
        batch_size=per_rank_batch,
        seed=args.seed + 101,
        device=str(device),
        mode=mode,
        direction_policy=args.direction_policy,
        finite=True,
        prefetch_shards=1,
        prefetch_workers=1,
        prefetch_batches=1,
        materialize_shards=args.loader_materialize_shards,
        pin_memory=args.loader_pin_memory,
        local_cache_dir=args.local_cache_dir,
        gpu_shards=args.loader_gpu_shards,
    )
    (
        estimated_scale,
        sampled_train_input_mean,
        sampled_train_target_mean,
    ) = estimate_scale_and_train_mean(
        mode,
        _iter_limited(
            init_loader.iter_epoch(0),
            _global_limit_to_local(args.normalization_samples, world_size),
        ),
        hidden_size,
        device,
        allow_legacy_v1=allow_legacy_v1,
    )
    scale = estimated_scale if args.normalize_activations else 1.0
    target_mode = _cross_target_mode(args.direction_policy) if mode == "cross" else mode
    exact_train_target_mean = cache_target_mean(
        train_manifest,
        target_mode,
        hidden_size=hidden_size,
        device=device,
    )
    train_target_mean = (
        exact_train_target_mean
        if exact_train_target_mean is not None
        else sampled_train_target_mean.to(device)
    )
    exact_train_input_mean = (
        cache_input_mean(
            train_manifest,
            mode,
            hidden_size=hidden_size,
            device=device,
        )
        if not (mode == "cross" and args.direction_policy != "both")
        else None
    )
    train_input_mean = (
        exact_train_input_mean
        if exact_train_input_mean is not None
        else sampled_train_input_mean.to(device)
    )
    directional_train_target_means = None
    if mode == "cross":
        a_to_b_mean = cache_target_mean(
            train_manifest,
            "cross_a_to_b",
            hidden_size=hidden_size,
            device=device,
        )
        b_to_a_mean = cache_target_mean(
            train_manifest,
            "cross_b_to_a",
            hidden_size=hidden_size,
            device=device,
        )
        if a_to_b_mean is not None and b_to_a_mean is not None:
            directional_train_target_means = {
                "a_to_b": a_to_b_mean,
                "b_to_a": b_to_a_mean,
            }

    model = BatchTopKSAE(
        hidden_size,
        args.dict_size,
        args.k,
        batch_topk_candidate_multiplier=args.batch_topk_candidate_multiplier,
        batch_topk_bf16_histogram=args.batch_topk_bf16_histogram,
        decoder_backend=args.decoder_backend,
    ).to(device)
    model.activation_scale.fill_(scale)
    model.pre_bias.copy_(train_input_mean * scale)
    model.decoder_bias.data.copy_(train_target_mean * scale)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        fused=bool(args.fused_adam and device.type == "cuda"),
    )
    config = _mode_config(
        args,
        mode=mode,
        hidden_size=hidden_size,
        scale=scale,
        train_cache_digest=train_cache_digest,
        train_cache_format=str(train_manifest.get("format", "unknown")),
        validation_cache_digest=validation_cache_digest,
        run_id=run_id,
    )
    config["train_target_mean_source"] = (
        "exact_full_training_cache"
        if exact_train_target_mean is not None
        else "normalization_sample_estimate"
    )
    config["train_input_mean_source"] = (
        "exact_full_training_cache_mean_inputs"
        if exact_train_input_mean is not None and mode == "cross"
        else "exact_full_training_cache"
        if exact_train_input_mean is not None
        else "normalization_sample_estimate"
    )
    config["train_input_mean_mode"] = (
        "mean"
        if mode == "cross"
        else "token"
        if mode == "temporal"
        else mode
    )
    config["train_target_mean_samples"] = (
        int(train_manifest["target_sufficient_statistics"]["count"])
        if exact_train_target_mean is not None
        else (
            int(cache_occurrences(train_manifest) or 0)
            if args.normalization_samples <= 0
            else min(
                int(args.normalization_samples),
                int(cache_occurrences(train_manifest) or args.normalization_samples),
            )
        )
    )

    start_step = 0
    samples_seen = 0
    accepted_samples_seen = 0
    total_filtered_rows = 0
    total_balance_dropped_rows = 0
    best_metric_value: float | None = None
    best_step: int | None = None
    best_snapshot: dict[str, torch.Tensor] | None = None
    best_snapshot_step: int | None = None
    best_snapshot_samples = 0
    best_snapshot_dirty = False
    resume_local_feature_counts: torch.Tensor | None = None
    resume_local_dead_ages: torch.Tensor | None = None
    if resume_mode:
        state = _load_resume_state(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            rank=rank,
        )
        _validate_resume_state(
            state,
            config=config,
            rank=rank,
            world_size=world_size,
        )
        start_step = int(state["step"])
        samples_seen = int(state.get("samples_seen", start_step * args.global_batch_size))
        accepted_samples_seen = int(
            state.get("accepted_training_rows", samples_seen)
        )
        total_filtered_rows = int(
            state.get("activation_norm_filtered_rows", 0)
        )
        total_balance_dropped_rows = int(
            state.get("activation_norm_balance_dropped_rows", 0)
        )
        best_metric_value = state.get("best_metric_value")
        best_step = state.get("best_step")
        _restore_rng_state(state["rng_by_rank"][rank])
        feature_counts_by_rank = state.get("feature_counts_by_rank")
        if (
            isinstance(feature_counts_by_rank, list)
            and rank < len(feature_counts_by_rank)
        ):
            resume_local_feature_counts = torch.as_tensor(
                feature_counts_by_rank[rank],
                dtype=model.feature_counts.dtype,
                device=model.feature_counts.device,
            )
        else:
            raise ValueError("resume checkpoint lacks per-rank feature counts")
        counts_since_fired = state.get("num_occurrences_since_fired_by_rank")
        if isinstance(counts_since_fired, list) and rank < len(counts_since_fired):
            resume_local_dead_ages = torch.as_tensor(
                counts_since_fired[rank],
                dtype=model.num_occurrences_since_fired.dtype,
                device=model.num_occurrences_since_fired.device,
            )
        run_id = str((state.get("config") or {}).get("run_id", run_id))
        config["run_id"] = run_id
        if start_step > args.steps:
            raise ValueError(
                f"checkpoint step {start_step} exceeds requested steps {args.steps}"
            )

    ddp = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=256,
            # AuxK starts contributing only after features cross the configured
            # activation age, so its autograd path is intentionally dynamic.
            static_graph=args.auxk_alpha <= 0,
        )
        if world_size > 1
        else model
    )
    if resume_local_feature_counts is not None:
        # DDP constructor state synchronization may broadcast buffers from rank
        # zero; restore this rank's local historical counts afterward.
        model.feature_counts.copy_(resume_local_feature_counts)
    if resume_local_dead_ages is not None:
        model.num_occurrences_since_fired.copy_(resume_local_dead_ages)
    autocast_dtype = dtype_from_name(args.autocast_dtype)
    train_adapter = OccurrenceLoaderAdapter(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        world_size=world_size,
        batch_size=per_rank_batch,
        seed=args.seed + 211,
        device=str(device),
        mode=mode,
        direction_policy=args.direction_policy,
        finite=bool(args.require_exact_coverage),
        prefetch_shards=args.loader_prefetch_shards,
        prefetch_workers=args.loader_prefetch_workers,
        prefetch_batches=args.loader_prefetch_batches,
        materialize_shards=args.loader_materialize_shards,
        pin_memory=args.loader_pin_memory,
        local_cache_dir=args.local_cache_dir,
        gpu_shards=args.loader_gpu_shards,
    )
    if args.require_exact_coverage:
        train_loader = iter(
            train_adapter.iter_epoch(
                0,
                start_offset=start_step * per_rank_batch,
            )
        )
    else:
        train_loader = iter(train_adapter)

    def validation_loader_factory(global_samples: int) -> Iterator[Any]:
        if validation_manifest is None or args.validation_cache_dir is None:
            return iter(())
        bounded_subset = global_samples > 0
        adapter = OccurrenceLoaderAdapter(
            args.validation_cache_dir,
            validation_manifest,
            rank=rank,
            world_size=world_size,
            batch_size=per_rank_validation_batch,
            seed=args.seed + 307,
            device=str(device),
            mode=mode,
            finite=True,
            prefetch_shards=(
                1 if bounded_subset else args.loader_prefetch_shards
            ),
            prefetch_workers=(
                1 if bounded_subset else args.loader_prefetch_workers
            ),
            prefetch_batches=args.loader_prefetch_batches,
            materialize_shards=args.loader_materialize_shards,
            pin_memory=args.loader_pin_memory,
            local_cache_dir=args.local_cache_dir,
            gpu_shards=args.loader_gpu_shards,
        )
        local_limit = _global_limit_to_local(global_samples, world_size)
        return _iter_limited(adapter.iter_epoch(0), local_limit)

    periodic_validation_batches: tuple[Any, ...] | None = None
    if (
        validation_manifest is not None
        and args.cache_periodic_validation
        and args.validation_samples > 0
    ):
        local_validation_samples = _global_limit_to_local(
            args.validation_samples, world_size
        )
        assert local_validation_samples is not None
        estimated_bytes = _estimated_validation_cache_bytes(
            mode=mode,
            local_samples=local_validation_samples,
            hidden_size=hidden_size,
            manifest=validation_manifest,
        )
        if estimated_bytes <= args.validation_cache_max_bytes:
            periodic_validation_batches = tuple(
                validation_loader_factory(args.validation_samples)
            )
            actual_bytes = sum(
                _batch_nbytes(batch) for batch in periodic_validation_batches
            )
            if rank == 0:
                log(
                    f"cached periodic validation for {mode}: "
                    f"{args.validation_samples} global rows, "
                    f"{actual_bytes / (1024**2):.1f} MiB/rank on {device}",
                    rank=rank,
                )
        elif rank == 0:
            log(
                f"periodic validation cache disabled for {mode}: estimated "
                f"{estimated_bytes / (1024**2):.1f} MiB/rank exceeds "
                f"{args.validation_cache_max_bytes / (1024**2):.1f} MiB",
                rank=rank,
            )

    def periodic_validation_loader_factory() -> Iterator[Any]:
        if periodic_validation_batches is not None:
            return iter(periodic_validation_batches)
        return validation_loader_factory(args.validation_samples)

    log_path = mode_dir / "metrics.jsonl"
    tensorboard_root = (
        Path(args.tensorboard_dir) / run_id / mode if args.tensorboard_dir else None
    )
    writer = TensorBoardLogger(
        tensorboard_root,
        rank=rank,
        flush_secs=args.tensorboard_flush_secs,
        max_queue=args.tensorboard_max_queue,
        purge_step=start_step + 1 if start_step else None,
        scalar_allowlist=CORE_TENSORBOARD_SCALARS,
        mode=mode,
        fidelity_reference_fve=args.fidelity_reference_fves.get(mode),
        gradient_clip=args.gradient_clip,
    )

    started = time.time()
    running = ReconstructionMetricAccumulator(device)
    running_steps = 0
    running_grad_norm = torch.zeros((), dtype=torch.float64, device=device)
    running_auxiliary_loss = 0.0
    running_auxiliary_steps = 0
    running_temporal_loss = 0.0
    running_temporal_accuracy = 0.0
    running_temporal_pairs = 0
    running_batch_wait = 0.0
    running_filtered_rows = 0
    running_balance_dropped_rows = 0
    running_window_started = time.perf_counter()
    threshold_initialized = bool(model.threshold.detach().cpu().item() >= 0)
    last_validation_metrics: dict[str, float | int | bool] | None = None
    final_full_validation_metrics: dict[str, float | int | bool] | None = None
    checkpoint_executor = (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"latest-checkpoint-{mode}",
        )
        if rank == 0 and args.async_latest_checkpoint
        else None
    )
    pending_latest: concurrent.futures.Future[None] | None = None
    try:
        if rank == 0:
            atomic_json_dump(config, mode_dir / "run_config.json")
            log(
                f"training {mode}: hidden={hidden_size} width={args.dict_size} k={args.k} "
                f"steps={start_step}->{args.steps} global_batch={args.global_batch_size} "
                f"scale={scale:.6g} cache={train_cache_digest[:12]}",
                rank=rank,
            )

        if validation_manifest is not None and start_step == 0:
            validation_metrics = validate_model(
                model=model,
                mode=mode,
                loader_factory=periodic_validation_loader_factory,
                train_target_mean=train_target_mean,
                device=device,
                use_threshold=args.validation_use_threshold,
                allow_legacy_v1=not cache_is_v2(validation_manifest),
                directional_train_target_means=directional_train_target_means,
            )
            last_validation_metrics = dict(validation_metrics)
            initial_value = _best_value(validation_metrics, args.best_metric)
            initial_is_best = _is_better(
                initial_value, best_metric_value, args.best_metric
            )
            if initial_is_best:
                best_metric_value = initial_value
                best_step = 0
            if rank == 0:
                row = {
                    "run_id": run_id,
                    "mode": mode,
                    "split": "validation",
                    "step": 0,
                    "is_best": initial_is_best,
                    **validation_metrics,
                }
                _write_jsonl(log_path, row)
                writer.add_scalars(validation_metrics, 0)
            if initial_is_best:
                if args.defer_best_checkpoint:
                    best_snapshot = _snapshot_best_model_collective(
                        model=model,
                        rank=rank,
                    )
                    best_snapshot_step = 0
                    best_snapshot_samples = 0
                    best_snapshot_dirty = True
                else:
                    _save_checkpoint_collective(
                        path=best_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=0,
                        samples_seen=0,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                    )

        for step_index in range(start_step, args.steps):
            batch_wait_started = time.perf_counter()
            batch = next(train_loader)
            running_batch_wait += time.perf_counter() - batch_wait_started
            inputs, targets, _ = select_occurrence_view(
                mode, batch, allow_legacy_v1=allow_legacy_v1
            )
            valid_mask, input_norm_threshold, target_norm_threshold = (
                activation_norm_ok_mask(
                    inputs,
                    targets,
                    args.max_activation_norm_multiple,
                )
            )
            local_filtered = inputs.shape[0] - int(valid_mask.sum().item())
            valid_mask, accepted_per_rank, balance_dropped = equalize_valid_rows(valid_mask)
            if accepted_per_rank <= 0:
                raise RuntimeError(
                    "activation norm filtering rejected every row on at least one rank"
                )
            accepted_global = accepted_per_rank * world_size
            filter_counts = torch.tensor(
                [local_filtered, balance_dropped],
                dtype=torch.int64,
                device=device,
            )
            if dist.is_initialized():
                dist.all_reduce(filter_counts, op=dist.ReduceOp.SUM)
            filtered_global = int(filter_counts[0].item())
            balance_dropped_global = int(filter_counts[1].item())
            optimizer.zero_grad(set_to_none=True)
            deduplication = (
                chunk_input_deduplication(batch, inputs.shape[0])
                if args.deduplicate_chunk_inputs and mode in {"mean", "cross"}
                else None
            )
            auxiliary_ids = model.auxiliary_feature_ids(
                args.auxk_activation_age,
                args.auxk_candidate_features,
            )
            temporal_previous = None
            temporal_pair_mask = None
            temporal_high_features = 0
            temporal_global_pairs = None
            if mode == "temporal":
                temporal_previous, temporal_pair_mask = select_temporal_pair_view(
                    batch
                )
                temporal_pair_mask = temporal_pair_mask & valid_mask
                temporal_pair_count = temporal_pair_mask.sum(dtype=torch.int64)
                if dist.is_initialized():
                    dist.all_reduce(temporal_pair_count, op=dist.ReduceOp.SUM)
                temporal_global_pairs = int(temporal_pair_count.item())
                if temporal_global_pairs <= 1:
                    raise RuntimeError(
                        "Temporal SAE batch contains fewer than two valid "
                        "within-chunk adjacent pairs"
                    )
                temporal_high_features = temporal_high_feature_count(
                    args.dict_size,
                    args.temporal_high_fraction,
                )
            with _autocast_context(device, autocast_dtype):
                forward_output = ddp(
                    inputs,
                    batch_topk=True,
                    distributed=world_size > 1,
                    return_activity_counts=True,
                    sample_mask=valid_mask,
                    global_sample_count=accepted_global,
                    unique_rows=(
                        None if deduplication is None else deduplication[0]
                    ),
                    dedup_inverse=(
                        None if deduplication is None else deduplication[1]
                    ),
                    auxiliary_feature_ids=auxiliary_ids,
                    auxiliary_k=args.auxk,
                    return_auxiliary=True,
                    temporal_previous=temporal_previous,
                    temporal_high_features=temporal_high_features,
                    temporal_sample_mask=temporal_pair_mask,
                    temporal_global_sample_count=temporal_global_pairs,
                )
                (
                    reconstructed,
                    features,
                    batch_threshold,
                    active_counts,
                    active_per_feature,
                    auxiliary_reconstruction,
                ) = forward_output[:6]
                scaled_targets = targets * model.activation_scale.to(targets.dtype)
                residual = reconstructed.float() - scaled_targets.float()
                primary_loss = residual[valid_mask].square().sum(dim=-1).mean()
                auxiliary_target = (scaled_targets - reconstructed.detach()).float()
                auxiliary_residual = (
                    auxiliary_reconstruction.float() - auxiliary_target
                )
                if auxiliary_ids.numel() and args.auxk_alpha > 0:
                    auxiliary_loss = (
                        auxiliary_residual[valid_mask].square().sum(dim=-1).mean()
                    )
                else:
                    auxiliary_loss = residual.new_zeros(())
                if mode == "temporal":
                    temporal_current_features = forward_output[6]
                    previous_features = forward_output[7]
                    high_reconstruction = forward_output[8]
                    high_residual = (
                        high_reconstruction.float()
                        - scaled_targets.float()
                    )
                    high_reconstruction_loss = (
                        high_residual[temporal_pair_mask]
                        .square()
                        .sum(dim=-1)
                        .mean()
                    )
                    assert temporal_pair_mask is not None
                    temporal_result = (
                        distributed_symmetric_temporal_contrastive_loss(
                        temporal_current_features[:, :temporal_high_features],
                        previous_features[:, :temporal_high_features],
                        temporal_pair_mask,
                        temperature=args.temporal_temperature,
                        )
                        if world_size > 1
                        else symmetric_temporal_contrastive_loss(
                            temporal_current_features[
                                :, :temporal_high_features
                            ],
                            previous_features[:, :temporal_high_features],
                            temporal_pair_mask,
                            temperature=args.temporal_temperature,
                            block_size=args.temporal_contrastive_block_size,
                        )
                    )
                    reconstruction_loss = (
                        args.temporal_high_reconstruction_weight
                        * high_reconstruction_loss
                        + args.temporal_full_reconstruction_weight
                        * primary_loss
                    ) / 2.0
                    loss = (
                        reconstruction_loss
                        + args.auxk_alpha * auxiliary_loss
                        + args.temporal_alpha * temporal_result.loss
                    )
                else:
                    high_reconstruction_loss = primary_loss.new_zeros(())
                    temporal_result = None
                    loss = primary_loss + args.auxk_alpha * auxiliary_loss
            loss.backward()
            model.remove_parallel_decoder_gradient_()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            multiplier = learning_rate_multiplier(
                step_index,
                args.steps,
                args.warmup_steps,
                args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            optimizer.step()
            model.normalize_decoder_()

            with torch.no_grad():
                if not threshold_initialized:
                    model.threshold.copy_(batch_threshold.float())
                    threshold_initialized = True
                else:
                    model.threshold.mul_(args.threshold_beta).add_(
                        batch_threshold.float(), alpha=1 - args.threshold_beta
                    )
                model.feature_counts.add_(active_per_feature)
                dead_activity_per_feature = active_per_feature
                if mode == "temporal":
                    dead_activity_per_feature = (
                        active_per_feature
                        + (temporal_current_features.detach() != 0).sum(
                            dim=0,
                            dtype=torch.int64,
                        )
                        + (previous_features.detach() != 0).sum(
                            dim=0,
                            dtype=torch.int64,
                        )
                    )
                model.update_dead_feature_stats_(
                    dead_activity_per_feature,
                    global_samples=accepted_global,
                    distributed=world_size > 1,
                )
                scaled_sse = primary_loss.detach() * accepted_per_rank
                raw_sse = scaled_sse / (scale * scale)
                baseline_sse = (
                    targets[valid_mask].float() - train_target_mean.float()
                ).square().sum()
            running.update_sufficient_statistics(
                    scaled_sse=scaled_sse,
                    raw_sse=raw_sse,
                    mean_predictor_sse=baseline_sse,
                    samples=accepted_per_rank,
                    elements=accepted_per_rank * targets.shape[1],
                active_counts=active_counts[valid_mask],
            )
            running_auxiliary_loss += float(auxiliary_loss.detach().item())
            running_auxiliary_steps += 1
            if temporal_result is not None:
                running_temporal_loss += float(
                    temporal_result.loss.detach().item()
                ) * temporal_result.pairs
                running_temporal_accuracy += float(
                    temporal_result.accuracy.detach().item()
                ) * temporal_result.pairs
                running_temporal_pairs += temporal_result.pairs
            running_steps += 1
            running_grad_norm.add_(grad_norm.detach().double())
            completed_step = step_index + 1
            samples_seen += args.global_batch_size
            accepted_samples_seen += accepted_global
            running_filtered_rows += filtered_global
            running_balance_dropped_rows += balance_dropped_global
            total_filtered_rows += filtered_global
            total_balance_dropped_rows += balance_dropped_global

            should_log = (
                completed_step == 1
                or completed_step % args.log_every == 0
                or completed_step == args.steps
            )
            if should_log:
                running_window_elapsed = (
                    time.perf_counter() - running_window_started
                )
                running.all_reduce_()
                summary = running.summary().as_dict()
                train_metrics: dict[str, float | int | bool] = {
                    "train/scaled_objective": summary["scaled_objective"],
                    "train/raw_mse": summary["raw_mse"],
                    "train/mean_predictor_mse": summary["mean_predictor_mse"],
                    "train/normalized_mse": summary["normalized_mse"],
                    "train/fve": summary["fve"],
                    "train/effective_l0": summary["effective_l0"],
                    "train/zero_code_fraction": summary["zero_code_fraction"],
                    "train/auxiliary_loss": running_auxiliary_loss
                    / max(1, running_auxiliary_steps),
                    "optimizer/lr": optimizer.param_groups[0]["lr"],
                    "optimizer/grad_norm": float(running_grad_norm.item())
                    / max(1, running_steps),
                    "optimizer/updates": completed_step,
                    "progress/samples_seen": samples_seen,
                    "progress/token_occurrences_seen": samples_seen,
                    "progress/accepted_training_rows": accepted_samples_seen,
                    "progress/unique_occurrence_coverage": (
                        samples_seen / exact_occurrences
                        if exact_occurrences is not None
                        else samples_seen
                        / max(1, int(cache_occurrences(train_manifest) or samples_seen))
                    ),
                    "system/elapsed_seconds": time.time() - started,
                    "system/samples_per_second": samples_seen
                    / max(time.time() - started, 1e-9),
                    "system/batch_wait_seconds": running_batch_wait,
                    "system/batch_wait_fraction": running_batch_wait
                    / max(running_window_elapsed, 1e-9),
                    "sparsity/threshold": model.threshold.item(),
                        "sparsity/dead_features": model.dead_feature_count(
                            args.dead_feature_threshold
                        ),
                        "sparsity/auxk_candidates": int(auxiliary_ids.numel()),
                    "sparsity/dead_fraction": model.dead_feature_count(
                        args.dead_feature_threshold
                    )
                    / max(1, args.dict_size),
                    "filter/activation_norm_multiple": args.max_activation_norm_multiple,
                    "filter/rejected_rows": running_filtered_rows,
                    "filter/balance_dropped_rows": running_balance_dropped_rows,
                    "filter/input_norm_threshold": float(input_norm_threshold.item()),
                    "filter/target_norm_threshold": float(target_norm_threshold.item()),
                }
                if mode == "temporal":
                    train_metrics.update(
                        {
                            "train/temporal_contrastive_loss": (
                                running_temporal_loss
                                / max(1, running_temporal_pairs)
                            ),
                            "train/temporal_contrastive_accuracy": (
                                running_temporal_accuracy
                                / max(1, running_temporal_pairs)
                            ),
                            "train/temporal_pairs": running_temporal_pairs,
                        }
                    )
                if rank == 0:
                    row = {
                        "run_id": run_id,
                        "mode": mode,
                        "split": "train",
                        "step": completed_step,
                        **train_metrics,
                    }
                    _write_jsonl(log_path, row)
                    writer.add_scalars(train_metrics, completed_step)
                    log(json.dumps(row, sort_keys=True), rank=rank)
                running.reset()
                running_steps = 0
                running_auxiliary_loss = 0.0
                running_auxiliary_steps = 0
                running_temporal_loss = 0.0
                running_temporal_accuracy = 0.0
                running_temporal_pairs = 0
                running_grad_norm.zero_()
                running_batch_wait = 0.0
                running_filtered_rows = 0
                running_balance_dropped_rows = 0
                running_window_started = time.perf_counter()

            validation_metrics = None
            should_validate = (
                validation_manifest is not None
                and (
                    (
                        args.validate_every > 0
                        and completed_step % args.validate_every == 0
                    )
                    or completed_step == args.steps
                )
            )
            if should_validate:
                validation_metrics = validate_model(
                    model=model,
                    mode=mode,
                    loader_factory=periodic_validation_loader_factory,
                    train_target_mean=train_target_mean,
                    device=device,
                    use_threshold=args.validation_use_threshold,
                    allow_legacy_v1=not cache_is_v2(validation_manifest),
                    directional_train_target_means=directional_train_target_means,
                )
                last_validation_metrics = dict(validation_metrics)
                current_value = _best_value(validation_metrics, args.best_metric)
                improved = _is_better(
                    current_value, best_metric_value, args.best_metric
                )
                if improved:
                    best_metric_value = current_value
                    best_step = completed_step
                if rank == 0:
                    row = {
                        "run_id": run_id,
                        "mode": mode,
                        "split": "validation",
                        "step": completed_step,
                        "is_best": improved,
                        **validation_metrics,
                    }
                    _write_jsonl(log_path, row)
                    writer.add_scalars(validation_metrics, completed_step)
                    writer.add_scalars(
                        {
                            "validation/is_best": improved,
                            "validation/best_step": best_step or 0,
                        },
                        completed_step,
                    )
                    writer.flush()

                if improved:
                    if args.defer_best_checkpoint:
                        best_snapshot = _snapshot_best_model_collective(
                            model=model,
                            rank=rank,
                        )
                        best_snapshot_step = completed_step
                        best_snapshot_samples = samples_seen
                        best_snapshot_dirty = True
                    else:
                        _save_checkpoint_collective(
                            path=best_path,
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            step=completed_step,
                            samples_seen=samples_seen,
                            best_metric_value=best_metric_value,
                            best_step=best_step,
                            rank=rank,
                            world_size=world_size,
                            accepted_samples_seen=accepted_samples_seen,
                            activation_norm_filtered_rows=total_filtered_rows,
                            activation_norm_balance_dropped_rows=(
                                total_balance_dropped_rows
                            ),
                        )

            should_save = (
                (args.save_every > 0 and completed_step % args.save_every == 0)
                or completed_step == args.steps
            )
            if should_save:
                link_best_from_latest = bool(
                    args.defer_best_checkpoint
                    and best_snapshot_dirty
                    and best_snapshot_step == completed_step
                )
                if (
                    args.defer_best_checkpoint
                    and best_snapshot_dirty
                    and not link_best_from_latest
                ):
                    assert best_snapshot_step is not None
                    _save_best_checkpoint_collective(
                        path=best_path,
                        tensors=best_snapshot,
                        config=config,
                        step=best_snapshot_step,
                        samples_seen=best_snapshot_samples,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                    )
                    best_snapshot = None
                    best_snapshot_dirty = False
                if pending_latest is not None:
                    pending_latest.result()
                    pending_latest = None
                if args.async_latest_checkpoint:
                    snapshot = _snapshot_checkpoint_collective(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=completed_step,
                        samples_seen=samples_seen,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                        accepted_samples_seen=accepted_samples_seen,
                        activation_norm_filtered_rows=total_filtered_rows,
                        activation_norm_balance_dropped_rows=(
                            total_balance_dropped_rows
                        ),
                    )
                    if rank == 0:
                        assert checkpoint_executor is not None and snapshot is not None
                        tensors, optimizer_state, checkpoint_state = snapshot
                        pending_latest = checkpoint_executor.submit(
                            save_training_checkpoint_snapshot,
                            tensors,
                            optimizer_state,
                            latest_path,
                            checkpoint_state,
                            latest_staging_path,
                        )
                else:
                    _save_checkpoint_collective(
                        path=latest_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=completed_step,
                        samples_seen=samples_seen,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                        accepted_samples_seen=accepted_samples_seen,
                        activation_norm_filtered_rows=total_filtered_rows,
                        activation_norm_balance_dropped_rows=(
                            total_balance_dropped_rows
                        ),
                    )
                if link_best_from_latest:
                    if pending_latest is not None:
                        pending_latest.result()
                        pending_latest = None
                    assert best_snapshot_step is not None
                    _save_best_checkpoint_collective(
                        path=best_path,
                        tensors=None,
                        config=config,
                        step=best_snapshot_step,
                        samples_seen=best_snapshot_samples,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                        source_weights=latest_path / "sae.safetensors",
                    )
                    best_snapshot = None
                    best_snapshot_dirty = False
                if rank == 0:
                    writer.flush()

        if args.defer_best_checkpoint and best_snapshot_dirty:
            assert best_snapshot_step is not None
            _save_best_checkpoint_collective(
                path=best_path,
                tensors=best_snapshot,
                config=config,
                step=best_snapshot_step,
                samples_seen=best_snapshot_samples,
                best_metric_value=best_metric_value,
                best_step=best_step,
                rank=rank,
                world_size=world_size,
            )
            best_snapshot = None
            best_snapshot_dirty = False

        if pending_latest is not None:
            pending_latest.result()
            pending_latest = None
        if dist.is_initialized():
            dist.barrier()

        if validation_manifest is not None:
            final_full_validation_metrics = validate_model(
                model=model,
                mode=mode,
                loader_factory=lambda: validation_loader_factory(
                    args.final_validation_samples
                ),
                train_target_mean=train_target_mean,
                device=device,
                use_threshold=args.validation_use_threshold,
                allow_legacy_v1=not cache_is_v2(validation_manifest),
                directional_train_target_means=directional_train_target_means,
            )
            if rank == 0:
                full_row = {
                    "run_id": run_id,
                    "mode": mode,
                    "split": "validation_full",
                    "step": args.steps,
                    "requested_samples": args.final_validation_samples,
                    **final_full_validation_metrics,
                }
                _write_jsonl(log_path, full_row)
                writer.add_scalars(
                    {
                        key.replace("validation/", "validation_full/", 1): value
                        for key, value in final_full_validation_metrics.items()
                    },
                    args.steps,
                )
                writer.flush()

        if args.require_exact_coverage:
            try:
                extra_batch = next(train_loader)
            except StopIteration:
                extra_batch = None
            if extra_batch is not None:
                raise RuntimeError("exact-coverage loader contains rows beyond configured steps")
            if samples_seen != exact_occurrences:
                raise RuntimeError(
                    f"exact coverage consumed {samples_seen} rows, expected {exact_occurrences}"
                )

        if dist.is_initialized():
            dist.all_reduce(model.feature_counts, op=dist.ReduceOp.SUM)
        if rank == 0:
            final_config = {
                **config,
                "steps_completed": args.steps,
                "samples_seen": samples_seen,
                "accepted_training_rows": accepted_samples_seen,
                "activation_norm_filtered_rows": total_filtered_rows,
                "activation_norm_balance_dropped_rows": total_balance_dropped_rows,
                "unique_occurrences_seen": (
                    exact_occurrences if args.require_exact_coverage else None
                ),
                "coverage_fraction": (
                    samples_seen / exact_occurrences
                    if exact_occurrences is not None
                    else None
                ),
                "best_metric_value": best_metric_value,
                "best_step": best_step,
                "tensorboard_dir": str(tensorboard_root) if tensorboard_root else None,
                "last_periodic_validation_metrics": last_validation_metrics,
                "final_full_validation_metrics": final_full_validation_metrics,
                "dead_features": model.dead_feature_count(
                    args.dead_feature_threshold
                ),
            }
            save_sae(model, mode_dir, final_config)
            alive = int((model.feature_counts > 0).sum())
            atomic_json_dump(
                {
                    "complete": True,
                    "run_id": run_id,
                    "mode": mode,
                    "steps": args.steps,
                    "samples_seen": samples_seen,
                    "accepted_training_rows": accepted_samples_seen,
                    "activation_norm_filtered_rows": total_filtered_rows,
                    "activation_norm_balance_dropped_rows": total_balance_dropped_rows,
                    "unique_occurrences_seen": (
                        exact_occurrences if args.require_exact_coverage else None
                    ),
                    "coverage_fraction": (
                        samples_seen / exact_occurrences
                        if exact_occurrences is not None
                        else None
                    ),
                    "exact_coverage": bool(args.require_exact_coverage),
                    "best_metric": args.best_metric,
                    "best_metric_value": best_metric_value,
                    "best_step": best_step,
                    "alive_features": alive,
                    "dead_features": model.dead_feature_count(
                        args.dead_feature_threshold
                    ),
                    "dead_feature_threshold": args.dead_feature_threshold,
                    "train_cache_digest": train_cache_digest,
                    "validation_cache_digest": validation_cache_digest,
                    "last_periodic_validation_metrics": last_validation_metrics,
                    "final_full_validation_metrics": final_full_validation_metrics,
                },
                mode_dir / "complete.json",
            )
            log(f"saved {mode} SAE; alive_features={alive}/{args.dict_size}", rank=rank)
        if dist.is_initialized():
            dist.barrier()
    finally:
        if pending_latest is not None:
            pending_latest.result()
        if checkpoint_executor is not None:
            checkpoint_executor.shutdown(wait=True, cancel_futures=False)
        writer.close()
        del ddp, model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_joint_chunk_mode(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    local_rank: int,
    hidden_size: int,
    train_manifest: Mapping[str, Any],
    train_cache_digest: str,
    validation_manifest: Mapping[str, Any] | None,
    validation_cache_digest: str | None,
    run_id: str,
) -> None:
    """Train Joint Chunk SAE with one shared code and two task readouts.

    Validation evaluates both outputs on the same paired occurrence rows. The
    occurrence loader and checkpoint protocol remain aligned with the
    established Mean/Cross runs.
    """

    if args.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    if args.steps <= 0 or args.log_every <= 0:
        raise ValueError("steps and log-every must be positive")
    if args.joint_chunk_alpha <= 0:
        raise ValueError("joint-chunk-alpha must be positive")
    cross_prefix = int(args.joint_cross_prefix)
    if cross_prefix < 0 or cross_prefix >= args.dict_size:
        raise ValueError(
            "joint-cross-prefix must be zero (two-head layout) or a non-empty "
            f"strict prefix below dict-size={args.dict_size}; got {cross_prefix}"
        )
    nested = cross_prefix > 0
    exact_occurrences = None
    if args.require_exact_coverage:
        exact_occurrences = validate_exact_training_coverage(
            train_manifest,
            world_size=world_size,
            global_batch_size=args.global_batch_size,
            steps=args.steps,
        )

    per_rank_batch = args.global_batch_size // world_size
    validation_global_batch = args.validation_batch_size or args.global_batch_size
    if validation_global_batch % world_size:
        raise ValueError("validation batch size must be divisible by world size")
    per_rank_validation_batch = validation_global_batch // world_size
    device = _device(local_rank)
    seed_everything(args.seed + rank)
    allow_legacy_v1 = not cache_is_v2(train_manifest)
    mode = "joint_chunk"
    mode_dir = Path(args.output_dir) / mode
    checkpoint_root = mode_dir / "checkpoints"
    latest_path = checkpoint_root / "latest"
    best_path = checkpoint_root / "best"
    latest_staging_path = (
        Path(args.checkpoint_staging_dir)
        / Path(args.output_dir).name
        / mode
        / "latest"
        if args.checkpoint_staging_dir
        else None
    )
    resume_path = Path(args.resume_from) / mode if args.resume_from else latest_path
    resume_mode = bool(args.resume_from) or (args.resume and resume_path.is_dir())
    if rank == 0:
        if mode_dir.exists() and not resume_mode:
            if args.overwrite_output:
                shutil.rmtree(mode_dir)
            elif any(mode_dir.iterdir()):
                raise FileExistsError(
                    f"output directory is not empty: {mode_dir}; "
                    "use --resume or --overwrite-output"
                )
        mode_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    init_loader = OccurrenceLoaderAdapter(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        world_size=world_size,
        batch_size=per_rank_batch,
        seed=args.seed + 101,
        device=str(device),
        mode=mode,
        direction_policy=args.direction_policy,
        finite=True,
        prefetch_shards=1,
        prefetch_workers=1,
        prefetch_batches=1,
        materialize_shards=args.loader_materialize_shards,
        pin_memory=args.loader_pin_memory,
        local_cache_dir=args.local_cache_dir,
        gpu_shards=args.loader_gpu_shards,
    )
    (
        estimated_scale,
        sampled_input_mean,
        sampled_cross_mean,
        sampled_mean_baseline,
        sampled_cross_baseline,
    ) = estimate_joint_chunk_stats(
        _iter_limited(
            init_loader.iter_epoch(0),
            _global_limit_to_local(args.normalization_samples, world_size),
        ),
        hidden_size,
        device,
        allow_legacy_v1=allow_legacy_v1,
    )
    scale = estimated_scale if args.normalize_activations else 1.0
    exact_input_mean = cache_input_mean(
        train_manifest,
        "mean",
        hidden_size=hidden_size,
        device=device,
    )
    exact_cross_mean = cache_target_mean(
        train_manifest,
        "cross",
        hidden_size=hidden_size,
        device=device,
    )
    input_mean = exact_input_mean if exact_input_mean is not None else sampled_input_mean.to(device)
    cross_mean = exact_cross_mean if exact_cross_mean is not None else sampled_cross_mean.to(device)
    directional_cross_means = None
    a_to_b_mean = cache_target_mean(
        train_manifest,
        "cross_a_to_b",
        hidden_size=hidden_size,
        device=device,
    )
    b_to_a_mean = cache_target_mean(
        train_manifest,
        "cross_b_to_a",
        hidden_size=hidden_size,
        device=device,
    )
    if a_to_b_mean is not None and b_to_a_mean is not None:
        directional_cross_means = {
            "a_to_b": a_to_b_mean,
            "b_to_a": b_to_a_mean,
        }
    model = JointChunkSAE(
        hidden_size,
        args.dict_size,
        args.k,
        batch_topk_candidate_multiplier=args.batch_topk_candidate_multiplier,
        batch_topk_bf16_histogram=args.batch_topk_bf16_histogram,
        decoder_backend=args.decoder_backend,
        cross_prefix=cross_prefix if nested else None,
    ).to(device)
    model.activation_scale.fill_(scale)
    model.pre_bias.copy_(input_mean * scale)
    model.decoder_bias.data.copy_(input_mean * scale)
    model.decoder_cross_bias.data.copy_(cross_mean * scale)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        fused=bool(args.fused_adam and device.type == "cuda"),
    )
    config = _mode_config(
        args,
        mode=mode,
        hidden_size=hidden_size,
        scale=scale,
        train_cache_digest=train_cache_digest,
        train_cache_format=str(train_manifest.get("format", "unknown")),
        validation_cache_digest=validation_cache_digest,
        run_id=run_id,
    )
    config.update(
        {
            "objective": (
                "joint_mean_full_cross_prefix_shared_decoder"
                if nested
                else "joint_mean_cross_two_decoder_heads"
            ),
            "decoder_heads": 1 if nested else 2,
            "decoder_head_names": (
                ["mean_full", "cross_prefix"]
                if nested
                else ["mean", "cross"]
            ),
            "joint_chunk_layout": (
                "nested_prefix" if nested else "independent_decoder_heads"
            ),
            "joint_chunk_implementation_version": (
                "nested-prefix-v1" if nested else "independent-heads-v1"
            ),
            "joint_cross_prefix": cross_prefix if nested else None,
            "feature_id_semantics": "shared_latent_coordinate",
            "decoder_vector_key": (
                "feature_id" if nested else "(head, feature_id)"
            ),
            "feature_role_policy": (
                "shared_if_feature_id_below_joint_cross_prefix_else_self_only"
                if nested
                else "both_heads"
            ),
            "joint_chunk_alpha": args.joint_chunk_alpha,
            "best_metric": "joint_cross_nmse",
            "joint_chunk_normalization": "train_only_constant_mean_predictor_sse",
            "joint_chunk_mean_baseline": sampled_mean_baseline,
            "joint_chunk_cross_baseline": sampled_cross_baseline,
            "joint_chunk_baseline_samples": min(
                int(args.normalization_samples),
                int(cache_occurrences(train_manifest) or args.normalization_samples),
            ),
            "joint_chunk_validation": validation_manifest is not None,
            "joint_chunk_auxk_policy": (
                "mean_only_full_dictionary"
                if nested
                else "mean_and_cross_separate_residuals"
            ),
            "train_input_mean": input_mean.detach().cpu().tolist(),
            "train_cross_target_mean": cross_mean.detach().cpu().tolist(),
        }
    )

    start_step = 0
    samples_seen = 0
    accepted_seen = 0
    total_filtered = 0
    total_balanced = 0
    best_metric_value: float | None = None
    best_step: int | None = None
    resume_local_feature_counts: torch.Tensor | None = None
    resume_local_dead_ages: torch.Tensor | None = None
    if resume_mode:
        state = _load_resume_state(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            rank=rank,
        )
        _validate_resume_state(
            state,
            config=config,
            rank=rank,
            world_size=world_size,
        )
        start_step = int(state["step"])
        samples_seen = int(
            state.get("samples_seen", start_step * args.global_batch_size)
        )
        accepted_seen = int(
            state.get("accepted_training_rows", samples_seen)
        )
        total_filtered = int(
            state.get("activation_norm_filtered_rows", 0)
        )
        total_balanced = int(
            state.get("activation_norm_balance_dropped_rows", 0)
        )
        best_metric_value = state.get("best_metric_value")
        best_step = state.get("best_step")
        _restore_rng_state(state["rng_by_rank"][rank])
        feature_counts_by_rank = state.get("feature_counts_by_rank")
        if (
            isinstance(feature_counts_by_rank, list)
            and rank < len(feature_counts_by_rank)
        ):
            resume_local_feature_counts = torch.as_tensor(
                feature_counts_by_rank[rank],
                dtype=model.feature_counts.dtype,
                device=model.feature_counts.device,
            )
        else:
            raise ValueError("resume checkpoint lacks per-rank feature counts")
        counts_since_fired = state.get("num_occurrences_since_fired_by_rank")
        if isinstance(counts_since_fired, list) and rank < len(counts_since_fired):
            resume_local_dead_ages = torch.as_tensor(
                counts_since_fired[rank],
                dtype=model.num_occurrences_since_fired.dtype,
                device=model.num_occurrences_since_fired.device,
            )
        else:
            raise ValueError("resume checkpoint lacks per-rank dead-feature ages")
        run_id = str((state.get("config") or {}).get("run_id", run_id))
        config["run_id"] = run_id
        if start_step > args.steps:
            raise ValueError(
                f"checkpoint step {start_step} exceeds requested steps {args.steps}"
            )

    ddp = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=256,
            static_graph=args.auxk_alpha <= 0,
        )
        if world_size > 1
        else model
    )
    if resume_local_feature_counts is not None:
        model.feature_counts.copy_(resume_local_feature_counts)
    if resume_local_dead_ages is not None:
        model.num_occurrences_since_fired.copy_(resume_local_dead_ages)
    if rank == 0:
        atomic_json_dump(config, mode_dir / "run_config.json")

    train_adapter = OccurrenceLoaderAdapter(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        world_size=world_size,
        batch_size=per_rank_batch,
        seed=args.seed + 211,
        device=str(device),
        mode=mode,
        direction_policy=args.direction_policy,
        finite=bool(args.require_exact_coverage),
        prefetch_shards=args.loader_prefetch_shards,
        prefetch_workers=args.loader_prefetch_workers,
        prefetch_batches=args.loader_prefetch_batches,
        materialize_shards=args.loader_materialize_shards,
        pin_memory=args.loader_pin_memory,
        local_cache_dir=args.local_cache_dir,
        gpu_shards=args.loader_gpu_shards,
    )
    train_loader = iter(
        train_adapter.iter_epoch(
            0,
            start_offset=start_step * per_rank_batch,
        )
        if args.require_exact_coverage
        else train_adapter
    )

    def validation_loader_factory(global_samples: int) -> Iterator[Any]:
        if validation_manifest is None or args.validation_cache_dir is None:
            return iter(())
        bounded_subset = global_samples > 0
        adapter = OccurrenceLoaderAdapter(
            args.validation_cache_dir,
            validation_manifest,
            rank=rank,
            world_size=world_size,
            batch_size=per_rank_validation_batch,
            seed=args.seed + 307,
            device=str(device),
            mode=mode,
            direction_policy=args.direction_policy,
            finite=True,
            prefetch_shards=(
                1 if bounded_subset else args.loader_prefetch_shards
            ),
            prefetch_workers=(
                1 if bounded_subset else args.loader_prefetch_workers
            ),
            prefetch_batches=args.loader_prefetch_batches,
            materialize_shards=args.loader_materialize_shards,
            pin_memory=args.loader_pin_memory,
            local_cache_dir=args.local_cache_dir,
            gpu_shards=args.loader_gpu_shards,
        )
        local_limit = _global_limit_to_local(global_samples, world_size)
        return _iter_limited(adapter.iter_epoch(0), local_limit)

    periodic_validation_batches: tuple[Any, ...] | None = None
    if (
        validation_manifest is not None
        and args.cache_periodic_validation
        and args.validation_samples > 0
    ):
        local_validation_samples = _global_limit_to_local(
            args.validation_samples,
            world_size,
        )
        assert local_validation_samples is not None
        estimated_bytes = _estimated_validation_cache_bytes(
            mode=mode,
            local_samples=local_validation_samples,
            hidden_size=hidden_size,
            manifest=validation_manifest,
        )
        if estimated_bytes <= args.validation_cache_max_bytes:
            periodic_validation_batches = tuple(
                validation_loader_factory(args.validation_samples)
            )
            if rank == 0:
                actual_bytes = sum(
                    _batch_nbytes(batch) for batch in periodic_validation_batches
                )
                log(
                    f"cached periodic validation for {mode}: "
                    f"{args.validation_samples} global rows, "
                    f"{actual_bytes / (1024**2):.1f} MiB/rank on {device}",
                    rank=rank,
                )
        elif rank == 0:
            log(
                f"periodic validation cache disabled for {mode}: estimated "
                f"{estimated_bytes / (1024**2):.1f} MiB/rank exceeds "
                f"{args.validation_cache_max_bytes / (1024**2):.1f} MiB",
                rank=rank,
            )

    def periodic_validation_loader_factory() -> Iterator[Any]:
        if periodic_validation_batches is not None:
            return iter(periodic_validation_batches)
        return validation_loader_factory(args.validation_samples)

    tensorboard_root = (
        Path(args.tensorboard_dir) / run_id / mode
        if args.tensorboard_dir
        else None
    )
    writer = TensorBoardLogger(
        tensorboard_root,
        rank=rank,
        flush_secs=args.tensorboard_flush_secs,
        max_queue=args.tensorboard_max_queue,
        purge_step=start_step + 1 if start_step else None,
        scalar_allowlist=JOINT_TENSORBOARD_SCALARS,
        mode=mode,
        fidelity_reference_fve=None,
        gradient_clip=args.gradient_clip,
    )
    log_path = mode_dir / "metrics.jsonl"
    started = time.time()
    running_mean = ReconstructionMetricAccumulator(device)
    running_cross = ReconstructionMetricAccumulator(device)
    running_steps = 0
    running_grad_norm = torch.zeros((), dtype=torch.float64, device=device)
    running_auxiliary_loss = 0.0
    running_auxiliary_steps = 0
    running_prefix_active = torch.zeros((), dtype=torch.float64, device=device)
    running_prefix_samples = torch.zeros((), dtype=torch.float64, device=device)
    running_batch_wait = 0.0
    running_filtered = 0
    running_balanced = 0
    running_window_started = time.perf_counter()
    threshold_initialized = bool(model.threshold.detach().cpu().item() >= 0)
    best_snapshot: dict[str, torch.Tensor] | None = None
    best_snapshot_step: int | None = None
    best_snapshot_samples = 0
    best_snapshot_dirty = False
    last_validation_metrics: dict[str, float | int | bool] | None = None
    final_full_validation_metrics: dict[str, float | int | bool] | None = None
    pending_latest: concurrent.futures.Future[None] | None = None
    checkpoint_executor = (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="latest-checkpoint-joint-chunk",
        )
        if rank == 0 and args.async_latest_checkpoint
        else None
    )
    autocast_dtype = dtype_from_name(args.autocast_dtype)
    baseline_mean_scaled = max(sampled_mean_baseline * scale * scale, 1e-12)
    baseline_cross_scaled = max(sampled_cross_baseline * scale * scale, 1e-12)
    try:
        if rank == 0:
            log(
                f"training {mode}: alpha={args.joint_chunk_alpha} hidden={hidden_size} "
                f"width={args.dict_size} steps={args.steps} "
                f"global_batch={args.global_batch_size} "
                f"layout={'nested_prefix' if nested else 'independent_decoder_heads'} "
                f"cross_prefix={cross_prefix if nested else 'all'}",
                rank=rank,
            )
        if validation_manifest is not None and start_step == 0:
            validation_metrics = validate_joint_chunk_model(
                model=model,
                loader_factory=periodic_validation_loader_factory,
                train_mean=input_mean,
                train_cross_mean=cross_mean,
                device=device,
                use_threshold=args.validation_use_threshold,
                allow_legacy_v1=not cache_is_v2(validation_manifest),
                directional_train_cross_means=directional_cross_means,
                deduplicate_chunk_inputs=args.deduplicate_chunk_inputs,
            )
            last_validation_metrics = dict(validation_metrics)
            best_metric_value = float(
                validation_metrics["validation/joint_cross_nmse"]
            )
            best_step = 0
            if args.defer_best_checkpoint:
                best_snapshot = _snapshot_best_model_collective(
                    model=model,
                    rank=rank,
                    device_snapshot=False,
                )
                best_snapshot_step = 0
                best_snapshot_dirty = True
            else:
                _save_checkpoint_collective(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=0,
                    samples_seen=0,
                    best_metric_value=best_metric_value,
                    best_step=best_step,
                    rank=rank,
                    world_size=world_size,
                )
            if rank == 0:
                row = {
                    "run_id": run_id,
                    "mode": mode,
                    "split": "validation",
                    "step": 0,
                    "is_best": True,
                    **validation_metrics,
                }
                _write_jsonl(log_path, row)
                writer.add_scalars(validation_metrics, 0)
                writer.flush()
        for step_index in range(start_step, args.steps):
            wait_started = time.perf_counter()
            batch = next(train_loader)
            running_batch_wait += time.perf_counter() - wait_started
            inputs, cross_targets, _ = select_joint_chunk_view(
                batch,
                allow_legacy_v1=allow_legacy_v1,
            )
            valid_mask, input_threshold, target_threshold = activation_norm_ok_mask(
                inputs,
                cross_targets,
                args.max_activation_norm_multiple,
            )
            local_filtered = inputs.shape[0] - int(valid_mask.sum().item())
            valid_mask, accepted_per_rank, balance_dropped = equalize_valid_rows(valid_mask)
            if accepted_per_rank <= 0:
                raise RuntimeError("activation norm filtering rejected every row on at least one rank")
            accepted_global = accepted_per_rank * world_size
            counts = torch.tensor(
                [local_filtered, balance_dropped], dtype=torch.int64, device=device
            )
            if dist.is_initialized():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            filtered_global, balanced_global = (int(counts[0]), int(counts[1]))
            optimizer.zero_grad(set_to_none=True)
            deduplication = (
                chunk_input_deduplication(batch, inputs.shape[0])
                if args.deduplicate_chunk_inputs
                else None
            )
            auxiliary_ids = model.auxiliary_feature_ids(
                args.auxk_activation_age,
                args.auxk_candidate_features,
            )
            with _autocast_context(device, autocast_dtype):
                output = ddp(
                    inputs,
                    joint=True,
                    batch_topk=True,
                    distributed=world_size > 1,
                    return_activity_counts=True,
                    sample_mask=valid_mask,
                    global_sample_count=accepted_global,
                    unique_rows=None if deduplication is None else deduplication[0],
                    dedup_inverse=None if deduplication is None else deduplication[1],
                    auxiliary_feature_ids=auxiliary_ids,
                    auxiliary_k=args.auxk,
                    return_auxiliary=True,
                )
                scaled_inputs = inputs * model.activation_scale.to(inputs.dtype)
                scaled_cross_targets = cross_targets * model.activation_scale.to(
                    cross_targets.dtype
                )
                if nested:
                    (
                        reconstructed_mean,
                        reconstructed_cross,
                        features,
                        batch_threshold,
                        active_counts,
                        active_per_feature,
                        auxiliary_mean,
                    ) = output
                    auxiliary_cross = None
                else:
                    (
                        reconstructed_mean,
                        reconstructed_cross,
                        features,
                        batch_threshold,
                        active_counts,
                        active_per_feature,
                        auxiliary_mean,
                        auxiliary_cross,
                    ) = output
                mean_residual = reconstructed_mean.float() - scaled_inputs.float()
                cross_residual = reconstructed_cross.float() - scaled_cross_targets.float()
                mean_error = mean_residual[valid_mask].square().sum(dim=-1).mean()
                cross_error = cross_residual[valid_mask].square().sum(dim=-1).mean()
                mean_normalized = mean_error / baseline_mean_scaled
                cross_normalized = cross_error / baseline_cross_scaled
                task_loss = (
                    mean_normalized + args.joint_chunk_alpha * cross_normalized
                ) / (1.0 + args.joint_chunk_alpha)
                aux_mean_residual = (
                    auxiliary_mean.float() - (scaled_inputs - reconstructed_mean.detach()).float()
                )
                aux_mean_error = aux_mean_residual[valid_mask].square().sum(dim=-1).mean()
                if nested:
                    auxiliary_loss = (
                        aux_mean_error / baseline_mean_scaled
                    ) / (1.0 + args.joint_chunk_alpha)
                else:
                    assert auxiliary_cross is not None
                    aux_cross_residual = (
                        auxiliary_cross.float()
                        - (scaled_cross_targets - reconstructed_cross.detach()).float()
                    )
                    aux_cross_error = (
                        aux_cross_residual[valid_mask].square().sum(dim=-1).mean()
                    )
                    auxiliary_loss = (
                        aux_mean_error / baseline_mean_scaled
                        + args.joint_chunk_alpha
                        * aux_cross_error
                        / baseline_cross_scaled
                    ) / (1.0 + args.joint_chunk_alpha)
                loss = task_loss + args.auxk_alpha * auxiliary_loss
            loss.backward()
            model.remove_parallel_decoder_gradient_()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            multiplier = learning_rate_multiplier(
                step_index, args.steps, args.warmup_steps, args.min_lr_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            optimizer.step()
            model.normalize_decoder_()
            with torch.no_grad():
                if not threshold_initialized:
                    model.threshold.copy_(batch_threshold.float())
                    threshold_initialized = True
                else:
                    model.threshold.mul_(args.threshold_beta).add_(
                        batch_threshold.float(), alpha=1 - args.threshold_beta
                    )
                model.feature_counts.add_(active_per_feature)
                model.update_dead_feature_stats_(
                    active_per_feature,
                    global_samples=accepted_global,
                    distributed=world_size > 1,
                )
            running_mean.update_sufficient_statistics(
                scaled_sse=mean_error.detach() * accepted_per_rank,
                raw_sse=mean_error.detach() * accepted_per_rank / (scale * scale),
                mean_predictor_sse=sampled_mean_baseline * accepted_per_rank,
                samples=accepted_per_rank,
                elements=accepted_per_rank * inputs.shape[1],
                active_counts=active_counts[valid_mask],
            )
            running_cross.update_sufficient_statistics(
                scaled_sse=cross_error.detach() * accepted_per_rank,
                raw_sse=cross_error.detach() * accepted_per_rank / (scale * scale),
                mean_predictor_sse=sampled_cross_baseline * accepted_per_rank,
                samples=accepted_per_rank,
                elements=accepted_per_rank * cross_targets.shape[1],
                active_counts=active_counts[valid_mask],
            )
            running_auxiliary_loss += float(auxiliary_loss.detach().item())
            running_auxiliary_steps += 1
            if nested:
                running_prefix_active.add_(
                    active_per_feature[:cross_prefix].sum().double()
                )
                running_prefix_samples.add_(accepted_per_rank)
            running_steps += 1
            running_grad_norm.add_(grad_norm.detach().double())
            running_batch_wait += 0.0
            running_filtered += filtered_global
            running_balanced += balanced_global
            total_filtered += filtered_global
            total_balanced += balanced_global
            accepted_seen += accepted_global
            samples_seen += args.global_batch_size
            completed_step = step_index + 1
            should_log = (
                completed_step == 1
                or completed_step % args.log_every == 0
                or completed_step == args.steps
            )
            if should_log:
                elapsed_window = max(time.perf_counter() - running_window_started, 1e-9)
                running_mean.all_reduce_()
                running_cross.all_reduce_()
                if nested and dist.is_initialized():
                    dist.all_reduce(running_prefix_active, op=dist.ReduceOp.SUM)
                    dist.all_reduce(running_prefix_samples, op=dist.ReduceOp.SUM)
                mean_summary = running_mean.summary().as_dict()
                cross_summary = running_cross.summary().as_dict()
                metrics = {
                    "train/joint_mean_scaled_objective": mean_summary["scaled_objective"],
                    "train/joint_cross_scaled_objective": cross_summary["scaled_objective"],
                    "train/joint_mean_normalized_mse": mean_summary["normalized_mse"],
                    "train/joint_cross_normalized_mse": cross_summary["normalized_mse"],
                    "train/joint_mean_fve": mean_summary["fve"],
                    "train/joint_cross_fve": cross_summary["fve"],
                    "train/joint_task_loss": (
                        mean_summary["scaled_objective"] / baseline_mean_scaled
                        + args.joint_chunk_alpha
                        * cross_summary["scaled_objective"] / baseline_cross_scaled
                    )
                    / (1.0 + args.joint_chunk_alpha),
                    "train/joint_alpha": args.joint_chunk_alpha,
                    "train/effective_l0": mean_summary["effective_l0"],
                    "train/zero_code_fraction": mean_summary["zero_code_fraction"],
                    "train/auxiliary_loss": running_auxiliary_loss / max(1, running_auxiliary_steps),
                    "optimizer/lr": optimizer.param_groups[0]["lr"],
                    "optimizer/grad_norm": float(running_grad_norm.item()) / max(1, running_steps),
                    "optimizer/updates": completed_step,
                    "progress/samples_seen": samples_seen,
                    "progress/token_occurrences_seen": samples_seen,
                    "progress/accepted_training_rows": accepted_seen,
                    "progress/unique_occurrence_coverage": (
                        samples_seen / exact_occurrences
                        if exact_occurrences is not None
                        else samples_seen / max(1, int(cache_occurrences(train_manifest) or samples_seen))
                    ),
                    "system/elapsed_seconds": time.time() - started,
                    "system/samples_per_second": samples_seen / max(time.time() - started, 1e-9),
                    "system/batch_wait_seconds": running_batch_wait,
                    "system/batch_wait_fraction": running_batch_wait / elapsed_window,
                    "sparsity/threshold": model.threshold.item(),
                    "sparsity/dead_features": model.dead_feature_count(args.dead_feature_threshold),
                    "sparsity/auxk_candidates": int(auxiliary_ids.numel()),
                    "sparsity/dead_fraction": model.dead_feature_count(args.dead_feature_threshold) / max(1, args.dict_size),
                    "filter/rejected_rows": running_filtered,
                    "filter/balance_dropped_rows": running_balanced,
                    "filter/input_norm_threshold": float(input_threshold.item()),
                    "filter/target_norm_threshold": float(target_threshold.item()),
                }
                if nested:
                    metrics["train/joint_k_prefix"] = float(
                        (
                            running_prefix_active
                            / running_prefix_samples.clamp_min(1.0)
                        ).item()
                    )
                if rank == 0:
                    row = {"run_id": run_id, "mode": mode, "split": "train", "step": completed_step, **metrics}
                    _write_jsonl(log_path, row)
                    writer.add_scalars(metrics, completed_step)
                    log(json.dumps(row, sort_keys=True), rank=rank)
                running_mean.reset()
                running_cross.reset()
                running_steps = 0
                running_grad_norm.zero_()
                running_auxiliary_loss = 0.0
                running_auxiliary_steps = 0
                running_prefix_active.zero_()
                running_prefix_samples.zero_()
                running_batch_wait = 0.0
                running_filtered = 0
                running_balanced = 0
                running_window_started = time.perf_counter()

            should_validate = (
                validation_manifest is not None
                and args.validate_every > 0
                and completed_step % args.validate_every == 0
            )
            if should_validate:
                validation_metrics = validate_joint_chunk_model(
                    model=model,
                    loader_factory=periodic_validation_loader_factory,
                    train_mean=input_mean,
                    train_cross_mean=cross_mean,
                    device=device,
                    use_threshold=args.validation_use_threshold,
                    allow_legacy_v1=not cache_is_v2(validation_manifest),
                    directional_train_cross_means=directional_cross_means,
                    deduplicate_chunk_inputs=args.deduplicate_chunk_inputs,
                )
                last_validation_metrics = dict(validation_metrics)
                current_value = float(
                    validation_metrics["validation/joint_cross_nmse"]
                )
                improved = _is_better(
                    current_value,
                    best_metric_value,
                    "nmse",
                )
                if improved:
                    best_metric_value = current_value
                    best_step = completed_step
                if rank == 0:
                    row = {
                        "run_id": run_id,
                        "mode": mode,
                        "split": "validation",
                        "step": completed_step,
                        "is_best": improved,
                        **validation_metrics,
                    }
                    _write_jsonl(log_path, row)
                    writer.add_scalars(validation_metrics, completed_step)
                    writer.add_scalars(
                        {
                            "validation/is_best": improved,
                            "validation/best_step": best_step or 0,
                        },
                        completed_step,
                    )
                    writer.flush()
                if improved:
                    if args.defer_best_checkpoint:
                        best_snapshot = _snapshot_best_model_collective(
                            model=model,
                            rank=rank,
                            device_snapshot=False,
                        )
                        best_snapshot_step = completed_step
                        best_snapshot_samples = samples_seen
                        best_snapshot_dirty = True
                    else:
                        _save_checkpoint_collective(
                            path=best_path,
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            step=completed_step,
                            samples_seen=samples_seen,
                            best_metric_value=best_metric_value,
                            best_step=best_step,
                            rank=rank,
                            world_size=world_size,
                            accepted_samples_seen=accepted_seen,
                            activation_norm_filtered_rows=total_filtered,
                            activation_norm_balance_dropped_rows=total_balanced,
                        )
            if (
                (args.save_every > 0 and completed_step % args.save_every == 0)
                or completed_step == args.steps
            ):
                link_best_from_latest = bool(
                    args.defer_best_checkpoint
                    and best_snapshot_dirty
                    and best_snapshot_step == completed_step
                )
                if (
                    args.defer_best_checkpoint
                    and best_snapshot_dirty
                    and not link_best_from_latest
                ):
                    assert best_snapshot_step is not None
                    _save_best_checkpoint_collective(
                        path=best_path,
                        tensors=best_snapshot,
                        config=config,
                        step=best_snapshot_step,
                        samples_seen=best_snapshot_samples,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                    )
                    best_snapshot = None
                    best_snapshot_dirty = False
                if pending_latest is not None:
                    pending_latest.result()
                    pending_latest = None
                if args.async_latest_checkpoint:
                    snapshot = _snapshot_checkpoint_collective(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=completed_step,
                        samples_seen=samples_seen,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                        accepted_samples_seen=accepted_seen,
                        activation_norm_filtered_rows=total_filtered,
                        activation_norm_balance_dropped_rows=total_balanced,
                    )
                    if rank == 0:
                        assert checkpoint_executor is not None and snapshot is not None
                        tensors, optimizer_state, checkpoint_state = snapshot
                        pending_latest = checkpoint_executor.submit(
                            save_training_checkpoint_snapshot,
                            tensors,
                            optimizer_state,
                            latest_path,
                            checkpoint_state,
                            latest_staging_path,
                        )
                else:
                    _save_checkpoint_collective(
                        path=latest_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=completed_step,
                        samples_seen=samples_seen,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                        accepted_samples_seen=accepted_seen,
                        activation_norm_filtered_rows=total_filtered,
                        activation_norm_balance_dropped_rows=total_balanced,
                    )
                if link_best_from_latest:
                    if pending_latest is not None:
                        pending_latest.result()
                        pending_latest = None
                    assert best_snapshot_step is not None
                    _save_best_checkpoint_collective(
                        path=best_path,
                        tensors=None,
                        config=config,
                        step=best_snapshot_step,
                        samples_seen=best_snapshot_samples,
                        best_metric_value=best_metric_value,
                        best_step=best_step,
                        rank=rank,
                        world_size=world_size,
                        source_weights=latest_path / "sae.safetensors",
                    )
                    best_snapshot = None
                    best_snapshot_dirty = False
                if rank == 0:
                    writer.flush()

        if args.defer_best_checkpoint and best_snapshot_dirty:
            assert best_snapshot_step is not None
            _save_best_checkpoint_collective(
                path=best_path,
                tensors=best_snapshot,
                config=config,
                step=best_snapshot_step,
                samples_seen=best_snapshot_samples,
                best_metric_value=best_metric_value,
                best_step=best_step,
                rank=rank,
                world_size=world_size,
            )
            best_snapshot = None
            best_snapshot_dirty = False
        if pending_latest is not None:
            pending_latest.result()
            pending_latest = None
        if dist.is_initialized():
            dist.barrier()
        if validation_manifest is not None:
            final_full_validation_metrics = validate_joint_chunk_model(
                model=model,
                loader_factory=lambda: validation_loader_factory(
                    args.final_validation_samples
                ),
                train_mean=input_mean,
                train_cross_mean=cross_mean,
                device=device,
                use_threshold=args.validation_use_threshold,
                allow_legacy_v1=not cache_is_v2(validation_manifest),
                directional_train_cross_means=directional_cross_means,
                deduplicate_chunk_inputs=args.deduplicate_chunk_inputs,
            )
            if rank == 0:
                full_row = {
                    "run_id": run_id,
                    "mode": mode,
                    "split": "validation_full",
                    "step": args.steps,
                    "requested_samples": args.final_validation_samples,
                    **{
                        key.replace("validation/", "validation_full/", 1): value
                        for key, value in final_full_validation_metrics.items()
                    },
                }
                _write_jsonl(log_path, full_row)
                writer.add_scalars(
                    {
                        key.replace("validation/", "validation_full/", 1): value
                        for key, value in final_full_validation_metrics.items()
                    },
                    args.steps,
                )
                writer.flush()
        if args.require_exact_coverage:
            try:
                extra_batch = next(train_loader)
            except StopIteration:
                extra_batch = None
            if extra_batch is not None or samples_seen != exact_occurrences:
                raise RuntimeError(
                    f"joint exact coverage consumed {samples_seen}, expected {exact_occurrences}"
                )
        if dist.is_initialized():
            dist.all_reduce(model.feature_counts, op=dist.ReduceOp.SUM)
        if rank == 0:
            final_config = {
                **config,
                "steps_completed": args.steps,
                "samples_seen": samples_seen,
                "accepted_training_rows": accepted_seen,
                "activation_norm_filtered_rows": total_filtered,
                "activation_norm_balance_dropped_rows": total_balanced,
                "unique_occurrences_seen": exact_occurrences,
                "coverage_fraction": samples_seen / exact_occurrences if exact_occurrences else None,
                "tensorboard_dir": str(tensorboard_root) if tensorboard_root else None,
                "best_metric": "joint_cross_nmse",
                "best_metric_value": best_metric_value,
                "best_step": best_step,
                "last_periodic_validation_metrics": last_validation_metrics,
                "final_full_validation_metrics": final_full_validation_metrics,
                "dead_features": model.dead_feature_count(args.dead_feature_threshold),
            }
            save_sae(model, mode_dir, final_config)
            alive = int((model.feature_counts > 0).sum())
            atomic_json_dump(
                {
                    "complete": True,
                    "run_id": run_id,
                    "mode": mode,
                    "steps": args.steps,
                    "samples_seen": samples_seen,
                    "accepted_training_rows": accepted_seen,
                    "activation_norm_filtered_rows": total_filtered,
                    "activation_norm_balance_dropped_rows": total_balanced,
                    "unique_occurrences_seen": exact_occurrences,
                    "coverage_fraction": samples_seen / exact_occurrences if exact_occurrences else None,
                    "exact_coverage": bool(args.require_exact_coverage),
                    "best_metric": "joint_cross_nmse",
                    "best_metric_value": best_metric_value,
                    "best_step": best_step,
                    "alive_features": alive,
                    "dead_features": model.dead_feature_count(args.dead_feature_threshold),
                    "train_cache_digest": train_cache_digest,
                    "validation_cache_digest": validation_cache_digest,
                    "last_periodic_validation_metrics": last_validation_metrics,
                    "final_full_validation_metrics": final_full_validation_metrics,
                    "validation_skipped": validation_manifest is None,
                },
                mode_dir / "complete.json",
            )
            log(f"saved {mode} SAE; alive_features={alive}/{args.dict_size}", rank=rank)
        if dist.is_initialized():
            dist.barrier()
    finally:
        if pending_latest is not None:
            pending_latest.result()
        if checkpoint_executor is not None:
            checkpoint_executor.shutdown(wait=True, cancel_futures=False)
        writer.close()
        del ddp, model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_modes_joint(
    args: argparse.Namespace,
    modes: list[str],
    rank: int,
    world_size: int,
    local_rank: int,
    hidden_size: int,
    train_manifest: Mapping[str, Any],
    train_cache_digest: str,
    validation_manifest: Mapping[str, Any] | None,
    validation_cache_digest: str | None,
    run_id: str,
) -> None:
    """Train multiple SAE objectives in lockstep from one activation-cache pass."""

    if args.resume or args.resume_from:
        raise ValueError(
            "joint-mode resume is intentionally fail-closed; resume modes "
            "sequentially, then start new joint runs from complete checkpoints"
        )
    if args.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    exact_occurrences = None
    if args.require_exact_coverage:
        exact_occurrences = validate_exact_training_coverage(
            train_manifest,
            world_size=world_size,
            global_batch_size=args.global_batch_size,
            steps=args.steps,
        )
    per_rank_batch = args.global_batch_size // world_size
    validation_global_batch = args.validation_batch_size or args.global_batch_size
    if validation_global_batch % world_size:
        raise ValueError("validation batch size must be divisible by world size")
    per_rank_validation_batch = validation_global_batch // world_size
    device = _device(local_rank)
    allow_legacy_v1 = not cache_is_v2(train_manifest)

    mode_dirs = {mode: Path(args.output_dir) / mode for mode in modes}
    for mode, mode_dir in mode_dirs.items():
        if rank == 0:
            if mode_dir.exists():
                if args.overwrite_output:
                    shutil.rmtree(mode_dir)
                elif any(mode_dir.iterdir()):
                    raise FileExistsError(
                        f"output directory is not empty: {mode_dir}; "
                        "use --overwrite-output"
                    )
            mode_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    init_loader = OccurrenceLoaderAdapter(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        world_size=world_size,
        batch_size=per_rank_batch,
        seed=args.seed + 101,
        device=str(device),
        mode="all",
        direction_policy=args.direction_policy,
        finite=True,
        prefetch_shards=1,
        prefetch_workers=1,
        prefetch_batches=1,
        materialize_shards=args.loader_materialize_shards,
        pin_memory=args.loader_pin_memory,
        local_cache_dir=args.local_cache_dir,
        gpu_shards=min(2, args.loader_gpu_shards),
    )
    estimates = estimate_joint_scales(
        modes,
        _iter_limited(
            init_loader.iter_epoch(0),
            _global_limit_to_local(args.normalization_samples, world_size),
        ),
        hidden_size,
        device,
        allow_legacy_v1=allow_legacy_v1,
    )

    states: dict[str, dict[str, Any]] = {}
    autocast_dtype = dtype_from_name(args.autocast_dtype)
    for mode in modes:
        estimated_scale, sampled_input_mean, sampled_target_mean = estimates[mode]
        scale = estimated_scale if args.normalize_activations else 1.0
        target_mode = _cross_target_mode(args.direction_policy) if mode == "cross" else mode
        exact_target_mean = cache_target_mean(
            train_manifest,
            target_mode,
            hidden_size=hidden_size,
            device=device,
        )
        target_mean = (
            exact_target_mean
            if exact_target_mean is not None
            else sampled_target_mean.to(device)
        )
        exact_input_mean = (
            cache_input_mean(
                train_manifest,
                mode,
                hidden_size=hidden_size,
                device=device,
            )
            if not (mode == "cross" and args.direction_policy != "both")
            else None
        )
        input_mean = (
            exact_input_mean
            if exact_input_mean is not None
            else sampled_input_mean.to(device)
        )
        directional = None
        if mode == "cross":
            a_to_b = cache_target_mean(
                train_manifest,
                "cross_a_to_b",
                hidden_size=hidden_size,
                device=device,
            )
            b_to_a = cache_target_mean(
                train_manifest,
                "cross_b_to_a",
                hidden_size=hidden_size,
                device=device,
            )
            if a_to_b is not None and b_to_a is not None:
                directional = {"a_to_b": a_to_b, "b_to_a": b_to_a}

        # The historical sequential path resets the same seed for every mode.
        seed_everything(args.seed + rank)
        model = BatchTopKSAE(
            hidden_size,
            args.dict_size,
            args.k,
            batch_topk_candidate_multiplier=args.batch_topk_candidate_multiplier,
            batch_topk_bf16_histogram=args.batch_topk_bf16_histogram,
            decoder_backend=args.decoder_backend,
        ).to(device)
        model.activation_scale.fill_(scale)
        model.pre_bias.copy_(input_mean * scale)
        model.decoder_bias.data.copy_(target_mean * scale)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            fused=bool(args.fused_adam and device.type == "cuda"),
        )
        ddp = (
            DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                bucket_cap_mb=256,
                # AuxK starts contributing only after features cross the configured
                # activation age, so its autograd path is intentionally dynamic.
                static_graph=args.auxk_alpha <= 0,
            )
            if world_size > 1
            else model
        )
        config = _mode_config(
            args,
            mode=mode,
            hidden_size=hidden_size,
            scale=scale,
            train_cache_digest=train_cache_digest,
            train_cache_format=str(train_manifest.get("format", "unknown")),
            validation_cache_digest=validation_cache_digest,
            run_id=run_id,
        )
        config["training_execution"] = "joint_lockstep_single_cache_pass"
        config["train_target_mean_source"] = (
            "exact_full_training_cache"
            if exact_target_mean is not None
            else "normalization_sample_estimate"
        )
        config["train_input_mean_source"] = (
            "exact_full_training_cache_mean_inputs"
            if exact_input_mean is not None and mode == "cross"
            else "exact_full_training_cache"
            if exact_input_mean is not None
            else "normalization_sample_estimate"
        )
        config["train_input_mean_mode"] = (
            "mean"
            if mode == "cross"
            else "token"
            if mode == "temporal"
            else mode
        )
        tensorboard_root = (
            Path(args.tensorboard_dir) / run_id / mode
            if args.tensorboard_dir
            else None
        )
        states[mode] = {
            "model": model,
            "ddp": ddp,
            "optimizer": optimizer,
            "config": config,
            "target_mean": target_mean,
            "directional": directional,
            "mode_dir": mode_dirs[mode],
            "latest_path": mode_dirs[mode] / "checkpoints" / "latest",
            "best_path": mode_dirs[mode] / "checkpoints" / "best",
            "latest_staging_path": (
                Path(args.checkpoint_staging_dir)
                / Path(args.output_dir).name
                / mode
                / "latest"
                if args.checkpoint_staging_dir
                else None
            ),
            "log_path": mode_dirs[mode] / "metrics.jsonl",
            "writer": TensorBoardLogger(
                tensorboard_root,
                rank=rank,
                flush_secs=args.tensorboard_flush_secs,
                max_queue=args.tensorboard_max_queue,
                scalar_allowlist=CORE_TENSORBOARD_SCALARS,
                mode=mode,
                fidelity_reference_fve=args.fidelity_reference_fves.get(mode),
                gradient_clip=args.gradient_clip,
            ),
            "running": ReconstructionMetricAccumulator(device),
            "running_steps": 0,
            "running_auxiliary_loss": 0.0,
            "running_auxiliary_steps": 0,
            "running_temporal_loss": 0.0,
            "running_temporal_accuracy": 0.0,
            "running_temporal_pairs": 0,
            "running_grad_norm": torch.zeros(
                (), dtype=torch.float64, device=device
            ),
            "running_filtered": 0,
            "running_balanced": 0,
            "accepted": 0,
            "filtered": 0,
            "balanced": 0,
            "threshold_initialized": False,
            "best_value": None,
            "best_step": None,
            "best_snapshot": None,
            "best_snapshot_step": None,
            "best_snapshot_samples": 0,
            "best_dirty": False,
            "last_validation": None,
            "final_validation": None,
            "checkpoint_executor": (
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"latest-checkpoint-{mode}",
                )
                if rank == 0 and args.async_latest_checkpoint
                else None
            ),
            "pending_latest": None,
        }
        if rank == 0:
            atomic_json_dump(config, mode_dirs[mode] / "run_config.json")

    train_adapter = OccurrenceLoaderAdapter(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        world_size=world_size,
        batch_size=per_rank_batch,
        seed=args.seed + 211,
        device=str(device),
        mode="all",
        direction_policy=args.direction_policy,
        finite=bool(args.require_exact_coverage),
        prefetch_shards=args.loader_prefetch_shards,
        prefetch_workers=args.loader_prefetch_workers,
        prefetch_batches=args.loader_prefetch_batches,
        materialize_shards=args.loader_materialize_shards,
        pin_memory=args.loader_pin_memory,
        local_cache_dir=args.local_cache_dir,
        gpu_shards=args.loader_gpu_shards,
    )
    train_loader = (
        iter(train_adapter.iter_epoch(0))
        if args.require_exact_coverage
        else iter(train_adapter)
    )

    def validation_loader_factory(global_samples: int) -> Iterator[Any]:
        if validation_manifest is None or args.validation_cache_dir is None:
            return iter(())
        bounded = global_samples > 0
        adapter = OccurrenceLoaderAdapter(
            args.validation_cache_dir,
            validation_manifest,
            rank=rank,
            world_size=world_size,
            batch_size=per_rank_validation_batch,
            seed=args.seed + 307,
            device=str(device),
            mode="all",
            finite=True,
            prefetch_shards=1 if bounded else args.loader_prefetch_shards,
            prefetch_workers=1 if bounded else args.loader_prefetch_workers,
            prefetch_batches=args.loader_prefetch_batches,
            materialize_shards=args.loader_materialize_shards,
            pin_memory=args.loader_pin_memory,
            local_cache_dir=args.local_cache_dir,
            gpu_shards=min(args.loader_gpu_shards, 4 if bounded else args.loader_gpu_shards),
        )
        return _iter_limited(
            adapter.iter_epoch(0),
            _global_limit_to_local(global_samples, world_size),
        )

    periodic_validation_batches: tuple[Any, ...] | None = None
    if (
        validation_manifest is not None
        and args.cache_periodic_validation
        and args.validation_samples > 0
    ):
        local_samples = _global_limit_to_local(
            args.validation_samples, world_size
        )
        assert local_samples is not None
        estimated = _estimated_validation_cache_bytes(
            mode="all",
            local_samples=local_samples,
            hidden_size=hidden_size,
            manifest=validation_manifest,
        )
        if estimated <= args.validation_cache_max_bytes:
            periodic_validation_batches = tuple(
                validation_loader_factory(args.validation_samples)
            )
            if rank == 0:
                actual = sum(
                    _batch_nbytes(batch)
                    for batch in periodic_validation_batches
                )
                log(
                    f"cached joint periodic validation: "
                    f"{actual / 2**20:.1f} MiB/rank",
                    rank=rank,
                )

    def periodic_loader() -> Iterator[Any]:
        if periodic_validation_batches is not None:
            return iter(periodic_validation_batches)
        return validation_loader_factory(args.validation_samples)

    started = time.time()
    samples_seen = 0
    running_batch_wait = 0.0
    running_window_started = time.perf_counter()

    def run_validation(loader: Iterable[Any], step: int) -> None:
        metrics_by_mode = validate_models_joint(
            models={mode: state["model"] for mode, state in states.items()},
            loader=loader,
            train_target_means={
                mode: state["target_mean"] for mode, state in states.items()
            },
            directional_train_target_means={
                mode: state["directional"] for mode, state in states.items()
            },
            use_threshold=args.validation_use_threshold,
            allow_legacy_v1=(
                validation_manifest is not None
                and not cache_is_v2(validation_manifest)
            ),
            deduplicate_chunk_inputs=args.deduplicate_chunk_inputs,
        )
        for mode, metrics in metrics_by_mode.items():
            state = states[mode]
            state["last_validation"] = dict(metrics)
            current = _best_value(metrics, args.best_metric)
            improved = _is_better(
                current, state["best_value"], args.best_metric
            )
            if improved:
                state["best_value"] = current
                state["best_step"] = step
                state["best_snapshot"] = _snapshot_best_model_collective(
                    model=state["model"],
                    rank=rank,
                )
                state["best_snapshot_step"] = step
                state["best_snapshot_samples"] = samples_seen
                state["best_dirty"] = True
            if rank == 0:
                row = {
                    "run_id": run_id,
                    "mode": mode,
                    "split": "validation",
                    "step": step,
                    "is_best": improved,
                    **metrics,
                }
                _write_jsonl(state["log_path"], row)
                state["writer"].add_scalars(metrics, step)
                state["writer"].add_scalars(
                    {
                        "validation/is_best": improved,
                        "validation/best_step": state["best_step"] or 0,
                    },
                    step,
                )

    try:
        if rank == 0:
            log(
                f"joint training modes={','.join(modes)} hidden={hidden_size} "
                f"width={args.dict_size} steps={args.steps} "
                f"global_batch={args.global_batch_size}",
                rank=rank,
            )
        if validation_manifest is not None:
            run_validation(periodic_loader(), 0)

        for step_index in range(args.steps):
            wait_started = time.perf_counter()
            batch = next(train_loader)
            running_batch_wait += time.perf_counter() - wait_started
            completed_step = step_index + 1
            for mode in modes:
                state = states[mode]
                model: BatchTopKSAE = state["model"]
                optimizer: torch.optim.Optimizer = state["optimizer"]
                inputs, targets, _ = select_occurrence_view(
                    mode,
                    batch,
                    allow_legacy_v1=allow_legacy_v1,
                )
                valid_mask, input_threshold, target_threshold = (
                    activation_norm_ok_mask(
                        inputs,
                        targets,
                        args.max_activation_norm_multiple,
                    )
                )
                local_filtered = inputs.shape[0] - int(valid_mask.sum())
                valid_mask, accepted_per_rank, balance_dropped = (
                    equalize_valid_rows(valid_mask)
                )
                if accepted_per_rank <= 0:
                    raise RuntimeError(
                        f"activation filtering removed all {mode} rows"
                    )
                accepted_global = accepted_per_rank * world_size
                filter_counts = torch.tensor(
                    [local_filtered, balance_dropped],
                    dtype=torch.int64,
                    device=device,
                )
                if dist.is_initialized():
                    dist.all_reduce(filter_counts, op=dist.ReduceOp.SUM)
                filtered_global = int(filter_counts[0])
                balanced_global = int(filter_counts[1])
                deduplication = (
                    chunk_input_deduplication(batch, inputs.shape[0])
                    if args.deduplicate_chunk_inputs
                    and mode in {"mean", "cross"}
                    else None
                )
                auxiliary_ids = model.auxiliary_feature_ids(
                    args.auxk_activation_age,
                    args.auxk_candidate_features,
                )
                state["auxk_candidates"] = int(auxiliary_ids.numel())
                temporal_previous = None
                temporal_pair_mask = None
                temporal_high_features = 0
                temporal_global_pairs = None
                if mode == "temporal":
                    temporal_previous, temporal_pair_mask = (
                        select_temporal_pair_view(batch)
                    )
                    temporal_pair_mask = temporal_pair_mask & valid_mask
                    temporal_pair_count = temporal_pair_mask.sum(
                        dtype=torch.int64
                    )
                    if dist.is_initialized():
                        dist.all_reduce(
                            temporal_pair_count,
                            op=dist.ReduceOp.SUM,
                        )
                    temporal_global_pairs = int(temporal_pair_count.item())
                    if temporal_global_pairs <= 1:
                        raise RuntimeError(
                            "Temporal SAE batch contains fewer than two valid "
                            "within-chunk adjacent pairs"
                        )
                    temporal_high_features = temporal_high_feature_count(
                        args.dict_size,
                        args.temporal_high_fraction,
                    )
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, autocast_dtype):
                    forward_output = state["ddp"](
                        inputs,
                        batch_topk=True,
                        distributed=world_size > 1,
                        return_activity_counts=True,
                        sample_mask=valid_mask,
                        global_sample_count=accepted_global,
                        unique_rows=(
                            None if deduplication is None else deduplication[0]
                        ),
                        dedup_inverse=(
                            None if deduplication is None else deduplication[1]
                        ),
                        auxiliary_feature_ids=auxiliary_ids,
                        auxiliary_k=args.auxk,
                        return_auxiliary=True,
                        temporal_previous=temporal_previous,
                        temporal_high_features=temporal_high_features,
                        temporal_sample_mask=temporal_pair_mask,
                        temporal_global_sample_count=temporal_global_pairs,
                    )
                    (
                        reconstructed,
                        features,
                        batch_threshold,
                        active_counts,
                        active_per_feature,
                        auxiliary_reconstruction,
                    ) = forward_output[:6]
                    scaled_targets = targets * model.activation_scale.to(
                        targets.dtype
                    )
                    residual = reconstructed.float() - scaled_targets.float()
                    primary_loss = residual[valid_mask].square().sum(dim=-1).mean()
                    auxiliary_target = (
                        scaled_targets - reconstructed.detach()
                    ).float()
                    auxiliary_residual = (
                        auxiliary_reconstruction.float() - auxiliary_target
                    )
                    if auxiliary_ids.numel() and args.auxk_alpha > 0:
                        auxiliary_loss = (
                            auxiliary_residual[valid_mask]
                            .square()
                            .sum(dim=-1)
                            .mean()
                        )
                    else:
                        auxiliary_loss = primary_loss.new_zeros(())
                    if mode == "temporal":
                        temporal_current_features = forward_output[6]
                        previous_features = forward_output[7]
                        high_reconstruction = forward_output[8]
                        high_residual = (
                            high_reconstruction.float()
                            - scaled_targets.float()
                        )
                        high_reconstruction_loss = (
                            high_residual[temporal_pair_mask]
                            .square()
                            .sum(dim=-1)
                            .mean()
                        )
                        assert temporal_pair_mask is not None
                        temporal_result = (
                            distributed_symmetric_temporal_contrastive_loss(
                                temporal_current_features[
                                    :, :temporal_high_features
                                ],
                                previous_features[:, :temporal_high_features],
                                temporal_pair_mask,
                                temperature=args.temporal_temperature,
                            )
                            if world_size > 1
                            else symmetric_temporal_contrastive_loss(
                                temporal_current_features[
                                    :, :temporal_high_features
                                ],
                                previous_features[:, :temporal_high_features],
                                temporal_pair_mask,
                                temperature=args.temporal_temperature,
                                block_size=args.temporal_contrastive_block_size,
                            )
                        )
                        reconstruction_loss = (
                            args.temporal_high_reconstruction_weight
                            * high_reconstruction_loss
                            + args.temporal_full_reconstruction_weight
                            * primary_loss
                        ) / 2.0
                        loss = (
                            reconstruction_loss
                            + args.auxk_alpha * auxiliary_loss
                            + args.temporal_alpha * temporal_result.loss
                        )
                    else:
                        temporal_result = None
                        loss = primary_loss + args.auxk_alpha * auxiliary_loss
                loss.backward()
                model.remove_parallel_decoder_gradient_()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                multiplier = learning_rate_multiplier(
                    step_index,
                    args.steps,
                    args.warmup_steps,
                    args.min_lr_ratio,
                )
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * multiplier
                optimizer.step()
                model.normalize_decoder_()
                with torch.no_grad():
                    if not state["threshold_initialized"]:
                        model.threshold.copy_(batch_threshold.float())
                        state["threshold_initialized"] = True
                    else:
                        model.threshold.mul_(args.threshold_beta).add_(
                            batch_threshold.float(),
                            alpha=1 - args.threshold_beta,
                        )
                    model.feature_counts.add_(active_per_feature)
                    dead_activity_per_feature = active_per_feature
                    if mode == "temporal":
                        dead_activity_per_feature = (
                            active_per_feature
                            + (
                                temporal_current_features.detach() != 0
                            ).sum(dim=0, dtype=torch.int64)
                            + (previous_features.detach() != 0).sum(
                                dim=0,
                                dtype=torch.int64,
                            )
                        )
                    model.update_dead_feature_stats_(
                        dead_activity_per_feature,
                        global_samples=accepted_global,
                        distributed=world_size > 1,
                    )
                    scaled_sse = primary_loss.detach() * accepted_per_rank
                    raw_sse = scaled_sse / (
                        float(model.activation_scale.item()) ** 2
                    )
                    baseline_sse = (
                        targets[valid_mask].float()
                        - state["target_mean"].float()
                    ).square().sum()
                state["running"].update_sufficient_statistics(
                    scaled_sse=scaled_sse,
                    raw_sse=raw_sse,
                    mean_predictor_sse=baseline_sse,
                    samples=accepted_per_rank,
                    elements=accepted_per_rank * targets.shape[1],
                    active_counts=active_counts[valid_mask],
                )
                state["running_auxiliary_loss"] += float(
                    auxiliary_loss.detach().item()
                )
                state["running_auxiliary_steps"] += 1
                if temporal_result is not None:
                    state["running_temporal_loss"] += float(
                        temporal_result.loss.detach().item()
                    ) * temporal_result.pairs
                    state["running_temporal_accuracy"] += float(
                        temporal_result.accuracy.detach().item()
                    ) * temporal_result.pairs
                    state["running_temporal_pairs"] += temporal_result.pairs
                state["running_steps"] += 1
                state["running_grad_norm"].add_(grad_norm.detach().double())
                state["running_filtered"] += filtered_global
                state["running_balanced"] += balanced_global
                state["filtered"] += filtered_global
                state["balanced"] += balanced_global
                state["accepted"] += accepted_global
                state["input_threshold"] = input_threshold
                state["target_threshold"] = target_threshold

            samples_seen += args.global_batch_size
            should_log = (
                completed_step == 1
                or completed_step % args.log_every == 0
                or completed_step == args.steps
            )
            if should_log:
                elapsed_window = time.perf_counter() - running_window_started
                for mode, state in states.items():
                    state["running"].all_reduce_()
                    summary = state["running"].summary().as_dict()
                    metrics = {
                        "train/scaled_objective": summary["scaled_objective"],
                        "train/raw_mse": summary["raw_mse"],
                        "train/mean_predictor_mse": summary[
                            "mean_predictor_mse"
                        ],
                        "train/normalized_mse": summary["normalized_mse"],
                        "train/fve": summary["fve"],
                        "train/effective_l0": summary["effective_l0"],
                        "train/zero_code_fraction": summary[
                            "zero_code_fraction"
                        ],
                        "train/auxiliary_loss": (
                            state["running_auxiliary_loss"]
                            / max(1, state["running_auxiliary_steps"])
                        ),
                        "sparsity/dead_fraction": state["model"].dead_feature_count(
                            args.dead_feature_threshold
                        )
                        / max(1, args.dict_size),
                        "optimizer/lr": state["optimizer"].param_groups[0]["lr"],
                        "optimizer/grad_norm": float(
                            state["running_grad_norm"].item()
                        )
                        / max(1, state["running_steps"]),
                        "optimizer/updates": completed_step,
                        "progress/samples_seen": samples_seen,
                        "progress/token_occurrences_seen": samples_seen,
                        "progress/accepted_training_rows": state["accepted"],
                        "progress/unique_occurrence_coverage": (
                            samples_seen / exact_occurrences
                            if exact_occurrences is not None
                            else 0.0
                        ),
                        "system/elapsed_seconds": time.time() - started,
                        "system/samples_per_second": samples_seen
                        / max(time.time() - started, 1e-9),
                        "system/batch_wait_seconds": running_batch_wait,
                        "system/batch_wait_fraction": running_batch_wait
                        / max(elapsed_window, 1e-9),
                        "sparsity/threshold": state["model"].threshold.item(),
                        "sparsity/dead_features": state[
                            "model"
                        ].dead_feature_count(args.dead_feature_threshold),
                        "sparsity/auxk_candidates": state.get(
                            "auxk_candidates", 0
                        ),
                        "sparsity/dead_fraction": state[
                            "model"
                        ].dead_feature_count(args.dead_feature_threshold)
                        / max(1, args.dict_size),
                        "filter/rejected_rows": state["running_filtered"],
                        "filter/balance_dropped_rows": state[
                            "running_balanced"
                        ],
                        "filter/input_norm_threshold": float(
                            state["input_threshold"].item()
                        ),
                        "filter/target_norm_threshold": float(
                            state["target_threshold"].item()
                        ),
                    }
                    if mode == "temporal":
                        metrics.update(
                            {
                                "train/temporal_contrastive_loss": (
                                    state["running_temporal_loss"]
                                    / max(1, state["running_temporal_pairs"])
                                ),
                                "train/temporal_contrastive_accuracy": (
                                    state["running_temporal_accuracy"]
                                    / max(1, state["running_temporal_pairs"])
                                ),
                                "train/temporal_pairs": state[
                                    "running_temporal_pairs"
                                ],
                            }
                        )
                    if rank == 0:
                        row = {
                            "run_id": run_id,
                            "mode": mode,
                            "split": "train",
                            "step": completed_step,
                            **metrics,
                        }
                        _write_jsonl(state["log_path"], row)
                        state["writer"].add_scalars(metrics, completed_step)
                        log(json.dumps(row, sort_keys=True), rank=rank)
                    state["running"].reset()
                    state["running_steps"] = 0
                    state["running_auxiliary_loss"] = 0.0
                    state["running_auxiliary_steps"] = 0
                    state["running_temporal_loss"] = 0.0
                    state["running_temporal_accuracy"] = 0.0
                    state["running_temporal_pairs"] = 0
                    state["running_grad_norm"].zero_()
                    state["running_filtered"] = 0
                    state["running_balanced"] = 0
                running_batch_wait = 0.0
                running_window_started = time.perf_counter()

            should_validate = (
                validation_manifest is not None
                and (
                    (
                        args.validate_every > 0
                        and completed_step % args.validate_every == 0
                    )
                    or completed_step == args.steps
                )
            )
            if should_validate:
                run_validation(periodic_loader(), completed_step)

            should_save = (
                (args.save_every > 0 and completed_step % args.save_every == 0)
                or completed_step == args.steps
            )
            if should_save:
                for mode, state in states.items():
                    if state["best_dirty"]:
                        _save_best_checkpoint_collective(
                            path=state["best_path"],
                            tensors=state["best_snapshot"],
                            config=state["config"],
                            step=state["best_snapshot_step"],
                            samples_seen=state["best_snapshot_samples"],
                            best_metric_value=state["best_value"],
                            best_step=state["best_step"],
                            rank=rank,
                            world_size=world_size,
                        )
                        state["best_snapshot"] = None
                        state["best_dirty"] = False
                    if state["pending_latest"] is not None:
                        state["pending_latest"].result()
                        state["pending_latest"] = None
                    if args.async_latest_checkpoint:
                        snapshot = _snapshot_checkpoint_collective(
                            model=state["model"],
                            optimizer=state["optimizer"],
                            config=state["config"],
                            step=completed_step,
                            samples_seen=samples_seen,
                            best_metric_value=state["best_value"],
                            best_step=state["best_step"],
                            rank=rank,
                            world_size=world_size,
                            accepted_samples_seen=state["accepted"],
                            activation_norm_filtered_rows=state["filtered"],
                            activation_norm_balance_dropped_rows=state["balanced"],
                        )
                        if rank == 0:
                            executor = state["checkpoint_executor"]
                            assert executor is not None and snapshot is not None
                            tensors, optimizer_state, checkpoint_state = snapshot
                            state["pending_latest"] = executor.submit(
                                save_training_checkpoint_snapshot,
                                tensors,
                                optimizer_state,
                                state["latest_path"],
                                checkpoint_state,
                                state["latest_staging_path"],
                            )
                    else:
                        _save_checkpoint_collective(
                            path=state["latest_path"],
                            model=state["model"],
                            optimizer=state["optimizer"],
                            config=state["config"],
                            step=completed_step,
                            samples_seen=samples_seen,
                            best_metric_value=state["best_value"],
                            best_step=state["best_step"],
                            rank=rank,
                            world_size=world_size,
                            accepted_samples_seen=state["accepted"],
                            activation_norm_filtered_rows=state["filtered"],
                            activation_norm_balance_dropped_rows=state["balanced"],
                        )

        for state in states.values():
            if state["pending_latest"] is not None:
                state["pending_latest"].result()
                state["pending_latest"] = None
        if dist.is_initialized():
            dist.barrier()

        if validation_manifest is not None:
            final_metrics = validate_models_joint(
                models={
                    mode: state["model"] for mode, state in states.items()
                },
                loader=validation_loader_factory(
                    args.final_validation_samples
                ),
                train_target_means={
                    mode: state["target_mean"]
                    for mode, state in states.items()
                },
                directional_train_target_means={
                    mode: state["directional"]
                    for mode, state in states.items()
                },
                use_threshold=args.validation_use_threshold,
                allow_legacy_v1=not cache_is_v2(validation_manifest),
                deduplicate_chunk_inputs=args.deduplicate_chunk_inputs,
            )
            for mode, metrics in final_metrics.items():
                states[mode]["final_validation"] = dict(metrics)
                if rank == 0:
                    _write_jsonl(
                        states[mode]["log_path"],
                        {
                            "run_id": run_id,
                            "mode": mode,
                            "split": "validation_full",
                            "step": args.steps,
                            "requested_samples": args.final_validation_samples,
                            **metrics,
                        },
                    )
                    states[mode]["writer"].add_scalars(
                        {
                            key.replace(
                                "validation/", "validation_full/", 1
                            ): value
                            for key, value in metrics.items()
                        },
                        args.steps,
                    )
                    states[mode]["writer"].flush()

        if args.require_exact_coverage:
            try:
                extra_batch = next(train_loader)
            except StopIteration:
                extra_batch = None
            if extra_batch is not None or samples_seen != exact_occurrences:
                raise RuntimeError("joint exact-coverage loader did not end exactly")

        for mode, state in states.items():
            model = state["model"]
            if dist.is_initialized():
                dist.all_reduce(model.feature_counts, op=dist.ReduceOp.SUM)
            if rank == 0:
                final_config = {
                    **state["config"],
                    "steps_completed": args.steps,
                    "samples_seen": samples_seen,
                    "accepted_training_rows": state["accepted"],
                    "activation_norm_filtered_rows": state["filtered"],
                    "activation_norm_balance_dropped_rows": state["balanced"],
                    "unique_occurrences_seen": (
                        exact_occurrences if args.require_exact_coverage else None
                    ),
                    "coverage_fraction": (
                        samples_seen / exact_occurrences
                        if exact_occurrences is not None
                        else None
                    ),
                    "best_metric_value": state["best_value"],
                    "best_step": state["best_step"],
                    "last_periodic_validation_metrics": state["last_validation"],
                    "final_full_validation_metrics": state["final_validation"],
                    "dead_features": model.dead_feature_count(
                        args.dead_feature_threshold
                    ),
                }
                save_sae(model, state["mode_dir"], final_config)
                alive = int((model.feature_counts > 0).sum())
                atomic_json_dump(
                    {
                        "complete": True,
                        "run_id": run_id,
                        "mode": mode,
                        "steps": args.steps,
                        "samples_seen": samples_seen,
                        "accepted_training_rows": state["accepted"],
                        "activation_norm_filtered_rows": state["filtered"],
                        "activation_norm_balance_dropped_rows": state["balanced"],
                        "unique_occurrences_seen": exact_occurrences,
                        "coverage_fraction": (
                            samples_seen / exact_occurrences
                            if exact_occurrences is not None
                            else None
                        ),
                        "exact_coverage": bool(args.require_exact_coverage),
                        "best_metric": args.best_metric,
                        "best_metric_value": state["best_value"],
                        "best_step": state["best_step"],
                        "alive_features": alive,
                        "dead_features": model.dead_feature_count(
                            args.dead_feature_threshold
                        ),
                        "dead_feature_threshold": args.dead_feature_threshold,
                        "train_cache_digest": train_cache_digest,
                        "validation_cache_digest": validation_cache_digest,
                        "last_periodic_validation_metrics": state[
                            "last_validation"
                        ],
                        "final_full_validation_metrics": state[
                            "final_validation"
                        ],
                    },
                    state["mode_dir"] / "complete.json",
                )
            if dist.is_initialized():
                dist.barrier()
    finally:
        for state in states.values():
            if state["pending_latest"] is not None:
                state["pending_latest"].result()
            executor = state["checkpoint_executor"]
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
            state["writer"].close()
        states.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _clip_tensor_parallel_grad_norm_(
    model: FeatureTensorParallelSAE,
    max_norm: float,
) -> torch.Tensor:
    device = model.local.decoder_weight.device
    local_squared = torch.zeros((), dtype=torch.float64, device=device)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.endswith("decoder_bias") and model.rank != 0:
            continue
        local_squared += parameter.grad.detach().double().square().sum()
    if dist.is_initialized() and model.world_size > 1:
        dist.all_reduce(
            local_squared,
            op=dist.ReduceOp.SUM,
            group=model.group,
        )
    norm = local_squared.sqrt()
    coefficient = min(1.0, float(max_norm) / max(float(norm), 1e-12))
    if coefficient < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return norm.float()


@torch.no_grad()
def _validate_tensor_parallel_model(
    *,
    model: FeatureTensorParallelSAE,
    mode: str,
    loader: Iterable[Any],
    train_target_mean: torch.Tensor,
    allow_legacy_v1: bool,
    directional_train_target_means: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, float | int | bool]:
    accumulators = {
        "all": ReconstructionMetricAccumulator(
            model.local.decoder_weight.device
        )
    }
    if mode == "cross":
        accumulators["a_to_b"] = ReconstructionMetricAccumulator(
            model.local.decoder_weight.device
        )
        accumulators["b_to_a"] = ReconstructionMetricAccumulator(
            model.local.decoder_weight.device
        )
    local_batch = None
    for batch in loader:
        inputs, targets, is_b = select_occurrence_view(
            mode,
            batch,
            allow_legacy_v1=allow_legacy_v1,
        )
        local_batch = inputs.shape[0]
        reconstructed, _features, _, active_counts, _active_per_feature = model(
            inputs,
            batch_topk=True,
            return_activity_counts=True,
        )
        start = model.rank * local_batch
        stop = start + local_batch
        local_reconstructed = reconstructed[start:stop]
        local_active = active_counts[start:stop]
        accumulators["all"].update(
            local_reconstructed,
            targets,
            activation_scale=model.activation_scale,
            mean_predictor_raw=train_target_mean,
            active_counts=local_active,
        )
        if mode == "cross":
            if is_b is None:
                raise ValueError("tensor-parallel cross validation lacks side")
            for label, mask in (("a_to_b", ~is_b), ("b_to_a", is_b)):
                if bool(mask.any()):
                    accumulators[label].update(
                        local_reconstructed[mask],
                        targets[mask],
                        activation_scale=model.activation_scale,
                        mean_predictor_raw=(
                            directional_train_target_means[label]
                            if directional_train_target_means is not None
                            else train_target_mean
                        ),
                        active_counts=local_active[mask],
                    )
    for accumulator in accumulators.values():
        accumulator.all_reduce_(group=model.group)
    macro = accumulators["all"].summary().as_dict()
    metrics = prefix_metrics("validation/batchtopk", macro)
    for key in (
        "samples",
        "elements",
        "raw_mse",
        "mean_predictor_mse",
        "normalized_mse",
        "nmse",
        "fve",
        "scaled_objective",
        "effective_l0",
        "zero_code_fraction",
    ):
        metrics[f"validation/{key}"] = macro[key]
    if mode == "cross":
        for label in ("a_to_b", "b_to_a"):
            metrics.update(
                prefix_metrics(
                    f"validation/{label}",
                    accumulators[label].summary().as_dict(),
                )
            )
    return metrics


def train_tensor_parallel_mode(
    args: argparse.Namespace,
    mode: str,
    rank: int,
    world_size: int,
    local_rank: int,
    hidden_size: int,
    train_manifest: Mapping[str, Any],
    train_cache_digest: str,
    validation_manifest: Mapping[str, Any] | None,
    validation_cache_digest: str | None,
    run_id: str,
) -> None:
    """Feature-sharded numerical-equivalence training backend.

    This backend writes full best/final SAE weights for evaluation and a
    rank-sharded resumable latest checkpoint.
    """

    if args.auxk_alpha > 0:
        raise ValueError(
            "feature-tensor parallelism does not implement the exact global "
            "AuxK candidate selection; use --parallelism ddp or set --auxk-alpha 0"
        )
    exact_occurrences = validate_exact_training_coverage(
        train_manifest,
        world_size=world_size,
        global_batch_size=args.global_batch_size,
        steps=args.steps,
    )
    per_rank_batch = args.global_batch_size // world_size
    device = _device(local_rank)
    allow_legacy_v1 = not cache_is_v2(train_manifest)
    mode_dir = Path(args.output_dir) / mode
    latest_path = mode_dir / "checkpoints" / "latest-tensor-parallel"
    resume_path = (
        Path(args.resume_from) / mode / "checkpoints" / "latest-tensor-parallel"
        if args.resume_from
        else latest_path
    )
    resume_mode = bool(args.resume_from) or (
        args.resume and (resume_path / "checkpoint_manifest.json").is_file()
    )
    if rank == 0:
        if mode_dir.exists() and not resume_mode:
            if args.overwrite_output:
                shutil.rmtree(mode_dir)
            elif any(mode_dir.iterdir()):
                raise FileExistsError(mode_dir)
        mode_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    init_loader = OccurrenceLoaderAdapter(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        world_size=world_size,
        batch_size=per_rank_batch,
        seed=args.seed + 101,
        device=str(device),
        mode=mode,
        direction_policy=args.direction_policy,
        finite=True,
        prefetch_shards=1,
        prefetch_workers=1,
        prefetch_batches=1,
        materialize_shards=args.loader_materialize_shards,
        pin_memory=args.loader_pin_memory,
        local_cache_dir=args.local_cache_dir,
        gpu_shards=min(2, args.loader_gpu_shards),
    )
    estimated_scale, sampled_input_mean, sampled_target_mean = (
        estimate_scale_and_train_mean(
        mode,
        _iter_limited(
            init_loader.iter_epoch(0),
            _global_limit_to_local(args.normalization_samples, world_size),
        ),
        hidden_size,
        device,
        allow_legacy_v1=allow_legacy_v1,
        )
    )
    scale = estimated_scale if args.normalize_activations else 1.0
    target_mode = _cross_target_mode(args.direction_policy) if mode == "cross" else mode
    target_mean = cache_target_mean(
        train_manifest,
        target_mode,
        hidden_size=hidden_size,
        device=device,
    )
    if target_mean is None:
        target_mean = sampled_target_mean
    directional_target_means = None
    if mode == "cross":
        a_to_b = cache_target_mean(
            train_manifest,
            "cross_a_to_b",
            hidden_size=hidden_size,
            device=device,
        )
        b_to_a = cache_target_mean(
            train_manifest,
            "cross_b_to_a",
            hidden_size=hidden_size,
            device=device,
        )
        if a_to_b is not None and b_to_a is not None:
            directional_target_means = {
                "a_to_b": a_to_b,
                "b_to_a": b_to_a,
            }

    model = FeatureTensorParallelSAE(
        hidden_size,
        args.dict_size,
        args.k,
        batch_topk_candidate_multiplier=args.batch_topk_candidate_multiplier,
        decoder_backend=args.decoder_backend,
    ).to(device)
    model.initialize_from_global_seed(args.seed)
    model.activation_scale.fill_(scale)
    input_mean = (
        cache_input_mean(
            train_manifest,
            mode,
            hidden_size=hidden_size,
            device=device,
        )
        if not (mode == "cross" and args.direction_policy != "both")
        else None
    )
    if input_mean is None:
        input_mean = sampled_input_mean
    model.pre_bias.copy_(input_mean * scale)
    model.decoder_bias.data.copy_(target_mean * scale)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        fused=bool(args.fused_adam and device.type == "cuda"),
    )
    config = _mode_config(
        args,
        mode=mode,
        hidden_size=hidden_size,
        scale=scale,
        train_cache_digest=train_cache_digest,
        train_cache_format=str(train_manifest.get("format", "unknown")),
        validation_cache_digest=validation_cache_digest,
        run_id=run_id,
    )
    config["parallelism"] = {
        "type": "feature_tensor",
        "world_size": world_size,
        "features_per_rank": model.local_dict_size,
    }
    config["train_target_mean_source"] = (
        "exact_full_training_cache"
        if cache_target_mean(
            train_manifest,
            mode,
            hidden_size=hidden_size,
            device=device,
        )
        is not None
        else "normalization_sample_estimate"
    )
    config["train_input_mean_source"] = (
        "exact_full_training_cache_mean_inputs"
        if cache_input_mean(
            train_manifest,
            mode,
            hidden_size=hidden_size,
            device=device,
        )
        is not None
        and mode == "cross"
        else "exact_full_training_cache"
        if cache_input_mean(
            train_manifest,
            mode,
            hidden_size=hidden_size,
            device=device,
        )
        is not None
        else "normalization_sample_estimate"
    )
    config["train_input_mean_mode"] = (
        "mean"
        if mode == "cross"
        else "token"
        if mode == "temporal"
        else mode
    )
    start_step = 0
    samples_seen = 0
    accepted_seen = 0
    total_filtered = 0
    total_balanced = 0
    best_value = None
    best_step = None
    if resume_mode:
        resume_state = load_sharded_training_checkpoint(
            model=model,
            optimizer=optimizer,
            input_dir=resume_path,
        )
        start_step = int(resume_state["step"])
        samples_seen = int(
            resume_state.get(
                "samples_seen",
                start_step * args.global_batch_size,
            )
        )
        accepted_seen = int(
            resume_state.get("accepted_training_rows", samples_seen)
        )
        total_filtered = int(
            resume_state.get("activation_norm_filtered_rows", 0)
        )
        total_balanced = int(
            resume_state.get("activation_norm_balance_dropped_rows", 0)
        )
        best_value = resume_state.get("best_metric_value")
        best_step = resume_state.get("best_step")
    if rank == 0:
        atomic_json_dump(config, mode_dir / "run_config.json")

    train_adapter = OccurrenceLoaderAdapter(
        args.activation_cache_dir,
        train_manifest,
        rank=rank,
        world_size=world_size,
        batch_size=per_rank_batch,
        seed=args.seed + 211,
        device=str(device),
        mode=mode,
        direction_policy=args.direction_policy,
        finite=True,
        prefetch_shards=args.loader_prefetch_shards,
        prefetch_workers=args.loader_prefetch_workers,
        prefetch_batches=args.loader_prefetch_batches,
        materialize_shards=args.loader_materialize_shards,
        pin_memory=args.loader_pin_memory,
        local_cache_dir=args.local_cache_dir,
        gpu_shards=args.loader_gpu_shards,
    )
    train_loader = iter(
        train_adapter.iter_epoch(
            0,
            start_offset=start_step * per_rank_batch,
        )
    )

    def validation_loader(global_samples: int) -> Iterator[Any]:
        if validation_manifest is None or args.validation_cache_dir is None:
            return iter(())
        adapter = OccurrenceLoaderAdapter(
            args.validation_cache_dir,
            validation_manifest,
            rank=rank,
            world_size=world_size,
            batch_size=per_rank_batch,
            seed=args.seed + 307,
            device=str(device),
            mode=mode,
            finite=True,
            prefetch_shards=2,
            prefetch_workers=2,
            prefetch_batches=args.loader_prefetch_batches,
            materialize_shards=args.loader_materialize_shards,
            pin_memory=args.loader_pin_memory,
            local_cache_dir=args.local_cache_dir,
            gpu_shards=min(4, args.loader_gpu_shards),
        )
        return _iter_limited(
            adapter.iter_epoch(0),
            _global_limit_to_local(global_samples, world_size),
        )

    log_path = mode_dir / "metrics.jsonl"
    writer = TensorBoardLogger(
        (
            Path(args.tensorboard_dir) / run_id / mode
            if args.tensorboard_dir
            else None
        ),
        rank=rank,
        flush_secs=args.tensorboard_flush_secs,
        max_queue=args.tensorboard_max_queue,
        scalar_allowlist=CORE_TENSORBOARD_SCALARS,
        mode=mode,
        fidelity_reference_fve=args.fidelity_reference_fves.get(mode),
        gradient_clip=args.gradient_clip,
    )
    running = ReconstructionMetricAccumulator(device)
    best_tensors = None
    if resume_mode and rank == 0:
        best_weights = mode_dir / "checkpoints" / "best" / "sae.safetensors"
        if best_weights.is_file():
            from safetensors.torch import load_file

            best_tensors = load_file(str(best_weights), device="cpu")
    threshold_initialized = bool(model.threshold.detach().cpu().item() >= 0)
    autocast_dtype = dtype_from_name(args.autocast_dtype)
    started = time.time()
    try:
        for step_index in range(start_step, args.steps):
            batch = next(train_loader)
            inputs, targets, _ = select_occurrence_view(
                mode,
                batch,
                allow_legacy_v1=allow_legacy_v1,
            )
            valid, _, _ = activation_norm_ok_mask(
                inputs,
                targets,
                args.max_activation_norm_multiple,
            )
            local_filtered = inputs.shape[0] - int(valid.sum())
            valid, accepted_per_rank, balanced = equalize_valid_rows(valid)
            counts = torch.tensor(
                [local_filtered, balanced],
                dtype=torch.int64,
                device=device,
            )
            if dist.is_initialized():
                dist.all_reduce(counts)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, autocast_dtype):
                (
                    reconstructed,
                    _features,
                    batch_threshold,
                    active_counts,
                    active_per_feature,
                ) = model(
                    inputs,
                    batch_topk=True,
                    return_activity_counts=True,
                    sample_mask=valid,
                )
                global_targets = all_gather_rows(targets, group=model.group)
                global_valid = all_gather_rows(
                    valid.to(torch.uint8),
                    group=model.group,
                ).bool()
                scaled_targets = global_targets * model.activation_scale.to(
                    global_targets.dtype
                )
                residual = reconstructed.float() - scaled_targets.float()
                primary_loss = residual[global_valid].square().sum(dim=-1).mean()
                auxiliary_loss = primary_loss.new_zeros(())
                loss = primary_loss
            loss.backward()
            model.remove_parallel_decoder_gradient_()
            grad_norm = _clip_tensor_parallel_grad_norm_(
                model,
                args.gradient_clip,
            )
            multiplier = learning_rate_multiplier(
                step_index,
                args.steps,
                args.warmup_steps,
                args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            optimizer.step()
            model.normalize_decoder_()
            accepted_global = accepted_per_rank * world_size
            with torch.no_grad():
                if not threshold_initialized:
                    model.threshold.copy_(batch_threshold.float())
                    threshold_initialized = True
                else:
                    model.threshold.mul_(args.threshold_beta).add_(
                        batch_threshold.float(),
                        alpha=1 - args.threshold_beta,
                    )
                model.feature_counts.add_(active_per_feature)
                model.update_dead_feature_stats_(
                    active_per_feature,
                    global_samples=accepted_global,
                    distributed=False,
                )
                start = rank * inputs.shape[0]
                stop = start + inputs.shape[0]
                local_residual = residual[start:stop][valid]
                scaled_sse = local_residual.square().sum()
                raw_sse = scaled_sse / (scale * scale)
                baseline = (
                    targets[valid].float() - target_mean.float()
                ).square().sum()
                running.update_sufficient_statistics(
                    scaled_sse=scaled_sse,
                    raw_sse=raw_sse,
                    mean_predictor_sse=baseline,
                    samples=accepted_per_rank,
                    elements=accepted_per_rank * hidden_size,
                    active_counts=active_counts[start:stop][valid],
                )
            samples_seen += args.global_batch_size
            accepted_seen += accepted_global
            total_filtered += int(counts[0])
            total_balanced += int(counts[1])
            completed = step_index + 1
            if (
                completed == 1
                or completed % args.log_every == 0
                or completed == args.steps
            ):
                running.all_reduce_(group=model.group)
                summary = running.summary().as_dict()
                metrics = {
                    "train/raw_mse": summary["raw_mse"],
                    "train/mean_predictor_mse": summary["mean_predictor_mse"],
                    "train/normalized_mse": summary["normalized_mse"],
                    "train/fve": summary["fve"],
                    "train/effective_l0": summary["effective_l0"],
                    "train/auxiliary_loss": float(auxiliary_loss.detach().item()),
                    "sparsity/dead_features": model.dead_feature_count(
                        args.dead_feature_threshold
                    ),
                    "sparsity/dead_fraction": model.dead_feature_count(
                        args.dead_feature_threshold
                    ) / max(1, args.dict_size),
                    "optimizer/grad_norm": float(grad_norm),
                    "optimizer/lr": optimizer.param_groups[0]["lr"],
                    "optimizer/updates": completed,
                    "progress/token_occurrences_seen": samples_seen,
                    "system/elapsed_seconds": time.time() - started,
                    "system/samples_per_second": samples_seen
                    / max(time.time() - started, 1e-9),
                    "progress/unique_occurrence_coverage": samples_seen
                    / exact_occurrences,
                }
                if rank == 0:
                    row = {
                        "run_id": run_id,
                        "mode": mode,
                        "split": "train",
                        "step": completed,
                        **metrics,
                    }
                    _write_jsonl(log_path, row)
                    writer.add_scalars(metrics, completed)
                running.reset()
            should_validate = (
                validation_manifest is not None
                and (
                    (
                        args.validate_every > 0
                        and completed % args.validate_every == 0
                    )
                    or completed == args.steps
                )
            )
            if should_validate:
                metrics = _validate_tensor_parallel_model(
                    model=model,
                    mode=mode,
                    loader=validation_loader(args.validation_samples),
                    train_target_mean=target_mean,
                    allow_legacy_v1=not cache_is_v2(validation_manifest),
                    directional_train_target_means=directional_target_means,
                )
                current = _best_value(metrics, args.best_metric)
                improved = _is_better(current, best_value, args.best_metric)
                if improved:
                    best_value = current
                    best_step = completed
                    best_tensors = model.gather_checkpoint_tensors()
                else:
                    # gather_checkpoint_tensors is collective only on improvement.
                    pass
                if rank == 0:
                    _write_jsonl(
                        log_path,
                        {
                            "run_id": run_id,
                            "mode": mode,
                            "split": "validation",
                            "step": completed,
                            "is_best": improved,
                            **metrics,
                        },
                    )
                    writer.add_scalars(metrics, completed)

            if (
                (args.save_every > 0 and completed % args.save_every == 0)
                or completed == args.steps
            ):
                save_sharded_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    output_dir=latest_path,
                    state={
                        "format": "chunk-saes-feature-tensor-state-v2",
                        "step": completed,
                        "samples_seen": samples_seen,
                        "accepted_training_rows": accepted_seen,
                        "activation_norm_filtered_rows": total_filtered,
                        "activation_norm_balance_dropped_rows": total_balanced,
                        "best_metric_value": best_value,
                        "best_step": best_step,
                        "config": config,
                    },
                )

        if samples_seen != exact_occurrences:
            raise RuntimeError("tensor-parallel exact coverage is incomplete")
        final_validation = None
        if validation_manifest is not None:
            final_validation = _validate_tensor_parallel_model(
                model=model,
                mode=mode,
                loader=validation_loader(args.final_validation_samples),
                train_target_mean=target_mean,
                allow_legacy_v1=not cache_is_v2(validation_manifest),
                directional_train_target_means=directional_target_means,
            )
        full_tensors = model.gather_checkpoint_tensors()
        if best_tensors is None:
            best_tensors = full_tensors
            best_step = args.steps
            best_value = float("nan")
        dead_features = model.dead_feature_count(args.dead_feature_threshold)
        if rank == 0:
            assert full_tensors is not None and best_tensors is not None
            save_inference_checkpoint(
                best_tensors,
                mode_dir / "checkpoints" / "best",
                {
                    "step": best_step,
                    "samples_seen": samples_seen,
                    "world_size": world_size,
                    "best_metric_value": best_value,
                    "best_step": best_step,
                    "config": config,
                },
            )
            from safetensors.torch import save_file

            save_file(full_tensors, str(mode_dir / "sae.safetensors"))
            final_config = {
                **config,
                "steps_completed": args.steps,
                "samples_seen": samples_seen,
                "accepted_training_rows": accepted_seen,
                "activation_norm_filtered_rows": total_filtered,
                "activation_norm_balance_dropped_rows": total_balanced,
                "unique_occurrences_seen": exact_occurrences,
                "coverage_fraction": 1.0,
                "best_metric_value": best_value,
                "best_step": best_step,
                "final_full_validation_metrics": final_validation,
            }
            atomic_json_dump(final_config, mode_dir / "config.json")
            atomic_json_dump(
                {
                    "complete": True,
                    "run_id": run_id,
                    "mode": mode,
                    "steps": args.steps,
                    "samples_seen": samples_seen,
                    "accepted_training_rows": accepted_seen,
                    "activation_norm_filtered_rows": total_filtered,
                    "activation_norm_balance_dropped_rows": total_balanced,
                    "unique_occurrences_seen": exact_occurrences,
                    "coverage_fraction": 1.0,
                    "exact_coverage": True,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_value,
                    "best_step": best_step,
                    "alive_features": int(
                        (full_tensors["feature_counts"] > 0).sum()
                    ),
                    "dead_features": dead_features,
                    "dead_feature_threshold": args.dead_feature_threshold,
                    "train_cache_digest": train_cache_digest,
                    "validation_cache_digest": validation_cache_digest,
                    "parallelism": "feature_tensor",
                    "final_full_validation_metrics": final_validation,
                },
                mode_dir / "complete.json",
            )
        if dist.is_initialized():
            dist.barrier()
    finally:
        writer.close()
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parser().parse_args()
    if args.target_granularity != "mean":
        raise ValueError(
            "--target-granularity sequence is implemented by "
            "src/train_sequence_cross_ablation.py"
        )
    args.fidelity_reference_fves = parse_fidelity_reference_fves(
        args.fidelity_reference_fves
    )
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be in [0, 1]")
    if args.auxk_alpha > 0:
        if args.auxk_activation_age <= 0:
            raise ValueError("auxk-activation-age must be positive when AuxK is enabled")
        if args.auxk_activation_age > args.dead_feature_threshold:
            raise ValueError(
                "auxk-activation-age must not exceed dead-feature-threshold"
            )
    preliminary_rank, preliminary_world_size, preliminary_local_rank = (
        distributed_info()
    )
    assigned_cpus = None
    if args.bind_cpu_affinity:
        assigned_cpus = bind_local_rank_cpu_affinity(
            local_rank=preliminary_local_rank,
            local_world_size=int(
                os.environ.get("LOCAL_WORLD_SIZE", preliminary_world_size)
            ),
        )
    configured_threads = configure_cpu_threads(default=4)
    rank, world_size, local_rank = initialize_distributed()
    if (
        rank,
        world_size,
        local_rank,
    ) != (
        preliminary_rank,
        preliminary_world_size,
        preliminary_local_rank,
    ):
        raise RuntimeError("distributed rank metadata changed during initialization")
    torch.set_float32_matmul_precision("high")
    try:
        train_manifest, train_digest = load_cache_manifest(args.activation_cache_dir)
        validation_manifest = None
        validation_digest = None
        if args.validation_cache_dir:
            validation_manifest, validation_digest = load_cache_manifest(
                args.validation_cache_dir
            )
        validate_cache_compatibility(train_manifest, validation_manifest)
        hidden_size = cache_hidden_size(train_manifest)
        if rank == 0:
            log(
                f"runtime CPU threads={torch.get_num_threads()} "
                f"rank0_affinity={assigned_cpus}",
                rank=rank,
            )
        if rank == 0:
            run_id = _new_run_id(args)
        else:
            run_id = None
        run_id = _broadcast_object(run_id, rank)
        modes = parse_csv(args.modes)
        invalid = set(modes) - {"token", "temporal", "mean", "cross", "joint_chunk"}
        if invalid:
            raise ValueError(f"unsupported modes: {sorted(invalid)}")
        if "joint_chunk" in modes:
            if len(modes) != 1:
                raise ValueError("joint_chunk must be trained as its own mode")
            if args.parallelism != "ddp":
                raise ValueError("joint_chunk currently requires --parallelism ddp")
            train_joint_chunk_mode(
                args,
                rank,
                world_size,
                local_rank,
                hidden_size,
                train_manifest,
                train_digest,
                validation_manifest,
                validation_digest,
                str(run_id),
            )
            return
        if args.parallelism == "feature-tensor" and "temporal" in modes:
            raise ValueError(
                "Temporal SAE currently requires --parallelism ddp because its "
                "high-prefix Matryoshka reconstruction and contrastive objective "
                "operate on the complete shared dictionary."
            )
        if args.parallelism == "feature-tensor":
            for mode in modes:
                train_tensor_parallel_mode(
                    args,
                    mode,
                    rank,
                    world_size,
                    local_rank,
                    hidden_size,
                    train_manifest,
                    train_digest,
                    validation_manifest,
                    validation_digest,
                    str(run_id),
                )
        elif args.joint_modes and len(modes) > 1 and not (
            args.resume or args.resume_from
        ):
            train_modes_joint(
                args,
                modes,
                rank,
                world_size,
                local_rank,
                hidden_size,
                train_manifest,
                train_digest,
                validation_manifest,
                validation_digest,
                str(run_id),
            )
        else:
            for mode in modes:
                train_mode(
                    args,
                    mode,
                    rank,
                    world_size,
                    local_rank,
                    hidden_size,
                    train_manifest,
                    train_digest,
                    validation_manifest,
                    validation_digest,
                    str(run_id),
                )
    except BaseException:
        # Emit the rank-local failure before process-group teardown.  A peer
        # can still be inside a collective when one rank fails, and NCCL
        # teardown may then block long enough to hide the original traceback.
        # Printing here preserves the actionable exception in the job log.
        import traceback

        traceback.print_exc()
        raise
    finally:
        # train_mode() already synchronizes successful mode completion. A
        # barrier here masks rank-local exceptions and leaves peers waiting for
        # the NCCL watchdog instead of letting torchrun fail the job promptly.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
