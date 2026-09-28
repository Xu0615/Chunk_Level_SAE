#!/usr/bin/env python
"""Train a provenance-tracked dense reference predictor for Cross-Chunk RFVE.

The reference deliberately receives exactly the same information as the
Cross-Chunk SAE: one owning chunk mean. A single direction-blind predictor is
shared across A→B and B→A examples. Validation selects capacity/checkpoint;
the document-disjoint test cache is evaluated exactly once after selection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from chunk_saes.artifacts import file_record, write_artifact_manifest
from chunk_saes.reference_fve import REFERENCE_MANIFEST_FORMAT
from chunk_saes.utils import atomic_json_dump, parse_int_csv, seed_everything
from train_chunk_saes import (
    OccurrenceLoaderAdapter,
    cache_hidden_size,
    cache_target_mean,
    load_cache_manifest,
    validate_cache_compatibility,
)


RESULT_FORMAT = "chunk-saes-cross-dense-reference-results-v1"
IMPLEMENTATION_VERSION = "direction-blind-residual-mlp-v1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fit a direction-blind dense Cross-Chunk predictor and publish a "
            "capacity sweep plus one independent-test FVE."
        )
    )
    p.add_argument("--train-cache-dir", required=True)
    p.add_argument("--validation-cache-dir", required=True)
    p.add_argument("--test-cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hidden-widths", default="1024,2048,4096")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument(
        "--activation-scale",
        type=float,
        default=None,
        help=(
            "Optional positive scalar used only for numerical conditioning. "
            "FVE is scale invariant. Publication runs should pass the Cross SAE "
            "activation_scale so teacher and SAE optimization use the same scale."
        ),
    )
    p.add_argument("--selection-tolerance", type=float, default=0.002)
    p.add_argument("--monitor-samples", type=int, default=262144)
    p.add_argument("--monitor-world-size", type=int, default=8)
    p.add_argument("--monitor-seed", type=int, default=349)
    p.add_argument("--overwrite", action="store_true")
    return p


class DenseCrossReference(nn.Module):
    """Residual MLP with no sparsity bottleneck and no direction input."""

    def __init__(
        self,
        activation_dim: int,
        hidden_width: int,
        *,
        input_mean: torch.Tensor,
        target_mean: torch.Tensor,
        activation_scale: float,
    ) -> None:
        super().__init__()
        self.activation_dim = int(activation_dim)
        self.hidden_width = int(hidden_width)
        self.register_buffer("input_mean", input_mean.float().clone())
        self.register_buffer("target_mean", target_mean.float().clone())
        self.register_buffer(
            "activation_scale",
            torch.tensor(float(activation_scale), dtype=torch.float32),
        )
        self.norm = nn.LayerNorm(self.activation_dim, elementwise_affine=False)
        self.skip = nn.Linear(self.activation_dim, self.activation_dim, bias=False)
        self.up = nn.Linear(self.activation_dim, self.hidden_width)
        self.down = nn.Linear(self.hidden_width, self.activation_dim)
        nn.init.eye_(self.skip.weight)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.activation_scale.to(dtype=x.dtype)
        centered = x * scale - self.input_mean.to(dtype=x.dtype) * scale
        return (
            self.target_mean.to(dtype=x.dtype) * scale
            + self.skip(centered)
            + self.down(F.silu(self.up(self.norm(centered))))
        )


def _rank_shards(cache_dir: Path, manifest: Mapping[str, Any]) -> list[Path]:
    return [
        cache_dir / str(shard["path"])
        for rank in manifest["ranks"]
        for shard in rank["shards"]
    ]


def _load_pair_shard(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        mean_a = handle.get_tensor("mean_a").clone()
        mean_b = handle.get_tensor("mean_b").clone()
        length_a = handle.get_tensor("length_a").float().clone()
        length_b = handle.get_tensor("length_b").float().clone()
    return mean_a, mean_b, length_a, length_b


def _autocast(device: torch.device, dtype: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and dtype == "bfloat16",
    )


def _epoch_shards(
    paths: list[Path],
    *,
    seed: int,
    epoch: int,
) -> Iterator[Path]:
    order = list(paths)
    random.Random(seed + epoch).shuffle(order)
    yield from order


def train_epoch(
    model: DenseCrossReference,
    optimizer: torch.optim.Optimizer,
    shard_paths: list[Path],
    *,
    device: torch.device,
    dtype: str,
    batch_size: int,
    gradient_clip: float,
    seed: int,
    epoch: int,
) -> dict[str, float | int]:
    model.train()
    loss_sum = 0.0
    updates = 0
    pairs = 0
    started = time.time()
    for shard_index, path in enumerate(
        _epoch_shards(shard_paths, seed=seed, epoch=epoch),
        start=1,
    ):
        mean_a, mean_b, length_a, length_b = _load_pair_shard(path)
        generator = torch.Generator().manual_seed(
            seed + epoch * 1_000_003 + shard_index
        )
        order = torch.randperm(mean_a.shape[0], generator=generator)
        for start in range(0, mean_a.shape[0], batch_size):
            indices = order[start : start + batch_size]
            a = mean_a.index_select(0, indices).to(device)
            b = mean_b.index_select(0, indices).to(device)
            weights_a = length_a.index_select(0, indices).to(device)
            weights_b = length_b.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, dtype):
                predicted_b = model(a)
                predicted_a = model(b)
                scale = model.activation_scale.to(dtype=a.dtype)
                losses = (
                    weights_a
                    * (predicted_b - b * scale).float().square().sum(dim=1)
                    + weights_b
                    * (predicted_a - a * scale).float().square().sum(dim=1)
                )
                loss = losses.sum() / (weights_a + weights_b).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            loss_sum += float(loss.detach())
            updates += 1
            pairs += int(indices.numel())
    return {
        "epoch": epoch + 1,
        "pairs": pairs,
        "updates": updates,
        "mean_scaled_objective": loss_sum / max(1, updates),
        "elapsed_seconds": time.time() - started,
    }


@torch.no_grad()
def evaluate(
    model: DenseCrossReference,
    shard_paths: list[Path],
    *,
    device: torch.device,
    dtype: str,
    batch_size: int,
    directional_target_means: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    totals = {
        "pooled": [0.0, 0.0, 0],
        "a_to_b": [0.0, 0.0, 0],
        "b_to_a": [0.0, 0.0, 0],
    }
    scale = float(model.activation_scale)
    pooled_mean = model.target_mean.to(device=device, dtype=torch.float32)
    a_to_b_mean = directional_target_means["a_to_b"].to(device)
    b_to_a_mean = directional_target_means["b_to_a"].to(device)
    for path in shard_paths:
        mean_a, mean_b, length_a, length_b = _load_pair_shard(path)
        for start in range(0, mean_a.shape[0], batch_size):
            a = mean_a[start : start + batch_size].to(device)
            b = mean_b[start : start + batch_size].to(device)
            weights_a = length_a[start : start + batch_size].to(device)
            weights_b = length_b[start : start + batch_size].to(device)
            with _autocast(device, dtype):
                predicted_b = model(a)
                predicted_a = model(b)
            residual_a = (predicted_b.float() - b.float() * scale).square().sum(1)
            residual_b = (predicted_a.float() - a.float() * scale).square().sum(1)
            baseline_pooled_a = (
                (b.float() - pooled_mean).square().sum(1) * scale * scale
            )
            baseline_pooled_b = (
                (a.float() - pooled_mean).square().sum(1) * scale * scale
            )
            baseline_a = (
                (b.float() - a_to_b_mean).square().sum(1) * scale * scale
            )
            baseline_b = (
                (a.float() - b_to_a_mean).square().sum(1) * scale * scale
            )
            weighted_residual_a = float((weights_a * residual_a).sum())
            weighted_residual_b = float((weights_b * residual_b).sum())
            totals["pooled"][0] += weighted_residual_a + weighted_residual_b
            totals["pooled"][1] += float(
                (weights_a * baseline_pooled_a).sum()
                + (weights_b * baseline_pooled_b).sum()
            )
            totals["pooled"][2] += int((weights_a + weights_b).sum())
            totals["a_to_b"][0] += weighted_residual_a
            totals["a_to_b"][1] += float((weights_a * baseline_a).sum())
            totals["a_to_b"][2] += int(weights_a.sum())
            totals["b_to_a"][0] += weighted_residual_b
            totals["b_to_a"][1] += float((weights_b * baseline_b).sum())
            totals["b_to_a"][2] += int(weights_b.sum())
    result: dict[str, Any] = {}
    for label, (sse, baseline_sse, occurrences) in totals.items():
        result[f"{label}_fve"] = 1.0 - sse / baseline_sse
        result[f"{label}_nmse"] = sse / baseline_sse
        result[f"{label}_occurrences"] = occurrences
    return result


@torch.no_grad()
def evaluate_monitor_subset(
    model: DenseCrossReference,
    cache_dir: Path,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: str,
    batch_size: int,
    global_samples: int,
    world_size: int,
    seed: int,
    directional_target_means: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Evaluate the exact fixed subset used by periodic SAE validation."""

    if global_samples <= 0 or global_samples % world_size:
        raise ValueError("monitor samples must be positive and divisible by world size")
    local_limit = global_samples // world_size
    model.eval()
    scale = float(model.activation_scale)
    pooled_mean = model.target_mean.to(device=device, dtype=torch.float32)
    a_to_b_mean = directional_target_means["a_to_b"].to(device)
    b_to_a_mean = directional_target_means["b_to_a"].to(device)
    totals = {
        "pooled": [0.0, 0.0, 0],
        "a_to_b": [0.0, 0.0, 0],
        "b_to_a": [0.0, 0.0, 0],
    }
    for rank in range(world_size):
        adapter = OccurrenceLoaderAdapter(
            cache_dir,
            manifest,
            rank=rank,
            world_size=world_size,
            batch_size=min(batch_size, local_limit),
            seed=seed,
            device=str(device),
            mode="cross",
            finite=True,
            prefetch_shards=1,
            prefetch_workers=1,
            prefetch_batches=0,
            materialize_shards=False,
            pin_memory=False,
            gpu_shards=0,
        )
        seen = 0
        iterator = adapter.iter_epoch(0)
        try:
            for batch in iterator:
                remaining = local_limit - seen
                if remaining <= 0:
                    break
                rows = int(batch["chunk_mean"].shape[0])
                if rows > remaining:
                    batch = {
                        key: value[:remaining]
                        for key, value in batch.items()
                    }
                    rows = remaining
                owning = batch["chunk_mean"]
                partner = batch["partner_mean"]
                is_b = batch["side"].bool()
                with _autocast(device, dtype):
                    prediction = model(owning)
                residual = (
                    prediction.float() - partner.float() * scale
                ).square().sum(1)
                pooled_baseline = (
                    (partner.float() - pooled_mean).square().sum(1)
                    * scale
                    * scale
                )
                totals["pooled"][0] += float(residual.sum())
                totals["pooled"][1] += float(pooled_baseline.sum())
                totals["pooled"][2] += rows
                for label, mask, baseline_mean in (
                    ("a_to_b", ~is_b, a_to_b_mean),
                    ("b_to_a", is_b, b_to_a_mean),
                ):
                    if not bool(mask.any()):
                        continue
                    totals[label][0] += float(residual[mask].sum())
                    totals[label][1] += float(
                        (
                            partner[mask].float() - baseline_mean
                        ).square().sum()
                        * scale
                        * scale
                    )
                    totals[label][2] += int(mask.sum())
                seen += rows
                if seen >= local_limit:
                    break
        finally:
            iterator.close()
        if seen != local_limit:
            raise ValueError(
                f"monitor rank {rank} yielded {seen} rows, expected {local_limit}"
            )
    result: dict[str, Any] = {}
    for label, (sse, baseline_sse, occurrences) in totals.items():
        result[f"{label}_fve"] = 1.0 - sse / baseline_sse
        result[f"{label}_nmse"] = sse / baseline_sse
        result[f"{label}_occurrences"] = occurrences
    return result


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }


def main() -> None:
    args = parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    widths = sorted(set(parse_int_csv(args.hidden_widths)))
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("--hidden-widths must contain positive integers")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.train_cache_dir).resolve()
    validation_dir = Path(args.validation_cache_dir).resolve()
    test_dir = Path(args.test_cache_dir).resolve()
    train_manifest, train_digest = load_cache_manifest(train_dir)
    validation_manifest, validation_digest = load_cache_manifest(validation_dir)
    test_manifest, test_digest = load_cache_manifest(test_dir)
    validate_cache_compatibility(train_manifest, validation_manifest)
    validate_cache_compatibility(train_manifest, test_manifest)
    if (validation_manifest.get("hf_corpus") or {}).get("logical_split") != "validation":
        raise ValueError("validation cache is not the logical validation split")
    if (test_manifest.get("hf_corpus") or {}).get("logical_split") != "test":
        raise ValueError("test cache is not the logical test split")
    hidden_size = cache_hidden_size(train_manifest)
    device = torch.device(args.device)
    seed_everything(args.seed)

    train_mean = cache_target_mean(
        train_manifest,
        "mean",
        hidden_size=hidden_size,
        device=torch.device("cpu"),
    )
    target_mean = cache_target_mean(
        train_manifest,
        "cross",
        hidden_size=hidden_size,
        device=torch.device("cpu"),
    )
    a_to_b_mean = cache_target_mean(
        train_manifest,
        "cross_a_to_b",
        hidden_size=hidden_size,
        device=torch.device("cpu"),
    )
    b_to_a_mean = cache_target_mean(
        train_manifest,
        "cross_b_to_a",
        hidden_size=hidden_size,
        device=torch.device("cpu"),
    )
    if any(
        value is None
        for value in (train_mean, target_mean, a_to_b_mean, b_to_a_mean)
    ):
        raise ValueError("training cache lacks exact Cross target means")
    assert train_mean is not None and target_mean is not None
    assert a_to_b_mean is not None and b_to_a_mean is not None
    if args.activation_scale is not None:
        activation_scale = float(args.activation_scale)
        if not math.isfinite(activation_scale) or activation_scale <= 0:
            raise ValueError("--activation-scale must be finite and positive")
        activation_scale_source = "explicit_cli"
    else:
        # The scalar cancels from FVE; this deterministic sample estimate is
        # solely for BF16 conditioning when no SAE scale is supplied.
        norm_sum = 0.0
        norm_count = 0
        for path in _rank_shards(train_dir, train_manifest)[:8]:
            mean_a, mean_b, length_a, length_b = _load_pair_shard(path)
            norm_sum += float((length_a * mean_a.float().norm(dim=1)).sum())
            norm_sum += float((length_b * mean_b.float().norm(dim=1)).sum())
            norm_count += int((length_a + length_b).sum())
        activation_scale = hidden_size**0.5 / max(
            norm_sum / norm_count,
            1e-12,
        )
        activation_scale_source = "deterministic_first_eight_train_shards"

    train_paths = _rank_shards(train_dir, train_manifest)
    validation_paths = _rank_shards(validation_dir, validation_manifest)
    test_paths = _rank_shards(test_dir, test_manifest)
    metrics_path = output_dir / "training_metrics.jsonl"
    sweep: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    for width in widths:
        seed_everything(args.seed)
        model = DenseCrossReference(
            hidden_size,
            width,
            input_mean=train_mean,
            target_mean=target_mean,
            activation_scale=activation_scale,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        best_fve = -math.inf
        best_epoch = -1
        best_state = None
        history = []
        for epoch in range(args.epochs):
            train_metrics = train_epoch(
                model,
                optimizer,
                train_paths,
                device=device,
                dtype=args.dtype,
                batch_size=args.batch_size,
                gradient_clip=args.gradient_clip,
                seed=args.seed,
                epoch=epoch,
            )
            validation_metrics = evaluate(
                model,
                validation_paths,
                device=device,
                dtype=args.dtype,
                batch_size=args.batch_size,
                directional_target_means={
                    "a_to_b": a_to_b_mean,
                    "b_to_a": b_to_a_mean,
                },
            )
            row = {
                "hidden_width": width,
                **train_metrics,
                "validation_full": validation_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            history.append(row)
            if validation_metrics["pooled_fve"] > best_fve:
                best_fve = float(validation_metrics["pooled_fve"])
                best_epoch = epoch + 1
                best_state = _state_dict_cpu(model)
        assert best_state is not None
        snapshots[width] = best_state
        sweep.append(
            {
                "hidden_width": width,
                "best_epoch": best_epoch,
                "validation_pooled_fve": best_fve,
                "validation_full": history[best_epoch - 1]["validation_full"],
            }
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    maximum = max(float(row["validation_pooled_fve"]) for row in sweep)
    selected_row = min(
        (
            row
            for row in sweep
            if float(row["validation_pooled_fve"])
            >= maximum - args.selection_tolerance
        ),
        key=lambda row: int(row["hidden_width"]),
    )
    selected_width = int(selected_row["hidden_width"])
    selected_model = DenseCrossReference(
        hidden_size,
        selected_width,
        input_mean=train_mean,
        target_mean=target_mean,
        activation_scale=activation_scale,
    ).to(device)
    selected_model.load_state_dict(snapshots[selected_width])
    checkpoint_path = checkpoint_dir / "reference.safetensors"
    save_file(_state_dict_cpu(selected_model), str(checkpoint_path))
    validation_monitor_metrics = evaluate_monitor_subset(
        selected_model,
        validation_dir,
        validation_manifest,
        device=device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        global_samples=args.monitor_samples,
        world_size=args.monitor_world_size,
        seed=args.monitor_seed,
        directional_target_means={
            "a_to_b": a_to_b_mean,
            "b_to_a": b_to_a_mean,
        },
    )
    test_metrics = evaluate(
        selected_model,
        test_paths,
        device=device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        directional_target_means={
            "a_to_b": a_to_b_mean,
            "b_to_a": b_to_a_mean,
        },
    )
    config = {
        "format": RESULT_FORMAT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture": "shared direction-blind residual MLP",
        "activation_dim": hidden_size,
        "hidden_widths": widths,
        "selected_hidden_width": selected_width,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "dtype": args.dtype,
        "activation_scale": activation_scale,
        "activation_scale_source": activation_scale_source,
        "selection_tolerance": args.selection_tolerance,
        "monitor_samples": args.monitor_samples,
        "monitor_world_size": args.monitor_world_size,
        "monitor_seed": args.monitor_seed,
        "direction_policy": "shared_direction_blind_bidirectional",
        "occurrence_weighting": (
            "A→B weighted by length_a; B→A weighted by length_b, matching "
            "the Cross-Chunk SAE occurrence objective"
        ),
    }
    config_path = output_dir / "config.json"
    atomic_json_dump(config, config_path)
    results = {
        "format": RESULT_FORMAT,
        "complete": True,
        "capacity_sweep": sweep,
        "selected_model": {
            "hidden_width": selected_width,
            "best_epoch": selected_row["best_epoch"],
            "selection_split": "validation",
            "selection_rule": (
                "smallest hidden width within selection_tolerance of the "
                "best validation pooled FVE"
            ),
        },
        "validation_monitor": validation_monitor_metrics,
        "validation_full": selected_row["validation_full"],
        "test": test_metrics,
        "test_evaluation_count": 1,
        "interpretation": (
            "The test pooled FVE is the empirical same-information dense "
            "reference denominator for Cross-Chunk RFVE."
        ),
    }
    results_path = output_dir / "test_metrics.json"
    atomic_json_dump(results, results_path)
    identity = {
        "mode": "cross",
        "implementation_version": IMPLEMENTATION_VERSION,
        "activation_dim": hidden_size,
        "layer": int(train_manifest["layer"]),
        "model_hash": train_manifest.get("model_hash"),
        "tokenizer_hash": train_manifest.get("tokenizer_hash"),
        "train_cache_digest": train_digest,
        "validation_cache_digest": validation_digest,
        "test_cache_digest": test_digest,
        "train_logical_split": (train_manifest.get("hf_corpus") or {}).get(
            "logical_split"
        ),
        "validation_logical_split": (
            validation_manifest.get("hf_corpus") or {}
        ).get("logical_split"),
        "test_logical_split": (test_manifest.get("hf_corpus") or {}).get(
            "logical_split"
        ),
        "direction_policy": "shared_direction_blind_bidirectional",
        "seed": args.seed,
    }
    manifest = write_artifact_manifest(
        {
            "format": REFERENCE_MANIFEST_FORMAT,
            "complete": True,
            "identity": identity,
            "files": {
                "config": file_record(config_path, relative_to=output_dir),
                "checkpoint": file_record(
                    checkpoint_path,
                    relative_to=output_dir,
                ),
                "training_metrics": file_record(
                    metrics_path,
                    relative_to=output_dir,
                ),
                "results": file_record(results_path, relative_to=output_dir),
            },
        },
        output_dir / "manifest.json",
    )
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "artifact_digest": manifest["artifact_digest"],
                "selected_hidden_width": selected_width,
                "test": test_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
