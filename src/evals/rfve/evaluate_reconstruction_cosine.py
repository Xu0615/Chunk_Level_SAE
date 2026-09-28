#!/usr/bin/env python
"""Evaluate input-output reconstruction cosine for frozen SAE checkpoints.

This is a post-training evaluation.  It loads validation-selected checkpoints
and reruns only the fixed held-out validation subset; no SAE parameters are
updated and no training job is resumed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from matplotlib.ticker import PercentFormatter

from chunk_saes.artifacts import file_record, write_artifact_manifest
from chunk_saes.plot_style import METHOD_COLORS, style_figure_text
from chunk_saes.sae import JointChunkSAE, load_sae
from chunk_saes.utils import atomic_json_dump
from train_chunk_saes import (
    OccurrenceLoaderAdapter,
    _global_limit_to_local,
    _iter_limited,
    cache_is_v2,
    chunk_input_deduplication,
    select_joint_chunk_view,
    select_occurrence_view,
)


RESULT_FORMAT = "chunk-saes-reconstruction-cosine-v1"
METHOD_ORDER = (
    "token",
    "temporal",
    "mean",
    "joint_alpha0p25",
    "joint_alpha0p5",
    "joint_alpha1",
    "joint_alpha1p5",
    "cross",
)
BASE_MODES = frozenset({"token", "temporal", "mean", "cross"})
SHORT_LABELS = {
    "token": "BatchTopK",
    "temporal": "Temporal",
    "mean": "Mean-Chunk",
    "joint_alpha0p25": "Joint α=0.25",
    "joint_alpha0p5": "Joint α=0.5",
    "joint_alpha1": "Joint α=1",
    "joint_alpha1p5": "Joint α=1.5",
    "cross": "Cross-Chunk",
}
COLORS = {
    method: METHOD_COLORS[method] for method in METHOD_ORDER
}


@dataclass
class CosineAccumulator:
    device: torch.device

    def __post_init__(self) -> None:
        self.totals = torch.zeros(
            3,
            dtype=torch.float64,
            device=self.device,
        )

    @torch.no_grad()
    def update(
        self,
        reconstructed_scaled: torch.Tensor,
        target_raw: torch.Tensor,
        *,
        activation_scale: torch.Tensor | float,
    ) -> None:
        scale = float(torch.as_tensor(activation_scale).item())
        reconstructed_raw = reconstructed_scaled.float() / scale
        target = target_raw.float()
        cosine = F.cosine_similarity(
            reconstructed_raw,
            target,
            dim=-1,
            eps=1e-12,
        ).double()
        self.totals[0] += cosine.sum()
        self.totals[1] += cosine.square().sum()
        self.totals[2] += target.shape[0]

    def all_reduce_(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.totals, op=dist.ReduceOp.SUM)

    def result(self) -> dict[str, float | int]:
        total, total_sq, count = self.totals.tolist()
        if count <= 0:
            raise ValueError("cosine evaluation accumulated no samples")
        mean = total / count
        variance = max(0.0, total_sq / count - mean * mean)
        return {
            "samples": int(round(count)),
            "mean_cosine_similarity": mean,
            "std_cosine_similarity": variance**0.5,
        }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-fidelity-results", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--joint-sae-root")
    p.add_argument(
        "--joint-method-root",
        action="append",
        default=[],
        metavar="METHOD=PATH",
    )
    p.add_argument("--validation-cache-dir", required=True)
    p.add_argument(
        "--validation-samples",
        type=int,
        default=None,
        help=(
            "Global rows from the fixed validation-cache prefix. Defaults to "
            "the evaluation_samples stored in training_fidelity.json."
        ),
    )
    p.add_argument(
        "--global-batch-size",
        "--batch-size",
        dest="global_batch_size",
        type=int,
        default=32_000,
        help=(
            "Global validation batch size. The default matches SAE training "
            "and therefore reproduces its exact distributed BatchTopK budget."
        ),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    p.add_argument("--figure-base", required=True)
    p.add_argument(
        "--reuse-results",
        action="store_true",
        help=(
            "Reuse a complete existing result JSON and regenerate only the "
            "figure/manifest. The stored protocol must match this request."
        ),
    )
    return p


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parse_joint_roots(specs: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for spec in specs:
        key, separator, raw_path = spec.partition("=")
        if not separator or not key.strip() or not raw_path.strip():
            raise ValueError(
                "--joint-method-root values must use METHOD=PATH syntax"
            )
        roots[key.strip()] = Path(raw_path.strip()).expanduser().resolve()
    return roots


def _joint_mode_dir(root: Path) -> Path:
    return root if (root / "metrics.jsonl").is_file() else root / "joint_chunk"


def _checkpoint_path(
    *,
    key: str,
    training: Mapping[str, Any],
    sae_root: Path,
    joint_sae_root: Path | None,
    joint_overrides: Mapping[str, Path],
) -> Path:
    selection = str(
        training.get("identity", {}).get("checkpoint_selection", "best")
    )
    if key in BASE_MODES:
        mode_dir = sae_root / key
    elif key == "joint":
        if joint_sae_root is None:
            raise ValueError(
                "training_fidelity contains joint but --joint-sae-root is absent"
            )
        mode_dir = _joint_mode_dir(joint_sae_root)
    else:
        root = joint_overrides.get(key)
        if root is None:
            extension = training.get("joint_extension", {})
            record = extension.get("joint_checkpoints", {}).get(key, {})
            if record.get("root"):
                root = Path(str(record["root"])).expanduser().resolve()
        if root is None:
            raise ValueError(f"cannot resolve checkpoint root for {key}")
        mode_dir = _joint_mode_dir(root)
    checkpoint = (
        mode_dir / "checkpoints" / "best"
        if selection == "best"
        else mode_dir
    )
    if not (checkpoint / "sae.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "sae.safetensors")
    return checkpoint


def _loader(
    *,
    cache_dir: Path,
    manifest: Mapping[str, Any],
    mode: str,
    samples: int,
    global_batch_size: int,
    device: torch.device,
    seed: int,
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if global_batch_size % world_size:
        raise ValueError(
            f"global batch size {global_batch_size} is not divisible by "
            f"world size {world_size}"
        )
    adapter = OccurrenceLoaderAdapter(
        cache_dir,
        manifest,
        rank=rank,
        world_size=world_size,
        batch_size=global_batch_size // world_size,
        seed=seed,
        device=str(device),
        mode=mode,
        finite=True,
        prefetch_shards=1,
        prefetch_workers=1,
        prefetch_batches=2,
        materialize_shards=True,
        pin_memory=device.type == "cuda",
        gpu_shards=1 if device.type == "cuda" else 0,
    )
    return _iter_limited(
        adapter.iter_epoch(0),
        _global_limit_to_local(samples, world_size),
    )


@torch.inference_mode()
def _evaluate_single(
    *,
    key: str,
    checkpoint: Path,
    cache_dir: Path,
    manifest: Mapping[str, Any],
    samples: int,
    global_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    loaded = load_sae(checkpoint, device=str(device))
    model = loaded.model
    model.eval()
    accumulator = CosineAccumulator(device)
    for batch in _loader(
        cache_dir=cache_dir,
        manifest=manifest,
        mode=key,
        samples=samples,
        global_batch_size=global_batch_size,
        device=device,
        seed=int(loaded.config.get("seed", 42)) + 307,
    ):
        inputs, targets, _ = select_occurrence_view(
            key,
            batch,
            allow_legacy_v1=not cache_is_v2(manifest),
        )
        reconstructed = model(
            inputs,
            batch_topk=True,
            distributed=dist.is_initialized() and dist.get_world_size() > 1,
        )[0]
        accumulator.update(
            reconstructed,
            targets,
            activation_scale=model.activation_scale,
        )
    accumulator.all_reduce_()
    result = accumulator.result()
    del model, loaded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def _evaluate_joint(
    *,
    checkpoint: Path,
    cache_dir: Path,
    manifest: Mapping[str, Any],
    samples: int,
    global_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    loaded = load_sae(checkpoint, device=str(device))
    model = loaded.model
    if not isinstance(model, JointChunkSAE):
        raise TypeError(f"{checkpoint} is not a JointChunkSAE")
    model.eval()
    mean_accumulator = CosineAccumulator(device)
    cross_accumulator = CosineAccumulator(device)
    for batch in _loader(
        cache_dir=cache_dir,
        manifest=manifest,
        mode="joint_chunk",
        samples=samples,
        global_batch_size=global_batch_size,
        device=device,
        seed=int(loaded.config.get("seed", 42)) + 307,
    ):
        inputs, cross_targets, _ = select_joint_chunk_view(
            batch,
            allow_legacy_v1=not cache_is_v2(manifest),
        )
        deduplication = (
            chunk_input_deduplication(batch, inputs.shape[0])
            if bool(loaded.config.get("deduplicate_chunk_inputs", True))
            else None
        )
        output = model(
            inputs,
            joint=True,
            batch_topk=True,
            distributed=dist.is_initialized() and dist.get_world_size() > 1,
            unique_rows=None if deduplication is None else deduplication[0],
            dedup_inverse=None if deduplication is None else deduplication[1],
        )
        reconstructed_mean, reconstructed_cross = output[:2]
        mean_accumulator.update(
            reconstructed_mean,
            inputs,
            activation_scale=model.activation_scale,
        )
        cross_accumulator.update(
            reconstructed_cross,
            cross_targets,
            activation_scale=model.activation_scale,
        )
    mean_accumulator.all_reduce_()
    cross_accumulator.all_reduce_()
    mean = mean_accumulator.result()
    cross = cross_accumulator.result()
    alpha = float(loaded.config["joint_chunk_alpha"])
    composite = (
        float(mean["mean_cosine_similarity"])
        + alpha * float(cross["mean_cosine_similarity"])
    ) / (1.0 + alpha)
    result = {
        "samples": int(mean["samples"]),
        "mean_cosine_similarity": composite,
        "components": {
            "mean": mean,
            "cross": cross,
        },
        "alpha": alpha,
        "composite_definition": (
            "(mean cosine + alpha * cross cosine) / (1 + alpha)"
        ),
    }
    del model, loaded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _plot(
    *,
    methods: Sequence[str],
    labels: Mapping[str, str],
    results: Mapping[str, Mapping[str, Any]],
    figure_base: Path,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFC",
            "savefig.facecolor": "white",
            "text.color": "#252932",
            "xtick.color": "#252932",
            "ytick.color": "#252932",
        }
    )
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    values = [
        float(results[key]["mean_cosine_similarity"]) for key in methods
    ]
    positions = list(range(len(methods)))
    lower = max(0.0, min(values) - 0.025)
    for key, position, value in zip(methods, positions, values, strict=True):
        ax.hlines(
            position,
            lower,
            value,
            color=COLORS[key],
            lw=4.5,
            alpha=0.42,
        )
        ax.scatter(
            [value],
            [position],
            s=115,
            color=COLORS[key],
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            value + 0.002,
            position,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=9.3,
            fontweight="bold",
        )
        components = results[key].get("components")
        if isinstance(components, Mapping):
            ax.text(
                value - 0.002,
                position + 0.24,
                (
                    f"M {float(components['mean']['mean_cosine_similarity']):.3f}\n"
                    f"C {float(components['cross']['mean_cosine_similarity']):.3f}"
                ),
                ha="right",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#5C6270",
            )
    ax.set_yticks(
        positions,
        [SHORT_LABELS.get(key, labels[key]) for key in methods],
    )
    ax.invert_yaxis()
    ax.set_xlim(lower, min(1.005, max(values) + 0.018))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Mean cosine similarity")
    ax.set_title(
        "Input/target-output reconstruction cosine on held-out validation",
        loc="left",
        fontsize=15,
        pad=18,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.02,
        (
            "Self-reconstruction uses input vs output; Cross uses partner target "
            "vs output · Joint is an α-weighted Mean/Cross composite"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#5C6270",
    )
    ax.grid(axis="x", color="#DDE2EA", lw=0.75)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.84, bottom=0.13)
    style_figure_text(fig, minimum_tick_size=8.5)
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [figure_base.with_suffix(".png"), figure_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=320, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    args = parser().parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "auto":
        device = (
            torch.device(f"cuda:{local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    training_path = Path(args.training_fidelity_results).expanduser().resolve()
    training = _read_json(training_path)
    published = training.get("methods")
    if not isinstance(published, Mapping):
        raise ValueError(f"{training_path}: missing methods")
    methods = tuple(
        key
        for key in training.get("method_order", METHOD_ORDER)
        if key in published
    )
    labels_payload = training.get("method_labels", {})
    labels = {
        key: str(labels_payload.get(key, SHORT_LABELS[key]))
        for key in methods
    }
    samples = args.validation_samples
    if samples is None:
        samples = min(
            int(published[key]["evaluation_samples"]) for key in methods
        )
    if samples <= 0:
        raise ValueError("--validation-samples must be positive")
    if samples % max(1, world_size):
        raise ValueError(
            f"validation samples {samples} are not divisible by world size "
            f"{world_size}"
        )

    cache_dir = Path(args.validation_cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    sae_root = Path(args.sae_root).expanduser().resolve()
    joint_sae_root = (
        Path(args.joint_sae_root).expanduser().resolve()
        if args.joint_sae_root
        else None
    )
    joint_overrides = _parse_joint_roots(args.joint_method_root)

    checkpoints: dict[str, Path] = {}
    for key in methods:
        checkpoints[key] = _checkpoint_path(
            key=key,
            training=training,
            sae_root=sae_root,
            joint_sae_root=joint_sae_root,
            joint_overrides=joint_overrides,
        )
    output = Path(args.output).expanduser().resolve()
    figure_base = Path(args.figure_base).expanduser().resolve()
    if args.reuse_results:
        if not output.is_file():
            raise FileNotFoundError(output)
        existing = _read_json(output)
        if existing.get("format") != RESULT_FORMAT or existing.get("complete") is not True:
            raise ValueError(f"cannot reuse incomplete/unexpected result: {output}")
        protocol = existing.get("protocol", {})
        if int(protocol.get("validation_samples", -1)) != samples:
            raise ValueError("reused cosine result has different validation_samples")
        if int(protocol.get("global_batch_size", -1)) != args.global_batch_size:
            raise ValueError("reused cosine result has different global_batch_size")
        if existing.get("method_order") != list(methods):
            raise ValueError("reused cosine result has different method_order")
        results = dict(existing["methods"])
        if rank != 0:
            if dist.is_initialized():
                dist.destroy_process_group()
            return
    else:
        results: dict[str, Any] = {}
        for key in methods:
            checkpoint = checkpoints[key]
            if rank == 0:
                print(f"[cosine] evaluating {key}: {checkpoint}", flush=True)
            if key in BASE_MODES:
                results[key] = _evaluate_single(
                    key=key,
                    checkpoint=checkpoint,
                    cache_dir=cache_dir,
                    manifest=manifest,
                    samples=samples,
                    global_batch_size=args.global_batch_size,
                    device=device,
                )
            else:
                results[key] = _evaluate_joint(
                    checkpoint=checkpoint,
                    cache_dir=cache_dir,
                    manifest=manifest,
                    samples=samples,
                    global_batch_size=args.global_batch_size,
                    device=device,
                )
            results[key]["checkpoint"] = str(checkpoint)
            if rank == 0:
                print(
                    f"[cosine] {key}="
                    f"{results[key]['mean_cosine_similarity']:.6f}",
                    flush=True,
                )
            if dist.is_initialized():
                dist.barrier()

        if rank != 0:
            if dist.is_initialized():
                dist.destroy_process_group()
            return

    figure_paths = _plot(
        methods=methods,
        labels=labels,
        results=results,
        figure_base=figure_base,
    )
    payload = {
        "format": RESULT_FORMAT,
        "complete": True,
        "metric": "mean per-sample cosine similarity between raw target and raw-space SAE reconstruction",
        "protocol": {
            "checkpoint_selection": training.get("identity", {}).get(
                "checkpoint_selection"
            ),
            "validation_cache": str(cache_dir),
            "validation_samples": samples,
            "validation_subset": (
                "deterministic prefix produced by seed=config.seed+307, matching "
                "periodic training validation"
            ),
            "inference": (
                "exact distributed BatchTopK with the same global validation "
                "batch size used during SAE training"
            ),
            "global_batch_size": args.global_batch_size,
            "joint_scalar": (
                "alpha-weighted arithmetic mean of Mean-head and Cross-head "
                "mean cosine similarities"
            ),
            "requires_retraining": False,
        },
        "method_order": list(methods),
        "method_labels": labels,
        "methods": results,
        "figures": [
            str(path.relative_to(output.parent)) for path in figure_paths
        ],
    }
    atomic_json_dump(payload, output)
    manifest_output = output.with_name("reconstruction_cosine_manifest.json")
    files: dict[str, Any] = {
        "results": file_record(output, relative_to=output.parent),
        "source_training_fidelity": file_record(
            training_path,
            relative_to=output.parent,
        ),
        "validation_cache_manifest": file_record(
            manifest_path,
            relative_to=output.parent,
        ),
    }
    for path in figure_paths:
        files[f"figure_{path.suffix.lstrip('.')}"] = file_record(
            path,
            relative_to=output.parent,
        )
    for key, checkpoint in checkpoints.items():
        files[f"checkpoint_config_{key}"] = file_record(
            checkpoint / "config.json",
            relative_to=output.parent,
        )
        files[f"checkpoint_weights_{key}"] = file_record(
            checkpoint / "sae.safetensors",
            relative_to=output.parent,
            hash_content=False,
        )
    write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": {
                "method_order": list(methods),
                "validation_samples": samples,
                "checkpoint_selection": payload["protocol"][
                    "checkpoint_selection"
                ],
            },
            "files": files,
        },
        manifest_output,
    )
    print(output, flush=True)
    for path in figure_paths:
        print(path, flush=True)
    print(manifest_output, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
