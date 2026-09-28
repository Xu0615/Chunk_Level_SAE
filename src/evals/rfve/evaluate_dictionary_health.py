#!/usr/bin/env python
"""Evaluate recent training inactivity and cross-document feature prevalence.

The protocol intentionally reuses the same frozen checkpoints, 2,048
length-stratified validation pairs, 8,192 uniformly sampled training-alive
features, and native chunk representations as the dictionary-utilization
evaluation. It reports two diagnostics with deliberately different evidence:

* recently inactive: the trainer's full-dictionary count of features that did
  not fire in the preceding 10M occurrences;
* ubiquitous: sampled features active in at least 50% of distinct held-out
  documents.

The first avoids declaring a rare semantic feature dead merely because a small
held-out set did not contain its concept. The second is a candidate
generic-feature rate; broad useful state can also be ubiquitous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter
from safetensors import safe_open

from chunk_saes.artifacts import file_record, write_artifact_manifest
from chunk_saes.plot_style import METHOD_COLORS, style_figure_text
from chunk_saes.utils import atomic_json_dump
from evals.dictionary_utilization import (
    analyze_adjacent_feature_consistency as adjacent,
)
from evals.joint_extension.common import FrozenJointEncoder


RESULT_FORMAT = "chunk-saes-heldout-dictionary-health-v1"
METHOD_ORDER = (
    "token",
    "temporal",
    "mean",
    "joint",
    "joint_alpha0p25",
    "joint_alpha0p5",
    "joint_alpha1",
    "joint_alpha1p5",
    "cross",
)
BASE_METHODS = frozenset({"token", "temporal", "mean", "cross"})
BASE_SAMPLE_SEEDS = {
    "token": 172,
    "temporal": 173,
    "mean": 174,
    "cross": 175,
}
JOINT_SAMPLE_SEEDS = {
    "joint": 176,
    "joint_alpha0p25": 176,
    "joint_alpha0p5": 177,
    "joint_alpha1": 178,
    "joint_alpha1p5": 179,
}
SHORT_LABELS = {
    "token": "BatchTopK",
    "temporal": "Temporal",
    "mean": "Mean-Chunk",
    "joint": "Joint α=0.25",
    "joint_alpha0p25": "Joint α=0.25",
    "joint_alpha0p5": "Joint α=0.5",
    "joint_alpha1": "Joint α=1",
    "joint_alpha1p5": "Joint α=1.5",
    "cross": "Cross-Chunk",
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
    p.add_argument("--sample-pairs", type=int, default=2_048)
    p.add_argument("--feature-sample-size", type=int, default=8_192)
    p.add_argument("--token-batch-size", type=int, default=256)
    p.add_argument("--mean-batch-size", type=int, default=64)
    p.add_argument(
        "--ubiquitous-support-fraction",
        type=float,
        default=0.50,
        help=(
            "A sampled feature is ubiquitous when active on at least this "
            "fraction of held-out chunks."
        ),
    )
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    p.add_argument("--figure-base", required=True)
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
    if key in BASE_METHODS:
        mode_dir = sae_root / key
    elif key in {"joint", "joint_alpha0p25"} and joint_sae_root is not None:
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


def _mode_dir_from_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.name == "best" and checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint


def _recent_inactivity_metrics(
    checkpoint: Path,
    *,
    selected_step: int,
    expected_dictionary_width: int,
) -> dict[str, Any]:
    mode_dir = _mode_dir_from_checkpoint(checkpoint)
    metrics_path = mode_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    selected_row: dict[str, Any] | None = None
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {metrics_path}:{line_number}"
                ) from error
            if (
                row.get("split") == "train"
                and row.get("step") is not None
                and int(row["step"]) <= selected_step
                and row.get("sparsity/dead_features") is not None
                and (
                    selected_row is None
                    or int(row["step"]) > int(selected_row["step"])
                )
            ):
                selected_row = row
    if selected_row is None:
        raise ValueError(
            f"{metrics_path} has no dead-feature row at or before "
            f"selected step {selected_step}"
        )
    config = _read_json(mode_dir / "config.json")
    threshold = int(config.get("dead_feature_threshold", 10_000_000))
    recently_inactive_features = int(
        selected_row["sparsity/dead_features"]
    )
    return {
        "training_metrics": str(metrics_path),
        "selected_checkpoint_step": selected_step,
        "inactivity_metric_step": int(selected_row["step"]),
        "inactivity_window_occurrences": threshold,
        "recently_inactive_features": recently_inactive_features,
        "recently_inactive_fraction": (
            recently_inactive_features / expected_dictionary_width
        ),
    }


def _sample_alive_feature_ids(
    checkpoint: Path,
    *,
    sample_size: int,
    seed: int,
) -> list[int]:
    with safe_open(
        str(checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        counts = handle.get_tensor("feature_counts")
    alive = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if alive.numel() == 0:
        raise ValueError(f"checkpoint has no training-alive features: {checkpoint}")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(alive.numel(), generator=generator)
    return (
        alive[order[: min(sample_size, int(alive.numel()))]]
        .sort()
        .values
        .tolist()
    )


def _encode_base(
    *,
    key: str,
    checkpoint: Path,
    means_a: torch.Tensor,
    means_b: torch.Tensor,
    tokens_a: list[torch.Tensor],
    tokens_b: list[torch.Tensor],
    feature_sample_size: int,
    token_batch_size: int,
    mean_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    encoder = adjacent.SampledEncoder(
        checkpoint,
        feature_sample_size=feature_sample_size,
        seed=BASE_SAMPLE_SEEDS[key],
        device=device,
    )
    try:
        if key in {"token", "temporal"}:
            codes_a = encoder.encode_token_mean(
                tokens_a,
                token_batch_size=token_batch_size,
            )
            codes_b = encoder.encode_token_mean(
                tokens_b,
                token_batch_size=token_batch_size,
            )
        else:
            codes_a = encoder.encode_means(
                means_a,
                batch_size=mean_batch_size,
            )
            codes_b = encoder.encode_means(
                means_b,
                batch_size=mean_batch_size,
            )
        return codes_a, codes_b, encoder.feature_ids.tolist()
    finally:
        encoder.close()


def _encode_joint(
    *,
    key: str,
    checkpoint: Path,
    means_a: torch.Tensor,
    means_b: torch.Tensor,
    feature_sample_size: int,
    mean_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    feature_ids = _sample_alive_feature_ids(
        checkpoint,
        sample_size=feature_sample_size,
        seed=JOINT_SAMPLE_SEEDS[key],
    )
    encoder = FrozenJointEncoder(checkpoint, device, feature_ids)
    try:
        codes_a = []
        codes_b = []
        for start in range(0, int(means_a.shape[0]), mean_batch_size):
            stop = min(int(means_a.shape[0]), start + mean_batch_size)
            codes_a.append(
                encoder.dense(means_a[start:stop]).float().cpu().numpy()
            )
            codes_b.append(
                encoder.dense(means_b[start:stop]).float().cpu().numpy()
            )
        return (
            np.concatenate(codes_a, axis=0),
            np.concatenate(codes_b, axis=0),
            feature_ids,
        )
    finally:
        encoder.close()


def _document_prevalence_metrics(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    document_hashes: np.ndarray,
    *,
    ubiquitous_support_fraction: float,
) -> dict[str, Any]:
    if codes_a.shape != codes_b.shape:
        raise ValueError("held-out code matrices must have equal shapes")
    document_hashes = np.asarray(document_hashes)
    if document_hashes.shape != (codes_a.shape[0],):
        raise ValueError("document hashes do not match held-out pair rows")
    pair_active = (codes_a > 0) | (codes_b > 0)
    _, inverse = np.unique(document_hashes, return_inverse=True)
    documents = int(inverse.max()) + 1
    document_active = np.zeros(
        (documents, pair_active.shape[1]),
        dtype=np.bool_,
    )
    for document_index in range(documents):
        document_active[document_index] = pair_active[
            inverse == document_index
        ].any(axis=0)
    support = document_active.sum(axis=0, dtype=np.int64)
    ubiquitous_min_support = int(
        np.ceil(ubiquitous_support_fraction * documents)
    )
    ubiquitous = support >= ubiquitous_min_support
    return {
        "heldout_pairs": int(pair_active.shape[0]),
        "heldout_chunks": int(2 * pair_active.shape[0]),
        "heldout_documents": documents,
        "sampled_training_alive_features": int(pair_active.shape[1]),
        "mean_document_support": float(support.mean()),
        "mean_document_support_fraction": float(support.mean() / documents),
        "ubiquitous_min_support_documents": ubiquitous_min_support,
        "ubiquitous_support_fraction": ubiquitous_support_fraction,
        "ubiquitous_features": int(ubiquitous.sum()),
        "ubiquitous_fraction": float(ubiquitous.mean()),
        "document_support_quantiles": {
            str(q): float(np.quantile(support, q))
            for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "ubiquitous_fraction_sensitivity": {
            str(threshold): float(
                np.mean(support >= np.ceil(threshold * documents))
            )
            for threshold in (0.01, 0.05, 0.10, 0.25, 0.50)
        },
    }


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
    matrix = np.asarray(
        [
            [
                float(results[key]["recently_inactive_fraction"]),
                float(results[key]["ubiquitous_fraction"]),
            ]
            for key in methods
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    vmax = max(0.01, float(matrix.max()) * 1.05)
    image = ax.imshow(
        matrix,
        cmap="Reds",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks(
        (0, 1),
        (
            "Recently inactive\nno firing in the last 10M occurrences",
            "Ubiquitous / always-on\nactive in ≥50% of held-out documents",
        ),
    )
    ax.set_yticks(
        np.arange(len(methods)),
        [SHORT_LABELS.get(key, labels[key]) for key in methods],
    )
    ax.scatter(
        np.full(len(methods), -0.035),
        np.arange(len(methods)),
        transform=ax.get_yaxis_transform(),
        marker="s",
        s=58,
        color=[METHOD_COLORS[key] for key in methods],
        edgecolor="white",
        linewidth=0.6,
        clip_on=False,
        zorder=5,
    )
    ax.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
    )
    for row_index, key in enumerate(methods):
        row = results[key]
        values = (
            (
                float(row["recently_inactive_fraction"]),
                int(row["recently_inactive_features"]),
                (
                    f"at step {int(row['inactivity_metric_step']):,}"
                ),
            ),
            (
                float(row["ubiquitous_fraction"]),
                int(row["ubiquitous_features"]),
                (
                    f"≥{int(row['ubiquitous_min_support_documents'])} "
                    f"of {int(row['heldout_documents'])} documents"
                ),
            ),
        )
        for column_index, (fraction, count, threshold) in enumerate(values):
            ax.text(
                column_index,
                row_index,
                (
                    f"{fraction:.3%}\n{count:,} / 65,536\n{threshold}"
                    if column_index == 0
                    else f"{fraction:.2%}\n{count:,} / 8,192\n{threshold}"
                ),
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="bold",
                color=(
                    "white"
                    if fraction >= 0.55 * vmax
                    else "#252932"
                ),
            )
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "Feature health at the validation-selected checkpoint",
        x=0.13,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.13,
        0.91,
        (
            "Recent inactivity uses the full training dictionary; ubiquitous "
            "uses the same 2,048 held-out pairs and 8,192-feature sample as Eval 2"
        ),
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#5C6270",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    colorbar.ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )
    colorbar.set_label("Feature fraction", fontweight="bold")
    fig.text(
        0.13,
        0.03,
        (
            "Recent inactivity is not biased by held-out topic coverage; "
            "ubiquitous is a candidate generic-feature rate, not a semantic verdict."
        ),
        ha="left",
        fontsize=8.3,
        color="#5C6270",
    )
    fig.subplots_adjust(left=0.19, right=0.91, top=0.80, bottom=0.10)
    style_figure_text(fig, minimum_tick_size=8.5)
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [figure_base.with_suffix(".png"), figure_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=320, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    args = parser().parse_args()
    if not 0.0 < args.ubiquitous_support_fraction <= 1.0:
        raise ValueError("--ubiquitous-support-fraction must be in (0, 1]")

    training_path = Path(args.training_fidelity_results).expanduser().resolve()
    training = _read_json(training_path)
    published = training.get("methods")
    if not isinstance(published, Mapping):
        raise ValueError(f"{training_path}: missing methods mapping")
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
    unexpected = set(methods) - set(METHOD_ORDER)
    if unexpected:
        raise ValueError(f"unsupported published methods: {sorted(unexpected)}")

    cache_dir = Path(args.validation_cache_dir).expanduser().resolve()
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = adjacent._read(cache_manifest_path)
    cache_identity = adjacent._cache_identity(cache_dir, cache_manifest)
    shard_paths = adjacent._shard_paths(cache_dir, cache_manifest)
    metadata, document_hashes = adjacent._scan_pair_metadata(shard_paths)
    selected = adjacent._stratified_sample(
        metadata,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
    )
    (
        means_a,
        means_b,
        tokens_a,
        tokens_b,
        pair_ids,
        lengths_a,
        lengths_b,
    ) = adjacent._load_selected_pairs(
        shard_paths,
        metadata,
        selected,
    )

    sae_root = Path(args.sae_root).expanduser().resolve()
    joint_sae_root = (
        Path(args.joint_sae_root).expanduser().resolve()
        if args.joint_sae_root
        else None
    )
    joint_overrides = _parse_joint_roots(args.joint_method_root)
    checkpoints = {
        key: _checkpoint_path(
            key=key,
            training=training,
            sae_root=sae_root,
            joint_sae_root=joint_sae_root,
            joint_overrides=joint_overrides,
        )
        for key in methods
    }
    device = torch.device(args.device)
    results: dict[str, Any] = {}
    for key in methods:
        print(f"[dictionary-health] encoding {labels[key]}", flush=True)
        if key in BASE_METHODS:
            codes_a, codes_b, feature_ids = _encode_base(
                key=key,
                checkpoint=checkpoints[key],
                means_a=means_a,
                means_b=means_b,
                tokens_a=tokens_a,
                tokens_b=tokens_b,
                feature_sample_size=args.feature_sample_size,
                token_batch_size=args.token_batch_size,
                mean_batch_size=args.mean_batch_size,
                device=device,
            )
        else:
            codes_a, codes_b, feature_ids = _encode_joint(
                key=key,
                checkpoint=checkpoints[key],
                means_a=means_a,
                means_b=means_b,
                feature_sample_size=args.feature_sample_size,
                mean_batch_size=args.mean_batch_size,
                device=device,
            )
        results[key] = {
            "checkpoint": str(checkpoints[key]),
            "feature_sample_seed": (
                BASE_SAMPLE_SEEDS[key]
                if key in BASE_METHODS
                else JOINT_SAMPLE_SEEDS[key]
            ),
            "sampled_feature_ids_digest": adjacent.json_digest(feature_ids),
            **_recent_inactivity_metrics(
                checkpoints[key],
                selected_step=int(published[key]["selected_step"]),
                expected_dictionary_width=65_536,
            ),
            **_document_prevalence_metrics(
                codes_a,
                codes_b,
                document_hashes[selected],
                ubiquitous_support_fraction=args.ubiquitous_support_fraction,
            ),
        }
        print(
            f"[dictionary-health] {key}: "
            f"recently_inactive="
            f"{results[key]['recently_inactive_fraction']:.4%}, "
            f"ubiquitous={results[key]['ubiquitous_fraction']:.4%}",
            flush=True,
        )

    output = Path(args.output).expanduser().resolve()
    figure_base = Path(args.figure_base).expanduser().resolve()
    figure_paths = _plot(
        methods=methods,
        labels=labels,
        results=results,
        figure_base=figure_base,
    )
    payload = {
        "format": RESULT_FORMAT,
        "complete": True,
        "question": (
            "How many features were inactive during the 10M occurrences before "
            "the selected checkpoint, and how many fire across a large fraction "
            "of distinct held-out documents?"
        ),
        "protocol": {
            "checkpoint_selection": training.get("identity", {}).get(
                "checkpoint_selection"
            ),
            "activation_cache": cache_identity,
            "sample_pairs": int(args.sample_pairs),
            "heldout_chunks": int(2 * args.sample_pairs),
            "pair_ids_digest": adjacent.json_digest(pair_ids.tolist()),
            "document_hashes_digest": adjacent.json_digest(
                [
                    value.hex()
                    for value in document_hashes[selected].tolist()
                ]
            ),
            "length_cells": sorted(
                {
                    f"{int(a)}x{int(b)}"
                    for a, b in zip(
                        lengths_a.tolist(),
                        lengths_b.tolist(),
                        strict=True,
                    )
                }
            ),
            "feature_sample_size": int(args.feature_sample_size),
            "feature_population": (
                "uniform sample from training-alive features in each selected "
                "checkpoint"
            ),
            "representation_protocol": (
                "token/temporal: mean of thresholded token activations per "
                "chunk; mean/cross/joint: thresholded activation of chunk "
                "mean; a feature is active when this native chunk-level code "
                "is positive"
            ),
            "recently_inactive_definition": (
                "full-dictionary trainer inactivity age >= the configured "
                "10,000,000-occurrence dead-feature threshold"
            ),
            "ubiquitous_definition": (
                "native thresholded code active in at least 50% of distinct "
                "held-out documents"
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
    manifest_path = output.with_name("dictionary_health_manifest.json")
    files: dict[str, Any] = {
        "results": file_record(output, relative_to=output.parent),
        "source_training_fidelity": file_record(
            training_path,
            relative_to=output.parent,
        ),
        "validation_cache_manifest": file_record(
            cache_manifest_path,
            relative_to=output.parent,
        ),
    }
    for path in figure_paths:
        files[f"figure_{path.suffix.lstrip('.')}"] = file_record(
            path,
            relative_to=output.parent,
        )
    for key, checkpoint in checkpoints.items():
        mode_dir = _mode_dir_from_checkpoint(checkpoint)
        files[f"checkpoint_config_{key}"] = file_record(
            checkpoint / "config.json",
            relative_to=output.parent,
        )
        files[f"checkpoint_weights_{key}"] = file_record(
            checkpoint / "sae.safetensors",
            relative_to=output.parent,
            hash_content=False,
        )
        files[f"training_metrics_{key}"] = file_record(
            mode_dir / "metrics.jsonl",
            relative_to=output.parent,
        )
    write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": {
                "method_order": list(methods),
                "sample_pairs": int(args.sample_pairs),
                "feature_sample_size": int(args.feature_sample_size),
                "ubiquitous_support_fraction": float(
                    args.ubiquitous_support_fraction
                ),
            },
            "files": files,
        },
        manifest_path,
    )
    print(output, flush=True)
    for path in figure_paths:
        print(path, flush=True)
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
