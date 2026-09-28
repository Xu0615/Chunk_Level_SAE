#!/usr/bin/env python
"""Plot SAE training-health diagnostics without modifying RFVE artifacts.

The existing ``training_fidelity`` figure answers whether each sparse model
captures the variance available to its task-specific reference.  This script
adds complementary diagnostics from the already-written training logs:

* train-versus-validation RFVE gap for all published SAE checkpoints;
* selected-checkpoint dictionary health, including activation concentration.

It deliberately writes new files only.  In particular, it never rewrites
``training_fidelity.json``, ``training_fidelity_manifest.json``, or either
``training_fidelity`` figure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from safetensors import safe_open

from chunk_saes.artifacts import file_record, write_artifact_manifest
from chunk_saes.plot_style import (
    METHOD_COLORS,
    METHOD_MARKERS,
    style_figure_text,
)
from chunk_saes.utils import atomic_json_dump


RESULT_FORMAT = "chunk-saes-training-health-v1"
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
DEFAULT_LABELS = {
    "token": "BatchTopK SAE",
    "temporal": "Temporal SAE",
    "mean": "Mean-Chunk SAE",
    "joint_alpha0p25": "Joint-Chunk SAE(alpha=0.25)",
    "joint_alpha0p5": "Joint-Chunk SAE(alpha=0.5)",
    "joint_alpha1": "Joint-Chunk SAE(alpha=1)",
    "joint_alpha1p5": "Joint-Chunk SAE(alpha=1.5)",
    "cross": "Cross-Chunk SAE",
}
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
MARKERS = {
    method: METHOD_MARKERS[method] for method in METHOD_ORDER
}


@dataclass(frozen=True)
class MethodLog:
    key: str
    label: str
    mode_dir: Path
    config: dict[str, Any]
    complete: dict[str, Any]
    train_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-fidelity-results", required=True)
    p.add_argument(
        "--sae-root",
        required=True,
        help="Root containing the token/temporal/mean/cross mode directories.",
    )
    p.add_argument(
        "--joint-sae-root",
        help=(
            "Run root for the published method key 'joint' (normally the "
            "alpha=0.25 nested Joint-Chunk SAE)."
        ),
    )
    p.add_argument(
        "--joint-method-root",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help=(
            "Optional override for an additional Joint method root. May be "
            "repeated, e.g. joint_alpha0p5=/path/to/run."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Existing or new rfve/figures directory.",
    )
    p.add_argument(
        "--summary-output",
        default=None,
        help=(
            "JSON summary path. Defaults to training_health.json beside "
            "training_fidelity.json."
        ),
    )
    return p


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {resolved}")
    return value


def _read_metric_rows(
    path: Path,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    train_by_step: dict[int, dict[str, Any]] = {}
    validation_by_step: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
            if not isinstance(row, dict) or row.get("step") is None:
                continue
            step = int(row["step"])
            if row.get("split") == "train":
                train_by_step[step] = row
            elif row.get("split") == "validation":
                validation_by_step[step] = row
    if not train_by_step:
        raise ValueError(f"{path} contains no train rows")
    if not validation_by_step:
        raise ValueError(f"{path} contains no periodic validation rows")
    return (
        tuple(train_by_step[step] for step in sorted(train_by_step)),
        tuple(
            validation_by_step[step] for step in sorted(validation_by_step)
        ),
    )


def _parse_joint_roots(specs: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for spec in specs:
        key, separator, raw_path = spec.partition("=")
        key = key.strip()
        raw_path = raw_path.strip()
        if not separator or not key or not raw_path:
            raise ValueError(
                "--joint-method-root values must use METHOD=PATH syntax"
            )
        if key in roots:
            raise ValueError(f"duplicate --joint-method-root for {key}")
        roots[key] = Path(raw_path).expanduser().resolve()
    return roots


def _joint_mode_dir(root: Path) -> Path:
    if (root / "metrics.jsonl").is_file():
        return root
    return root / "joint_chunk"


def _resolve_method_log(
    *,
    key: str,
    label: str,
    training: Mapping[str, Any],
    sae_root: Path,
    joint_sae_root: Path | None,
    joint_overrides: Mapping[str, Path],
) -> MethodLog:
    if key in BASE_MODES:
        mode_dir = sae_root / key
    elif key == "joint":
        if joint_sae_root is None:
            raise ValueError(
                "training_fidelity contains 'joint' but --joint-sae-root "
                "was not provided"
            )
        mode_dir = _joint_mode_dir(joint_sae_root)
    else:
        root = joint_overrides.get(key)
        if root is None:
            extension = training.get("joint_extension")
            checkpoints = (
                extension.get("joint_checkpoints", {})
                if isinstance(extension, Mapping)
                else {}
            )
            record = checkpoints.get(key)
            if isinstance(record, Mapping) and record.get("root"):
                root = Path(str(record["root"])).expanduser().resolve()
        if root is None:
            raise ValueError(
                f"cannot resolve checkpoint root for published method {key!r}; "
                "provide --joint-method-root METHOD=PATH"
            )
        mode_dir = _joint_mode_dir(root)

    metrics_path = mode_dir / "metrics.jsonl"
    config_path = mode_dir / "config.json"
    complete_path = mode_dir / "complete.json"
    for path in (metrics_path, config_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_rows, validation_rows = _read_metric_rows(metrics_path)
    return MethodLog(
        key=key,
        label=label,
        mode_dir=mode_dir,
        config=_read_json(config_path),
        complete=_read_json(complete_path),
        train_rows=train_rows,
        validation_rows=validation_rows,
    )


def _is_joint(method: Mapping[str, Any]) -> bool:
    return method.get("alpha") is not None and isinstance(
        method.get("components"), Mapping
    )


def _task_rfve(
    method: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    split: str,
) -> float:
    if _is_joint(method):
        alpha = float(method["alpha"])
        components = method["components"]
        mean_reference = float(components["mean"]["reference_fve"])
        cross_reference = float(components["cross"]["reference_fve"])
        mean_fve = float(row[f"{split}/joint_mean_fve"])
        cross_fve = float(row[f"{split}/joint_cross_fve"])
        fve = (mean_fve + alpha * cross_fve) / (1.0 + alpha)
        reference = (
            mean_reference + alpha * cross_reference
        ) / (1.0 + alpha)
        return fve / reference
    reference = float(method["reference"]["reference_fve"])
    return float(row[f"{split}/fve"]) / reference


def _row_at_or_before(
    rows: Sequence[Mapping[str, Any]],
    step: int,
) -> Mapping[str, Any]:
    candidates = [row for row in rows if int(row["step"]) <= step]
    if not candidates:
        raise ValueError(f"no metric row at or before step {step}")
    return max(candidates, key=lambda row: int(row["step"]))


def _row_at_step(
    rows: Sequence[Mapping[str, Any]],
    step: int,
) -> Mapping[str, Any]:
    for row in rows:
        if int(row["step"]) == step:
            return row
    raise ValueError(f"no metric row at selected step {step}")


def _samples_seen(
    *,
    step: int,
    train_row: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> int:
    if train_row is not None and train_row.get("progress/samples_seen") is not None:
        return int(train_row["progress/samples_seen"])
    return step * int(config.get("global_batch_size", 32_000))


def _selected_checkpoint_path(
    source: MethodLog,
    checkpoint_selection: str,
) -> Path:
    if checkpoint_selection == "best":
        return source.mode_dir / "checkpoints" / "best" / "sae.safetensors"
    if checkpoint_selection == "final":
        return source.mode_dir / "sae.safetensors"
    raise ValueError(
        f"unsupported checkpoint selection {checkpoint_selection!r}"
    )


def _read_selected_feature_counts(
    source: MethodLog,
    checkpoint_selection: str,
) -> tuple[Any | None, Path | None]:
    checkpoint = _selected_checkpoint_path(source, checkpoint_selection)
    if not checkpoint.is_file():
        return None, None
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        if "feature_counts" not in handle.keys():
            return None, checkpoint
        return handle.get_tensor("feature_counts"), checkpoint


def _validation_code_metrics(
    method: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[float, float]:
    if _is_joint(method):
        effective_l0 = min(
            float(row["validation/joint_mean_effective_l0"]),
            float(row["validation/joint_cross_effective_l0"]),
        )
        zero_code_fraction = max(
            float(row["validation/joint_mean_zero_code_fraction"]),
            float(row["validation/joint_cross_zero_code_fraction"]),
        )
        return effective_l0, zero_code_fraction
    return (
        float(row["validation/effective_l0"]),
        float(row["validation/zero_code_fraction"]),
    )


def _feature_usage_summary(feature_counts: Any | None) -> dict[str, float] | None:
    if feature_counts is None:
        return None
    counts = feature_counts.detach().cpu().numpy().astype(np.float64, copy=False)
    total = float(counts.sum())
    if total <= 0.0:
        return {
            "effective_features": 0.0,
            "effective_feature_fraction": 0.0,
            "top_1pct_activity_share": 0.0,
        }
    probabilities = counts[counts > 0.0] / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    effective_features = float(np.exp(entropy))
    top_count = max(1, int(np.ceil(0.01 * counts.size)))
    top_share = float(np.partition(counts, -top_count)[-top_count:].sum() / total)
    return {
        "effective_features": effective_features,
        "effective_feature_fraction": effective_features / counts.size,
        "activation_entropy_nats": entropy,
        "normalized_activation_entropy": entropy / np.log(counts.size),
        "top_1pct_activity_share": top_share,
    }


def _smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    width = min(window, values.size if values.size % 2 else values.size - 1)
    width = max(3, width)
    pad = width // 2
    return np.convolve(
        np.pad(values, (pad, pad), mode="edge"),
        np.full(width, 1.0 / width),
        mode="valid",
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFC",
            "savefig.facecolor": "white",
            "xtick.color": "#252932",
            "ytick.color": "#252932",
            "text.color": "#252932",
        }
    )


def _save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    if base.parent.name != "figures":
        raise ValueError(f"figure output must be inside a figures/ directory: {base}")
    base.parent.mkdir(parents=True, exist_ok=True)
    style_figure_text(fig, minimum_tick_size=8.2)
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=320, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _health_statuses(
    *,
    effective_l0: float,
    k: int,
    zero_code_fraction: float,
    never_fired: int,
    recent_dead: int,
    dictionary_width: int,
) -> tuple[list[int], list[str]]:
    l0_error = abs(effective_l0 - k) / max(1, k)
    never_fraction = never_fired / max(1, dictionary_width)
    recent_fraction = recent_dead / max(1, dictionary_width)

    statuses = [
        2 if l0_error <= 0.005 else 1 if l0_error <= 0.05 else 0,
        (
            2
            if zero_code_fraction == 0.0
            else 1
            if zero_code_fraction <= 0.01
            else 0
        ),
        2 if never_fired == 0 else 1 if never_fraction <= 0.001 else 0,
        2 if recent_dead == 0 else 1 if recent_fraction <= 0.001 else 0,
    ]
    annotations = [
        f"{effective_l0:.1f} / {k}",
        f"{100.0 * zero_code_fraction:.3f}%",
        f"{never_fired:,}",
        f"{recent_dead:,}",
    ]
    return statuses, annotations


def _plot_generalization_stability(
    *,
    methods: Sequence[str],
    labels: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.4, 6.5))
    for key in methods:
        row = summaries[key]
        gap = row["generalization_gap_trajectory"]
        x = np.asarray(
            [float(point["samples_seen"]) / 1_000_000_000 for point in gap]
        )
        y = np.asarray(
            [100.0 * float(point["train_minus_validation_rfve"]) for point in gap]
        )
        ax.plot(x, y, color=COLORS[key], lw=0.7, alpha=0.12)
        ax.plot(
            x,
            _smooth(y),
            color=COLORS[key],
            lw=2.4,
            ls=(0, (5, 2.2)) if key.startswith("joint") else "-",
            label=labels[key],
        )
        selected = row["selected_checkpoint"]
        ax.scatter(
            [float(selected["samples_seen"]) / 1_000_000_000],
            [100.0 * float(selected["train_minus_validation_rfve"])],
            marker=MARKERS[key],
            s=60,
            color=COLORS[key],
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )

    ax.axhline(0.0, color="#667085", lw=1.0, ls=(0, (4, 4)))
    ax.set_xlabel("Training occurrences seen (billions)")
    ax.set_ylabel("Train − validation RFVE (percentage points)")
    ax.set_title(
        "Generalization and stability across training",
        loc="left",
        fontsize=15,
        pad=18,
    )
    ax.grid(color="#DDE2EA", lw=0.75, alpha=0.9)
    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=8.0,
        columnspacing=1.1,
        handlelength=2.4,
    )
    ax.text(
        0.0,
        1.02,
        (
            "All seven published SAEs · thin = logged windows · thick = "
            "9-point moving average · markers = validation-selected checkpoints"
        ),
        transform=ax.transAxes,
        fontsize=9.0,
        color="#5C6270",
        va="bottom",
    )
    ax.text(
        0.0,
        -0.14,
        "Thin lines are logged windows; thick lines are 9-point moving averages. "
        "Positive means train RFVE is higher.",
        transform=ax.transAxes,
        fontsize=8.0,
        color="#5C6270",
        va="top",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.19)
    return _save_figure(
        fig,
        output_dir / "generalization_stability",
    )


def _plot_dictionary_health(
    *,
    methods: Sequence[str],
    labels: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    columns = (
        "Fraction alive",
        "Recently dead\n(last 10M)",
        "Activation entropy\n(normalized)",
        "Top-1% activity\nshare",
    )
    matrix = np.zeros((len(methods), len(columns)), dtype=float)
    annotations: list[list[str]] = []
    for row_index, key in enumerate(methods):
        selected = summaries[key]["selected_checkpoint"]
        width = int(selected["dictionary_width"])
        alive = int(selected["alive_features"])
        dead = int(selected["recent_dead_features"])
        entropy = float(selected["normalized_activation_entropy"])
        top_share = float(selected["top_1pct_activity_share"])
        matrix[row_index] = (
            alive / max(1, width),
            dead / max(1, width),
            entropy,
            top_share,
        )
        annotations.append(
            [
                f"{100.0 * alive / width:.3f}%\n{alive:,}/{width:,}",
                f"{100.0 * dead / width:.4f}%\n{dead:,} features",
                (
                    f"{100.0 * entropy:.1f}%\n"
                    f"H={float(selected['activation_entropy_nats']):.2f}"
                ),
                f"{100.0 * top_share:.1f}%",
            ]
        )

    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(columns)), columns)
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
        color=[COLORS[key] for key in methods],
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
    for row_index, row in enumerate(annotations):
        for column_index, value in enumerate(row):
            background = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                value,
                ha="center",
                va="center",
                fontsize=8.7,
                fontweight="bold",
                color="white" if background >= 0.62 else "#252932",
            )
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "Dictionary health at the validation-selected checkpoint",
        x=0.12,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.905,
        (
            "Alive/dead measure coverage and recent inactivity; entropy and "
            "top-1% share measure concentration of cumulative feature_counts"
        ),
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#5C6270",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025)
    colorbar.ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )
    colorbar.set_label("Displayed fraction", fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.94, top=0.78, bottom=0.08)
    return _save_figure(
        fig,
        output_dir / "dictionary_health",
    )


def _build_summary(
    *,
    training: Mapping[str, Any],
    method_logs: Mapping[str, MethodLog],
    methods: Sequence[str],
) -> dict[str, Any]:
    checkpoint_selection = str(
        training.get("identity", {}).get("checkpoint_selection", "best")
    )
    summary_methods: dict[str, Any] = {}
    for key in methods:
        published = training["methods"][key]
        source = method_logs[key]
        selected_step = int(published["selected_step"])
        selected_validation = _row_at_step(
            source.validation_rows,
            selected_step,
        )
        selected_train = _row_at_or_before(source.train_rows, selected_step)
        effective_l0, zero_code_fraction = _validation_code_metrics(
            published,
            selected_validation,
        )
        dictionary_width = int(source.config["dict_size"])
        k = int(source.config["k"])
        feature_counts, _ = _read_selected_feature_counts(
            source,
            checkpoint_selection,
        )
        feature_usage = _feature_usage_summary(feature_counts)
        checkpoint_file = _selected_checkpoint_path(
            source,
            checkpoint_selection,
        )
        if feature_counts is not None:
            alive_features = int((feature_counts > 0).sum().item())
            alive_source = "selected_checkpoint_feature_counts"
        else:
            alive = source.complete.get("alive_features")
            if alive is None:
                raise ValueError(
                    f"{source.mode_dir}: no selected-checkpoint feature_counts "
                    "or complete.alive_features"
                )
            alive_features = int(alive)
            alive_source = "final_complete_fallback"
            checkpoint_file = None
        recent_dead = int(selected_train.get("sparsity/dead_features", 0))
        train_rfve = _task_rfve(published, selected_train, split="train")
        validation_rfve = _task_rfve(
            published,
            selected_validation,
            split="validation",
        )

        train_by_step = {
            int(row["step"]): row for row in source.train_rows
        }
        gap_trajectory = []
        for validation_row in source.validation_rows:
            step = int(validation_row["step"])
            if step <= 0 or step not in train_by_step:
                continue
            train_row = train_by_step[step]
            train_value = _task_rfve(
                published,
                train_row,
                split="train",
            )
            validation_value = _task_rfve(
                published,
                validation_row,
                split="validation",
            )
            gap_trajectory.append(
                {
                    "step": step,
                    "samples_seen": _samples_seen(
                        step=step,
                        train_row=train_row,
                        config=source.config,
                    ),
                    "train_rfve": train_value,
                    "validation_rfve": validation_value,
                    "train_minus_validation_rfve": (
                        train_value - validation_value
                    ),
                }
            )
        if not gap_trajectory:
            raise ValueError(f"{source.mode_dir} has no aligned train/validation rows")

        selected: dict[str, Any] = {
            "selected_step": selected_step,
            "samples_seen": _samples_seen(
                step=selected_step,
                train_row=selected_train,
                config=source.config,
            ),
            "validation_rfve": validation_rfve,
            "train_rfve": train_rfve,
            "train_minus_validation_rfve": train_rfve - validation_rfve,
            "effective_l0": effective_l0,
            "k": k,
            "zero_code_fraction": zero_code_fraction,
            "dictionary_width": dictionary_width,
            "alive_features": alive_features,
            "never_fired_features": dictionary_width - alive_features,
            "alive_feature_source": alive_source,
            "recent_dead_features": recent_dead,
            "recent_dead_fraction": recent_dead / max(1, dictionary_width),
            "dead_feature_window_occurrences": int(
                source.config.get("dead_feature_threshold", 0)
            ),
            "effective_features": (
                feature_usage["effective_features"]
                if feature_usage is not None
                else None
            ),
            "effective_feature_fraction": (
                feature_usage["effective_feature_fraction"]
                if feature_usage is not None
                else None
            ),
            "activation_entropy_nats": (
                feature_usage["activation_entropy_nats"]
                if feature_usage is not None
                else None
            ),
            "normalized_activation_entropy": (
                feature_usage["normalized_activation_entropy"]
                if feature_usage is not None
                else None
            ),
            "top_1pct_activity_share": (
                feature_usage["top_1pct_activity_share"]
                if feature_usage is not None
                else None
            ),
        }
        if _is_joint(published):
            prefix_width = int(source.config["joint_cross_prefix"])
            if feature_counts is not None:
                prefix_alive_features = int(
                    (feature_counts[:prefix_width] > 0).sum().item()
                )
                outer_alive_features = int(
                    (feature_counts[prefix_width:] > 0).sum().item()
                )
            else:
                prefix_alive_features = None
                outer_alive_features = None
            selected.update(
                {
                    "joint_mean_rfve": (
                        float(selected_validation["validation/joint_mean_fve"])
                        / float(
                            published["components"]["mean"]["reference_fve"]
                        )
                    ),
                    "joint_cross_rfve": (
                        float(selected_validation["validation/joint_cross_fve"])
                        / float(
                            published["components"]["cross"]["reference_fve"]
                        )
                    ),
                    "joint_prefix_active": float(
                        selected_validation["validation/joint_k_prefix"]
                    ),
                    "joint_prefix_active_share": float(
                        selected_validation["validation/joint_k_prefix"]
                    )
                    / max(1, k),
                    "joint_cross_prefix_width": prefix_width,
                    "joint_prefix_alive_features": prefix_alive_features,
                    "joint_outer_alive_features": outer_alive_features,
                }
            )

        temporal_objective = None
        if key == "temporal":
            temporal_objective = [
                {
                    "step": int(row["step"]),
                    "samples_seen": _samples_seen(
                        step=int(row["step"]),
                        train_row=row,
                        config=source.config,
                    ),
                    "contrastive_loss": float(
                        row["train/temporal_contrastive_loss"]
                    ),
                    "contrastive_accuracy": float(
                        row["train/temporal_contrastive_accuracy"]
                    ),
                    "pairs": int(row["train/temporal_pairs"]),
                }
                for row in source.train_rows
                if row.get("train/temporal_contrastive_loss") is not None
                and row.get("train/temporal_contrastive_accuracy") is not None
            ]

        joint_objective = None
        if _is_joint(published):
            joint_objective = [
                {
                    "step": int(row["step"]),
                    "samples_seen": _samples_seen(
                        step=int(row["step"]),
                        train_row=row,
                        config=source.config,
                    ),
                    "prefix_active": float(row["train/joint_k_prefix"]),
                    "prefix_active_share": float(row["train/joint_k_prefix"])
                    / max(1, k),
                    "mean_fve": float(row["train/joint_mean_fve"]),
                    "cross_fve": float(row["train/joint_cross_fve"]),
                    "task_loss": float(row["train/joint_task_loss"]),
                }
                for row in source.train_rows
                if row.get("train/joint_k_prefix") is not None
            ]

        summary_methods[key] = {
            "label": source.label,
            "mode_dir": str(source.mode_dir),
            "metrics_path": str(source.mode_dir / "metrics.jsonl"),
            "selected_checkpoint_path": (
                str(checkpoint_file) if checkpoint_file is not None else None
            ),
            "selected_checkpoint": selected,
            "training_extrema": {
                "maximum_recent_dead_features": max(
                    int(row.get("sparsity/dead_features", 0))
                    for row in source.train_rows
                ),
                "maximum_train_zero_code_fraction": max(
                    float(row.get("train/zero_code_fraction", 0.0))
                    for row in source.train_rows
                ),
            },
            "generalization_gap_trajectory": gap_trajectory,
            "temporal_objective": temporal_objective,
            "joint_objective": joint_objective,
        }

    return {
        "format": RESULT_FORMAT,
        "complete": True,
        "definition": {
            "generalization_gap": (
                "task-normalized train RFVE minus task-normalized validation RFVE "
                "at the same logged step"
            ),
            "effective_l0": (
                "mean number of nonzero code entries per validation sample"
            ),
            "zero_code_fraction": (
                "fraction of validation samples with no active SAE feature"
            ),
            "alive_features": (
                "features with positive cumulative feature_counts in the "
                "selected checkpoint"
            ),
            "recent_dead_features": (
                "features whose occurrences-since-fired counter exceeds the "
                "configured dead_feature_threshold"
            ),
        },
        "interpretation": (
            "These are optimization, sparsity, and objective-specific diagnostics. "
            "They do not establish semantic interpretability; smoothness, feature "
            "explanation quality, coverage, and causal sufficiency require held-out "
            "post-training evaluations."
        ),
        "checkpoint_selection": checkpoint_selection,
        "method_order": list(methods),
        "method_labels": {
            key: method_logs[key].label for key in methods
        },
        "methods": summary_methods,
    }


def main() -> None:
    args = parser().parse_args()
    _style()

    training_path = Path(args.training_fidelity_results).expanduser().resolve()
    training = _read_json(training_path)
    published_methods = training.get("methods")
    if not isinstance(published_methods, Mapping):
        raise ValueError(f"{training_path}: missing methods mapping")
    methods = tuple(
        key
        for key in training.get("method_order", METHOD_ORDER)
        if key in published_methods
    )
    if not methods:
        raise ValueError(f"{training_path}: no published methods")
    unexpected = set(methods) - set(METHOD_ORDER)
    if unexpected:
        raise ValueError(f"unsupported published methods: {sorted(unexpected)}")

    labels_payload = training.get("method_labels", {})
    labels = {
        key: str(labels_payload.get(key, DEFAULT_LABELS[key]))
        for key in methods
    }
    sae_root = Path(args.sae_root).expanduser().resolve()
    joint_sae_root = (
        Path(args.joint_sae_root).expanduser().resolve()
        if args.joint_sae_root
        else None
    )
    joint_overrides = _parse_joint_roots(args.joint_method_root)
    method_logs = {
        key: _resolve_method_log(
            key=key,
            label=labels[key],
            training=training,
            sae_root=sae_root,
            joint_sae_root=joint_sae_root,
            joint_overrides=joint_overrides,
        )
        for key in methods
    }

    summary = _build_summary(
        training=training,
        method_logs=method_logs,
        methods=methods,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.name != "figures":
        raise ValueError("--output-dir must point to an rfve/figures directory")
    generalization_paths = _plot_generalization_stability(
        methods=methods,
        labels=labels,
        summaries=summary["methods"],
        output_dir=output_dir,
    )
    for legacy_base in (
        "training_health_diagnostics",
        "objective_specific_diagnostics",
    ):
        for suffix in (".png", ".pdf"):
            (output_dir / f"{legacy_base}{suffix}").unlink(missing_ok=True)

    summary_output = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else training_path.with_name("training_health.json")
    )
    summary["figures"] = [
        str(path.relative_to(summary_output.parent))
        for path in generalization_paths
    ]
    atomic_json_dump(summary, summary_output)

    files: dict[str, Any] = {
        "source_training_fidelity": file_record(
            training_path,
            relative_to=summary_output.parent,
        ),
        "results": file_record(
            summary_output,
            relative_to=summary_output.parent,
        ),
    }
    for path in generalization_paths:
        files[f"figure_{path.stem}_{path.suffix.lstrip('.')}"] = file_record(
            path,
            relative_to=summary_output.parent,
        )
    for key, source in method_logs.items():
        files[f"metrics_{key}"] = file_record(
            source.mode_dir / "metrics.jsonl",
            relative_to=summary_output.parent,
        )
        files[f"config_{key}"] = file_record(
            source.mode_dir / "config.json",
            relative_to=summary_output.parent,
        )
        files[f"complete_{key}"] = file_record(
            source.mode_dir / "complete.json",
            relative_to=summary_output.parent,
        )
        checkpoint_value = summary["methods"][key].get(
            "selected_checkpoint_path"
        )
        if checkpoint_value is not None:
            checkpoint = Path(str(checkpoint_value))
            files[f"selected_checkpoint_{key}"] = file_record(
                checkpoint,
                relative_to=summary_output.parent,
                hash_content=False,
            )

    manifest_path = summary_output.with_name("training_health_manifest.json")
    write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": {
                "checkpoint_selection": summary["checkpoint_selection"],
                "method_order": list(methods),
                "source_training_fidelity_format": training.get("format"),
            },
            "files": files,
        },
        manifest_path,
    )
    print(summary_output, flush=True)
    for path in generalization_paths:
        print(path, flush=True)
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
