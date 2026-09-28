#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy.spatial import ConvexHull

from chunk_saes.artifacts import load_artifact_manifest
from chunk_saes.plot_style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LIGHT_COLORS,
    METHOD_SHORT_LABELS,
    METHOD_TEXT_COLORS,
    METHODS,
    style_figure_text,
)


COLORS = METHOD_COLORS
LIGHT_COLORS = METHOD_LIGHT_COLORS
DOMAIN_COLORS = {
    "CS": "#4C78A8",
    "Economics": "#F58518",
    "Electrical Engineering": "#E45756",
    "Mathematics": "#72B7B2",
    "Physics": "#54A24B",
    "Quantitative Biology": "#EECA3B",
    "Quantitative Finance": "#B279A2",
    "Statistics": "#FF9DA6",
}
DOMAIN_SHORT = {
    "CS": "CS",
    "Economics": "Econ",
    "Electrical Engineering": "EE",
    "Mathematics": "Math",
    "Physics": "Phys",
    "Quantitative Biology": "Q-Bio",
    "Quantitative Finance": "Q-Fin",
    "Statistics": "Stats",
}
REPRESENTATIONS = {
    "token": "token_sae_{token_aggregation}",
    "temporal": "temporal_sae_mean",
    "mean": "mean_chunk_sae",
    "cross": "cross_chunk_sae",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Create publication-quality Chunk-SAE figures inspired by semantic "
            "manifold and sequence-transition visualizations."
        )
    )
    p.add_argument("--training-fidelity-results", required=True)
    p.add_argument("--probe-results", required=True)
    p.add_argument("--adjacent-consistency-results", required=True)
    p.add_argument(
        "--dictionary-utilization-results",
        default=None,
        help=(
            "Standalone Eval 2 Effective feature fraction artifact. If omitted, "
            "the dictionary_utilization section of adjacent-consistency results "
            "is used for backward compatibility."
        ),
    )
    p.add_argument("--representation-geometry-results", required=True)
    p.add_argument("--representation-embeddings", required=True)
    p.add_argument("--training-fidelity-manifest", required=True)
    p.add_argument("--probe-manifest", required=True)
    p.add_argument("--adjacent-consistency-manifest", required=True)
    p.add_argument("--representation-geometry-manifest", required=True)
    # Canonical task-specific destinations.  The ``evalN`` names remain as
    # compatibility aliases for older launchers, but new runs should use the
    # descriptive task names so each task owns exactly one primary plot.
    p.add_argument(
        "--rfve-figure-dir",
        "--rfve-plot-dir",
        dest="rfve_figure_dir",
        default=None,
    )
    p.add_argument(
        "--dictionary-utilization-figure-dir",
        "--dictionary-utilization-plot-dir",
        dest="dictionary_utilization_figure_dir",
        default=None,
    )
    p.add_argument(
        "--semantic-geometry-figure-dir",
        "--semantic-geometry-plot-dir",
        dest="semantic_geometry_figure_dir",
        default=None,
    )
    p.add_argument(
        "--label-efficiency-figure-dir",
        "--label-efficiency-plot-dir",
        dest="label_efficiency_figure_dir",
        default=None,
    )
    p.add_argument(
        "--temporal-robustness-figure-dir",
        "--temporal-robustness-plot-dir",
        dest="temporal_robustness_figure_dir",
        default=None,
    )
    p.add_argument("--eval1-plot-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--eval2-plot-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--eval3-plot-dir", default=None, help=argparse.SUPPRESS)
    return p


def _read(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFC",
            "savefig.facecolor": "white",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )


def _save(
    fig: plt.Figure,
    base: Path,
    *,
    overwrite_png: bool = False,
) -> list[Path]:
    """Save the PNG and PDF together in the task's ``figures/`` directory."""
    if base.parent.name != "figures":
        raise ValueError(f"figure output must be inside a figures/ directory: {base}")
    base.parent.mkdir(parents=True, exist_ok=True)
    style_figure_text(fig, minimum_tick_size=8.5)
    png_path = base.with_suffix(".png")
    paths = [png_path, base.with_suffix(".pdf")]
    if overwrite_png or not png_path.is_file():
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _method_legend() -> list[Line2D]:
    return _legend_for_modes(METHODS)


def _available_modes(*method_payloads: object) -> tuple[str, ...]:
    mappings = [
        payload
        for payload in method_payloads
        if isinstance(payload, dict)
    ]
    return tuple(
        mode
        for mode in METHODS
        if all(mode in mapping for mapping in mappings)
    )


def _legend_for_modes(modes: tuple[str, ...] | list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[mode],
            lw=3,
            marker="o",
            markersize=6,
            label=METHOD_LABELS[mode],
        )
        for mode in modes
    ]


def plot_training_fidelity(
    training: dict,
    base: Path,
    *,
    overwrite_png: bool = False,
) -> list[Path]:
    methods = training.get("methods", {})
    modes = tuple(
        mode
        for mode in ("token", "temporal", "mean", "joint", "cross")
        if mode in methods
    )
    fig, ax = plt.subplots(figsize=(10.6, 6.1))
    selected_values: dict[str, float] = {}

    def smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
        if values.size < 3:
            return values.copy()
        width = min(window, values.size if values.size % 2 else values.size - 1)
        width = max(3, width)
        padding = width // 2
        padded = np.pad(values, (padding, padding), mode="edge")
        return np.convolve(
            padded,
            np.full(width, 1.0 / width),
            mode="valid",
        )

    for mode in modes:
        row = training["methods"][mode]
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"{mode} lacks an RFVE training trajectory")
        occurrences = np.asarray(
            [float(point["samples_seen"]) / 1_000_000_000 for point in trajectory]
        )
        values = np.asarray([float(point["rfve"]) for point in trajectory])
        selected_step = float(row["selected_step"])
        selected_value = float(row["rfve"])
        selected_occurrences = selected_step * 32_000 / 1_000_000_000
        selected_values[mode] = selected_value
        line_style = (0, (5, 2.2)) if mode == "joint" else "-"
        ax.plot(
            occurrences,
            values,
            color=COLORS[mode],
            lw=0.8,
            ls=line_style,
            alpha=0.13,
            zorder=1,
        )
        ax.scatter(
            occurrences,
            values,
            color=COLORS[mode],
            s=6,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            occurrences,
            smooth(values),
            color=COLORS[mode],
            lw=3.0 if mode == "joint" else 3.1 if mode == "cross" else 2.7,
            ls=line_style,
            alpha=0.97,
            label=f"{METHOD_LABELS[mode]}   {selected_value:.3f}",
            zorder=4 if mode == "joint" else 3 if mode == "cross" else 2,
        )
        if mode == "joint":
            ax.scatter(
                selected_occurrences,
                selected_value,
                s=118,
                facecolor="white",
                edgecolor=COLORS[mode],
                linewidth=2.2,
                marker="o",
                zorder=6,
            )
        else:
            ax.scatter(
                selected_occurrences,
                selected_value,
                s=80,
                color=COLORS[mode],
                edgecolor="white",
                linewidth=1.2,
                marker="D",
                zorder=5,
            )

    total_occurrences = max(
        float(point["samples_seen"]) / 1_000_000_000
        for mode in modes
        for point in training["methods"][mode]["trajectory"]
    )
    ax.axhline(
        1.0,
        color="#667085",
        lw=1.2,
        ls=(0, (4, 4)),
        alpha=0.9,
        zorder=1,
    )
    ax.text(
        0.995,
        1.0,
        "dense / identity reference",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#667085",
    )
    ax.set_xlim(0, total_occurrences * 1.012)
    lower = min(
        float(point["rfve"])
        for mode in modes
        for point in training["methods"][mode]["trajectory"]
    )
    ax.set_ylim(
        max(-0.05, lower - 0.02),
        max(1.035, max(selected_values.values()) + 0.04),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ticks = np.linspace(0, total_occurrences, 6)
    ax.set_xticks(ticks, [f"{tick:.1f}" for tick in ticks])
    ax.set_xlabel("Training occurrences seen (billions)")
    ax.set_ylabel("Reference FVE (RFVE)")
    ax.grid(axis="both", color="#DDE1E8", lw=0.75, alpha=0.8)
    ax.legend(
        loc="lower right",
        ncol=2,
        handlelength=2.6,
        columnspacing=1.5,
        title=(
            "Selected-checkpoint RFVE\nJoint = α-weighted Mean/Cross composite"
            if "joint" in modes
            else "Selected-checkpoint RFVE"
        ),
    )

    status = (
        "Provenance-tracked Cross reference verified"
        if training.get("publication_ready")
        else "audit-only Cross reference; replace before publication"
    )
    status_color = "#287A52" if training.get("publication_ready") else "#8A5A14"
    cross_reference = training["methods"].get("cross", {}).get("reference", {})
    reference_note = status
    if training.get("publication_ready") and cross_reference.get("artifact_digest"):
        selected = cross_reference.get("selected_model") or {}
        widths = [
            str(int(item["hidden_width"]))
            for item in cross_reference.get("capacity_sweep", [])
        ]
        reference_note = (
            f"{status} · shared direction-blind dense MLP · "
            f"capacity sweep {'/'.join(widths)} · selected {selected.get('hidden_width')} · "
            f"independent test audit · artifact {str(cross_reference['artifact_digest'])[:10]}"
        )
    fig.text(
        0.105,
        0.035,
        reference_note,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="semibold",
        color=status_color,
    )
    fig.suptitle(
        "Reference FVE (RFVE) across 1B training occurrences",
        x=0.095,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.095,
        0.925,
        (
            "RFVE = sparse-SAE explained variance / same-information reference "
            "explained variance  ·  marker = validation-selected checkpoint"
        ),
        ha="left",
        va="top",
        fontsize=9.2,
        color="#5C6270",
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.15)
    return _save(fig, base, overwrite_png=overwrite_png)


def plot_semantic_manifold(
    geometry: dict,
    embeddings: dict[str, np.ndarray],
    base: Path,
) -> list[Path]:
    try:
        from evals.feature_dynamics.plot_cross_feature_semantic_manifold import (
            render_semantic_manifold,
        )

        # The standard evaluation layout places this plot at
        # <eval-root>/semantic_geometry/figures/semantic_geometry.
        eval_root = base.parents[2]
        if (eval_root / "shared/downstream_transfer/probe_features" / "features-test.npz").is_file():
            return render_semantic_manifold(
                eval_root=eval_root,
                base=base,
                dpi=320,
            )
    except (ImportError, FileNotFoundError, KeyError, ValueError):
        # Keep the generic fallback usable for unit tests or nonstandard
        # external artifact layouts.
        pass

    labels = embeddings["labels"].astype(str)
    unique_labels = list(DOMAIN_COLORS)
    modes = [
        mode
        for mode in METHODS
        if mode in geometry.get("methods", {})
        and f"{mode}_xy" in embeddings
    ]
    if not modes:
        raise ValueError(
            "semantic manifold plot has no methods with both metrics and embeddings"
        )
    fig, axes = plt.subplots(
        1,
        len(modes),
        figsize=(4.5 * len(modes), 4.5),
    )
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes, strict=True):
        coordinates = embeddings[f"{mode}_xy"]
        for label in unique_labels:
            mask = labels == label
            points = coordinates[mask]
            color = DOMAIN_COLORS[label]
            if points.shape[0] >= 4:
                try:
                    hull = ConvexHull(points)
                    polygon = points[hull.vertices]
                    ax.fill(
                        polygon[:, 0],
                        polygon[:, 1],
                        color=color,
                        alpha=0.055,
                        lw=0,
                    )
                except Exception:
                    pass
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=8,
                color=color,
                alpha=0.58,
                linewidth=0,
                rasterized=True,
            )
            centroid = np.median(points, axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                DOMAIN_SHORT[label],
                fontsize=7,
                ha="center",
                va="center",
                fontweight="semibold",
                color="#222222",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )
        metrics = geometry["methods"][mode]["geometry"]
        ax.set_title(
            (
                f"{METHOD_LABELS[mode]}\n"
                f"silhouette {metrics['silhouette']:.3f} · "
                f"neighbor purity {metrics['neighbor_purity']:.3f}"
            ),
            color=COLORS[mode],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(
                COLORS[mode] if mode == "cross" else "#D0D2D8"
            )
            spine.set_linewidth(2.2 if mode == "cross" else 0.8)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DOMAIN_COLORS[label],
            markeredgecolor="none",
            markersize=6,
            label=DOMAIN_SHORT[label],
        )
        for label in unique_labels
    ]
    fig.legend(
        handles=handles,
        ncol=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, base)


def plot_downstream_scaling(probes: dict, base: Path) -> list[Path]:
    token_aggregation = probes["chosen_token_aggregation"]
    available_representations = probes.get("representations", {})
    modes = tuple(
        mode
        for mode in METHODS
        if REPRESENTATIONS[mode].format(
            token_aggregation=token_aggregation
        )
        in available_representations
    )
    names = {
        mode: template.format(token_aggregation=token_aggregation)
        for mode, template in REPRESENTATIONS.items()
        if mode in modes
    }
    label_budgets = [
        int(value) for value in probes["metadata"]["low_label_budgets"]
    ]
    feature_budgets = [
        int(value) for value in probes["metadata"]["feature_budgets"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for mode in modes:
        row = probes["representations"][names[mode]]
        label_means = np.asarray(
            [
                row["low_label"][str(budget)]["mean_accuracy"]
                for budget in label_budgets
            ]
        )
        label_stds = np.asarray(
            [
                row["low_label"][str(budget)]["std_accuracy"]
                for budget in label_budgets
            ]
        )
        axes[0].plot(
            label_budgets,
            label_means,
            color=COLORS[mode],
            lw=2.5,
            marker="o",
            markersize=5,
            label=METHOD_LABELS[mode],
        )
        axes[0].fill_between(
            label_budgets,
            label_means - label_stds,
            label_means + label_stds,
            color=COLORS[mode],
            alpha=0.13,
            linewidth=0,
        )
        feature_values = [
            row["acc_at"][str(budget)]["accuracy"]
            for budget in feature_budgets
        ] + [row["full"]["accuracy"]]
        axes[1].plot(
            np.arange(len(feature_values)),
            feature_values,
            color=COLORS[mode],
            lw=2.5,
            marker="o",
            markersize=5,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(label_budgets, [str(value) for value in label_budgets])
    axes[0].set_xlabel("Training examples per class")
    axes[0].set_ylabel("8-way test accuracy")
    axes[0].set_title("(a) Label efficiency")
    axes[0].grid(color="#DDDEE4", lw=0.7)
    axes[0].legend(handles=_legend_for_modes(modes))
    axes[1].set_xticks(
        np.arange(len(feature_budgets) + 1),
        [str(value) for value in feature_budgets] + ["Dense"],
    )
    axes[1].set_xlabel("Selected SAE features")
    axes[1].set_ylabel("8-way test accuracy")
    axes[1].set_title("(b) Feature-budget efficiency")
    axes[1].grid(color="#DDDEE4", lw=0.7)
    fig.tight_layout()
    return _save(fig, base)


def plot_ood_robustness(
    probes: dict,
    geometry: dict,
    base: Path,
) -> list[Path]:
    token_aggregation = probes["chosen_token_aggregation"]
    names = {
        "token": f"token_sae_{token_aggregation}",
        "temporal": "temporal_sae_mean",
        "mean": "mean_chunk_sae",
        "cross": "cross_chunk_sae",
    }
    modes = tuple(
        mode
        for mode in METHODS
        if names[mode] in probes.get("representations", {})
        and mode in geometry.get("methods", {})
    )
    labels = list(probes["metadata"]["labels"])
    matrix = np.asarray(
        [
            [
                probes["representations"][names[mode]]["ood"][
                    "per_class_accuracy"
                ][label]
                for label in labels
            ]
            for mode in modes
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    image = axes[0].imshow(
        matrix,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0.60,
        vmax=0.90,
    )
    axes[0].set_xticks(
        np.arange(len(labels)),
        [DOMAIN_SHORT[label] for label in labels],
        rotation=35,
        ha="right",
    )
    axes[0].set_yticks(
        np.arange(len(modes)),
        [METHOD_SHORT_LABELS[mode] for mode in modes],
        fontweight="bold",
    )
    axes[0].set_title("(a) Time-OOD accuracy by scientific domain")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if matrix[row, column] > 0.80 else "#222222",
            )
    fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.025)

    cross_index = modes.index("cross")
    order = np.argsort(matrix[cross_index] - matrix[0])
    y = np.arange(len(labels))
    references = tuple(mode for mode in modes if mode != "cross")
    cross_delta_by_reference = {
        reference: matrix[cross_index, order] - matrix[index, order]
        for index, reference in enumerate(modes)
        if reference != "cross"
    }
    offsets = {
        mode: (
            (len(references) - 1) / 2 - index
        )
        * 0.18
        for index, mode in enumerate(references)
    }
    axes[1].axvline(0, color="#666666", lw=1)
    for reference, deltas in cross_delta_by_reference.items():
        offset = offsets[reference]
        axes[1].scatter(
            deltas,
            y + offset,
            s=42,
            color=COLORS[reference],
            label=(
                f"{METHOD_SHORT_LABELS['cross']} − "
                f"{METHOD_SHORT_LABELS[reference]}"
            ),
            zorder=3,
        )
        for index in range(len(labels)):
            axes[1].plot(
                [0, deltas[index]],
                [index + offset, index + offset],
                color=LIGHT_COLORS[reference],
                lw=2,
            )
    axes[1].set_yticks(
        y,
        [DOMAIN_SHORT[labels[index]] for index in order],
    )
    axes[1].set_xlabel("Accuracy difference")
    axes[1].set_title("(b) Cross-Chunk SAE advantage by domain")
    axes[1].grid(axis="x", color="#DDDEE4", lw=0.7)
    axes[1].legend(loc="lower right")
    worst = {
        mode: geometry["methods"][mode]["ood_class_balance"][
            "worst_class_accuracy"
        ]
        for mode in modes
    }
    axes[1].text(
        0.98,
        -0.22,
        (
            "Worst-domain OOD\n"
            f"BatchTopK {worst['token']:.3f} · "
            f"Temporal {worst['temporal']:.3f} · "
            f"Mean-Chunk {worst['mean']:.3f} · "
            f"Cross-Chunk {worst['cross']:.3f}"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": COLORS["cross"],
            "alpha": 0.9,
        },
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.25,
        top=0.90,
        wspace=0.34,
    )
    return _save(fig, base)


def plot_information_utilization(
    dictionary: dict,
    base: Path,
) -> list[Path]:
    """Render the one Eval 2 question: how evenly is the dictionary used?

    ``effective_feature_fraction`` is an entropy-equivalent fraction of the
    dictionary.  Multiplying it by the model's K=128 activation budget gives
    a compact, intuitive slot-equivalent scale without claiming that those
    slots are a fixed set of features on every example.
    """

    # Other publication figures may install their own rcParams before this
    # function is called; reset them so the artifact is deterministic.
    _style()
    modes = _available_modes(dictionary.get("methods", {}))
    if not modes:
        raise ValueError("dictionary utilization contains no complete methods")

    def method_fraction(mode: str) -> float:
        row = dictionary["methods"][mode]
        if "dictionary_utilization" in row:
            row = row["dictionary_utilization"]
        return float(row["effective_feature_fraction"])

    activation_budget = 128
    fractions = np.asarray(
        [method_fraction(mode) for mode in modes],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(fractions)) or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("effective feature fractions must lie in [0, 1]")
    equivalent_slots = activation_budget * fractions

    y = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.barh(
        y,
        np.full(len(modes), activation_budget, dtype=np.float64),
        height=0.62,
        color="#E8ECF2",
        edgecolor="white",
        linewidth=1.2,
        label="Remaining capacity",
        zorder=1,
    )
    ax.barh(
        y,
        equivalent_slots,
        height=0.62,
        color=[COLORS[mode] for mode in modes],
        edgecolor="white",
        linewidth=1.2,
        label="Entropy-equivalent content capacity",
        zorder=2,
    )
    ax.set_xlim(0, activation_budget * 1.23)
    ax.set_xticks(np.arange(0, activation_budget + 1, 16))
    ax.set_xlabel("Entropy-equivalent content slots in the K=128 budget")
    ax.set_ylabel("")
    ax.set_yticks(y, [METHOD_SHORT_LABELS[mode] for mode in modes])
    ax.invert_yaxis()
    ax.set_title(
        "Effective dictionary fraction: how many K=128 slots carry distinct content?",
        loc="left",
        pad=18,
    )
    ax.text(
        0.0,
        1.015,
        "Higher means activation mass is spread across more dictionary features; "
        "lower means a few features do most of the work. (8,192 alive-feature sample)",
        transform=ax.transAxes,
        color="#5C6270",
        fontsize=9.2,
        va="bottom",
    )
    for row, (mode, fraction, slots) in enumerate(
        zip(modes, fractions, equivalent_slots, strict=True)
    ):
        ax.text(
            slots + 1.5,
            row,
            f"{slots:.1f} / {activation_budget}  ({fraction:.1%})",
            va="center",
            ha="left",
            color=METHOD_TEXT_COLORS[mode],
            fontsize=10.5,
            fontweight="bold",
            clip_on=False,
        )
    ax.legend(
        handles=[
            Patch(
                facecolor="#E8ECF2",
                edgecolor="none",
                label="Capacity not represented as distinct, equally used slots",
            ),
            Patch(
                facecolor=COLORS["cross"],
                edgecolor="none",
                label="Effective content capacity",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        ncol=1,
        fontsize=8.3,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    ax.grid(axis="x", color="#DDDEE4", lw=0.7, zorder=0)
    ax.text(
        0.0,
        -0.20,
        "Effective fraction = exp(entropy of the dictionary-level activation distribution) "
        "/ sampled alive-feature count.  The 128-slot bar is an entropy-equivalent visualization.",
        transform=ax.transAxes,
        color="#5C6270",
        fontsize=8.3,
        va="top",
    )
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.22, top=0.84)
    return _save(fig, base)


def main() -> None:
    args = parser().parse_args()
    _style()
    training = _read(args.training_fidelity_results)
    probes = _read(args.probe_results)
    adjacent = _read(args.adjacent_consistency_results)
    dictionary = (
        _read(args.dictionary_utilization_results)
        if args.dictionary_utilization_results
        else adjacent
    )
    geometry = _read(args.representation_geometry_results)
    with np.load(args.representation_embeddings, allow_pickle=True) as data:
        embeddings = {key: data[key] for key in data.files}

    load_artifact_manifest(
        args.training_fidelity_manifest,
        expected_format="chunk-saes-training-fidelity-v2",
        verify_files=True,
    )
    load_artifact_manifest(
        args.probe_manifest,
        expected_format="chunk-saes-linear-probe-results-v2",
        verify_files=True,
    )
    load_artifact_manifest(
        args.adjacent_consistency_manifest,
        expected_format="chunk-saes-adjacent-feature-consistency-v1",
        verify_files=True,
    )
    load_artifact_manifest(
        args.representation_geometry_manifest,
        expected_format="chunk-saes-representation-geometry-v1",
        verify_files=True,
    )
    def destination(canonical: str, legacy: str | None, label: str) -> Path:
        value = getattr(args, canonical) or legacy
        if not value:
            raise ValueError(
                f"missing --{canonical.replace('_', '-')} (or legacy --{label}-plot-dir)"
            )
        return Path(value)

    rfve = destination("rfve_figure_dir", args.eval1_plot_dir, "eval1")
    dictionary_dir = destination(
        "dictionary_utilization_figure_dir", args.eval2_plot_dir, "eval2"
    )
    # A legacy Eval 3 destination receives all three plots.  Canonical runs
    # route each question to its own task directory.
    semantic_value = args.semantic_geometry_figure_dir or args.eval3_plot_dir
    label_value = args.label_efficiency_figure_dir or args.eval3_plot_dir
    temporal_value = args.temporal_robustness_figure_dir or args.eval3_plot_dir
    if not semantic_value or not label_value or not temporal_value:
        raise ValueError(
            "missing --semantic-geometry-figure-dir "
            "(or legacy --eval3-plot-dir)"
        )
    semantic_geometry = Path(semantic_value)
    label_efficiency = Path(label_value)
    temporal_robustness = Path(temporal_value)
    plot_training_fidelity(
        training,
        rfve / "training_fidelity",
        # RFVE is a published protected raster in this layout. New
        # evaluations may refresh the vector exports, but must never
        # replace the checked-in PNG implicitly when an optional joint
        # reference is present.
        overwrite_png=False,
    )
    plot_semantic_manifold(
        geometry,
        embeddings,
        semantic_geometry / "semantic_geometry",
    )
    plot_downstream_scaling(
        probes,
        label_efficiency / "label_efficiency",
    )
    plot_ood_robustness(
        probes,
        geometry,
        temporal_robustness / "temporal_robustness",
    )
    plot_information_utilization(
        dictionary,
        dictionary_dir / "dictionary_utilization",
    )


if __name__ == "__main__":
    main()
