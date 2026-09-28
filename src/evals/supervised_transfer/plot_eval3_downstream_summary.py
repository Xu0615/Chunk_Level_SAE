#!/usr/bin/env python
"""Render a compact endpoint summary of Eval 3 downstream utility.

The historical Eval 3 plots show curves and domain-level matrices separately.
This figure keeps only the final, decision-relevant values so the label,
feature, and time-shift results can be read together at a glance.  It writes
to a new output base and never overwrites the legacy plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from chunk_saes.plot_style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_PALE_COLORS,
    METHOD_SHORT_LABELS,
    METHODS,
    style_figure_text,
)


WHITE = "#FFFFFF"
PANEL = "#F7F8FB"
DARK = "#20242C"
MID = "#667085"
GRID = "#DDE2EA"
EDGE = "#C8CFD9"
CHANCE = "#98A2B3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the compact final-endpoint Eval 3 summary figure."
    )
    parser.add_argument("--probe-results", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--dpi", type=int, default=320)
    return parser


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_rows(probes: dict) -> dict[str, dict]:
    token_aggregation = probes["chosen_token_aggregation"]
    names = {
        "token": f"token_sae_{token_aggregation}",
        "temporal": "temporal_sae_mean",
        "mean": "mean_chunk_sae",
        "cross": "cross_chunk_sae",
    }
    representations = probes.get("representations", {})
    missing = [
        f"{mode}: {name}"
        for mode, name in names.items()
        if name not in representations
    ]
    if missing:
        raise KeyError("missing canonical probe representations: " + ", ".join(missing))
    return {mode: representations[name] for mode, name in names.items()}


def _trapezoid(values: np.ndarray, x: np.ndarray) -> float:
    # numpy renamed trapz in newer releases; keep the script portable across
    # the environments used to render the other evaluation figures.
    integrate = (
        np.trapezoid
        if hasattr(np, "trapezoid")
        else np.trapz
    )
    return float(integrate(values, x))


def _metrics(probes: dict, rows: dict[str, dict]) -> dict[str, dict[str, float]]:
    budgets = np.asarray(
        [int(value) for value in probes["metadata"]["low_label_budgets"]],
        dtype=np.float64,
    )
    log_budgets = np.log2(budgets)
    span = max(float(log_budgets[-1] - log_budgets[0]), 1e-12)
    result: dict[str, dict[str, float]] = {}
    for mode in METHODS:
        row = rows[mode]
        low_curve = np.asarray(
            [
                float(row["low_label"][str(int(budget))]["mean_accuracy"])
                for budget in budgets
            ],
            dtype=np.float64,
        )
        full = float(row["full"]["accuracy"])
        ood = float(row["ood"]["accuracy"])
        per_class = row["ood"]["per_class_accuracy"]
        worst = float(min(float(value) for value in per_class.values()))
        result[mode] = {
            "low_label_auc": _trapezoid(low_curve, log_budgets) / span,
            # ``full`` is the final Dense endpoint in the feature-budget plot.
            "dense_accuracy": full,
            "ood_accuracy": ood,
            "worst_ood_accuracy": worst,
            "ood_retention": ood / max(full, 1e-12),
        }
    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "font.weight": "medium",
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelweight": "bold",
            "axes.facecolor": PANEL,
            "axes.edgecolor": EDGE,
            "axes.linewidth": 0.9,
            "xtick.color": MID,
            "ytick.color": MID,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.2,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "text.color": DARK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_title(ax: plt.Axes, letter: str, title: str, subtitle: str) -> None:
    ax.set_title(f"{letter}   {title}", loc="left", pad=18, color=DARK)
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="medium",
        color=MID,
    )


def _finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, lw=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=3, color=MID)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)


def _highlight_cross(ax: plt.Axes, y: np.ndarray) -> None:
    cross_index = list(METHODS).index("cross")
    ax.axhspan(
        y[cross_index] - 0.43,
        y[cross_index] + 0.43,
        color=METHOD_PALE_COLORS["cross"],
        alpha=0.9,
        zorder=0,
    )


def _lollipop_panel(
    ax: plt.Axes,
    values: dict[str, float],
    *,
    letter: str,
    title: str,
    subtitle: str,
    xlim: tuple[float, float],
    xticks: list[float],
    xlabel: str,
    value_format: str = ".3f",
) -> None:
    _panel_title(ax, letter, title, subtitle)
    y = np.arange(len(METHODS), dtype=np.float64)
    _highlight_cross(ax, y)
    for index, mode in enumerate(METHODS):
        value = float(values[mode])
        color = METHOD_COLORS[mode]
        ax.hlines(
            index,
            xlim[0],
            value,
            color=color,
            lw=4.5 if mode == "cross" else 3.5,
            alpha=0.42 if mode != "cross" else 0.72,
            zorder=2,
        )
        ax.scatter(
            value,
            index,
            s=112 if mode == "cross" else 92,
            color=color,
            edgecolor=WHITE,
            linewidth=1.4,
            zorder=3,
        )
        ax.text(
            value + (xlim[1] - xlim[0]) * 0.018,
            index,
            format(value, value_format),
            ha="left",
            va="center",
            fontsize=9.4,
            fontweight="bold" if mode == "cross" else "medium",
            color=DARK,
            clip_on=False,
            zorder=4,
        )
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([format(value, ".2f") for value in xticks])
    ax.set_xlabel(xlabel, labelpad=9)
    ax.set_yticks(y, [METHOD_SHORT_LABELS[mode] for mode in METHODS])
    ax.invert_yaxis()
    _finish_axis(ax)


def _ood_panel(ax: plt.Axes, metrics: dict[str, dict[str, float]]) -> None:
    _panel_title(
        ax,
        "C",
        "Time-OOD accuracy",
        "Overall and worst domain across 8 scientific classes",
    )
    y = np.arange(len(METHODS), dtype=np.float64)
    _highlight_cross(ax, y)
    x_min, x_max = 0.60, 0.86
    for index, mode in enumerate(METHODS):
        overall = metrics[mode]["ood_accuracy"]
        worst = metrics[mode]["worst_ood_accuracy"]
        color = METHOD_COLORS[mode]
        ax.hlines(
            index,
            worst,
            overall,
            color=color,
            lw=5.0 if mode == "cross" else 4.0,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            overall,
            index,
            s=108 if mode == "cross" else 90,
            color=color,
            edgecolor=WHITE,
            linewidth=1.4,
            zorder=4,
        )
        ax.scatter(
            worst,
            index,
            s=78 if mode == "cross" else 66,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=2.0,
            zorder=4,
        )
        ax.text(
            overall + 0.006,
            index - 0.16,
            f"{overall:.3f}",
            ha="left",
            va="center",
            fontsize=8.7,
            fontweight="bold" if mode == "cross" else "medium",
            color=DARK,
            clip_on=False,
        )
        ax.text(
            worst - 0.006,
            index + 0.18,
            f"{worst:.3f}",
            ha="right",
            va="center",
            fontsize=8.1,
            color=MID,
            clip_on=False,
        )
    ax.set_xlim(x_min, x_max)
    ax.set_xticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    ax.set_xticklabels([f"{value:.2f}" for value in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]])
    ax.set_xlabel("8-way accuracy", labelpad=9)
    ax.set_yticks(y, [METHOD_SHORT_LABELS[mode] for mode in METHODS])
    ax.invert_yaxis()
    _finish_axis(ax)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=METHOD_COLORS["cross"],
                markeredgecolor=WHITE,
                markeredgewidth=1.0,
                markersize=7,
                label="Overall OOD",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=WHITE,
                markeredgecolor=METHOD_COLORS["cross"],
                markeredgewidth=1.6,
                markersize=7,
                label="Worst domain",
            ),
        ],
        loc="lower right",
        fontsize=8.0,
        handlelength=1.0,
        frameon=True,
        facecolor=WHITE,
        edgecolor=GRID,
        framealpha=0.96,
        borderpad=0.45,
    )


def _retention_panel(ax: plt.Axes, metrics: dict[str, dict[str, float]]) -> None:
    values = {mode: metrics[mode]["ood_retention"] for mode in METHODS}
    _lollipop_panel(
        ax,
        values,
        letter="D",
        title="OOD retention",
        subtitle="OOD accuracy divided by ordinary test accuracy",
        xlim=(0.925, 1.005),
        xticks=[0.93, 0.95, 0.97, 0.99, 1.00],
        xlabel="Retained performance (%)",
        value_format=".1%",
    )
    ax.set_xticklabels(["93", "95", "97", "99", "100"])
    ax.axvline(1.0, color=CHANCE, lw=1.1, ls=(0, (3, 3)), zorder=1)
    ax.text(
        0.995,
        0.03,
        "1.00 = no time-shift loss",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=MID,
    )


def _save(fig: plt.Figure, base: Path, dpi: int) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    style_figure_text(fig, minimum_tick_size=8.5)
    paths = [base.with_suffix(suffix) for suffix in (".png", ".pdf")]
    fig.savefig(
        paths[0],
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.10,
        metadata={
            "Title": "Eval 3 downstream utility endpoint summary",
            "Creator": "Chunk-SAE evaluation pipeline",
        },
    )
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return paths


def render(probes: dict, output_base: Path, dpi: int) -> list[Path]:
    rows = _canonical_rows(probes)
    metrics = _metrics(probes, rows)
    _style()

    fig = plt.figure(figsize=(16.0, 9.0), facecolor=WHITE)
    fig.text(
        0.065,
        0.955,
        "Frozen SAE codes: final downstream utility",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=DARK,
    )
    fig.text(
        0.065,
        0.915,
        "Eight-way ArXiv domain probe | external multinomial logistic regression | endpoint view",
        ha="left",
        va="top",
        fontsize=10.5,
        color=MID,
    )
    fig.add_artist(
        plt.Line2D(
            [0.065, 0.935],
            [0.885, 0.885],
            transform=fig.transFigure,
            color=METHOD_COLORS["cross"],
            lw=2.4,
            solid_capstyle="round",
        )
    )
    cross = metrics["cross"]
    fig.text(
        0.935,
        0.955,
        "CROSS-CHUNK SAE",
        ha="right",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=METHOD_COLORS["cross"],
    )
    fig.text(
        0.935,
        0.915,
        (
            f"AUC {cross['low_label_auc']:.3f}  |  Dense {cross['dense_accuracy']:.3f}  |  "
            f"OOD {cross['ood_accuracy']:.3f}  |  Retained {cross['ood_retention']:.1%}"
        ),
        ha="right",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=DARK,
    )

    grid = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.935,
        bottom=0.145,
        top=0.835,
        wspace=0.24,
        hspace=0.42,
        height_ratios=[1.0, 1.10],
    )
    low_auc = {mode: metrics[mode]["low_label_auc"] for mode in METHODS}
    dense = {mode: metrics[mode]["dense_accuracy"] for mode in METHODS}
    _lollipop_panel(
        fig.add_subplot(grid[0, 0]),
        low_auc,
        letter="A",
        title="Label efficiency",
        subtitle="Low-label AUC over 1-256 labels per class (log2 area)",
        xlim=(0.695, 0.742),
        xticks=[0.70, 0.72, 0.74],
        xlabel="Low-label AUC",
    )
    _lollipop_panel(
        fig.add_subplot(grid[0, 1]),
        dense,
        letter="B",
        title="Feature budget",
        subtitle="Dense endpoint = all 65,536 SAE features",
        xlim=(0.81, 0.862),
        xticks=[0.82, 0.84, 0.86],
        xlabel="Dense 8-way test accuracy",
    )
    _ood_panel(fig.add_subplot(grid[1, 0]), metrics)
    _retention_panel(fig.add_subplot(grid[1, 1]), metrics)

    fig.text(
        0.085,
        0.075,
        "OOD split: first submission year >= 2023 | Worst domain: minimum of the 8 OOD class accuracies",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MID,
    )
    fig.text(
        0.935,
        0.075,
        "Higher is better",
        ha="right",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=DARK,
    )
    return _save(fig, output_base, dpi)


def main() -> None:
    args = _parser().parse_args()
    paths = render(
        _read(Path(args.probe_results)),
        Path(args.output_base),
        args.dpi,
    )
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
