#!/usr/bin/env python
"""Render one 2:1 comparison figure for each selected summary metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

try:
    from .evaluation_summary_data import (
        METHODS,
        METHOD_COLORS,
        METHOD_LABELS,
        METHOD_MARKERS,
        MetricSpec,
        load_metric_specs,
        serializable_metric_summary,
    )
except ImportError:
    from evaluation_summary_data import (  # type: ignore[no-redef]
        METHODS,
        METHOD_COLORS,
        METHOD_LABELS,
        METHOD_MARKERS,
        MetricSpec,
        load_metric_specs,
        serializable_metric_summary,
    )


TEXT = "#252932"
MUTED = "#626A76"
GRID = "#E2E6EC"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--eval-root", required=True)
    result.add_argument("--output-dir", default=None)
    result.add_argument("--dpi", type=int, default=240)
    return result


def _format_value(metric: MetricSpec, value: float) -> str:
    if metric.display == "percent":
        return f"{100.0 * value:.1f}%"
    if metric.display == "decimal3":
        return f"{value:.3f}"
    if metric.display == "score1":
        return f"{value:.1f}"
    raise ValueError(f"unknown display mode: {metric.display}")


def _render_metric(
    metric: MetricSpec,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(12.0, 6.0), facecolor="white")
    fig.subplots_adjust(left=0.335, right=0.945, top=0.76, bottom=0.18)
    ax.set_facecolor("white")
    show_annotations = metric.key != "causal_steering"
    font_scale = 1.2 if metric.key == "causal_steering" else 1.0

    y_positions = np.arange(len(METHODS), dtype=np.float64)[::-1]
    if metric.display == "score1":
        axis_low, axis_high = 0.0, 100.0
    elif metric.key in {"feature_persistence_lift", "dictionary_utilization"}:
        axis_low, axis_high = 0.0, 0.35
    elif metric.key == "high_level_feature_fraction":
        axis_low, axis_high = 0.0, 0.40
    else:
        axis_low, axis_high = 0.0, 1.0
    span = axis_high - axis_low
    label_offset = 0.012 * span

    bar_values = [float(metric.values[method]) for method in METHODS]
    bars = ax.barh(
        y_positions,
        bar_values,
        height=0.62,
        color=[METHOD_COLORS[method] for method in METHODS],
        edgecolor="white",
        linewidth=1.3,
        zorder=2,
    )
    if metric.key == "causal_steering":
        best_bar = bars[int(np.argmax(bar_values))]
        best_bar.set_edgecolor(TEXT)
        best_bar.set_linewidth(2.5)
        best_bar.set_clip_on(False)

    for y, method in zip(y_positions, METHODS, strict=True):
        value = float(metric.values[method])
        interval = (
            metric.confidence_intervals.get(method)
            if metric.confidence_intervals is not None
            else None
        )
        if interval is not None:
            lower, upper = (float(interval[0]), float(interval[1]))
            xerr = np.asarray(
                [[max(value - lower, 0.0)], [max(upper - value, 0.0)]],
                dtype=np.float64,
            )
            ax.errorbar(
                [value],
                [y],
                xerr=xerr,
                fmt="none",
                ecolor=METHOD_COLORS[method],
                elinewidth=2.4,
                capsize=5.0,
                capthick=2.0,
                zorder=3,
            )
            label_anchor = max(value, upper)
        else:
            label_anchor = value

        place_left = label_anchor > axis_high - 0.075 * span
        text_x = (
            (interval[0] if interval is not None else value) - label_offset
            if place_left
            else label_anchor + label_offset
        )
        ax.text(
            text_x,
            y,
            _format_value(metric, value),
            ha="right" if place_left else "left",
            va="center",
            fontsize=15.0 * font_scale,
            fontweight="bold",
            color=TEXT,
        )

    if show_annotations and metric.neutral_value is not None:
        ax.axvline(
            metric.neutral_value,
            color="#7A828E",
            linewidth=1.6,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        ax.text(
            metric.neutral_value + 0.008 * span,
            -0.72,
            f"calibrated null = {_format_value(metric, metric.neutral_value)}",
            ha="left",
            va="center",
            fontsize=10.5,
            color=MUTED,
        )

    ax.set_xlim(axis_low, axis_high)
    ax.set_ylim(-0.75, len(METHODS) - 0.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [METHOD_LABELS[method] for method in METHODS],
        fontsize=14.0 * font_scale,
        fontweight="bold",
    )
    ax.set_xlabel(
        metric.axis_label,
        fontsize=19.0 if metric.key == "causal_steering" else 14.0 * font_scale,
        fontweight="bold",
        labelpad=12,
    )
    ax.tick_params(axis="x", labelsize=12.0 * font_scale)
    if metric.key == "causal_steering":
        for tick_label in ax.get_xticklabels():
            tick_label.set_color("black")
            tick_label.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=12)
    if metric.display == "percent":
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.xaxis.grid(True, color=GRID, linewidth=1.1)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9DA5B1")
    ax.spines["bottom"].set_linewidth(1.1)

    if show_annotations:
        fig.text(
            0.07,
            0.925,
            metric.title,
            ha="left",
            va="top",
            fontsize=24.0,
            fontweight="bold",
            color=TEXT,
        )
        fig.text(
            0.07,
            0.855,
            metric.subtitle,
            ha="left",
            va="top",
            fontsize=13.0,
            color=MUTED,
        )
        fig.text(
            0.07,
            0.035,
            f"Source: {metric.source}",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#7A828E",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metric.order:02d}_{metric.key}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def render(eval_root: Path, output_dir: Path, dpi: int) -> list[Path]:
    metrics = load_metric_specs(eval_root)
    outputs: list[Path] = []
    for metric in metrics:
        outputs.extend(_render_metric(metric, output_dir, dpi))

    manifest = output_dir.parent / "selected_metrics.json"
    manifest.write_text(
        json.dumps(
            serializable_metric_summary(metrics),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(manifest)
    return outputs


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else eval_root / "evaluation_summary" / "figures"
    )
    print(
        "\n".join(str(path) for path in render(eval_root, output_dir, int(args.dpi))),
        flush=True,
    )


if __name__ == "__main__":
    main()
