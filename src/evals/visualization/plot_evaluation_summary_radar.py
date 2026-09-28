#!/usr/bin/env python
"""Render the selected six-by-six SAE evaluation summary radar figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

try:
    from .evaluation_summary_data import (
        METHODS,
        METHOD_COLORS,
        METHOD_LABELS,
        METHOD_MARKERS,
        MetricSpec,
        load_metric_specs,
    )
except ImportError:
    from evaluation_summary_data import (  # type: ignore[no-redef]
        METHODS,
        METHOD_COLORS,
        METHOD_LABELS,
        METHOD_MARKERS,
        MetricSpec,
        load_metric_specs,
    )


TEXT = "#272C38"
DEEP_BLUE = "#0C445E"
ACCENT_BLUE = "#307DCA"
BRICK_RED = "#893A37"
SLATE = "#637484"
GRID = "#D5E2EA"
OUTER_GRID = "#99BAD1"

GROUP_COLORS = (DEEP_BLUE, BRICK_RED, SLATE)

RADAR_LABELS = {
    "reconstruction_cosine": "Reconstruction\ncosine",
    "context_autointerp": "Focused\nInterpretability",
    "high_level_feature_fraction": "High-level\nfeature fraction",
    "feature_persistence_lift": "Feature persistence\nlift",
    "dictionary_utilization": "Dictionary\nutilization",
    "reasoning_native_recall": "Native reasoning\nrecall",
    "reasoning_generalization": "Reasoning\ngeneralization",
    "document_recall_at_5": "Document\nRecall@5",
}

METHOD_TEXT_COLORS = {
    "token": "#547E92",
    "temporal": ACCENT_BLUE,
    "mean": "#A95F49",
    "joint_alpha0p25": "#9B6A1D",
    "cross": BRICK_RED,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--eval-root", required=True)
    result.add_argument("--output-dir", default=None)
    result.add_argument("--output-name", default="evaluation_summary")
    result.add_argument("--dpi", type=int, default=240)
    return result


def _normalise(value: float, metric: MetricSpec) -> float:
    width = metric.radar_high - metric.radar_low
    if width <= 0.0:
        raise ValueError(f"invalid radar range for {metric.key}")
    return float(np.clip((value - metric.radar_low) / width, 0.0, 1.0))


def _format_axis_tick(metric: MetricSpec, value: float) -> str:
    if metric.display == "percent":
        return f"{100.0 * value:.0f}%"
    if metric.display == "score1":
        return f"{value:.0f}"
    if metric.key in {
        "reconstruction_cosine",
        "feature_persistence_lift",
        "arxiv_low_label_auc",
    }:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _add_metric_ticks(
    ax: plt.Axes,
    metrics: tuple[MetricSpec, ...],
    angles: np.ndarray,
    *,
    label_colors: tuple[str, ...],
    fontsize: float = 15.5,
) -> None:
    """Label each spoke with its own metric-scale values inside the radar."""

    ring_positions = (0.25, 0.50, 0.75)
    for angle, metric, color in zip(
        angles,
        metrics,
        label_colors,
        strict=True,
    ):
        # Put labels consistently on the clockwise side of each spoke.
        tick_angle = float(angle + 0.042)
        display_angle = float(
            (
                np.degrees(
                    ax.get_theta_offset()
                    + ax.get_theta_direction() * float(angle)
                )
                + 360.0
            )
            % 360.0
        )
        horizontal = float(np.cos(np.radians(display_angle)))
        if abs(horizontal) < 0.15:
            alignment = "left"
        elif horizontal > 0.0:
            alignment = "left"
        else:
            alignment = "right"
            tick_angle = float(angle - 0.042)

        for radius in ring_positions:
            value = metric.radar_low + radius * (
                metric.radar_high - metric.radar_low
            )
            ax.text(
                tick_angle,
                radius,
                _format_axis_tick(metric, value),
                ha=alignment,
                va="center",
                fontsize=fontsize,
                fontweight="semibold",
                fontstretch="condensed",
                color=color,
                zorder=12,
            )


def _add_metric_labels(
    ax: plt.Axes,
    metrics: tuple[MetricSpec, ...],
    angles: np.ndarray,
    *,
    label_colors: tuple[str, ...],
    radius: float = 1.165,
    fontsize: float = 21.0,
) -> None:
    """Place metric names outside the radar, tangent to its circumference."""

    for angle, metric, color in zip(
        angles,
        metrics,
        label_colors,
        strict=True,
    ):
        display_angle = float(
            (
                np.degrees(
                    ax.get_theta_offset()
                    + ax.get_theta_direction() * float(angle)
                )
                + 360.0
            )
            % 360.0
        )
        rotation = display_angle - 90.0
        while rotation > 90.0:
            rotation -= 180.0
        while rotation < -90.0:
            rotation += 180.0
        ax.text(
            float(angle),
            radius,
            RADAR_LABELS.get(metric.key, metric.radar_label),
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            multialignment="center",
            linespacing=0.92,
            fontsize=fontsize,
            fontweight="bold",
            fontstretch="condensed",
            color=color,
            clip_on=False,
            zorder=15,
        )
        # Every label uses the same radial anchor. Since the text is tangent
        # to the circle, this gives diagonal labels the same visual clearance
        # as the labels on the horizontal left/right spokes. Do not use the
        # axis-aligned rendered bounding box to push diagonal text outward:
        # that box substantially overestimates its inward extent.


def _draw_radar(
    ax: plt.Axes,
    metrics: tuple[MetricSpec, ...],
    *,
    label_colors: tuple[str, ...],
    metric_label_radius: float = 1.165,
    metric_label_fontsize: float = 21.0,
    tick_fontsize: float = 15.5,
) -> np.ndarray:
    count = len(metrics)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    # Rotate the complete radar 30 degrees clockwise from the conventional
    # north-up orientation.
    ax.set_theta_offset(0.5 * np.pi - np.deg2rad(30.0))
    ax.set_theta_direction(-1)
    ax.set_thetamin(0.0)
    ax.set_thetamax(360.0)
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, 1.06)
    ax.set_facecolor("white")

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", pad=0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.95)
    ax.spines["polar"].set_color(OUTER_GRID)
    ax.spines["polar"].set_linewidth(1.5)

    for method in METHODS:
        values = np.asarray(
            [_normalise(float(metric.values[method]), metric) for metric in metrics],
            dtype=np.float64,
        )
        values = np.r_[values, values[0]]
        is_token_level = method in {"token", "temporal"}
        ax.plot(
            closed_angles,
            values,
            color=METHOD_COLORS[method],
            linewidth=3.0,
            linestyle=(0, (4, 2.4)) if is_token_level else "-",
            marker=METHOD_MARKERS[method],
            markersize=8.2,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=5,
        )
        ax.fill(
            closed_angles,
            values,
            color=METHOD_COLORS[method],
            alpha=0.055,
            zorder=2,
        )
    _add_metric_ticks(
        ax,
        metrics,
        angles,
        label_colors=label_colors,
        fontsize=tick_fontsize,
    )
    _add_metric_labels(
        ax,
        metrics,
        angles,
        label_colors=label_colors,
        radius=metric_label_radius,
        fontsize=metric_label_fontsize,
    )
    return angles


def _add_pair_brace(
    fig: plt.Figure,
    ax: plt.Axes,
    angles: np.ndarray,
    first: int,
    second: int,
    label: str,
    *,
    color: str,
) -> None:
    """Draw a compact standard brace outside a pair of metric labels."""

    # Work in physical inches so the brace is not distorted by the 2:1 canvas.
    dpi = float(fig.dpi)
    center = np.asarray(ax.transAxes.transform((0.5, 0.5)), dtype=np.float64) / dpi

    # Place the brace on a fixed outer ring. Its long axis is tangent to the
    # radar at the midpoint of the two grouped metrics.
    first_angle = float(angles[first])
    second_angle = float(angles[second])
    if second_angle <= first_angle:
        second_angle += 2.0 * np.pi
    midpoint_angle = 0.5 * (first_angle + second_angle)
    # Keep the grouping brace visually close to the concrete metric names.
    # The group heading remains attached just outside the brace below.
    brace_radius = 1.36
    endpoints = [
        np.asarray(
            ax.transData.transform((angle, brace_radius)),
            dtype=np.float64,
        )
        / dpi
        for angle in (first_angle, second_angle)
    ]
    tangent = endpoints[1] - endpoints[0]
    tangent /= float(np.linalg.norm(tangent))
    tangent_angle = float(np.arctan2(tangent[1], tangent[0]))
    brace_center = (
        np.asarray(
            ax.transData.transform((midpoint_angle, brace_radius)),
            dtype=np.float64,
        )
        / dpi
    )
    outward = brace_center - center
    outward /= float(np.linalg.norm(outward))
    right_normal = np.asarray([tangent[1], -tangent[0]])

    # A vertical brace is rotated tangentially. Choose the orientation whose
    # center cusp points outward toward the group heading.
    brace_character = "}" if float(np.dot(right_normal, outward)) >= 0.0 else "{"
    glyph = TextPath(
        (0.0, 0.0),
        brace_character,
        size=1.0,
        prop=FontProperties(family="STIXGeneral"),
    )
    bounds = glyph.get_extents()
    brace_depth_inches = 0.105
    brace_length_inches = 2.50
    transform = (
        Affine2D()
        .translate(
            -bounds.x0 - 0.5 * bounds.width,
            -bounds.y0 - 0.5 * bounds.height,
        )
        .scale(
            brace_depth_inches / bounds.width,
            brace_length_inches / bounds.height,
        )
        # The source brace's long direction is vertical.
        .rotate(tangent_angle - 0.5 * np.pi)
        .translate(float(brace_center[0]), float(brace_center[1]))
    )
    patch = PathPatch(
        glyph,
        transform=transform + fig.dpi_scale_trans,
        facecolor=color,
        edgecolor="none",
        clip_on=False,
        zorder=20,
    )
    fig.add_artist(patch)

    label_style = {
        "ha": "center",
        "va": "center",
        "fontsize": 23.0,
        "fontweight": "bold",
        "fontstretch": "condensed",
        "color": color,
    }
    label_rotation = float(np.degrees(tangent_angle))
    while label_rotation > 90.0:
        label_rotation -= 180.0
    while label_rotation < -90.0:
        label_rotation += 180.0
    # First lay the text out without rotation. Once rotated tangentially, its
    # unrotated height is exactly its thickness in the radial direction. This
    # avoids using an axis-aligned bbox for slanted text, which would
    # overestimate the radial thickness and push diagonal headings too far
    # away from their braces.
    label_artist = fig.text(
        0.0,
        0.0,
        label,
        **label_style,
        rotation=0.0,
        rotation_mode="anchor",
        multialignment="center",
        linespacing=0.94,
        zorder=21,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    brace_vertices = (
        patch.get_path().transformed(patch.get_transform()).vertices / dpi
    )
    brace_outer_edge = float(
        np.max((brace_vertices - brace_center) @ outward)
    )
    label_bounds = label_artist.get_window_extent(renderer=renderer)
    clearance_inches = 0.035
    label_half_depth_inches = 0.5 * float(label_bounds.height) / dpi
    label_center_distance_inches = (
        brace_outer_edge + clearance_inches + label_half_depth_inches
    )
    label_position_inches = (
        brace_center + label_center_distance_inches * outward
    )
    label_artist.set_position(
        (
            float(label_position_inches[0] / fig.get_figwidth()),
            float(label_position_inches[1] / fig.get_figheight()),
        )
    )
    label_artist.set_rotation(label_rotation)


def render(
    eval_root: Path,
    output_dir: Path,
    output_name: str,
    dpi: int,
) -> list[Path]:
    metrics = load_metric_specs(eval_root)
    intrinsic = tuple(metric for metric in metrics if metric.group == "intrinsic")
    downstream = tuple(metric for metric in metrics if metric.group == "downstream")
    if len(intrinsic) != 6 or len(downstream) != 6:
        raise RuntimeError(
            f"expected six metrics per radar, got {len(intrinsic)} and "
            f"{len(downstream)}"
        )

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Match the construction DPI to the export DPI. The grouping braces are
    # positioned in physical units so this also keeps them identical in PNG
    # and PDF output.
    fig = plt.figure(figsize=(18.0, 9.0), dpi=dpi, facecolor="white")
    # Leave a little more room between the panels for the slanted headings
    # outside the inward-facing braces.  Keep each polar axes physically
    # square on the 2:1 canvas.
    left = fig.add_axes([0.097, 0.257, 0.295, 0.590], projection="polar")
    # Give the downstream radar a slightly larger physical diameter. Its
    # longer metric names then occupy a wider circumference and read less
    # densely without changing their equal angular spacing.
    right = fig.add_axes([0.591, 0.247, 0.315, 0.630], projection="polar")
    label_colors = tuple(color for color in GROUP_COLORS for _ in range(2))
    left_angles = _draw_radar(
        left,
        intrinsic,
        label_colors=label_colors,
    )
    right_angles = _draw_radar(
        right,
        downstream,
        label_colors=label_colors,
        metric_label_radius=1.170,
    )

    # The left radar is organized into three paired dimensions.
    _add_pair_brace(
        fig,
        left,
        left_angles,
        0,
        1,
        "Reconstruction",
        color=GROUP_COLORS[0],
    )
    _add_pair_brace(
        fig,
        left,
        left_angles,
        2,
        3,
        "Interpretability",
        color=GROUP_COLORS[1],
    )
    _add_pair_brace(
        fig,
        left,
        left_angles,
        4,
        5,
        "Feature-level\nproperties",
        color=GROUP_COLORS[2],
    )
    # The first four downstream axes form two paired evaluations.
    _add_pair_brace(
        fig,
        right,
        right_angles,
        0,
        1,
        "Classification transfer",
        color=GROUP_COLORS[0],
    )
    _add_pair_brace(
        fig,
        right,
        right_angles,
        2,
        3,
        "Reasoning",
        color=GROUP_COLORS[1],
    )
    _add_pair_brace(
        fig,
        right,
        right_angles,
        4,
        5,
        "Steering &\nretrieval",
        color=GROUP_COLORS[2],
    )

    fig.text(
        0.251,
        0.984,
        "Representation quality and interpretability",
        ha="center",
        va="top",
        fontsize=26.0,
        fontweight="bold",
        color=DEEP_BLUE,
    )
    fig.text(
        0.7525,
        0.984,
        "Downstream utility",
        ha="center",
        va="top",
        fontsize=26.0,
        fontweight="bold",
        color=BRICK_RED,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=3.2,
            linestyle=(0, (4, 2.4)) if method in {"token", "temporal"} else "-",
            marker=METHOD_MARKERS[method],
            markersize=9.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.012, 0.022, 0.976, 0.078),
        mode="expand",
        ncol=5,
        frameon=False,
        fontsize=20.5,
        handlelength=1.85,
        handletextpad=0.30,
        columnspacing=0.30,
        borderaxespad=0.0,
    )
    for method, text in zip(METHODS, legend.get_texts(), strict=True):
        text.set_fontweight("bold")
        text.set_fontstretch("condensed")
        text.set_color(METHOD_TEXT_COLORS[method])

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / output_name
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else eval_root / "evaluation_summary" / "figures"
    )
    print(
        "\n".join(
            str(path)
            for path in render(
                eval_root,
                output_dir,
                str(args.output_name),
                int(args.dpi),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
