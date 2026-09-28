"""Camera-ready motivation figure for the token-SAE granularity gap.

This module is intentionally independent of the experiment implementation.
It consumes the stored diagnostics and turns them into a clean visual
argument:

    token-level SAE family -> shared granularity defect -> passage failure
    -> design requirement for a high-level, interpretable SAE.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


# Editorial palette: cool colors describe the existing token-SAE family,
# coral marks the failure, and green marks the proposed design direction.
BG = "#F4F6F9"
PAPER = "#FFFFFF"
INK = "#122033"
MUTED = "#5D6A7C"
FAINT = "#8D99A8"
BORDER = "#DDE4EC"
HAIRLINE = "#E9EDF2"

NAVY = "#183B56"
BLUE = "#2D6CDF"
BLUE_2 = "#4E8BE8"
BLUE_PALE = "#EDF4FF"
CYAN = "#21A6C7"

CORAL = "#D64B5F"
CORAL_DARK = "#A92F43"
CORAL_PALE = "#FFF0F2"
AMBER = "#E6A23C"
AMBER_PALE = "#FFF7E7"

GREEN = "#13866F"
GREEN_DARK = "#0B6958"
GREEN_PALE = "#EAF7F3"
MINT = "#CDEDE4"

PURPLE = "#6F58A5"
PURPLE_PALE = "#F2EEFA"


def _mix(color: str, amount: float = 0.88) -> tuple[float, float, float]:
    """Mix a matplotlib color with white."""

    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    return tuple(rgb * (1.0 - amount) + amount)


def _card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = PAPER,
    edgecolor: str = BORDER,
    linewidth: float = 0.9,
    radius: float = 0.018,
    shadow: bool = False,
    zorder: float = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    if shadow:
        patch.set_path_effects(
            [
                path_effects.SimplePatchShadow(
                    offset=(1.8, -1.8),
                    shadow_rgbFace="#1B2A40",
                    alpha=0.10,
                    rho=0.98,
                ),
                path_effects.Normal(),
            ]
        )
    ax.add_patch(patch)
    return patch


def _pill(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    color: str,
    edgecolor: str = "none",
    fontsize: float = 7.2,
    weight: str = "bold",
    zorder: float = 5,
) -> None:
    _card(
        ax,
        x,
        y,
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        radius=height / 2,
        zorder=zorder,
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        zorder=zorder + 1,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.7,
    mutation_scale: float = 14,
    connectionstyle: str = "arc3",
    zorder: float = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def _section_header(
    ax: plt.Axes,
    x: float,
    y: float,
    number: str,
    label: str,
    *,
    color: str,
) -> None:
    ax.add_patch(
        Circle(
            (x + 0.014, y),
            0.014,
            facecolor=color,
            edgecolor="none",
            zorder=5,
        )
    )
    ax.text(
        x + 0.014,
        y,
        number,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=PAPER,
        zorder=6,
    )
    ax.text(
        x + 0.037,
        y,
        label,
        ha="left",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=color,
        zorder=6,
    )


def _method_icon(
    ax: plt.Axes,
    kind: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    color: str,
) -> None:
    """Draw a tiny, self-contained glyph for each token-SAE variant."""

    left = x + 0.012
    right = x + width - 0.012
    bottom = y + 0.014
    top = y + height - 0.014
    mid_y = (bottom + top) / 2

    if kind == "jump":
        ax.plot(
            [left, left + 0.025, left + 0.025, right],
            [bottom + 0.004, bottom + 0.004, top - 0.004, top - 0.004],
            color=color,
            linewidth=1.5,
            solid_capstyle="round",
            zorder=7,
        )
        ax.plot(
            [left + 0.025, left + 0.025],
            [bottom, top],
            color=FAINT,
            linewidth=0.7,
            linestyle=(0, (2, 2)),
            zorder=6,
        )
    elif kind == "topk":
        heights = (0.012, 0.027, 0.018, 0.036, 0.022)
        for index, bar_height in enumerate(heights):
            bx = left + index * 0.0105
            active = index in (1, 3)
            ax.add_patch(
                FancyBboxPatch(
                    (bx, bottom),
                    0.0065,
                    bar_height,
                    boxstyle="round,pad=0,rounding_size=0.002",
                    facecolor=color if active else "#D8E1EC",
                    edgecolor="none",
                    zorder=7,
                )
            )
    elif kind == "batch":
        for row in range(3):
            for col in range(5):
                active = (row, col) in {(0, 1), (1, 4), (2, 0), (2, 3)}
                ax.add_patch(
                    Circle(
                        (left + col * 0.011, top - row * 0.014),
                        0.0033,
                        facecolor=color if active else "#D8E1EC",
                        edgecolor="none",
                        zorder=7,
                    )
                )
    elif kind == "temporal":
        xs = np.linspace(left, right, 6)
        ys = mid_y + np.asarray((-0.012, 0.010, -0.002, 0.014, -0.008, 0.008))
        ax.plot(xs, ys, color=color, linewidth=1.4, zorder=7)
        for px, py in zip(xs, ys, strict=True):
            ax.add_patch(
                Circle(
                    (px, py),
                    0.0031,
                    facecolor=PAPER,
                    edgecolor=color,
                    linewidth=0.9,
                    zorder=8,
                )
            )
        _arrow(
            ax,
            (right - 0.006, ys[-1]),
            (right + 0.006, ys[-1]),
            color=color,
            linewidth=1.0,
            mutation_scale=8,
            zorder=8,
        )


def _method_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    label: str,
    kind: str,
    *,
    color: str,
) -> None:
    _card(
        ax,
        x,
        y,
        width,
        0.078,
        facecolor="#FAFCFF",
        edgecolor="#CEDAEC",
        linewidth=0.8,
        radius=0.012,
        zorder=3,
    )
    _method_icon(ax, kind, x, y + 0.024, width, 0.050, color=color)
    ax.text(
        x + width / 2,
        y + 0.014,
        label,
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        zorder=8,
    )


def _token(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    text: str,
    *,
    highlight: bool = False,
) -> None:
    _card(
        ax,
        x,
        y,
        width,
        0.043,
        facecolor=CORAL_PALE if highlight else PAPER,
        edgecolor=CORAL if highlight else "#C9D6E5",
        linewidth=1.0 if highlight else 0.8,
        radius=0.008,
        zorder=5,
    )
    ax.text(
        x + width / 2,
        y + 0.0215,
        text,
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold" if highlight else "normal",
        color=CORAL_DARK if highlight else INK,
        zorder=6,
    )


def _sparse_code(
    ax: plt.Axes,
    center_x: float,
    y: float,
    *,
    seed: int,
    color: str = BLUE,
) -> None:
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.15, 1.0, size=7)
    active = set(rng.choice(7, size=2, replace=False).tolist())
    start = center_x - 0.023
    for index, value in enumerate(values):
        height = 0.008 + 0.028 * float(value)
        ax.add_patch(
            FancyBboxPatch(
                (start + index * 0.0072, y),
                0.0045,
                height,
                boxstyle="round,pad=0,rounding_size=0.0015",
                facecolor=color if index in active else "#DCE4EE",
                edgecolor="none",
                zorder=5,
            )
        )


def _feature_chip(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    token: str,
    feature_id: int,
    *,
    color: str,
) -> None:
    _card(
        ax,
        x,
        y,
        width,
        0.052,
        facecolor=_mix(color, 0.93),
        edgecolor=_mix(color, 0.58),
        linewidth=0.8,
        radius=0.010,
        zorder=5,
    )
    ax.text(
        x + width / 2,
        y + 0.033,
        f"‹{token}›",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=color,
        zorder=6,
    )
    ax.text(
        x + width / 2,
        y + 0.012,
        f"f{feature_id:,} · 10/10",
        ha="center",
        va="center",
        fontsize=4.7,
        fontweight="bold",
        color=INK,
        zorder=6,
    )


def _scatter_cloud(
    ax: plt.Axes,
    center: tuple[float, float],
    *,
    width: float,
    height: float,
    seed: int = 11,
) -> None:
    cx, cy = center
    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []
    while len(points) < 74:
        px = rng.normal(0, 0.34)
        py = rng.normal(0, 0.30)
        if (px / 0.95) ** 2 + (py / 0.90) ** 2 <= 1.0:
            points.append((cx + px * width, cy + py * height))
    points_array = np.asarray(points)
    colors = rng.choice(
        [BLUE, BLUE_2, CYAN, "#A7C7EE", CORAL],
        size=len(points),
        p=[0.26, 0.27, 0.22, 0.20, 0.05],
    )
    ax.scatter(
        points_array[:, 0],
        points_array[:, 1],
        s=rng.uniform(5, 17, size=len(points)),
        c=colors,
        linewidths=0,
        alpha=0.88,
        zorder=5,
    )


def _check(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.add_patch(
        Circle(
            (x, y),
            0.010,
            facecolor=GREEN,
            edgecolor="none",
            zorder=7,
        )
    )
    ax.plot(
        [x - 0.0042, x - 0.0005, x + 0.0052],
        [y - 0.0002, y - 0.0041, y + 0.0047],
        color=PAPER,
        linewidth=1.25,
        solid_capstyle="round",
        zorder=8,
    )
    ax.text(
        x + 0.017,
        y,
        text,
        ha="left",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        color=INK,
        zorder=8,
    )


def _high_level_feature(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    feature_id: str,
    label: str,
    *,
    accent: str,
) -> None:
    _card(
        ax,
        x,
        y,
        width,
        0.052,
        facecolor=PAPER,
        edgecolor="#B7DCCF",
        linewidth=0.8,
        radius=0.010,
        zorder=6,
    )
    _pill(
        ax,
        x + 0.007,
        y + 0.010,
        0.038,
        0.032,
        feature_id,
        facecolor=_mix(accent, 0.88),
        color=accent,
        fontsize=6.0,
        zorder=7,
    )
    ax.text(
        x + 0.052,
        y + 0.026,
        label,
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=INK,
        zorder=8,
    )


def render_infographic(
    results: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    """Render the redesigned 2:1 motivation figure."""

    trigger = results["token_trigger_concentration"]
    support = results["passage_support_expansion"]
    target = results["semantic_target"]
    cards = trigger["feature_cards"]
    discarded = 100.0 * (
        1.0 - support["median_activation_mass_retained_by_top128"]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    # Exact 2:1 aspect ratio. Saving without tight cropping preserves it for
    # PNG, PDF, and SVG.
    fig = plt.figure(figsize=(16, 8), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Header.
    _pill(
        ax,
        0.038,
        0.925,
        0.090,
        0.033,
        "MOTIVATION",
        facecolor=BLUE_PALE,
        color=BLUE,
        fontsize=7.2,
    )
    ax.text(
        0.038,
        0.875,
        "Token-level sparsity is not semantic abstraction",
        ha="left",
        va="center",
        fontsize=25.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.039,
        0.829,
        "Changing the sparsifier changes code selection—not the semantic unit optimized by the model.",
        ha="left",
        va="center",
        fontsize=10.2,
        color=MUTED,
    )
    ax.plot([0.038, 0.962], [0.798, 0.798], color=BORDER, linewidth=1.0)

    # Main cards.
    left = (0.035, 0.168, 0.255, 0.600)
    middle = (0.315, 0.168, 0.390, 0.600)
    right = (0.730, 0.168, 0.235, 0.600)
    _card(ax, *left, shadow=True, zorder=1)
    _card(ax, *middle, shadow=True, zorder=1)
    _card(
        ax,
        *right,
        facecolor="#F8FCFA",
        edgecolor="#BFDCD3",
        shadow=True,
        zorder=1,
    )

    # Inter-card flow arrows.
    _arrow(
        ax,
        (left[0] + left[2] + 0.006, 0.480),
        (middle[0] - 0.007, 0.480),
        color=BLUE,
        linewidth=2.0,
        mutation_scale=15,
    )
    _arrow(
        ax,
        (middle[0] + middle[2] + 0.006, 0.480),
        (right[0] - 0.007, 0.480),
        color=GREEN,
        linewidth=2.0,
        mutation_scale=15,
    )

    # ------------------------------------------------------------------
    # 01 — Token-level SAE family.
    # ------------------------------------------------------------------
    _section_header(
        ax,
        left[0] + 0.018,
        0.735,
        "01",
        "TOKEN-LEVEL SAE FAMILY",
        color=BLUE,
    )
    ax.text(
        left[0] + 0.020,
        0.690,
        "Different mechanisms.\nSame token-indexed bottleneck.",
        ha="left",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color=INK,
        linespacing=1.18,
    )

    method_width = 0.099
    method_left = left[0] + 0.020
    method_right = left[0] + 0.136
    _method_card(
        ax,
        method_left,
        0.563,
        method_width,
        "JumpReLU",
        "jump",
        color=BLUE,
    )
    _method_card(
        ax,
        method_right,
        0.563,
        method_width,
        "TopK",
        "topk",
        color=BLUE,
    )
    _method_card(
        ax,
        method_left,
        0.470,
        method_width,
        "BatchTopK",
        "batch",
        color=CYAN,
    )
    _method_card(
        ax,
        method_right,
        0.470,
        method_width,
        "Temporal",
        "temporal",
        color=PURPLE,
    )

    ax.text(
        left[0] + 0.020,
        0.430,
        "All still emit a sparse code at token position t",
        ha="left",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=MUTED,
    )
    ax.plot(
        [left[0] + 0.020, left[0] + left[2] - 0.020],
        [0.410, 0.410],
        color=HAIRLINE,
        linewidth=0.9,
    )

    token_specs = (
        ("volcano", 0.052, False),
        ("erupted", 0.049, False),
        (",", 0.022, True),
        ("evacuate", 0.052, False),
    )
    total_width = sum(item[1] for item in token_specs) + 0.009 * 3
    token_x = left[0] + (left[2] - total_width) / 2
    token_centers: list[float] = []
    for text, width, highlight in token_specs:
        _token(ax, token_x, 0.351, width, text, highlight=highlight)
        token_centers.append(token_x + width / 2)
        token_x += width + 0.009
    for index, center_x in enumerate(token_centers):
        ax.plot(
            [center_x, center_x],
            [0.348, 0.330],
            color="#B7C9DD",
            linewidth=0.8,
            zorder=3,
        )
        _sparse_code(ax, center_x, 0.285, seed=31 + index)

    _card(
        ax,
        left[0] + 0.020,
        0.207,
        left[2] - 0.040,
        0.054,
        facecolor=NAVY,
        edgecolor="none",
        radius=0.012,
        zorder=4,
    )
    ax.text(
        left[0] + left[2] / 2,
        0.234,
        r"$h_t\ \longrightarrow\ z_t\ \longrightarrow\ \hat{h}_t$",
        ha="center",
        va="center",
        fontsize=12.3,
        fontweight="bold",
        color=PAPER,
        zorder=5,
    )
    ax.text(
        left[0] + left[2] / 2,
        0.188,
        "Sparse competition + reconstruction are applied per t",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=BLUE,
    )

    # ------------------------------------------------------------------
    # 02 — Root cause and resulting passage-level failure.
    # ------------------------------------------------------------------
    _section_header(
        ax,
        middle[0] + 0.018,
        0.735,
        "02",
        "SHARED GRANULARITY DEFECT",
        color=CORAL,
    )

    root_x = middle[0] + 0.018
    root_y = 0.493
    root_w = middle[2] - 0.036
    root_h = 0.214
    _card(
        ax,
        root_x,
        root_y,
        root_w,
        root_h,
        facecolor="#FFFDFD",
        edgecolor="#F0C9D0",
        linewidth=0.9,
        radius=0.014,
        zorder=2,
    )
    ax.text(
        root_x + 0.016,
        root_y + root_h - 0.027,
        "ROOT CAUSE",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=CORAL,
        zorder=5,
    )
    ax.text(
        root_x + 0.016,
        root_y + 0.137,
        "The bottleneck is attached to a position,\nnot to the high-level idea.",
        ha="left",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
        linespacing=1.22,
        zorder=5,
    )
    ax.text(
        root_x + 0.016,
        root_y + 0.076,
        "Even when hₜ contains context, no loss asks one\ncoordinate to summarize the whole passage.",
        ha="left",
        va="center",
        fontsize=6.9,
        color=MUTED,
        linespacing=1.26,
        zorder=5,
    )
    _pill(
        ax,
        root_x + 0.016,
        root_y + 0.017,
        0.157,
        0.034,
        "activation rule  ≠  semantic unit",
        facecolor=CORAL_PALE,
        color=CORAL_DARK,
        edgecolor="#EDB5BF",
        fontsize=6.4,
        zorder=5,
    )

    evidence_x = root_x + 0.188
    evidence_w = root_w - 0.204
    _card(
        ax,
        evidence_x,
        root_y + 0.018,
        evidence_w,
        root_h - 0.036,
        facecolor=CORAL_PALE,
        edgecolor="none",
        radius=0.012,
        zorder=3,
    )
    ax.text(
        evidence_x + 0.012,
        root_y + root_h - 0.041,
        "OBSERVED IN A BATCHTOPK TOKEN SAE",
        ha="left",
        va="center",
        fontsize=5.7,
        fontweight="bold",
        color=CORAL_DARK,
        zorder=6,
    )
    ax.text(
        evidence_x + 0.012,
        root_y + 0.113,
        f"{trigger['same_token_all_10']:,} / {trigger['sampled_features']:,}",
        ha="left",
        va="center",
        fontsize=18.5,
        fontweight="bold",
        color=CORAL_DARK,
        zorder=6,
    )
    ax.text(
        evidence_x + 0.012,
        root_y + 0.083,
        "exact-token anchors",
        ha="left",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=INK,
        zorder=6,
    )

    chip_width = (evidence_w - 0.032) / 3
    chip_colors = (CORAL, AMBER, BLUE)
    chip_x = evidence_x + 0.008
    for row, color in zip(cards, chip_colors, strict=True):
        _feature_chip(
            ax,
            chip_x,
            root_y + 0.023,
            chip_width,
            row["dominant_surface"],
            row["feature_id"],
            color=color,
        )
        chip_x += chip_width + 0.008

    consequence_x = middle[0] + 0.018
    consequence_y = 0.190
    consequence_w = middle[2] - 0.036
    consequence_h = 0.278
    _card(
        ax,
        consequence_x,
        consequence_y,
        consequence_w,
        consequence_h,
        facecolor="#FCFDFE",
        edgecolor="#D5DFEA",
        linewidth=0.9,
        radius=0.014,
        zorder=2,
    )
    ax.text(
        consequence_x + 0.016,
        consequence_y + consequence_h - 0.028,
        "CONSEQUENCE AT PASSAGE SCALE",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=NAVY,
        zorder=5,
    )
    ax.text(
        consequence_x + 0.016,
        consequence_y + consequence_h - 0.059,
        "Pooling accumulates local detectors—it does not create a global concept.",
        ha="left",
        va="center",
        fontsize=7.0,
        color=MUTED,
        zorder=5,
    )

    # Many token codes flow into a large union of coordinates.
    code_base_x = consequence_x + 0.018
    for index in range(6):
        _sparse_code(
            ax,
            code_base_x + 0.012 + index * 0.019,
            consequence_y + 0.090 + (index % 2) * 0.010,
            seed=80 + index,
            color=BLUE_2,
        )
    ax.text(
        consequence_x + 0.071,
        consequence_y + 0.070,
        "128 token codes",
        ha="center",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color=MUTED,
        zorder=6,
    )
    _arrow(
        ax,
        (consequence_x + 0.125, consequence_y + 0.122),
        (consequence_x + 0.153, consequence_y + 0.136),
        color=BLUE,
        linewidth=1.5,
        mutation_scale=11,
    )

    cloud_center = (consequence_x + 0.195, consequence_y + 0.132)
    _card(
        ax,
        cloud_center[0] - 0.052,
        cloud_center[1] - 0.068,
        0.104,
        0.136,
        facecolor=BLUE_PALE,
        edgecolor="#AFC9ED",
        linewidth=0.9,
        radius=0.052,
        zorder=3,
    )
    _scatter_cloud(
        ax,
        cloud_center,
        width=0.115,
        height=0.115,
        seed=17,
    )
    ax.text(
        cloud_center[0],
        cloud_center[1] + 0.013,
        f"{support['median_unique_features']:,.0f}",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color=INK,
        zorder=8,
        path_effects=[
            path_effects.withStroke(linewidth=4.5, foreground=BLUE_PALE)
        ],
    )
    ax.text(
        cloud_center[0],
        cloud_center[1] - 0.019,
        "active coordinates",
        ha="center",
        va="center",
        fontsize=5.7,
        fontweight="bold",
        color=MUTED,
        zorder=8,
        path_effects=[
            path_effects.withStroke(linewidth=3.5, foreground=BLUE_PALE)
        ],
    )

    fork_x = consequence_x + 0.274
    _arrow(
        ax,
        (cloud_center[0] + 0.058, cloud_center[1] + 0.016),
        (fork_x, consequence_y + 0.177),
        color=CORAL,
        linewidth=1.25,
        mutation_scale=10,
        connectionstyle="arc3,rad=-0.12",
    )
    _arrow(
        ax,
        (cloud_center[0] + 0.058, cloud_center[1] - 0.010),
        (fork_x, consequence_y + 0.091),
        color=AMBER,
        linewidth=1.25,
        mutation_scale=10,
        connectionstyle="arc3,rad=0.12",
    )
    _card(
        ax,
        fork_x,
        consequence_y + 0.145,
        0.068,
        0.061,
        facecolor=CORAL_PALE,
        edgecolor="#EAB2BC",
        linewidth=0.8,
        radius=0.010,
        zorder=5,
    )
    ax.text(
        fork_x + 0.034,
        consequence_y + 0.184,
        "KEEP ALL",
        ha="center",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color=CORAL_DARK,
        zorder=6,
    )
    ax.text(
        fork_x + 0.034,
        consequence_y + 0.162,
        "DENSE",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=CORAL_DARK,
        zorder=6,
    )
    _card(
        ax,
        fork_x,
        consequence_y + 0.058,
        0.068,
        0.061,
        facecolor=AMBER_PALE,
        edgecolor="#E8C47F",
        linewidth=0.8,
        radius=0.010,
        zorder=5,
    )
    ax.text(
        fork_x + 0.034,
        consequence_y + 0.097,
        "TOP-128",
        ha="center",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color="#9A6419",
        zorder=6,
    )
    ax.text(
        fork_x + 0.034,
        consequence_y + 0.075,
        f"{discarded:.1f}% lost",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#9A6419",
        zorder=6,
    )
    ax.text(
        consequence_x + consequence_w / 2,
        consequence_y + 0.025,
        "Local interpretability does not automatically compose into semantic interpretability.",
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=CORAL_DARK,
        zorder=6,
    )

    # ------------------------------------------------------------------
    # 03 — Design target: a new high-level feature SAE.
    # ------------------------------------------------------------------
    _section_header(
        ax,
        right[0] + 0.018,
        0.735,
        "03",
        "DESIGN TARGET",
        color=GREEN,
    )
    ax.text(
        right[0] + 0.020,
        0.690,
        "A new high-level\nfeature SAE",
        ha="left",
        va="top",
        fontsize=14.2,
        fontweight="bold",
        color=INK,
        linespacing=1.14,
    )
    ax.text(
        right[0] + 0.020,
        0.625,
        "Train the bottleneck on the semantic object\nthat the feature is expected to explain.",
        ha="left",
        va="top",
        fontsize=6.8,
        color=MUTED,
        linespacing=1.24,
    )

    # Whole passage enters as one native observation.
    passage_x = right[0] + 0.020
    passage_y = 0.520
    passage_w = right[2] - 0.040
    _card(
        ax,
        passage_x,
        passage_y,
        passage_w,
        0.075,
        facecolor=PAPER,
        edgecolor="#B8D9CE",
        linewidth=0.9,
        radius=0.012,
        zorder=4,
    )
    ax.add_patch(
        FancyBboxPatch(
            (passage_x, passage_y),
            0.006,
            0.075,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=GREEN,
            edgecolor="none",
            zorder=5,
        )
    )
    ax.text(
        passage_x + 0.016,
        passage_y + 0.055,
        "WHOLE PASSAGE / SEMANTIC UNIT",
        ha="left",
        va="center",
        fontsize=5.5,
        fontweight="bold",
        color=GREEN_DARK,
        zorder=6,
    )
    passage_text = target["passage"].replace("nearby ", "")
    ax.text(
        passage_x + 0.016,
        passage_y + 0.027,
        f"“{passage_text}”",
        ha="left",
        va="center",
        fontsize=6.3,
        color=INK,
        zorder=6,
    )
    _arrow(
        ax,
        (right[0] + right[2] / 2, passage_y - 0.005),
        (right[0] + right[2] / 2, 0.486),
        color=GREEN,
        linewidth=1.6,
        mutation_scale=11,
    )

    # Encoder / sparse high-level bottleneck.
    module_x = right[0] + 0.052
    module_y = 0.431
    module_w = right[2] - 0.104
    _card(
        ax,
        module_x,
        module_y,
        module_w,
        0.055,
        facecolor=GREEN_DARK,
        edgecolor="none",
        radius=0.014,
        zorder=5,
    )
    ax.text(
        module_x + module_w / 2,
        module_y + 0.028,
        r"HIGH-LEVEL SAE   $\rightarrow\ z_{\mathrm{H}}$",
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="bold",
        color=PAPER,
        zorder=6,
    )
    _arrow(
        ax,
        (right[0] + right[2] / 2, module_y - 0.004),
        (right[0] + right[2] / 2, 0.399),
        color=GREEN,
        linewidth=1.5,
        mutation_scale=10,
    )

    feature_x = right[0] + 0.020
    feature_w = right[2] - 0.040
    _high_level_feature(
        ax,
        feature_x,
        0.337,
        feature_w,
        "H17",
        "volcanic hazard",
        accent=GREEN,
    )
    _high_level_feature(
        ax,
        feature_x,
        0.278,
        feature_w,
        "H42",
        "evacuation intent",
        accent=BLUE,
    )
    _high_level_feature(
        ax,
        feature_x,
        0.219,
        feature_w,
        "H73",
        "cause  →  protective action",
        accent=PURPLE,
    )

    # Interpretability properties.
    _pill(
        ax,
        right[0] + 0.020,
        0.179,
        0.085,
        0.029,
        "✓  SPARSE CODE",
        facecolor=GREEN_PALE,
        color=GREEN_DARK,
        edgecolor="#B7DCCF",
        fontsize=5.7,
        zorder=6,
    )
    _pill(
        ax,
        right[0] + 0.111,
        0.179,
        0.104,
        0.029,
        "✓  INTERPRETABLE",
        facecolor=GREEN_PALE,
        color=GREEN_DARK,
        edgecolor="#B7DCCF",
        fontsize=5.7,
        zorder=6,
    )

    # Footer: the single design principle the reader should retain.
    _card(
        ax,
        0.035,
        0.052,
        0.930,
        0.082,
        facecolor=NAVY,
        edgecolor="none",
        radius=0.018,
        shadow=True,
        zorder=3,
    )
    ax.text(
        0.058,
        0.093,
        "DESIGN PRINCIPLE",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#A9C8E6",
        zorder=5,
    )
    ax.text(
        0.175,
        0.093,
        "The unit of sparsity must match the unit of explanation.",
        ha="left",
        va="center",
        fontsize=12.1,
        fontweight="bold",
        color=PAPER,
        zorder=5,
    )
    _pill(
        ax,
        0.742,
        0.071,
        0.200,
        0.044,
        "HIGH-LEVEL  ×  SPARSE  ×  INTERPRETABLE",
        facecolor=GREEN,
        color=PAPER,
        fontsize=7.2,
        zorder=6,
    )

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": figures / "token_sae_granularity_gap.png",
        "pdf": figures / "token_sae_granularity_gap.pdf",
        "svg": figures / "token_sae_granularity_gap.svg",
    }
    metadata = {
        "Title": "Token-level sparsity is not semantic abstraction",
        "Creator": "Matplotlib",
        "Subject": (
            "Token-level SAE family, shared granularity defect, passage-scale "
            "failure, and the design target for high-level interpretable SAEs"
        ),
    }
    fig.savefig(
        paths["png"],
        dpi=dpi,
        facecolor=BG,
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
        metadata=metadata,
    )
    fig.savefig(
        paths["pdf"],
        facecolor=BG,
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
        metadata=metadata,
    )
    fig.savefig(
        paths["svg"],
        facecolor=BG,
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
        metadata={
            "Title": metadata["Title"],
            "Creator": metadata["Creator"],
            "Description": metadata["Subject"],
        },
    )
    plt.close(fig)
    return {
        key: str(path.relative_to(output_dir)) for key, path in paths.items()
    }
