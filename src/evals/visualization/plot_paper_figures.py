#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

from chunk_saes.plot_style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_PALE_COLORS,
    METHOD_SHORT_LABELS,
    METHOD_TEXT_COLORS,
    METHODS,
    style_figure_text,
)

COLORS = METHOD_COLORS
PALE = METHOD_PALE_COLORS
DARK = "#20242C"
MID = "#667085"
GRID = "#DDE2EA"
PANEL = "#F7F8FB"
WHITE = "#FFFFFF"
GOLD = "#D69E2E"
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render the two compact 2:1 camera-ready Chunk-SAE figures."
    )
    p.add_argument("--eval-root", required=True)
    p.add_argument(
        "--document-linking-dir",
        default="document_linking",
        help=(
            "Lexically controlled document-linking artifact directory "
            "relative to --eval-root."
        ),
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate protected paper figures instead of reusing them.",
    )
    return p


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "font.weight": "medium",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.labelweight": "bold",
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#C8CFD9",
            "axes.linewidth": 0.9,
            "xtick.color": MID,
            "ytick.color": MID,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "text.color": DARK,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _downstream_style() -> None:
    """Larger type for the dense five-panel downstream utility figure."""

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "font.weight": "bold",
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
        }
    )


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    # Figure captions carry the global title in the paper.  Keep this helper as
    # a no-op for backward-compatible call sites and reserve the full canvas
    # for data and panel labels.
    del fig, title, subtitle


def _panel_title(
    ax: plt.Axes,
    letter: str,
    title: str,
    *,
    fontsize: float = 12,
    pad: float = 11,
) -> None:
    ax.set_title(
        f"{letter}   {title}",
        loc="left",
        pad=pad,
        fontsize=fontsize,
        fontweight="bold",
        color=DARK,
    )


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    linewidth: float = 1.2,
    radius: float = 0.025,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        transform=ax.transAxes,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def _badge(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    size: float = 8.2,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=size,
        fontweight="bold",
        color=WHITE,
        bbox={
            "boxstyle": "round,pad=0.30,rounding_size=0.16",
            "facecolor": color,
            "edgecolor": "none",
        },
    )


def _feature_contrast(
    ax: plt.Axes,
    invariance: dict,
    linking: dict,
) -> None:
    # Keep the illustrative document-linking panel separate from the
    # dictionary-level high-level-feature census.  ``invariance`` remains in
    # the signature for backward compatibility with existing callers.
    del invariance
    ax.set_axis_off()
    ax.set_facecolor(PANEL)
    _panel_title(ax, "A", "Low-overlap chunks: wording changes, meaning persists")

    left_x, right_x = 0.02, 0.53
    card_y, card_w, card_h = 0.08, 0.40, 0.79
    _box(
        ax,
        left_x,
        card_y,
        card_w,
        card_h,
        face="#F4F6F9",
        edge=MID,
        linewidth=1.8,
    )
    _box(
        ax,
        right_x,
        card_y,
        card_w,
        card_h,
        face=PALE["cross"],
        edge=COLORS["cross"],
        linewidth=2.2,
    )

    _badge(ax, left_x + 0.025, 0.815, "CHUNK A", MID)
    _badge(ax, right_x + 0.025, 0.815, "CHUNK B", COLORS["cross"])
    examples = list(linking.get("examples") or [])
    example = examples[0] if examples else {}
    query = _clean_chunk_excerpt(
        str(
            example.get(
                "query",
                "British weather and a wind-resistant umbrella frame.",
            )
        )
    )
    partner = _clean_chunk_excerpt(
        str(
            example.get(
                "correct_partner",
                "Product listing for a floral Joules umbrella.",
            )
        )
    )
    ax.text(
        left_x + 0.025,
        0.725,
        "Same source document",
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        right_x + 0.025,
        0.725,
        "Same source document",
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        left_x + 0.025,
        0.665,
        "Use description",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        right_x + 0.025,
        0.665,
        "Product listing",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=DARK,
    )

    ax.text(
        left_x + 0.025,
        0.56,
        textwrap.fill(query, width=46),
        transform=ax.transAxes,
        fontsize=9.1,
        color=DARK,
        va="top",
        linespacing=1.35,
    )
    ax.text(
        right_x + 0.025,
        0.56,
        textwrap.fill(partner, width=46),
        transform=ax.transAxes,
        fontsize=9.1,
        color=DARK,
        va="top",
        linespacing=1.35,
    )

    ax.annotate(
        "",
        xy=(0.505, 0.46),
        xytext=(0.445, 0.46),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 3.0,
            "color": COLORS["cross"],
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.text(
        0.475,
        0.545,
        "LOW OVERLAP\nSAME DOCUMENT",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=COLORS["cross"],
        linespacing=1.15,
    )

    ranks = example.get("ranks") or {}
    preference_text = "DOCUMENT LINKING"
    if ranks.get("cross") is not None and ranks.get("token") is not None:
        preference_text += (
            f"   ·   LINK RANK {float(ranks['cross']):.0f} "
            f"VS {float(ranks['token']):.0f}"
        )
    ax.text(
        0.475,
        0.025,
        preference_text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=DARK,
    )


def _feature_frontier(
    ax: plt.Axes,
    high_level: dict,
    adjacent: dict,
) -> None:
    _panel_title(ax, "B", "Feature-quality frontier")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    modes = _available_modes(
        high_level.get("methods", {}),
        adjacent.get("methods", {}),
    )
    label_positions = {
        "token": (0.443, 0.158),
        "temporal": (0.415, 0.235),
        "mean": (0.350, 0.302),
        "cross": (0.448, 0.329),
    }
    for mode in modes:
        abstraction = high_level["methods"][mode]["cross_document_abstraction"]
        persistence = adjacent["methods"][mode]["feature_metrics"]
        x = float(abstraction["mean"])
        y = float(persistence["mean_persistence_lift"])
        xerr = np.asarray(
            [
                x - float(abstraction["95ci"][0]),
                float(abstraction["95ci"][1]) - x,
            ]
        ).reshape(2, 1)
        yerr = np.asarray(
            [
                y - float(persistence["mean_persistence_lift_95ci"][0]),
                float(persistence["mean_persistence_lift_95ci"][1]) - y,
            ]
        ).reshape(2, 1)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=COLORS[mode],
            elinewidth=2.0,
            capsize=4,
            zorder=2,
        )
        if mode == "cross":
            ax.scatter(
                x,
                y,
                s=900,
                facecolor=COLORS["cross"],
                edgecolor=WHITE,
                linewidth=4,
                alpha=0.18,
                zorder=2,
            )
        ax.scatter(
            x,
            y,
            s=260 if mode != "cross" else 330,
            facecolor=COLORS[mode],
            edgecolor=WHITE,
            linewidth=2.2,
            zorder=3,
        )
        lx, ly = label_positions[mode]
        ax.text(
            lx,
            ly,
            METHOD_LABELS[mode],
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=METHOD_TEXT_COLORS[mode],
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": WHITE,
                "edgecolor": "none",
                "alpha": 0.93,
            },
            zorder=5,
        )

    cross_abs = high_level["methods"]["cross"]["cross_document_abstraction"][
        "mean"
    ]
    best_abs = max(
        high_level["methods"][mode]["cross_document_abstraction"]["mean"]
        for mode in modes
        if mode != "cross"
    )
    cross_persist = adjacent["methods"]["cross"]["feature_metrics"][
        "mean_persistence_lift"
    ]
    best_persist = max(
        adjacent["methods"][mode]["feature_metrics"][
            "mean_persistence_lift"
        ]
        for mode in modes
        if mode != "cross"
    )
    ax.text(
        0.03,
        0.05,
        (
            f"+{100 * (cross_abs / best_abs - 1):.1f}% abstraction\n"
            f"+{100 * (cross_persist / best_persist - 1):.1f}% persistence"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["cross"],
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.40,rounding_size=0.14",
            "facecolor": WHITE,
            "edgecolor": COLORS["cross"],
            "linewidth": 1.2,
        },
    )
    ax.set_xlim(0.345, 0.515)
    ax.set_ylim(0.145, 0.34)
    ax.set_xlabel("Cross-document abstraction  →", fontweight="bold")
    ax.set_ylabel("Adjacent feature persistence  →", fontweight="bold")


def _bar_metric(
    ax: plt.Axes,
    adjacent: dict,
    *,
    metric: str,
    higher_is_better: bool,
) -> None:
    modes = _available_modes(adjacent.get("methods", {}))
    order = tuple(mode for mode in METHODS if mode in modes)
    ordered_values = [
        adjacent["methods"][mode]["dictionary_utilization"][metric]
        for mode in order
    ]
    y = np.arange(len(order))
    ax.barh(
        y,
        ordered_values,
        color=[COLORS[mode] for mode in order],
        height=0.53,
        alpha=0.94,
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.65, zorder=0)
    maximum = max(ordered_values) * 1.22
    left_margin = maximum * 0.55
    ax.set_xlim(-left_margin, maximum)
    for index, (mode, value) in enumerate(zip(order, ordered_values, strict=True)):
        ax.text(
            -left_margin * 0.94,
            index,
            METHOD_SHORT_LABELS[mode],
            va="center",
            ha="left",
            fontsize=6.7,
            fontweight="bold",
            color=METHOD_TEXT_COLORS[mode],
            linespacing=0.90,
            zorder=3,
        )
        ax.text(
            value + maximum * 0.025,
            index,
            f"{value:.1%}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=METHOD_TEXT_COLORS[mode],
        )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _dictionary_panel(
    effective_ax: plt.Axes,
    concentration_ax: plt.Axes,
    adjacent: dict,
) -> None:
    effective_ax.set_title(
        "C   Effective feature fraction ↑",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    concentration_ax.set_title(
        "Top 1% activity share ↓",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    _bar_metric(
        effective_ax,
        adjacent,
        metric="effective_feature_fraction",
        higher_is_better=True,
    )
    _bar_metric(
        concentration_ax,
        adjacent,
        metric="top_1pct_activity_share",
        higher_is_better=False,
    )


def _blend(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    return tuple(rgb * (1.0 - amount) + amount)


def _shorten(text: str, width: int) -> str:
    return textwrap.shorten(
        " ".join(str(text).split()),
        width=width,
        placeholder="…",
    )


def _clean_chunk_excerpt(text: str) -> str:
    """Remove obvious boundary fragments while preserving the source wording."""

    value = " ".join(str(text).split())
    first_period = value.find(".")
    if 0 <= first_period < 12:
        remainder = value[first_period + 1 :].lstrip()
        if remainder[:1].isupper():
            value = remainder
    value = re.sub(r",\s*\d+\s*$", "", value)
    last_stop = max(value.rfind("."), value.rfind("!"), value.rfind("?"))
    if 0 <= last_stop and len(value) - last_stop - 1 <= 8:
        value = value[: last_stop + 1]
    if value and value[-1] not in ".!?…":
        value += "…"
    return value


def _probe_representation_names(probes: dict) -> dict[str, str]:
    token_name = (
        "token_sae_max"
        if probes["chosen_token_aggregation"] == "max"
        else "token_sae_mean"
    )
    return {
        "token": token_name,
        "temporal": "temporal_sae_mean",
        "mean": "mean_chunk_sae",
        "cross": "cross_chunk_sae",
    }


def _ood_matrix(
    probes: dict,
) -> tuple[tuple[str, ...], list[str], np.ndarray]:
    names = _probe_representation_names(probes)
    modes = tuple(
        mode
        for mode in METHODS
        if names[mode] in probes.get("representations", {})
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
    return modes, labels, matrix


def _document_linking_panel(ax: plt.Axes, linking: dict) -> None:
    _panel_title(
        ax,
        "A",
        "Semantic document linking  (lexical invariance)",
        fontsize=12.8,
        pad=10,
    )
    methods = tuple(mode for mode in METHODS if mode in linking["methods"])
    controls = linking["controls"]
    labels = [
        "Word Jaccard",
        *[METHOD_SHORT_LABELS[m] for m in methods],
    ]
    values = [
        float(controls["word_jaccard"]["recall_at_1"]),
        *[
            float(linking["methods"][mode]["recall_at_1"])
            for mode in methods
        ],
    ]
    colors = ["#AEB5BF", *[COLORS[mode] for mode in methods]]
    y = np.arange(len(labels))
    ax.barh(
        y,
        values,
        height=0.62,
        color=colors,
        edgecolor=[
            "#929AA6",
            *[
                COLORS["cross"] if mode == "cross" else WHITE
                for mode in methods
            ],
        ],
        linewidth=[
            0.8,
            *[2.4 if mode == "cross" else 1.2 for mode in methods],
        ],
        zorder=2,
    )
    dense = float(controls["raw_mean_hidden"]["recall_at_1"])
    ax.axvline(
        dense,
        color="#735C50",
        lw=2.2,
        ls=(0, (5, 3)),
        zorder=1,
    )
    ax.text(
        dense,
        0.965,
        f"Dense hidden  {dense:.3f}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        color="#604C42",
        bbox={
            "boxstyle": "round,pad=0.20,rounding_size=0.10",
            "facecolor": WHITE,
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )
    ax.set_yticks(y, labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.61)
    ax.set_xticks(np.arange(0, 0.61, 0.1))
    ax.set_xlabel("Recall@1")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for index, (value, label) in enumerate(zip(values, labels, strict=True)):
        is_cross = label == METHOD_SHORT_LABELS["cross"]
        if is_cross:
            x_position = max(0.012, value - 0.012)
            horizontal_alignment = "right"
            text_color = WHITE
        else:
            x_position = min(value + 0.012, 0.592)
            horizontal_alignment = "left"
            text_color = DARK
        ax.text(
            x_position,
            index,
            f"{value:.3f}",
            ha=horizontal_alignment,
            va="center",
            fontsize=11.5 if is_cross else 10.5,
            fontweight="bold",
            color=text_color,
            zorder=4,
        )
    gain = linking["comparisons"]["cross_minus_raw_mean_hidden"][
        "recall_at_1"
    ]["point"]
    p_value = linking["comparisons"]["cross_minus_raw_mean_hidden"][
        "paired_randomization_p"
    ]
    effect_text = (
        f"+{100 * gain:.1f} pp"
        if p_value < 1e-5
        else f"+{100 * gain:.1f} pp  ·  p = {p_value:.1e}"
    )
    ax.text(
        0.985,
        0.078,
        effect_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.6,
        fontweight="bold",
        color=COLORS["cross"],
        bbox={
            "boxstyle": "round,pad=0.32,rounding_size=0.14",
            "facecolor": WHITE,
            "edgecolor": COLORS["cross"],
            "linewidth": 1.4,
        },
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _document_linking_example(ax: plt.Axes, linking: dict) -> None:
    ax.set_axis_off()
    ax.set_title(
        "Example · zero shared words",
        loc="left",
        pad=10,
        fontsize=12.0,
        fontweight="bold",
        color=DARK,
    )
    examples = linking.get("examples", [])
    if not examples:
        ax.text(
            0.5,
            0.5,
            "Qualitative example unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        return
    example = min(
        examples,
        key=lambda row: (
            float(row.get("lexical_jaccard", 1.0)),
            float(row["ranks"]["cross"]),
            -float(row["ranks"]["token"]),
        ),
    )
    query_excerpt = _shorten(
        _clean_chunk_excerpt(example["query"]),
        90,
    )
    partner_excerpt = _shorten(
        _clean_chunk_excerpt(example["correct_partner"]),
        90,
    )
    gallery_sizes = linking["controls"]["chance"][
        "gallery_sizes_by_target_length"
    ]
    gallery_size = int(
        gallery_sizes.get(
            str(int(example["partner_length"])),
            max(int(value) for value in gallery_sizes.values()),
        )
    )
    ax.text(
        0.50,
        0.91,
        "Find the same-document chunk despite zero shared words.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        0.50,
        0.805,
        f"0 shared content words  ·  Jaccard = {float(example['lexical_jaccard']):.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["cross"],
    )
    _box(
        ax,
        0.03,
        0.57,
        0.94,
        0.17,
        face="#F4F7FA",
        edge="#CAD3DE",
        linewidth=1.1,
        radius=0.025,
    )
    _box(
        ax,
        0.03,
        0.30,
        0.94,
        0.17,
        face=PALE["cross"],
        edge=_blend(COLORS["cross"], 0.45),
        linewidth=1.4,
        radius=0.025,
    )
    ax.text(
        0.065,
        0.715,
        "QUERY CHUNK",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=MID,
        va="top",
    )
    ax.text(
        0.065,
        0.445,
        "CORRECT SAME-DOCUMENT PARTNER",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["cross"],
        va="top",
    )
    ax.text(
        0.065,
        0.652,
        textwrap.fill(f"“{query_excerpt}”", width=66),
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=DARK,
        va="top",
        linespacing=1.22,
    )
    ax.text(
        0.065,
        0.382,
        textwrap.fill(f"“{partner_excerpt}”", width=66),
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=DARK,
        va="top",
        linespacing=1.22,
    )
    ax.annotate(
        "",
        xy=(0.50, 0.475),
        xytext=(0.50, 0.555),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.8,
            "color": COLORS["cross"],
        },
    )
    ax.text(
        0.53,
        0.515,
        "same topic · different wording",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=MID,
    )
    ax.text(
        0.50,
        0.225,
        (
            f"CORRECT CHUNK POSITION AMONG {gallery_size:,} CANDIDATES"
            "  ·  #1 = TOP RESULT"
        ),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=DARK,
    )
    modes = tuple(mode for mode in METHODS if mode in example["ranks"])
    xs = np.linspace(0.13, 0.87, len(modes))
    for x, mode in zip(xs, modes, strict=True):
        selected = mode == "cross"
        rank = int(example["ranks"][mode])
        ax.text(
            x,
            0.095,
            (
                f"{METHOD_SHORT_LABELS[mode]}\n"
                f"found at #{rank}"
                f"{'  ✓' if selected else ''}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.7 if selected else 9.3,
            fontweight="bold",
            color=WHITE if selected else DARK,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.38,rounding_size=0.16",
                "facecolor": (
                    COLORS["cross"] if selected else _blend(COLORS[mode], 0.82)
                ),
                "edgecolor": COLORS[mode],
                "linewidth": 2.0 if selected else 1.2,
            },
        )


def _ood_absolute_panel(
    ax: plt.Axes,
    colorbar_ax: plt.Axes,
    probes: dict,
) -> tuple[tuple[str, ...], list[str], np.ndarray]:
    _panel_title(
        ax,
        "B",
        "Temporal domain transfer  (robustness to shift)",
        fontsize=12.8,
        pad=10,
    )
    modes, labels, matrix = _ood_matrix(probes)
    overall = matrix.mean(axis=1)
    display_matrix = np.column_stack([matrix, overall])
    display_labels = [DOMAIN_SHORT[label] for label in labels] + ["Overall"]
    image = ax.imshow(
        display_matrix,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0.60,
        vmax=0.90,
    )
    ax.set_xticks(
        np.arange(len(display_labels)),
        display_labels,
        rotation=24,
        ha="right",
        fontweight="bold",
    )
    ax.set_yticks(
        np.arange(len(modes)),
        [METHOD_SHORT_LABELS[mode] for mode in modes],
        fontweight="bold",
    )
    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            value = display_matrix[row, column]
            ax.text(
                column,
                row,
                (
                    f"{value:.1%}"
                    if column == display_matrix.shape[1] - 1
                    else f"{value:.0%}"
                ),
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color=WHITE if value >= 0.82 else DARK,
            )
    overall_column = display_matrix.shape[1] - 1
    ax.axvline(
        overall_column - 0.5,
        color=WHITE,
        lw=3.0,
        zorder=4,
    )
    ax.get_xticklabels()[-1].set_color(COLORS["cross"])
    cross_index = modes.index("cross")
    ax.add_patch(
        FancyBboxPatch(
            (-0.48, cross_index - 0.47),
            len(display_labels) - 0.04,
            0.94,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            fill=False,
            edgecolor=COLORS["cross"],
            linewidth=2.5,
        )
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(
        image,
        cax=colorbar_ax,
        orientation="vertical",
    )
    colorbar.set_ticks([0.60, 0.70, 0.80, 0.90])
    colorbar.ax.tick_params(labelsize=8.8)
    colorbar.ax.set_title(
        "OOD\naccuracy",
        fontsize=8.5,
        fontweight="bold",
        pad=5,
    )
    return modes, labels, matrix


def _ood_advantage_panel(
    ax: plt.Axes,
    *,
    modes: tuple[str, ...],
    labels: list[str],
    matrix: np.ndarray,
) -> None:
    ax.set_title(
        "Cross-Chunk gain over each baseline",
        loc="left",
        pad=10,
        fontsize=12.0,
        fontweight="bold",
        color=DARK,
    )
    cross_index = modes.index("cross")
    references = tuple(mode for mode in modes if mode != "cross")
    baseline_indices = [modes.index(mode) for mode in references]
    best_baseline = np.max(matrix[baseline_indices], axis=0)
    order = np.argsort(matrix[cross_index] - best_baseline)[::-1]
    y = np.arange(len(labels))
    offsets = np.linspace(-0.19, 0.19, len(references))
    ax.axvline(0, color="#727780", lw=1.2, zorder=0)
    for mode, offset in zip(references, offsets, strict=True):
        deltas = 100 * (
            matrix[cross_index, order] - matrix[modes.index(mode), order]
        )
        for row, value in enumerate(deltas):
            ax.plot(
                [0, value],
                [row + offset, row + offset],
                color=_blend(COLORS[mode], 0.48),
                lw=3.2,
                solid_capstyle="round",
                zorder=1,
            )
        ax.scatter(
            deltas,
            y + offset,
            s=72,
            color=COLORS[mode],
            edgecolor=WHITE,
            linewidth=1.2,
            label=f"vs {METHOD_SHORT_LABELS[mode]}",
            zorder=3,
        )
    ax.set_yticks(
        y,
        [DOMAIN_SHORT[labels[index]] for index in order],
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.set_xlim(-4.5, 12.5)
    ax.set_xticks([-4, 0, 4, 8, 12])
    ax.set_xlabel("Cross-Chunk accuracy gain (pp)")
    ax.grid(axis="x", color=GRID, lw=0.75, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend = ax.legend(
        loc="lower right",
        fontsize=8.8,
        handlelength=1.8,
        borderaxespad=0.6,
        frameon=True,
        facecolor=WHITE,
        edgecolor="#D7DCE4",
        framealpha=0.97,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _low_label_scaling(ax: plt.Axes, probes: dict) -> None:
    _panel_title(
        ax,
        "C",
        "Label efficiency before full-supervision parity",
        fontsize=13.5,
        pad=10,
    )
    names = _probe_representation_names(probes)
    modes = tuple(
        mode
        for mode in METHODS
        if names[mode] in probes.get("representations", {})
    )
    budgets = [
        int(value)
        for value in probes["metadata"]["low_label_budgets"]
    ]
    x = np.arange(len(budgets))
    ax.axvspan(
        0.55,
        5.45,
        color=PALE["cross"],
        alpha=0.68,
        zorder=0,
    )
    ax.text(
        0.46,
        0.955,
        "Cross-Chunk advantage regime",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["cross"],
    )
    for mode in modes:
        rows = probes["representations"][names[mode]]["low_label"]
        means = np.asarray(
            [float(rows[str(budget)]["mean_accuracy"]) for budget in budgets]
        )
        stds = np.asarray(
            [float(rows[str(budget)]["std_accuracy"]) for budget in budgets]
        )
        line_width = 3.4 if mode == "cross" else 2.3
        marker_size = 8.0 if mode == "cross" else 6.3
        ax.fill_between(
            x,
            100 * (means - stds),
            100 * (means + stds),
            color=COLORS[mode],
            alpha=0.13 if mode == "cross" else 0.08,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            100 * means,
            color=COLORS[mode],
            lw=line_width,
            marker="o",
            ms=marker_size,
            markeredgecolor=WHITE,
            markeredgewidth=1.2,
            label=METHOD_LABELS[mode],
            zorder=3 if mode == "cross" else 2,
        )
    ax.set_xticks(x, [str(value) for value in budgets], fontweight="bold")
    ax.set_xlabel("Training labels per class")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(-0.30, 6.30)
    ax.set_ylim(39, 87)
    ax.set_yticks([40, 50, 60, 70, 80])
    ax.grid(color=GRID, lw=0.75)
    legend = ax.legend(
        loc="lower right",
        ncol=2,
        fontsize=8.1,
        handlelength=1.9,
        columnspacing=0.75,
        handletextpad=0.40,
        borderaxespad=0.55,
        frameon=True,
        facecolor=WHITE,
        edgecolor="#D7DCE4",
        framealpha=0.98,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    temporal_one = probes["representations"][names["temporal"]]["low_label"][
        "1"
    ]["mean_accuracy"]
    cross_one = probes["representations"][names["cross"]]["low_label"]["1"][
        "mean_accuracy"
    ]
    ax.annotate(
        (
            f"Extreme 1-shot:\nTemporal {temporal_one:.1%} > "
            f"Cross {cross_one:.1%}"
        ),
        xy=(0, 100 * temporal_one),
        xytext=(0.18, 43.3),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=DARK,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": WHITE,
            "edgecolor": COLORS["temporal"],
            "linewidth": 1.0,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["temporal"],
            "lw": 1.4,
        },
    )


def _ood_heatmap(
    ax: plt.Axes,
    probes: dict,
    *,
    label_ax: plt.Axes | None = None,
) -> None:
    title_ax = label_ax if label_ax is not None else ax
    _panel_title(
        title_ax,
        "D",
        "Time-OOD gain by domain",
        fontsize=12.8,
        pad=10,
    )
    labels = list(probes["metadata"]["labels"])
    names = _probe_representation_names(probes)
    modes = tuple(
        mode
        for mode in METHODS
        if names[mode] in probes.get("representations", {})
    )
    actual = np.asarray(
        [
            [
                probes["representations"][names[mode]]["ood"][
                    "per_class_accuracy"
                ][label]
                for label in labels
            ]
            for mode in modes
        ]
    )
    cross_index = modes.index("cross")
    references = tuple(mode for mode in modes if mode != "cross")
    gains = 100 * np.stack(
        [
            actual[cross_index] - actual[modes.index(reference)]
            for reference in references
        ],
        axis=0,
    )
    maximum = 11.0
    cmap = LinearSegmentedColormap.from_list(
        "gain",
        [COLORS["token"], "#FAFAFB", COLORS["cross"]],
    )
    image = ax.imshow(
        gains,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
    )
    ax.set_xticks(
        np.arange(len(labels)),
        [DOMAIN_SHORT[label] for label in labels],
        rotation=24,
        ha="right",
        fontweight="bold",
    )
    if label_ax is None:
        ax.set_yticks(
            np.arange(len(references)),
            [METHOD_SHORT_LABELS[mode] for mode in references],
            fontweight="bold",
        )
    else:
        ax.set_yticks(np.arange(len(references)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        label_ax.set_axis_off()
        label_ax.set_xlim(0, 1)
        label_ax.set_ylim(len(references) - 0.5, -0.5)
        label_ax.text(
            0.88,
            -0.31,
            "BASELINE",
            ha="right",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=MID,
        )
        for row, mode in enumerate(references):
            label_ax.text(
                0.88,
                row,
                METHOD_SHORT_LABELS[mode],
                ha="right",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color=DARK,
            )
    for row in range(gains.shape[0]):
        for column in range(len(labels)):
            value = gains[row, column]
            ax.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=9.3,
                fontweight="bold",
                color=WHITE if abs(value) >= 6 else DARK,
            )
    stats = labels.index("Statistics")
    ax.add_patch(
        FancyBboxPatch(
            (stats - 0.48, -0.48),
            0.96,
            gains.shape[0] - 0.04,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            fill=False,
            edgecolor=GOLD,
            linewidth=2.8,
        )
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.11,
        pad=0.30,
        aspect=35,
    )
    colorbar.set_label(
        "Cross-Chunk − baseline accuracy (pp)",
        fontsize=9.5,
    )
    colorbar.ax.tick_params(labelsize=8.7)
    if label_ax is not None:
        label_position = label_ax.get_position()
        heat_position = ax.get_position()
        label_ax.set_position(
            [
                label_position.x0,
                heat_position.y0,
                label_position.width,
                heat_position.height,
            ]
        )


def _downstream_kpis(
    ax: plt.Axes,
    geometry: dict,
    probes: dict,
) -> None:
    ax.set_axis_off()
    _panel_title(
        ax,
        "E",
        "OOD summary",
        fontsize=11.8,
        pad=10,
    )
    names = _probe_representation_names(probes)
    modes = tuple(
        mode
        for mode in METHODS
        if mode in geometry.get("methods", {})
        and names[mode] in probes.get("representations", {})
    )
    ood_accuracy = {
        mode: float(
            probes["representations"][names[mode]]["ood"]["accuracy"]
        )
        for mode in modes
    }
    worst = {
        mode: float(
            probes["representations"][names[mode]]["ood"][
                "worst_class_accuracy"
            ]
        )
        for mode in modes
    }
    nmi = {
        mode: float(geometry["methods"][mode]["semantic_information_nmi"])
        for mode in modes
    }
    cards = [
        (
            0.67,
            f"{ood_accuracy['cross']:.1%}",
            "OOD ACCURACY",
            (
                f"+{100 * (ood_accuracy['cross'] - max(ood_accuracy[m] for m in modes if m != 'cross')):.1f} pp "
                "vs best"
            ),
        ),
        (
            0.36,
            f"{worst['cross']:.1%}",
            "WORST DOMAIN",
            (
                f"+{100 * (worst['cross'] - max(worst[m] for m in modes if m != 'cross')):.1f} pp "
                "vs best"
            ),
        ),
        (
            0.05,
            f"{nmi['cross']:.3f}",
            "SEMANTIC NMI",
            (
                f"+{100 * (nmi['cross'] - max(nmi[m] for m in modes if m != 'cross')):.1f} pp "
                "vs best"
            ),
        ),
    ]
    for y, value, label, delta in cards:
        _box(
            ax,
            0.04,
            y,
            0.92,
            0.24,
            face=WHITE,
            edge=COLORS["cross"],
            linewidth=1.8,
            radius=0.030,
        )
        ax.text(
            0.50,
            y + 0.158,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=21.0,
            fontweight="bold",
            color=COLORS["cross"],
        )
        ax.text(
            0.50,
            y + 0.090,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=DARK,
        )
        ax.text(
            0.50,
            y + 0.035,
            delta,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=COLORS["cross"],
        )


def _save(
    fig: plt.Figure,
    base: Path,
    dpi: int,
    *,
    overwrite: bool = False,
) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    style_figure_text(fig, minimum_tick_size=9.5)
    png_path = base.with_suffix(".png")
    if overwrite or not png_path.is_file():
        fig.savefig(
            png_path,
            dpi=dpi,
            facecolor=WHITE,
            bbox_inches=None,
            pad_inches=0,
        )
    fig.savefig(
        base.with_suffix(".pdf"),
        dpi=dpi,
        facecolor=WHITE,
        bbox_inches=None,
        pad_inches=0,
        metadata={
            "Title": base.name,
            "Creator": "Chunk-SAE evaluation pipeline",
        },
    )
    plt.close(fig)


def plot_feature_quality(
    eval_root: Path,
    *,
    high_level: dict,
    adjacent: dict,
    invariance: dict,
    linking: dict,
    dpi: int,
    overwrite: bool = False,
) -> None:
    if (eval_root / "paper_figure1_feature_quality.png").is_file() and not overwrite:
        return
    fig = plt.figure(figsize=(16, 8), facecolor=WHITE)
    _header(
        fig,
        "Sparse features: illustrative concepts and aggregate controls",
        (
            "The low-overlap same-document example evaluates retrieval only; "
            "the high-level feature census is a separate held-out audit."
        ),
    )
    outer = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.955,
        bottom=0.075,
        top=0.95,
        height_ratios=[0.92, 1.08],
        width_ratios=[1.05, 1.0],
        hspace=0.32,
        wspace=0.16,
    )
    feature_ax = fig.add_subplot(outer[0, :])
    frontier_ax = fig.add_subplot(outer[1, 0])
    dictionary_grid = outer[1, 1].subgridspec(
        1,
        2,
        wspace=0.30,
    )
    effective_ax = fig.add_subplot(dictionary_grid[0, 0])
    concentration_ax = fig.add_subplot(dictionary_grid[0, 1])
    _feature_contrast(feature_ax, invariance, linking)
    _feature_frontier(frontier_ax, high_level, adjacent)
    _dictionary_panel(
        effective_ax,
        concentration_ax,
        adjacent,
    )
    _save(
        fig,
        eval_root / "paper_figure1_feature_quality",
        dpi,
        overwrite=overwrite,
    )


def plot_downstream_value(
    eval_root: Path,
    *,
    geometry: dict,
    probes: dict,
    linking: dict,
    dpi: int,
    overwrite: bool = False,
) -> None:
    if (eval_root / "paper_figure2_downstream_transfer.png").is_file() and not overwrite:
        return
    _downstream_style()
    fig = plt.figure(figsize=(16, 8), facecolor=WHITE)
    _header(
        fig,
        "Cross-Chunk codes improve downstream retrieval and OOD transfer",
        (
            "These task-level gains do not by themselves imply a larger "
            "fraction of individually interpretable high-level features."
        ),
    )
    outer = fig.add_gridspec(
        1,
        1,
        left=0.080,
        right=0.975,
        bottom=0.075,
        top=0.955,
    )
    tasks = outer[0].subgridspec(
        2,
        1,
        height_ratios=[0.90, 1.10],
        hspace=0.34,
    )
    retrieval = tasks[0].subgridspec(
        1,
        2,
        width_ratios=[1.48, 0.92],
        wspace=0.08,
    )
    ood = tasks[1].subgridspec(
        1,
        2,
        width_ratios=[1.50, 0.92],
        wspace=0.22,
    )
    linking_ax = fig.add_subplot(retrieval[0, 0])
    example_ax = fig.add_subplot(retrieval[0, 1])
    ood_left = ood[0, 0].subgridspec(
        1,
        2,
        width_ratios=[1.0, 0.030],
        wspace=0.018,
    )
    ood_absolute_ax = fig.add_subplot(ood_left[0, 0])
    ood_colorbar_ax = fig.add_subplot(ood_left[0, 1])
    ood_advantage_ax = fig.add_subplot(ood[0, 1])
    _document_linking_panel(linking_ax, linking)
    _document_linking_example(example_ax, linking)
    modes, labels, matrix = _ood_absolute_panel(
        ood_absolute_ax,
        ood_colorbar_ax,
        probes,
    )
    _ood_advantage_panel(
        ood_advantage_ax,
        modes=modes,
        labels=labels,
        matrix=matrix,
    )
    _save(
        fig,
        eval_root / "paper_figure2_downstream_transfer",
        dpi,
        overwrite=overwrite,
    )


def main() -> None:
    args = parser().parse_args()
    _style()
    eval_root = Path(args.eval_root).resolve()
    high_level = _read(
        eval_root
        / "shared"
        / "high_level_feature_analysis"
        / "high_level_feature_results.json"
    )
    adjacent = _read(
        eval_root
        / "shared"
        / "feature_consistency"
        / "adjacent_consistency"
        / "adjacent_feature_consistency.json"
    )
    invariance_path = eval_root / "semantic_invariance" / "results.json"
    invariance = _read(invariance_path) if invariance_path.is_file() else {}
    geometry = _read(
        eval_root
        / "semantic_geometry"
        / "representation_geometry"
        / "representation_geometry.json"
    )
    probes = _read(eval_root / "label_efficiency" / "linear_probe_results.json")
    linking = _read(
        eval_root
        / args.document_linking_dir
        / "document_linking_results.json"
    )
    plot_feature_quality(
        eval_root,
        high_level=high_level,
        adjacent=adjacent,
        invariance=invariance,
        linking=linking,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    plot_downstream_value(
        eval_root,
        geometry=geometry,
        probes=probes,
        linking=linking,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(eval_root / "paper_figure1_feature_quality.png", flush=True)
    print(eval_root / "paper_figure2_downstream_transfer.png", flush=True)


if __name__ == "__main__":
    main()
