#!/usr/bin/env python
"""Render the single at-a-glance plot for the code-page retrieval task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from chunk_saes.plot_style import METHOD_COLORS


METHODS = (
    ("token", "BatchTopK SAE", METHOD_COLORS["token"]),
    ("temporal", "Temporal SAE", METHOD_COLORS["temporal"]),
    ("mean", "Mean-Chunk SAE", METHOD_COLORS["mean"]),
    ("cross", "Cross-Chunk SAE", METHOD_COLORS["cross"]),
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def render(summary_path: Path, output: Path) -> Path:
    if output.parent.name != "figures":
        raise ValueError(f"--output must be inside a figures/ directory: {output}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("retrieval summary has no queries")
    labels = [str(row.get("title") or row.get("query_id")) for row in queries]
    values = {
        key: [
            100.0
            * float((row.get("methods") or {}).get(key, {}).get("top20_keyword_hits", 0))
            / 20.0
            for row in queries
        ]
        for key, _label, _color in METHODS
    }
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    positions = list(range(len(labels)))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for method_index, ((key, label, color), offset) in enumerate(
        zip(METHODS, offsets, strict=True)
    ):
        bars = ax.bar(
            [x + offset * width for x in positions],
            values[key],
            width=width,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.7,
            label=label,
        )
        for bar, value in zip(bars, values[key], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5 + method_index * 3.2,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#252932",
            )
    ax.set_title(
        "Code-page retrieval: topic hits in each method's Top-20",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=28,
    )
    ax.text(
        0,
        1.075,
        "Higher values mean more of the retrieved pages satisfy the pre-registered topic rule; this is a lexical audit, not a human relevance score.",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#667085",
        va="bottom",
    )
    ax.set_ylabel("Top-20 keyword-rule hits (%)")
    ax.set_xticks(positions, labels, rotation=24, ha="right")
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    ax.grid(axis="y", color="#DDE2EA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncol=4,
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(0, 1.02),
        columnspacing=1.2,
    )
    fig.text(
        0.01,
        0.015,
        f"{len(queries)} locked queries · {int(payload.get('candidate_documents', 0)):,} candidate pages · seed and candidate pool fixed",
        fontsize=8,
        color="#667085",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.24)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parser().parse_args()
    print(render(args.summary, args.output), flush=True)


if __name__ == "__main__":
    main()
