"""Shared camera-ready plotting vocabulary for all Chunk-SAE evaluations."""

from __future__ import annotations


METHODS = ("token", "temporal", "mean", "cross")
JOINT_METHODS = (
    "joint_alpha0p25",
    "joint_alpha0p5",
    "joint_alpha1",
    "joint_alpha1p5",
)
ALL_METHODS = (
    "token",
    "temporal",
    "mean",
    *JOINT_METHODS,
    "cross",
)

METHOD_LABELS = {
    "token": "BatchTopK SAE",
    "temporal": "Temporal SAE",
    "mean": "Mean-Chunk SAE",
    # ``joint`` is the legacy alias for the published alpha=0.25 checkpoint.
    "joint": "Joint-Chunk SAE (α=0.25)",
    "joint_alpha0p25": "Joint-Chunk SAE (α=0.25)",
    "joint_alpha0p5": "Joint-Chunk SAE (α=0.5)",
    "joint_alpha1": "Joint-Chunk SAE (α=1.0)",
    "joint_alpha1p5": "Joint-Chunk SAE (α=1.5)",
    "cross": "Cross-Chunk SAE",
}

METHOD_SHORT_LABELS = {
    "token": "BatchTopK",
    "temporal": "Temporal",
    "mean": "Mean-Chunk",
    "joint": "Joint α=0.25",
    "joint_alpha0p25": "Joint α=0.25",
    "joint_alpha0p5": "Joint α=0.5",
    "joint_alpha1": "Joint α=1.0",
    "joint_alpha1p5": "Joint α=1.5",
    "cross": "Cross-Chunk",
}

# Exact camera-ready palette used by the frozen-explanation AutoInterp figure.
# Every evaluation and paper figure must use these colors for method identity.
METHOD_COLORS = {
    "token": "#b8d7ea",
    "temporal": "#317ecb",
    "mean": "#feb499",
    "joint": "#f5c26b",
    "joint_alpha0p25": "#f5c26b",
    "joint_alpha0p5": "#e78b55",
    "joint_alpha1": "#b55d91",
    "joint_alpha1p5": "#7a5195",
    "cross": "#be223d",
}

METHOD_LIGHT_COLORS = {
    "token": "#b8d7ea",
    "temporal": "#317ecb",
    "mean": "#feb499",
    "joint": "#f5c26b",
    "joint_alpha0p25": "#f5c26b",
    "joint_alpha0p5": "#e78b55",
    "joint_alpha1": "#b55d91",
    "joint_alpha1p5": "#7a5195",
    "cross": "#be223d",
}

METHOD_PALE_COLORS = {
    "token": "#f3f8fb",
    "temporal": "#eaf3fb",
    "mean": "#fff3eb",
    "joint": "#fbf3de",
    "joint_alpha0p25": "#fbf3de",
    "joint_alpha0p5": "#fbe9df",
    "joint_alpha1": "#f5e8f0",
    "joint_alpha1p5": "#eeeaf5",
    "cross": "#f8e9ec",
}

METHOD_MARKERS = {
    "token": "o",
    "temporal": "s",
    "mean": "^",
    "joint": "v",
    "joint_alpha0p25": "v",
    "joint_alpha0p5": "P",
    "joint_alpha1": "*",
    "joint_alpha1p5": "X",
    "cross": "D",
}

# The two light method colors are intentionally too pale for small text on a
# white page. Data marks retain the exact palette above; these darker family
# tones are used only for method-name text so labels remain accessible.
METHOD_TEXT_COLORS = {
    mode: "#252932" for mode in (*ALL_METHODS, "joint")
}


def style_figure_text(fig, *, minimum_tick_size: float | None = None) -> None:
    """Bold axis labels, tick labels, and legends without changing data ink."""

    axes = list(fig.axes)
    for ax in axes:
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        ax.title.set_fontweight("bold")
        tick_labels = [*ax.get_xticklabels(), *ax.get_yticklabels()]
        for label in tick_labels:
            label.set_fontweight("bold")
            label.set_color("#252932")
            if (
                minimum_tick_size is not None
                and label.get_fontsize() < minimum_tick_size
            ):
                label.set_fontsize(minimum_tick_size)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontweight("bold")
                text.set_color("#252932")
            legend.get_title().set_fontweight("bold")
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_color("#252932")
        legend.get_title().set_fontweight("bold")
