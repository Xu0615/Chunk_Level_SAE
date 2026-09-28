#!/usr/bin/env python
"""Compare additional Joint-Chunk weights with the two token-level baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from chunk_saes.plot_style import (
    METHOD_COLORS as SHARED_METHOD_COLORS,
    METHOD_TEXT_COLORS as SHARED_METHOD_TEXT_COLORS,
)

try:
    from . import plot_evaluation_summary_radar as radar
    from .evaluation_summary_data import load_metric_specs
except ImportError:  # pragma: no cover
    import plot_evaluation_summary_radar as radar  # type: ignore
    from evaluation_summary_data import load_metric_specs  # type: ignore


BASELINES = ("token", "temporal")
ALPHAS = ("joint_alpha0p5", "joint_alpha1", "joint_alpha1p5")
METHODS = (*BASELINES, *ALPHAS)
ALPHA_LABELS = {
    "joint_alpha0p5": r"Joint-Chunk SAE ($\alpha=0.5$)",
    "joint_alpha1": r"Joint-Chunk SAE ($\alpha=1.0$)",
    "joint_alpha1p5": r"Joint-Chunk SAE ($\alpha=1.5$)",
}
ALPHA_MARKERS = {
    "joint_alpha0p5": "s",
    "joint_alpha1": "D",
    "joint_alpha1p5": "P",
}
METHOD_LABELS = {
    **{method: radar.METHOD_LABELS[method] for method in BASELINES},
    **ALPHA_LABELS,
}
# Share method colors with rfve/plot_training_health.py, including all alphas.
METHOD_COLORS = {method: SHARED_METHOD_COLORS[method] for method in METHODS}
METHOD_MARKERS = {
    **{method: radar.METHOD_MARKERS[method] for method in BASELINES},
    **ALPHA_MARKERS,
}
METHOD_TEXT_COLORS = {
    method: SHARED_METHOD_TEXT_COLORS[method] for method in METHODS
}


def _read(root: Path, relative: str) -> dict:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {relative}")
    return value


def _alpha_metric_values(root: Path) -> dict[str, dict[str, float]]:
    """Read alpha-specific values from the same canonical artifacts as the main radar."""
    rfve = _read(root, "rfve/training_fidelity.json")["methods"]
    cosine = _read(root, "rfve/reconstruction_cosine.json")["methods"]
    context = _read(root, "autointerp/autointerp_exact1000/results/summary.json")["methods"]
    high = _read(root, "semantic_invariance/results.json")["methods"]
    dictionary = _read(root, "dictionary_utilization/dictionary_utilization.json")["methods"]
    temporal = _read(root, "temporal_robustness/temporal_robustness.json")["methods"]
    probes = _read(root, "label_efficiency/linear_probe_results.json")["high_level_transfer"]["methods"]
    reasoning = _read(root, "reasoning/results_summary.json")
    steering = _read(root, "steering/steering_summary.json")["methods"]
    linking = _read(root, "document_linking/document_linking_results.json")["methods"]

    result: dict[str, dict[str, float]] = {alpha: {} for alpha in ALPHAS}
    for alpha in ALPHAS:
        result[alpha].update(
            rfve=float(rfve[alpha]["rfve"]),
            reconstruction_cosine=float(cosine[alpha]["mean_cosine_similarity"]),
            context_autointerp=float(context[alpha]["supported_selectivity"]),
            high_level_feature_fraction=float(high[alpha]["high_level_fraction"]),
            feature_persistence_lift=float(
                dictionary[alpha]["feature_persistence_metrics"]["mean_persistence_lift"]
            ),
            dictionary_utilization=float(dictionary[alpha]["effective_feature_fraction"]),
            arxiv_ood_accuracy=float(temporal[alpha]["ood_accuracy"]),
            arxiv_low_label_auc=float(probes[alpha]["low_label_auc"]),
            reasoning_native_recall=float(reasoning["native_recall_mean"][alpha]),
            reasoning_generalization=0.5
            * (
                float(reasoning["cue_free_recall_mean"][alpha])
                + 1.0
                - float(reasoning["cue_only_false_activation_mean"][alpha])
            ),
            causal_steering=float(steering[alpha]["mean"]),
            document_recall_at_5=float(linking[alpha]["recall_at_5"]),
        )
    return result


def _patch_style() -> None:
    """Reuse the camera-ready radar geometry while changing only the series."""
    radar.METHODS = METHODS
    radar.METHOD_LABELS = METHOD_LABELS
    radar.METHOD_COLORS = METHOD_COLORS
    radar.METHOD_MARKERS = METHOD_MARKERS
    radar.METHOD_TEXT_COLORS = METHOD_TEXT_COLORS


def render(eval_root: Path, output_dir: Path, output_name: str, dpi: int) -> list[Path]:
    _patch_style()
    base_metrics = load_metric_specs(eval_root)
    values = _alpha_metric_values(eval_root)
    metrics = tuple(
        replace(
            metric,
            values={
                **{method: metric.values[method] for method in BASELINES},
                **{alpha: values[alpha][metric.key] for alpha in ALPHAS},
            },
        )
        for metric in base_metrics
    )
    intrinsic = tuple(metric for metric in metrics if metric.group == "intrinsic")
    downstream = tuple(metric for metric in metrics if metric.group == "downstream")
    if len(intrinsic) != 6 or len(downstream) != 6:
        raise RuntimeError("expected six intrinsic and six downstream metrics")

    radar.matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": radar.TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(18.0, 9.0), dpi=dpi, facecolor="white")
    left = fig.add_axes([0.097, 0.257, 0.295, 0.590], projection="polar")
    right = fig.add_axes([0.591, 0.247, 0.315, 0.630], projection="polar")
    label_colors = tuple(color for color in radar.GROUP_COLORS for _ in range(2))
    left_angles = radar._draw_radar(left, intrinsic, label_colors=label_colors)
    right_angles = radar._draw_radar(
        right, downstream, label_colors=label_colors, metric_label_radius=1.170
    )
    for ax, angles, pairs in (
        (left, left_angles, ((0, 1, "Reconstruction"), (2, 3, "Interpretability"), (4, 5, "Feature-level\nproperties"))),
        (right, right_angles, ((0, 1, "Classification transfer"), (2, 3, "Reasoning"), (4, 5, "Steering &\nretrieval"))),
    ):
        for first, second, label in pairs:
            radar._add_pair_brace(fig, ax, angles, first, second, label, color=radar.GROUP_COLORS[pairs.index((first, second, label))])

    fig.text(0.251, 0.984, "Representation quality and interpretability", ha="center", va="top", fontsize=26.0, fontweight="bold", color=radar.DEEP_BLUE)
    fig.text(0.7525, 0.984, "Downstream utility", ha="center", va="top", fontsize=26.0, fontweight="bold", color=radar.BRICK_RED)
    handles = [
        Line2D(
            [0], [0], color=METHOD_COLORS[method], linewidth=3.2,
            linestyle=(0, (4, 2.4)) if method in BASELINES else "-",
            marker=METHOD_MARKERS[method], markersize=9.5,
            markeredgecolor="white", markeredgewidth=0.9,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.018, 0.022, 0.964, 0.078), mode="expand", ncol=5, frameon=False, fontsize=17.5, handlelength=1.85, handletextpad=0.25, columnspacing=0.25, borderaxespad=0.0)
    for method, text in zip(METHODS, legend.get_texts(), strict=True):
        text.set_fontweight("bold")
        text.set_fontstretch("condensed")
        text.set_color(METHOD_TEXT_COLORS[method])
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / output_name
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default="joint_alpha_evaluation_summary")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    root = Path(args.eval_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "evaluation_summary" / "figures"
    print("\n".join(str(p) for p in render(root, out, args.output_name, args.dpi)), flush=True)


if __name__ == "__main__":
    main()
