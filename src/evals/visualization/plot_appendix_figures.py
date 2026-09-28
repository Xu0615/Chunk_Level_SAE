#!/usr/bin/env python
"""Render the paper's appendix figures from frozen evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from chunk_saes.plot_style import (
    ALL_METHODS,
    METHOD_COLORS,
    METHOD_MARKERS,
    METHOD_SHORT_LABELS,
    METHODS,
    style_figure_text,
)


FIGURES = (
    "training_health",
    "sequence_activation_tsne",
    "semantic_geometry",
    "label_efficiency",
    "joint_alpha_evaluation_summary",
    "reasoning_matched_exposure",
    "steering_strength_curve",
)
MAIN_METHODS = ("token", "temporal", "mean", "joint_alpha0p25", "cross")
DARK = "#252932"
BLUE = "#0C445E"
GRID = "#D5E2EA"
LABELS = {m: METHOD_SHORT_LABELS[m] for m in ALL_METHODS}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "font.weight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.edgecolor": "#99BAD1",
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "text.color": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


def method_handle(method: str, *, dashed: bool = False) -> Line2D:
    return Line2D(
        [], [], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
        linewidth=2.6, markersize=7, markeredgecolor="white",
        markeredgewidth=0.5, linestyle="--" if dashed else "-",
        label=LABELS[method],
    )


def method_legend(fig, methods, *, columns=4, y=0.995) -> None:
    fig.legend(
        handles=[method_handle(m) for m in methods], loc="upper center",
        bbox_to_anchor=(0.5, y), ncol=columns, columnspacing=1.3,
        handletextpad=0.5, handlelength=2, borderaxespad=0,
    )


def clean_axis(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=BLUE, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(length=3.5, pad=5)


def save_figure(fig, output: Path, name: str, dpi: int) -> dict:
    style_figure_text(fig)
    fig.canvas.draw()
    paths = []
    for suffix in ("pdf", "png"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
        paths.append(path.name)
    size = fig.get_size_inches().tolist()
    plt.close(fig)
    return {"files": paths, "canvas_inches": size}


def training_health(root: Path, output: Path, dpi: int) -> dict:
    source = root / "rfve/training_health.json"
    data = read_json(source)
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 7.0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.85, bottom=0.10,
                        hspace=0.46, wspace=0.27)
    method_legend(fig, ALL_METHODS)
    titles = ("A  Train-validation RFVE gap", "B  Effective sparsity",
              "C  Recently inactive features", "D  Zero-code rate")
    for ax, title in zip(axs.flat, titles):
        clean_axis(ax, title)
        ax.set_xlim(-0.01, 1.01)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Training occurrences (billions)")
    trajectories = {}
    for index, method in enumerate(ALL_METHODS):
        row = data["methods"][method]
        with Path(row["metrics_path"]).open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        # Resumed runs can repeat steps; the last logged record owns each step.
        validation = {r["step"]: r for r in records if r.get("split") == "validation"}
        train = {r["step"]: r for r in records if r.get("split") == "train"}
        lkey = "validation/effective_l0"
        zkey = "validation/zero_code_fraction"
        if method.startswith("joint"):
            lkey = "validation/joint_mean_effective_l0"
            zkey = "validation/joint_mean_zero_code_fraction"
        valid = [validation[s] for s in sorted(validation) if lkey in validation[s]]
        x = np.array([r["step"] * 32000 / 1e9 for r in valid])
        l0 = np.array([r[lkey] for r in valid])
        zero = np.array([r[zkey] for r in valid]) * 100
        if method.startswith("joint"):
            l0 = np.minimum(l0, [r["validation/joint_cross_effective_l0"] for r in valid])
            zero = np.maximum(zero, np.array([
                r["validation/joint_cross_zero_code_fraction"] for r in valid
            ]) * 100)
        active = [train[s] for s in sorted(train) if "sparsity/dead_features" in train[s]]
        tx = np.array([r["step"] * 32000 / 1e9 for r in active])
        dead = np.array([r["sparsity/dead_features"] for r in active]) / 65536 * 100
        gap = row["generalization_gap_trajectory"]
        gx = np.array([r["samples_seen"] / 1e9 for r in gap])
        gy = np.array([r["train_minus_validation_rfve"] for r in gap])
        color = METHOD_COLORS[method]
        # Keep raw observations visible; smoothing applies only to the RFVE gap.
        width = min(9, len(gy) if len(gy) % 2 else len(gy) - 1)
        smooth = np.convolve(np.pad(gy, (width // 2,) * 2, mode="edge"),
                             np.ones(width) / width, mode="valid")
        axs[0, 0].plot(gx, gy, color=color, linewidth=0.7, alpha=0.16)
        axs[0, 0].plot(gx, smooth, color=color, linewidth=2.2)
        for ax, xx, yy in ((axs[0, 1], x, l0), (axs[1, 0], tx, dead),
                           (axs[1, 1], x, zero)):
            ax.plot(xx, yy, color=color, linewidth=2.0)
            # Stagger markers along observed x positions to expose coincident curves.
            mark = np.unique(np.linspace(0, len(xx) - 1, 25, dtype=int)[index::8])
            ax.plot(xx[mark], yy[mark], linestyle="none", color=color,
                    marker=METHOD_MARKERS[method], markersize=6.5,
                    markeredgecolor="white", markeredgewidth=0.45, zorder=5)
        trajectories[method] = {
            "metrics_path": row["metrics_path"],
            "rfve_gap": {"occurrences_billions": gx.tolist(), "raw": gy.tolist(),
                         "moving_average_9": smooth.tolist()},
            "validation": {"occurrences_billions": x.tolist(), "effective_l0": l0.tolist(),
                           "zero_code_percent": zero.tolist()},
            "inactivity": {"occurrences_billions": tx.tolist(), "percent": dead.tolist()},
        }
    axs[0, 0].axhline(0, color=DARK, linewidth=0.8, linestyle="--")
    axs[0, 0].set_ylabel("RFVE gap")
    axs[0, 1].set_ylabel("Mean active features")
    axs[0, 1].set_ylim(124, 132)
    axs[0, 1].set_yticks([124, 128, 132])
    axs[0, 1].text(0.5, 0.82, r"All methods: $L_0 = 128$", ha="center",
                   transform=axs[0, 1].transAxes, fontsize=15)
    axs[1, 0].set_ylabel("Dictionary fraction (%)")
    axs[1, 0].set_ylim(-0.001, 0.026)
    axs[1, 0].set_yticks([0, 0.01, 0.02], ["0", "0.01", "0.02"])
    axs[1, 1].set_ylabel("Empty codes (%)")
    axs[1, 1].set_yscale("symlog", linthresh=0.001)
    axs[1, 1].set_ylim(-0.00025, 2.5)
    axs[1, 1].set_yticks([0, 0.001, 0.01, 0.1, 1], ["0", "0.001", "0.01", "0.1", "1"])
    info = save_figure(fig, output, "training_health", dpi)
    info.update(sources=[str(source), *[v["metrics_path"] for v in trajectories.values()]],
                protocol={"rfve_gap_smoothing": "9 logged observations; raw curves shown faintly",
                          "dead_feature_window_occurrences": 10000000,
                          "zero_axis": "symlog, linear threshold 0.001 percent",
                          "joint_code_metrics": "minimum target L0; maximum target zero-code fraction"},
                plotted_data=trajectories)
    return info


def sequence_activation_tsne(root: Path, output: Path, dpi: int) -> dict:
    """Keep the published asset path, showing only the document embeddings."""
    from evals.feature_dynamics.plot_cross_feature_semantic_manifold import (
        DOMAIN_COLORS, DOMAIN_SHORT, _robust_ellipse,
    )
    embedding_source = root / "semantic_geometry/representation_geometry/representation_display_embeddings.npz"
    metric_source = root / "feature_dynamics/cross_concept_manifold_figure.json"
    geometry_source = root / "semantic_geometry/representation_geometry/representation_geometry.json"
    embeddings = load_arrays(embedding_source)
    metrics = read_json(metric_source)["manifold_metrics"]
    geometry = read_json(geometry_source)["methods"]
    highlighted = ("Quantitative Biology", "Quantitative Finance")
    margins = {
        domain: metrics["cross"]["domain_neighbor_purity"][domain]
        - max(metrics[m]["domain_neighbor_purity"][domain] for m in METHODS if m != "cross")
        for domain in DOMAIN_COLORS
    }
    assert tuple(sorted(margins, key=margins.get, reverse=True)[:2]) == highlighted
    label_positions = {
        "token": ((0.035, 0.92), (0.965, 0.065)),
        "temporal": ((0.035, 0.92), (0.965, 0.065)),
        "mean": ((0.035, 0.065), (0.965, 0.92)),
        "cross": ((0.035, 0.92), (0.965, 0.065)),
    }
    purity = {m: float(geometry[m]["geometry"]["neighbor_purity"]) for m in METHODS}
    best = max(METHODS, key=purity.get)
    gain = 100 * (purity[best] - purity["token"])
    fig, axs = plt.subplots(1, 4, figsize=(13.4, 4.3))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.82, bottom=0.125, wspace=0.08)
    fig.text(0.02, 0.975, "Neighborhood purity in 50-D (%) | Higher is better",
             fontsize=15, ha="left", va="top", color=BLUE)
    fig.text(0.98, 0.975, f"Highest: {LABELS[best]} (+{gain:.1f} pp vs BatchTopK)",
             fontsize=15, ha="right", va="top", color=METHOD_COLORS[best])
    plotted = {}
    for index, (ax, method) in enumerate(zip(axs, METHODS)):
        xy = embeddings[f"{method}_xy"]
        labels = embeddings["labels"]
        assert xy.shape == (2048, 2) and len(labels) == 2048
        assert set(labels) == set(DOMAIN_COLORS)
        np.testing.assert_allclose(metrics[method]["neighbor_purity"], purity[method])
        domain_purity = metrics[method]["domain_neighbor_purity"]
        np.testing.assert_allclose(np.mean(list(domain_purity.values())), purity[method])
        for domain, color in DOMAIN_COLORS.items():
            mask = labels == domain
            ax.scatter(xy[mask, 0], xy[mask, 1], s=7, c=color, alpha=0.70,
                       linewidths=0, rasterized=True)
        ax.set_title(f"{chr(65 + index)}  {LABELS[method]} | {purity[method] * 100:.1f}%",
                     fontsize=15.5, pad=10, color=BLUE)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.04)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(METHOD_COLORS[method])
            spine.set_linewidth(1.5)
        ax.apply_aspect()
        plotted[method] = {"overall_purity_percent": purity[method] * 100, "highlights": {}}
        for domain, position in zip(highlighted, label_positions[method]):
            center, width, height, angle = _robust_ellipse(xy[labels == domain])
            ax.add_patch(Ellipse(center, width, height, angle=angle, fill=False,
                                 edgecolor=DOMAIN_COLORS[domain], linewidth=1.8, zorder=4))
            # End the arrow at the ellipse edge, leaving the central points visible.
            label_data = ax.transData.inverted().transform(ax.transAxes.transform(position))
            direction = label_data - center
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            local_direction = rotation.T @ direction
            radius = np.sqrt((local_direction[0] / (width / 2)) ** 2
                             + (local_direction[1] / (height / 2)) ** 2)
            target = center + direction / radius
            ax.annotate(f"{DOMAIN_SHORT[domain]} {domain_purity[domain] * 100:.1f}%",
                        xy=target, xytext=position, textcoords="axes fraction",
                        ha="left" if position[0] < 0.5 else "right", va="center",
                        fontsize=14, color=DARK,
                        bbox={"boxstyle": "round,pad=0.25,rounding_size=0.15",
                              "facecolor": "white", "edgecolor": DOMAIN_COLORS[domain],
                              "linewidth": 1.0, "alpha": 0.97},
                        arrowprops={"arrowstyle": "-|>", "color": DOMAIN_COLORS[domain],
                                    "lw": 1.4, "mutation_scale": 11,
                                    "connectionstyle": "arc3,rad=0.12", "shrinkA": 3, "shrinkB": 1},
                        zorder=5)
            plotted[method]["highlights"][domain] = {
                "domain_purity_percent": domain_purity[domain] * 100,
                "ellipse_center": center.tolist(), "ellipse_width": float(width),
                "ellipse_height": float(height), "ellipse_angle": float(angle),
                "label_axes_fraction": list(position), "arrow_target": target.tolist(),
            }
    handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=7,
                      label=DOMAIN_SHORT[domain]) for domain, color in DOMAIN_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.002),
               ncol=8, fontsize=14, columnspacing=1.0, handletextpad=0.3)
    info = save_figure(fig, output, "sequence_activation_tsne", dpi)
    info.update(sources=[str(embedding_source), str(embedding_source.with_suffix(".json")),
                         str(metric_source), str(geometry_source)],
                methods=list(METHODS),
                protocol={"tsne": read_json(embedding_source.with_suffix(".json"))["identity"],
                          "tsne_samples": len(embeddings["labels"]),
                          "panels": "four independent document t-SNE projections; axes not aligned",
                          "domain_colors": DOMAIN_COLORS,
                          "purity": "cached 50-D label-free SVD, non-self cosine neighbor ranks 2-11; not a 2-D or circled-region metric",
                          "highlighted_domains": list(highlighted),
                          "highlight_margins_pp": {domain: margins[domain] * 100 for domain in highlighted},
                          "selection": "the two domains with the largest Cross-Chunk purity gains over the strongest other displayed method",
                          "ellipse": "shared robust ellipse: remove farthest 8% from domain median, covariance scale 1.15; illustrative, not a confidence region",
                          "best_method": best, "best_minus_batchtopk_pp": gain,
                          "removed": "feature traces already shown in the main text"},
                plotted_data=plotted)
    return info


def semantic_geometry(root: Path, output: Path, dpi: int) -> dict:
    from evals.feature_dynamics.plot_cross_feature_semantic_manifold import DOMAIN_SHORT

    source = root / "feature_dynamics/cross_concept_manifold_figure.json"
    domain_data = read_json(source)["manifold_metrics"]
    geometry_source = root / "semantic_geometry/representation_geometry/representation_geometry.json"
    geometry = read_json(geometry_source)["methods"]
    domains = list(DOMAIN_SHORT)
    purity = np.array([[domain_data[m]["domain_neighbor_purity"][d] for d in domains] for m in METHODS])
    test = np.array([geometry[m]["geometry"]["neighbor_purity"] for m in METHODS])
    ood = np.array([geometry[m]["cross_time_neighbors"]["neighbor_purity"] for m in METHODS])
    np.testing.assert_allclose(purity.mean(axis=1), test, rtol=0, atol=1e-10)
    fig = plt.figure(figsize=(12.2, 4.65))
    grid = fig.add_gridspec(1, 2, width_ratios=(2.65, 1.0), left=0.13, right=0.98,
                           top=0.80, bottom=0.27, wspace=0.20)
    ax = fig.add_subplot(grid[0])
    cmap = LinearSegmentedColormap.from_list("paper_purity", ["#f3f8fb", METHOD_COLORS["token"], METHOD_COLORS["temporal"], BLUE])
    ax.imshow(purity * 100, cmap=cmap, vmin=35, vmax=90, aspect="auto")
    ax.set_title("A  Within-domain neighbor purity (%)", loc="left", color=BLUE, pad=16)
    ax.set_xticks(np.arange(8), list(DOMAIN_SHORT.values()), fontsize=12.5)
    ax.set_yticks(np.arange(4), [LABELS[m] for m in METHODS])
    ax.tick_params(length=0, pad=9)
    for i in range(4):
        for j in range(8):
            ax.text(j, i, f"{purity[i, j] * 100:.1f}", ha="center", va="center",
                    fontsize=14, color="white" if purity[i, j] >= 0.715 else DARK)
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    right = fig.add_subplot(grid[1], sharey=ax)
    clean_axis(right, "B  Across the year split")
    for i, method in enumerate(METHODS):
        color = METHOD_COLORS[method]
        right.plot([ood[i] * 100, test[i] * 100], [i, i], color=color, lw=3)
        right.plot(test[i] * 100, i, "o", color=color, ms=10, markeredgecolor=DARK,
                   markeredgewidth=0.6)
        right.plot(ood[i] * 100, i, "D", color=color, ms=8, markerfacecolor="white",
                   markeredgewidth=2)
    right.tick_params(axis="y", left=False, labelleft=False)
    right.set_xlim(60, 72)
    right.set_xticks([60, 64, 68, 72])
    right.set_xlabel("Neighbor purity (%)", labelpad=9)
    right.spines["left"].set_visible(False)
    right.grid(axis="x", color=GRID, linewidth=0.8)
    handles = [Line2D([], [], marker="o", color=DARK, linestyle="none", markersize=8,
                      label="Test within test"),
               Line2D([], [], marker="D", color=DARK, markerfacecolor="white",
                      linestyle="none", markersize=7, label="OOD querying test")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, 0.005),
               ncol=2, fontsize=13, columnspacing=2)
    info = save_figure(fig, output, "semantic_geometry", dpi)
    info.update(sources=[str(source), str(geometry_source)],
                protocol={"space": "50-D SVD fitted on test+OOD without labels",
                          "neighbors": "cached test audit uses non-self cosine neighbor ranks 2-11; OOD uses nearest 10 test codes",
                          "cross_time_metric": "neighbor label purity, distinct from majority-vote accuracy",
                          "sample_counts": "2048 test, 2048 OOD; 256 documents per class"},
                plotted_data={"methods": list(METHODS), "domains": domains,
                              "test_purity_by_domain": purity.tolist(),
                              "test_purity": test.tolist(), "cross_time_purity": ood.tolist()})
    return info


def label_efficiency(root: Path, output: Path, dpi: int) -> dict:
    source = root / "label_efficiency/linear_probe_results.json"
    data = read_json(source)
    budgets = data["high_level_transfer"]["low_label_budgets"]
    values = {}
    for method in MAIN_METHODS:
        representation = data["high_level_transfer"]["methods"][method]["representation"]
        rows = data["representations"][representation]["low_label"]
        values[method] = np.array([rows[str(b)]["seeds"] for b in budgets])
        for b, mean in zip(budgets, values[method].mean(axis=1)):
            np.testing.assert_allclose(mean, data["high_level_transfer"]["methods"][method][f"low_label_{b}"])
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.15, wspace=0.23)
    method_legend(fig, MAIN_METHODS, columns=5)
    for ax, title in zip(axs, ("A  Accuracy across label budgets", "B  Paired gain over BatchTopK")):
        clean_axis(ax, title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("Labeled documents per class")
        ax.set_xlim(0.85, 310)
    for method in MAIN_METHODS:
        color = METHOD_COLORS[method]
        for index, ax in enumerate(axs):
            if index == 1 and method == "token":
                continue
            runs = values[method] if index == 0 else values[method] - values["token"]
            mean = runs.mean(axis=1) * 100
            sem = runs.std(axis=1, ddof=1) / np.sqrt(runs.shape[1]) * 100
            ax.fill_between(budgets, mean - sem, mean + sem, color=color, alpha=0.12,
                            linewidth=0)
            ax.plot(budgets, mean, color=color, linewidth=2.5,
                    marker=METHOD_MARKERS[method], markersize=7,
                    markeredgecolor="white", markeredgewidth=0.6)
    axs[0].set_ylabel("Test accuracy (%)")
    axs[0].set_ylim(41, 87)
    axs[0].set_yticks([45, 55, 65, 75, 85])
    axs[1].set_ylabel("Accuracy gain (percentage points)")
    axs[1].axhline(0, color=METHOD_COLORS["token"], linewidth=1.5, linestyle="--")
    axs[1].set_ylim(-3.6, 6.3)
    axs[1].set_yticks([-2, 0, 2, 4, 6])
    info = save_figure(fig, output, "label_efficiency", dpi)
    info.update(sources=[str(source)], methods=list(MAIN_METHODS),
                protocol={"budgets": budgets, "seeds": 5, "bands": "plus/minus one SEM",
                          "difference": "within-seed paired difference from BatchTopK before mean/SEM"},
                plotted_data={m: values[m].tolist() for m in MAIN_METHODS})
    return info


def joint_alpha_evaluation_summary(root: Path, output: Path, dpi: int) -> dict:
    from evals.visualization.plot_joint_alpha_summary_radar import render

    paths = render(root, output, "joint_alpha_evaluation_summary", dpi)
    return {"files": [p.name for p in paths],
            "sources": [str(root / p) for p in (
                "rfve/training_fidelity.json", "rfve/reconstruction_cosine.json",
                "autointerp/autointerp_exact1000/results/summary.json",
                "semantic_invariance/results.json", "dictionary_utilization/dictionary_utilization.json",
                "temporal_robustness/temporal_robustness.json", "label_efficiency/linear_probe_results.json",
                "reasoning/results_summary.json", "steering/steering_summary.json",
                "document_linking/document_linking_results.json")],
            "protocol": "same 6+6 spokes, ranges, palette and renderer as the existing appendix alpha radar"}


def reasoning_matched_exposure(root: Path, output: Path, dpi: int) -> dict:
    folder = root / "reasoning"
    source = folder / "matched_exposure_metrics.csv"
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(r["view"], r["method"], r["category"]): r for r in rows}
    views = {
        "native": ("native_recall.csv", "validation_native", "A  Native recall"),
        "cue_free": ("cue_free_recall.csv", "validation_cue_free", "B  Cue-free recall"),
        "cue_only": ("cue_only_false_activation.csv", "validation_cue_only_negative", "C  Cue-only rejection"),
    }
    original = {}
    for view, (filename, _, _) in views.items():
        with (folder / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                original[(view, row["method"], row["category"])] = row
    native = load_arrays(folder / "frozen_feature_activations.npz")
    matched = load_arrays(folder / "matched_exposure_scores.npz")
    with (folder / "row_manifest.jsonl").open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    record_by_id = {int(r["row_id"]): r for r in records}
    assert len(record_by_id) == len(records)
    records = [record_by_id[int(row_id)] for row_id in native["row_ids"]]
    # Align by document identity and split, never by the two archives' row order.
    matched_lookup = {(str(h), str(s)): i for i, (h, s) in enumerate(
        zip(matched["row_order"], matched["row_splits"])
    )}
    assert len(matched_lookup) == len(matched["row_order"])
    matched_rows = np.array([
        matched_lookup[(r["document_hash"], r["split"])] for r in records
    ])
    for key in ("methods", "categories", "feature_ids"):
        np.testing.assert_array_equal(native[key], matched[key])
    before = native["calibrated_active"].astype(np.int8)
    after = (matched["document_scores"][matched_rows] > matched["calibration_cutoffs"]).astype(np.int8)
    columns = {(str(m), str(c)): i for i, (m, c) in enumerate(
        zip(native["methods"], native["categories"])
    )}
    for (method, category), col in columns.items():
        if method in ("mean", "cross"):
            subset = [j for j, r in enumerate(records)
                      if r["category"] == category and r["split"] in {v[1] for v in views.values()}]
            np.testing.assert_array_equal(before[subset, col], after[subset, col])
    categories = ("causal_mechanism", "planning", "backtracking", "conditional_assumption", "induction")
    names = ("Causal\nmechanism", "Planning", "Backtracking", "Conditional\nassumption", "Induction")
    methods = ("token", "temporal")
    bootstrap_seed, bootstrap_samples = 20260916, 20000
    rng = np.random.default_rng(bootstrap_seed)
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.8), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.81, bottom=0.16, wspace=0.20)
    method_legend(fig, methods, columns=2)
    plotted = {}
    axis_specs = (((-45, 15), [-40, -20, 0, 10]),
                  ((-13, 13), [-10, -5, 0, 5, 10]),
                  ((-9, 39), [0, 10, 20, 30]))
    for ax, (view, (_, split, title)), (limits, ticks) in zip(axs, views.items(), axis_specs):
        clean_axis(ax, title)
        plotted[view] = {}
        for i, method in enumerate(methods):
            details = []
            for category in categories:
                col = columns[(method, category)]
                subset = [j for j, r in enumerate(records)
                          if r["split"] == split and r["category"] == category]
                old_row, new_row = original[(view, method, category)], indexed[(view, method, category)]
                assert len(subset) == int(old_row["n"]) == int(new_row["n"]) == 100
                assert int(old_row["feature_id"]) == int(new_row["feature_id"]) == native["feature_ids"][col]
                old, new = before[subset, col], after[subset, col]
                assert int(old.sum()) == int(old_row["active_count"])
                assert int(new.sum()) == int(new_row["active_count"])
                np.testing.assert_allclose([old.mean(), new.mean()],
                                           [float(old_row["rate"]), float(new_row["rate"])])
                if view == "cue_only":
                    old, new = 1 - old, 1 - new
                difference = new - old
                # Resample document pairs; separate Wilson bounds are not intervals for a difference.
                samples = rng.choice(difference, size=(bootstrap_samples, len(subset))).mean(axis=1) * 100
                low, high = np.quantile(samples, [0.025, 0.975])
                details.append({"category": category, "n": len(subset),
                                "native_percent": float(old.mean() * 100),
                                "matched_percent": float(new.mean() * 100),
                                "delta_pp": float(difference.mean() * 100),
                                "paired_bootstrap_95_low": float(low),
                                "paired_bootstrap_95_high": float(high),
                                "improved_count": int((difference > 0).sum()),
                                "worsened_count": int((difference < 0).sum())})
            delta = np.array([r["delta_pp"] for r in details])
            low = np.array([r["paired_bootstrap_95_low"] for r in details])
            high = np.array([r["paired_bootstrap_95_high"] for r in details])
            y = np.arange(5) + (i - 0.5) * 0.30
            errors = np.array([delta - low, high - delta])
            assert np.min(errors) >= -1e-12
            assert np.all(low > limits[0]) and np.all(high < limits[1])
            ax.errorbar(delta, y, xerr=np.maximum(errors, 0),
                        fmt=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                        elinewidth=2.4, capsize=3, markersize=7,
                        markeredgecolor=DARK, markeredgewidth=0.55, zorder=3)
            plotted[view][method] = details
        ax.axvline(0, color=DARK, linestyle="--", linewidth=1.1)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_xlabel("Change (percentage points)", fontsize=12.5)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_yticks(np.arange(5), names)
        ax.set_ylim(4.5, -0.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    info = save_figure(fig, output, "reasoning_matched_exposure", dpi)
    info.update(sources=[str(source), *[str(folder / filename) for filename, _, _ in views.values()],
                         *[str(folder / filename) for filename in (
                             "frozen_feature_activations.npz", "matched_exposure_scores.npz",
                             "row_manifest.jsonl", "matched_exposure_manifest.json")]],
                methods=list(methods),
                protocol={"categories": list(categories), "n_per_category_view": 100,
                          "statistic": "matched minus native-resolution rate, in percentage points",
                          "cue_only": "rejection; negate the false-activation difference",
                          "interval": "paired-document percentile bootstrap 95%, conditional on frozen IDs and cutoffs",
                          "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
                          "row_alignment": "native row ID to manifest, then document hash and split to matched cache",
                          "omitted_methods": "Mean-Chunk and Cross-Chunk: all document decisions in the five target-relation panels unchanged",
                          "x_axis": "panel-specific ranges; dashed zero is unchanged rate",
                          "scores": "frozen feature IDs, shared 128-token chunks, recalibrated cutoff"},
                plotted_data=plotted)
    return info


def steering_strength_curve(root: Path, output: Path, dpi: int) -> dict:
    source = root / "steering/steering_summary.json"
    data = read_json(source)
    base = data["protocol"]["base_steering_strengths"]
    rescue = data["protocol"]["rescue_strengths"]
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 5.15))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.75, bottom=0.14, wspace=0.27)
    method_legend(fig, ALL_METHODS)
    plotted = {}
    for ax, title in zip(axs, ("A  Full feature panel", "B  Adaptive rescue", "C  Rescue sample sizes")):
        clean_axis(ax, title)
        ax.set_xlabel("Steering strength")
    for method in ALL_METHODS:
        rows = data["methods"][method]["per_strength"]
        means = [[rows[f"{s:g}"]["mean"] for s in grid] for grid in (base, rescue)]
        counts = [rows[f"{s:g}"]["n"] for s in rescue]
        assert all(rows[f"{s:g}"]["n"] == 100 for s in base)
        assert all(a >= b for a, b in zip(counts[:-1], counts[1:]))
        for index, (grid, values) in enumerate(((base, means[0]), (rescue, means[1]), (rescue, counts))):
            axs[index].plot(grid, values, color=METHOD_COLORS[method],
                            marker=METHOD_MARKERS[method], markersize=6,
                            markeredgecolor="white", markeredgewidth=0.5,
                            linewidth=2.1, linestyle="-" if index == 0 else "--")
        plotted[method] = {"base_mean": means[0], "rescue_mean": means[1], "rescue_n": counts}
    for ax, grid in zip(axs, (base, rescue, rescue)):
        ax.set_xticks(grid, [f"{s:g}" for s in grid])
        ax.margins(x=0.05)
    for ax in axs[:2]:
        ax.set_ylim(48.5, 70)
        ax.set_yticks([50, 55, 60, 65, 70])
        ax.axhline(50, color=DARK, linestyle=":", linewidth=1.1)
    axs[0].set_ylabel("Mean steering score")
    axs[1].tick_params(labelleft=False)
    axs[2].set_ylabel("Features evaluated")
    axs[2].set_ylim(0, 58)
    axs[2].set_yticks([0, 15, 30, 45])
    totals = [sum(plotted[m]["rescue_n"][i] for m in ALL_METHODS) for i in range(len(rescue))]
    expected = data["protocol"]["adaptive_rescue"]["attempts_by_strength"]
    assert totals == [expected[f"{s:g}"] for s in rescue]
    info = save_figure(fig, output, "steering_strength_curve", dpi)
    info.update(sources=[str(source)],
                protocol={"base_strengths": base, "rescue_strengths": rescue,
                          "base_n_per_method": 100, "rescue_total_n": totals,
                          "rescue_selection": "unchanged at every earlier strength; stop at first changed continuation",
                          "no_effect_score": 50, "y_axis": "score axis zooms to 48.5-70"},
                plotted_data=plotted)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = args.eval_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "appendix_figures_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"figures": {}}
    manifest.update(format="chunk-saes-appendix-figures-v1", eval_root=str(root),
                    method_colors={m: METHOD_COLORS[m] for m in ALL_METHODS}, dpi=args.dpi,
                    generator=str(Path(__file__).resolve()),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name in FIGURES:
        if name not in args.figures:
            continue
        style()
        print(f"Rendering {name} ...", flush=True)
        info = globals()[name](root, output, args.dpi)
        info["source_sha256"] = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in info["sources"]
        }
        manifest["figures"][name] = info
        manifest["complete"] = all(name in manifest["figures"] for name in FIGURES)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        print(f"Saved {name}.pdf and {name}.png", flush=True)


if __name__ == "__main__":
    main()
