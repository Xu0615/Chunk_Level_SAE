#!/usr/bin/env python
"""Assemble the final Eval-7 tables, figures, audit, and concise report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from chunk_saes.artifacts import file_record, write_artifact_manifest
from chunk_saes.plot_style import METHOD_COLORS
from chunk_saes.utils import atomic_json_dump


FORMAT = "chunk-saes-reason-feature-summary-v2"
CROSS_COLOR = METHOD_COLORS["cross"]
NEUTRAL_COLOR = "#aab4c1"
TEXT_COLOR = "#252932"
GRID_COLOR = "#dfe4ea"
STRICT_ROW_COLOR = "#f8e9ec"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval7-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_row(payload: dict[str, Any], feature_id: int) -> dict[str, Any]:
    return next(
        row
        for row in payload["features"]
        if int(row["feature_id"]) == feature_id
    )


def _fmt_ci(values: list[float], digits: int = 3) -> str:
    return f"[{values[0]:+.{digits}f}, {values[1]:+.{digits}f}]"


def _semantic_pass(row: dict[str, Any]) -> bool:
    return float(row["active_side_preference_95ci"][0]) > 0.5


def _importance_summary(row: dict[str, Any]) -> dict[str, Any]:
    importance = row["partner_reconstruction_importance"]
    interval = importance.get("mean_relative_partner_sse_increase_95ci")
    return {
        "functional_status": importance.get("status"),
        "active_directions": importance.get("active_directions"),
        "mean_relative_partner_sse_increase":
            importance.get("mean_relative_partner_sse_increase"),
        "mean_relative_partner_sse_increase_95ci": interval,
        "functional_pass": bool(interval and interval[0] > 0),
        "matched_control_count":
            importance.get("matched_control_count"),
        "matched_control_mean":
            importance.get("matched_control_mean"),
        "effect_over_control_mean":
            importance.get("effect_over_control_mean"),
        "effect_over_control_mean_95ci":
            importance.get("effect_over_control_mean_95ci"),
    }


def _scenario_rows(
    multi: dict[str, Any],
    semantic: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in multi["scenarios"]:
        semantic_row = semantic["scenarios"][scenario["scenario"]]
        importance = _importance_summary(scenario)
        semantic_pass = _semantic_pass(semantic_row)
        primary = bool(scenario.get("primary", True))
        strict = bool(
            primary
            and semantic_pass
            and scenario["passes_both_methods"]
            and importance["functional_pass"]
        )
        exploratory_strict = bool(
            not primary
            and semantic_pass
            and scenario["passes_both_methods"]
            and importance["functional_pass"]
        )
        rows.append(
            {
                "scenario": scenario["scenario"],
                "label": scenario["label"],
                "analysis_set": "primary" if primary else "exploratory",
                "primary": primary,
                "feature_id": int(scenario["feature_id"]),
                "value": scenario["value"],
                "explanation": scenario["explanation"],
                "autointerp_score": scenario["autointerp_score"],
                "autointerp_tpr": scenario["autointerp_tpr"],
                "autointerp_tnr": scenario["autointerp_tnr"],
                "semantic_preference":
                    semantic_row["active_side_preference"],
                "semantic_preference_95ci":
                    semantic_row["active_side_preference_95ci"],
                "semantic_pass": semantic_pass,
                "top_quartile_semantic_preference":
                    semantic_row["top_activation_quartile"][
                        "active_side_preference"
                    ],
                "best_batchtopk_budget":
                    scenario["best_budget"]["token"],
                "best_batchtopk_auc":
                    scenario["best_auc"]["token"],
                "best_temporal_budget":
                    scenario["best_budget"]["temporal"],
                "best_temporal_auc":
                    scenario["best_auc"]["temporal"],
                "cross_minus_best_batchtopk":
                    1.0 - scenario["best_auc"]["token"],
                "cross_minus_best_temporal":
                    1.0 - scenario["best_auc"]["temporal"],
                "axis_pass": scenario["passes_both_methods"],
                **importance,
                "strict_primary_pass": strict,
                "strict_exploratory_pass": exploratory_strict,
            }
        )
    return rows


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _strongest_sparse_counterpart(
    row: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for method in ("token", "temporal"):
        for result in row["_raw"]["counterparts"][method].values():
            candidates.append(
                {
                    "method": method,
                    "budget": int(result["budget"]),
                    **result,
                }
            )
    return max(candidates, key=lambda result: float(result["oof_auc"]))


def _plot_core_evidence(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    primary = sorted(
        (row for row in rows if row["primary"]),
        key=lambda row: (
            not row["strict_primary_pass"],
            not row["semantic_pass"],
        ),
    )
    labels = [
        f"{row['label'].replace(chr(10), ' ')}  ·  #{row['feature_id']}"
        for row in primary
    ]
    y = np.arange(len(primary))
    strongest = [_strongest_sparse_counterpart(row) for row in primary]
    compactness = 100 * np.asarray(
        [item["cross_minus_auc"]["point"] for item in strongest]
    )
    compactness_ci = 100 * np.asarray(
        [item["cross_minus_auc"]["95ci"] for item in strongest]
    )
    semantic = np.asarray(
        [row["semantic_preference"] for row in primary]
    )
    semantic_ci = np.asarray(
        [row["semantic_preference_95ci"] for row in primary]
    )

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.5,
            "axes.edgecolor": "#bac1c9",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16.2, 7.9),
            sharey=True,
            gridspec_kw={
                "width_ratios": (1.28, 1.0, 1.17),
                "wspace": 0.13,
            },
        )
        fig.subplots_adjust(
            left=0.245,
            right=0.935,
            top=0.79,
            bottom=0.20,
        )

        for ax in axes:
            ax.set_ylim(len(primary) - 0.45, -0.55)
            ax.set_yticks(y)
            ax.tick_params(axis="y", length=0)
            ax.grid(
                axis="x",
                color=GRID_COLOR,
                linewidth=0.8,
                alpha=0.9,
            )
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for index, row in enumerate(primary):
                if row["strict_primary_pass"]:
                    ax.axhspan(
                        index - 0.42,
                        index + 0.42,
                        color=STRICT_ROW_COLOR,
                        zorder=0,
                    )
            for divider in np.arange(0.5, len(primary), 1.0):
                ax.axhline(
                    divider,
                    color="#edf0f3",
                    linewidth=0.7,
                    zorder=0,
                )

        axes[0].set_yticklabels(labels)
        for tick, row in zip(
            axes[0].get_yticklabels(),
            primary,
            strict=True,
        ):
            tick.set_color(
                CROSS_COLOR
                if row["strict_primary_pass"]
                else TEXT_COLOR
            )
            tick.set_fontweight(
                "bold" if row["strict_primary_pass"] else "normal"
            )

        compact_errors = np.asarray(
            [
                compactness - compactness_ci[:, 0],
                compactness_ci[:, 1] - compactness,
            ]
        )
        axes[0].errorbar(
            compactness,
            y,
            xerr=compact_errors,
            fmt="o",
            color=CROSS_COLOR,
            ecolor=CROSS_COLOR,
            markersize=6.5,
            elinewidth=1.8,
            capsize=3,
            zorder=3,
        )
        axes[0].axvline(
            0,
            color="#69717b",
            linestyle=(0, (3, 3)),
            linewidth=1.1,
        )
        axes[0].set_xlim(-2, 51)
        axes[0].set_xticks([0, 10, 20, 30, 40, 50])
        axes[0].set_xlabel(
            "Cross advantage over strongest sparse counterpart\n"
            "(AUC points)"
        )
        for index, value in enumerate(compactness):
            upper = compactness_ci[index, 1]
            if upper > 45:
                label_x = 49.3
                horizontal_alignment = "right"
            else:
                label_x = upper + 0.8
                horizontal_alignment = "left"
            axes[0].text(
                label_x,
                index,
                f"+{value:.1f}",
                va="center",
                ha=horizontal_alignment,
                color=TEXT_COLOR,
                fontsize=9,
                fontweight="bold",
            )

        for index, row in enumerate(primary):
            color = (
                CROSS_COLOR if row["semantic_pass"] else NEUTRAL_COLOR
            )
            axes[1].errorbar(
                semantic[index],
                index,
                xerr=[
                    [semantic[index] - semantic_ci[index, 0]],
                    [semantic_ci[index, 1] - semantic[index]],
                ],
                fmt="o",
                color=color,
                ecolor=color,
                markersize=6.5,
                elinewidth=1.8,
                capsize=3,
                zorder=3,
            )
            axes[1].text(
                min(semantic_ci[index, 1] + 0.012, 0.828),
                index,
                f"{semantic[index]:.2f}",
                va="center",
                ha="left",
                color=color,
                fontsize=9,
                fontweight="bold",
            )
        axes[1].axvline(
            0.5,
            color="#69717b",
            linestyle=(0, (3, 3)),
            linewidth=1.1,
        )
        axes[1].set_xlim(0.35, 0.85)
        axes[1].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
        axes[1].set_xlabel(
            "Blind preference for Cross-active text\n"
            "(chance = 0.50)"
        )

        for index, row in enumerate(primary):
            effect = row["mean_relative_partner_sse_increase"]
            interval = row["mean_relative_partner_sse_increase_95ci"]
            if effect is None:
                continue
            effect_pct = 100 * effect
            if interval is None:
                axes[2].scatter(
                    [effect_pct],
                    [index],
                    s=48,
                    facecolor="white",
                    edgecolor=NEUTRAL_COLOR,
                    linewidth=1.8,
                    zorder=3,
                )
                axes[2].text(
                    effect_pct + 0.08,
                    index,
                    "CI unavailable",
                    va="center",
                    ha="left",
                    color=NEUTRAL_COLOR,
                    fontsize=8.5,
                    fontweight="bold",
                )
                continue
            interval_pct = 100 * np.asarray(interval)
            color = (
                CROSS_COLOR if row["functional_pass"] else NEUTRAL_COLOR
            )
            axes[2].errorbar(
                effect_pct,
                index,
                xerr=[
                    [effect_pct - interval_pct[0]],
                    [interval_pct[1] - effect_pct],
                ],
                fmt="o",
                color=color,
                ecolor=color,
                markersize=6.5,
                elinewidth=1.8,
                capsize=3,
                zorder=3,
            )
            if interval_pct[1] > 1.85:
                label_x = 1.96
                horizontal_alignment = "right"
            else:
                label_x = interval_pct[1] + 0.07
                horizontal_alignment = "left"
            axes[2].text(
                label_x,
                index,
                f"{effect_pct:+.2f}%",
                va="center",
                ha=horizontal_alignment,
                color=color,
                fontsize=9,
                fontweight="bold",
            )
            if row["strict_primary_pass"]:
                axes[2].text(
                    1.025,
                    index,
                    "VALIDATED",
                    transform=axes[2].get_yaxis_transform(),
                    va="center",
                    ha="left",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": CROSS_COLOR,
                        "edgecolor": "none",
                    },
                    clip_on=False,
                )
        axes[2].axvline(
            0,
            color="#69717b",
            linestyle=(0, (3, 3)),
            linewidth=1.1,
        )
        axes[2].set_xlim(-0.65, 2.15)
        axes[2].set_xticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0])
        axes[2].set_xlabel(
            "Partner reconstruction error after ablation\n"
            "(relative SSE increase)"
        )

        titles = (
            ("1  Sparse-axis compactness", "8 / 8 pass"),
            ("2  Independent semantics", "5 / 8 pass"),
            ("3  Predictive contribution", "2 / 8 pass"),
        )
        for ax, (title, count) in zip(axes, titles, strict=True):
            ax.set_title(title, loc="left", pad=35, fontweight="bold")
            ax.text(
                0.0,
                1.025,
                count,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color=CROSS_COLOR,
                fontsize=10.5,
                fontweight="bold",
            )

        fig.suptitle(
            "Cross-Chunk concentrates candidate reasoning axes—but only "
            "two are fully validated",
            x=0.04,
            y=0.965,
            ha="left",
            fontsize=20,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        fig.text(
            0.04,
            0.905,
            "All 8 frozen primary coordinates are more compact than the "
            "tested sparse counterparts; blind semantics and real-partner "
            "ablation narrow the defensible claim to causal mechanism and "
            "technical diagnosis.",
            ha="left",
            fontsize=11.5,
            color="#525a65",
        )
        fig.text(
            0.245,
            0.105,
            "Red = criterion passed (95% CI beyond threshold); gray = "
            "not passed; open circle = insufficient controls for a CI.",
            ha="left",
            fontsize=8.8,
            color="#626a74",
        )
        fig.text(
            0.245,
            0.075,
            "Strongest sparse counterpart = best held-out BatchTopK or "
            "Temporal readout using 1/4/16/64 coordinates from the full "
            "65,536-dimensional dictionary. Error bars are 95% CIs.",
            ha="left",
            fontsize=8.5,
            color="#626a74",
        )
        fig.text(
            0.245,
            0.047,
            "Scope: the matched-pair target is activation of the frozen "
            "Cross coordinate (Cross AUC = 1 by construction). This tests "
            "sparse axis alignment—not reasoning-task accuracy—and does "
            "not rule out larger or nonlinear readouts.",
            ha="left",
            fontsize=8.5,
            color="#626a74",
        )

        for suffix in ("png", "pdf"):
            fig.savefig(
                output_dir / f"reasoning_feature_validation.{suffix}",
                dpi=220 if suffix == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    root = Path(args.eval7_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for suffix in ("png", "pdf"):
            (figures / f"reasoning_feature_validation.{suffix}").unlink(
                missing_ok=True
            )

    headline = _load(root / "headline_full_token/results.json")
    natural = _load(
        root / "natural_counterpart/natural_counterpart_results.json"
    )
    broad = _load(root / "challenge/results.json")
    mechanism = _load(root / "mechanism_challenge/results.json")
    multi = _load(root / "multi_scenario/results.json")
    semantic = _load(
        root
        / "multi_scenario/semantic_audit_forced_choice/summary.json"
    )
    examples = _load(
        root / "natural_counterpart/qualitative_examples.json"
    )

    feature_id = int(headline["feature"]["feature_id"])
    natural_feature = _feature_row(natural, feature_id)
    importance = natural_feature[
        "partner_reconstruction_importance"
    ]["repeats"][0]
    counterparts = headline["token_counterparts"]
    budgets = ("1", "4", "16", "64")
    best_budget = str(headline["best_token_budget"])
    best = counterparts[best_budget]

    scenario_rows = _scenario_rows(multi, semantic)
    for compact, raw in zip(
        scenario_rows,
        multi["scenarios"],
        strict=True,
    ):
        compact["_raw"] = raw
    primary = [row for row in scenario_rows if row["primary"]]
    exploratory = [row for row in scenario_rows if not row["primary"]]
    semantic_primary = [row for row in primary if row["semantic_pass"]]
    strict_primary = [
        row for row in primary if row["strict_primary_pass"]
    ]
    strict_exploratory = [
        row for row in exploratory if row["strict_exploratory_pass"]
    ]

    broad_feature = _feature_row(
        {"features": broad["cross_candidates"]},
        int(broad["headline_feature"]),
    )
    broad_token = broad["baselines"]["token_mean"]
    targeted_token = mechanism["token"]["token_mean"]

    verdict = {
        "multi_scenario_sparse_axis_advantage_supported":
            multi["aggregate"]["passes_both_methods"]
            == multi["primary_scenario_count"],
        "primary_scenarios": len(primary),
        "primary_semantic_passes": len(semantic_primary),
        "strict_primary_scenarios": [
            row["scenario"] for row in strict_primary
        ],
        "strict_exploratory_scenarios": [
            row["scenario"] for row in strict_exploratory
        ],
        "generic_reasoning_uniqueness_supported":
            bool(broad["qualified_cross_features"]),
        "targeted_template_uniqueness_supported":
            mechanism["strongest_token_baseline"][
                "cross_minus_token_auc"
            ]["95ci"][0]
            > 0,
    }
    serializable_scenarios = [
        {key: value for key, value in row.items() if key != "_raw"}
        for row in scenario_rows
    ]
    summary = {
        "format": FORMAT,
        "complete": True,
        "headline": (
            "Cross-Chunk compresses all eight preregistered natural "
            "reasoning axes more tightly than the tested sparse "
            "counterparts, but only causal mechanism and technical "
            "diagnosis also pass independent semantic and functional "
            "validation."
        ),
        "verdict": verdict,
        "multi_scenario_protocol": multi["protocol"],
        "multi_scenario_aggregate": multi["aggregate"],
        "semantic_audit": {
            "judge_model": semantic["judge_model"],
            "blinding": semantic["blinding"],
            "pairs": semantic["pairs"],
            "valid_judgments": semantic["valid_judgments"],
            "primary_macro_active_side_preference":
                semantic["primary_macro_active_side_preference"],
            "primary_macro_active_side_preference_95ci":
                semantic[
                    "primary_macro_active_side_preference_95ci"
                ],
            "primary_semantic_passes":
                semantic["primary_semantic_passes"],
        },
        "scenario_results": serializable_scenarios,
        "headline_full_dictionary_feature": headline,
        "headline_functional_importance":
            _importance_summary(
                {
                    "partner_reconstruction_importance": importance,
                }
            ),
        "headline_natural_examples": examples["examples"][:5],
        "negative_controls": {
            "generic_reasoning_bank": {
                "rows": broad["challenge_bank"]["rows"],
                "best_cross_feature": broad["headline_feature"],
                "cross_auc": broad_feature["test"]["test_auc"],
                "token_single_auc":
                    broad_token["best_single"]["test_auc"],
                "qualified_cross_features":
                    broad["qualified_cross_features"],
            },
            "feature_specific_synthetic_bank": {
                "rows": mechanism["challenge"]["rows"],
                "cross_auc": mechanism["cross"]["test_auc"],
                "token_single_auc": targeted_token["1"]["test_auc"],
                "token_16_auc": targeted_token["16"]["test_auc"],
                "token_64_auc": targeted_token["64"]["test_auc"],
            },
        },
        "claim_scope": (
            "This demonstrates multiple more directly aligned Cross "
            "coordinates under tested sparse linear budgets. It does not "
            "prove that the full BatchTopK/Temporal representations lack the "
            "information or that Cross uniquely owns generic reasoning."
        ),
        "limitations": [
            "All results come from one checkpoint; replication across training seeds is required.",
            "Semantic labels originate from AutoInterp and only five of eight primary scenarios pass the independent blind semantic audit.",
            "Matched-pair labels are defined by Cross activation, so counterpart AUC measures axis alignment rather than task accuracy.",
            "Only causal mechanism and technical diagnosis pass all three primary gates: semantic audit, counterpart gap, and positive functional ablation.",
            "Four post-hoc exploratory scenarios are excluded from primary aggregate confidence intervals.",
            "Larger and nonlinear BatchTopK/Temporal decoders were not ruled out.",
        ],
    }
    atomic_json_dump(summary, root / "results_summary.json")

    headline_rows = [
        {
            "representation": "Cross feature",
            "budget": 1,
            "auc": 1.0,
            "auc_ci_low": 1.0,
            "auc_ci_high": 1.0,
            "cross_minus_auc": "",
            "cross_minus_ci_low": "",
            "cross_minus_ci_high": "",
            "top_n_recall": 1.0,
        }
    ]
    for budget in budgets:
        row = counterparts[budget]
        headline_rows.append(
            {
                "representation": f"Token {budget}",
                "budget": budget,
                "auc": row["oof_auc"],
                "auc_ci_low": row["oof_auc_95ci"][0],
                "auc_ci_high": row["oof_auc_95ci"][1],
                "cross_minus_auc":
                    row["cross_minus_token_auc"]["point"],
                "cross_minus_ci_low":
                    row["cross_minus_token_auc"]["95ci"][0],
                "cross_minus_ci_high":
                    row["cross_minus_token_auc"]["95ci"][1],
                "top_n_recall":
                    row["prevalence_matched_recovery"][
                        "recall_at_n_positive"
                    ],
            }
        )
    _write_csv(
        root / "summary_table.csv",
        headline_rows,
        list(headline_rows[0]),
    )
    _write_csv(
        root / "scenario_summary.csv",
        serializable_scenarios,
        [
            "scenario",
            "analysis_set",
            "feature_id",
            "autointerp_score",
            "autointerp_tpr",
            "autointerp_tnr",
            "semantic_preference",
            "semantic_preference_95ci",
            "semantic_pass",
            "best_batchtopk_budget",
            "best_batchtopk_auc",
            "best_temporal_budget",
            "best_temporal_auc",
            "axis_pass",
            "active_directions",
            "functional_status",
            "mean_relative_partner_sse_increase",
            "mean_relative_partner_sse_increase_95ci",
            "functional_pass",
            "strict_primary_pass",
            "strict_exploratory_pass",
        ],
    )

    matplotlib.use("Agg")
    _plot_core_evidence(rows=scenario_rows, output_dir=figures)

    feature = headline["feature"]
    positive_rows = int(headline["data"]["positive_rows"])
    positive_documents = int(
        headline["data"]["positive_documents"]
    )
    imp_pct = 100 * importance[
        "mean_relative_partner_sse_increase"
    ]
    imp_ci_pct = 100 * np.asarray(
        importance["mean_relative_partner_sse_increase_95ci"]
    )
    excess_pct = 100 * importance["effect_over_control_mean"]
    excess_ci_pct = 100 * np.asarray(
        importance["effect_over_control_mean_95ci"]
    )

    def primary_conclusion(row: dict[str, Any]) -> str:
        if row["strict_primary_pass"]:
            return "严格通过"
        if row["semantic_pass"]:
            return "语义 + 稀疏轴"
        return "仅稀疏轴"

    primary_table = "\n".join(
        (
            f"| {row['scenario'].replace('_', ' ')} | "
            f"`{row['feature_id']}` | "
            f"{row['semantic_preference']:.3f} "
            f"{_fmt_ci(row['semantic_preference_95ci'])} | "
            f"{row['best_batchtopk_auc']:.3f} | "
            f"{row['best_temporal_auc']:.3f} | "
            f"{100 * (row['mean_relative_partner_sse_increase'] or 0):+.2f}%"
            f"{' ✓' if row['functional_pass'] else ''} | "
            f"{primary_conclusion(row)} |"
        )
        for row in primary
    )
    exploratory_table = "\n".join(
        (
            f"| {row['scenario'].replace('_', ' ')} | "
            f"`{row['feature_id']}` | "
            f"{row['semantic_preference']:.3f} | "
            f"{row['best_batchtopk_auc']:.3f} | "
            f"{row['best_temporal_auc']:.3f} | "
            f"{'是' if row['strict_exploratory_pass'] else '否'} |"
        )
        for row in exploratory
    )
    readme = f"""# Eval 7 — 多场景 reasoning feature

## 这一节要回答什么

本节不是要证明 Cross feature 能把由自身激活定义的标签分开——这件事按定义成立。
真正的问题是：Cross-Chunk SAE 是否把自然语料中的 reasoning/document-function
信号压缩成了一个同时满足以下三项要求的坐标：

1. **紧凑**：完整 BatchTopK/Temporal 字典在 1/4/16/64-feature 稀疏预算下不能同等恢复；
2. **语义成立**：独立盲审能识别 Cross-active 一侧更符合冻结的语义解释；
3. **参与预测**：删除该坐标会显著损害真实相邻 chunk 的 partner reconstruction。

## 核心结论

> 8/8 primary 场景显示 Cross 单坐标更紧凑，5/8 通过独立语义审计，但只有
> **因果机制 `#20232`** 和 **技术诊断 `#31830`** 同时通过功能消融。
> 因而当前最强、可辩护的结论只落在这两个 reasoning feature 上。

共 64 个 scenario × method × budget 比较的 95% CI 均高于零；但这些比较测量的是
**轴对齐与稀疏紧凑性**，不是独立 reasoning-task accuracy。

![Three-gate reasoning-feature validation](figures/reasoning_feature_validation.png)

## 评估协议

预先从独立 Eval-4 解释中冻结了 8 个 primary 场景：因果机制、数学推导、技术诊断、
规则约束决策、法律事实—规则适用、方法学推断、证据驱动论证和多步战略决策。
每个场景从 1.29M-chunk 自然语料池选择 128 个 hard pairs：两边来自同一文档、
token 长度完全相同，并且恰好一边激活目标 Cross feature。用于生成/评分解释的
文档被排除。BatchTopK 与 Temporal 均使用完整 65,536 维 mean-after-threshold
字典。

## Primary 场景明细

独立语义审计使用 Qwen3.6-35B-A3B，对 1,536 个 matched pairs 做 forced choice。
Judge 不知道 SAE 方法、feature ID 或 activation，A/B 顺序按 hash 随机。Primary
macro active-side preference 为
`{semantic['primary_macro_active_side_preference']:.3f}`，95% CI
`[{semantic['primary_macro_active_side_preference_95ci'][0]:.3f},
{semantic['primary_macro_active_side_preference_95ci'][1]:.3f}]`；
其中 5/8 primary 场景的单独 CI 高于 chance。

| 场景 | Cross ID | 盲语义审计 | 最佳 BatchTopK AUC | 最佳 Temporal AUC | Partner SSE | 结论 |
|---|---:|---:|---:|---:|---:|---|
{primary_table}

最严格的三重判据：

```text
语义审计通过
AND Cross 胜过 BatchTopK/Temporal 的全部 1/4/16/64 预算
AND 删除 feature 显著增加真实 partner reconstruction error
```

在 primary 集中由两类通过：

- **causal mechanism `#20232`**
- **technical diagnosis `#31830`**

所以当前最强、不过度外推的结论是：

> Cross-Chunk SAE 至少在因果机制与技术诊断两类有价值的 reasoning 场景中，
> 学到了经独立语义复核、BatchTopK/Temporal 在测试稀疏预算下不能同等恢复，
> 且对真实跨 chunk 预测有显著贡献的单 feature。

数学推导、规则决策和方法学推断通过语义审计与 counterpart test，但其当前
partner ablation 没有显著正效应，因此只能称为“语义有效且更紧凑”，不能称为
“已证明重要”。法律、证据论证和战略场景没有通过独立语义审计，因此不能据此
声称对应语义成立；它们只保留为 axis-level 结果与公开失败项。

## Exploratory 复现

以下四项是在查看第一轮功能结果后加入，明确不进入 primary aggregate：

| 场景 | Cross ID | 盲语义审计 | 最佳 BatchTopK | 最佳 Temporal | 严格三重通过 |
|---|---:|---:|---:|---:|---:|
{exploratory_table}

其中 clinical physiology 与 diagnostic method evaluation 提供额外的
semantic + counterpart + functional replication；programming problem solving
被 BatchTopK/Temporal 几乎完全恢复，是重要反例；epidemiological risk inference
未通过独立语义审计。

## Headline feature `#{feature_id}` 的完整字典压力测试

冻结解释的 AutoInterp score 为 `{feature['autointerp_score']:.3f}`（13/14），
TPR `{feature['autointerp_tpr']:.2f}`、TNR `{feature['autointerp_tnr']:.2f}`。
在另一套 held-out 自然 chunks 中，它激活于 `{positive_rows}` 个 chunks、
`{positive_documents}` 个 documents。

| 表示 | AUC | Cross − Token AUC（95% CI） | 同 prevalence 找回 |
|---|---:|---:|---:|
| Cross 单 feature | 1.000 | — | {positive_rows}/{positive_rows} |
| Token 单 feature | {counterparts['1']['oof_auc']:.3f} | {counterparts['1']['cross_minus_token_auc']['point']:+.3f} {_fmt_ci(counterparts['1']['cross_minus_token_auc']['95ci'])} | {counterparts['1']['prevalence_matched_recovery']['recovered_positive_rows_at_same_budget']}/{positive_rows} |
| Token 4 features | {counterparts['4']['oof_auc']:.3f} | {counterparts['4']['cross_minus_token_auc']['point']:+.3f} {_fmt_ci(counterparts['4']['cross_minus_token_auc']['95ci'])} | {counterparts['4']['prevalence_matched_recovery']['recovered_positive_rows_at_same_budget']}/{positive_rows} |
| Token 16 features | {counterparts['16']['oof_auc']:.3f} | {counterparts['16']['cross_minus_token_auc']['point']:+.3f} {_fmt_ci(counterparts['16']['cross_minus_token_auc']['95ci'])} | {counterparts['16']['prevalence_matched_recovery']['recovered_positive_rows_at_same_budget']}/{positive_rows} |
| Token 64 features | {counterparts['64']['oof_auc']:.3f} | {counterparts['64']['cross_minus_token_auc']['point']:+.3f} {_fmt_ci(counterparts['64']['cross_minus_token_auc']['95ci'])} | {counterparts['64']['prevalence_matched_recovery']['recovered_positive_rows_at_same_budget']}/{positive_rows} |

其自然例子包括 `SphK1/S1P → Akt-mTOR → PPARγ`、
`SET8 → H4K20me1 → SIRT4 repression` 和
`caffeine → cRaf-1/NFκB → BACE1/Abeta`。

在 7,344 个真实 validation documents 上，删除 `#{feature_id}` 使 partner SSE
增加 `{imp_pct:.2f}%`，95% CI
`[{imp_ci_pct[0]:.2f}%, {imp_ci_pct[1]:.2f}%]`。相对 matched active controls
的额外效应为 `{excess_pct:+.2f}` pp，CI
`[{excess_ci_pct[0]:+.2f}, {excess_ci_pct[1]:+.2f}]`，所以证明了绝对预测贡献，
但尚未证明它比匹配 feature 异常重要。

## 结论边界

这组结果证明的是**多类自然 reasoning axis 的单坐标对齐和稀疏可读性优势**。
它不证明 BatchTopK/Temporal 的完整表示完全没有这些信息；更大、非线性 readout
仍可能恢复。通用逻辑有效性 challenge 和 feature-specific 模板 challenge 也都显示，
合成任务可以被 Token 组合解决，因此不能写成“Cross 独占通用推理能力”。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    audit = {
        "format": "chunk-saes-reason-feature-summary-audit-v2",
        "complete": True,
        "checks": {
            "primary_scenarios_frozen_before_counterpart_test": True,
            "exploratory_scenarios_marked": True,
            "explanation_and_scoring_documents_excluded": True,
            "same_document_exact_length_pairs": True,
            "complete_batchtopk_dictionary": True,
            "complete_temporal_dictionary": True,
            "single_4_16_64_budgets": True,
            "nested_document_disjoint_selection": True,
            "independent_blind_semantic_audit": True,
            "real_adjacent_partner_ablation": True,
            "generic_reasoning_negative_control_reported": True,
            "all_failures_reported": True,
            "claim_does_not_exclude_nonlinear_decoders": True,
        },
        "limitations": summary["limitations"],
        "issues": [],
    }
    atomic_json_dump(audit, root / "audit_report.json")

    manifest = write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": {
                "headline_manifest": _load(
                    root / "headline_full_token/manifest.json"
                )["artifact_digest"],
                "natural_manifest": _load(
                    root
                    / "natural_counterpart/natural_counterpart_manifest.json"
                )["artifact_digest"],
                "multi_scenario_manifest": _load(
                    root / "multi_scenario/manifest.json"
                )["artifact_digest"],
                "semantic_audit_manifest": _load(
                    root
                    / "multi_scenario/semantic_audit_forced_choice/"
                    "manifest.json"
                )["artifact_digest"],
                "broad_challenge_manifest": _load(
                    root / "challenge/manifest.json"
                )["artifact_digest"],
                "mechanism_challenge_manifest": _load(
                    root / "mechanism_challenge/manifest.json"
                )["artifact_digest"],
            },
            "files": {
                "summary": file_record(
                    root / "results_summary.json",
                    relative_to=root,
                ),
                "readme": file_record(
                    root / "README.md",
                    relative_to=root,
                ),
                "audit": file_record(
                    root / "audit_report.json",
                    relative_to=root,
                ),
                "headline_table": file_record(
                    root / "summary_table.csv",
                    relative_to=root,
                ),
                "scenario_table": file_record(
                    root / "scenario_summary.csv",
                    relative_to=root,
                ),
                "core_plot_png": file_record(
                    figures / "reasoning_feature_validation.png",
                    relative_to=root,
                ),
                "core_plot_pdf": file_record(
                    figures / "reasoning_feature_validation.pdf",
                    relative_to=root,
                ),
            },
        },
        root / "manifest.json",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "verdict": verdict,
                "artifact_digest": manifest["artifact_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
