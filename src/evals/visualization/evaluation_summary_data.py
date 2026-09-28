"""Shared data definitions for the layer-21 evaluation summary figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping

from chunk_saes.plot_style import (
    METHOD_COLORS as SHARED_METHOD_COLORS,
    METHOD_MARKERS as SHARED_METHOD_MARKERS,
)


METHODS = (
    "token",
    "temporal",
    "mean",
    "joint_alpha0p25",
    "cross",
)

METHOD_LABELS = {
    "token": "BatchTopK SAE",
    "temporal": "Temporal SAE",
    "mean": "Mean-Chunk SAE",
    "joint_alpha0p25": "Joint-Chunk SAE (α=0.25)",
    "cross": "Cross-Chunk SAE",
}

METHOD_SHORT_LABELS = {
    "token": "BatchTopK",
    "temporal": "Temporal",
    "mean": "Mean-Chunk",
    "joint_alpha0p25": "Joint\nalpha=0.25",
    "cross": "Cross-Chunk",
}

# Match the task-level camera-ready palette.
METHOD_COLORS = {
    method: SHARED_METHOD_COLORS[method] for method in METHODS
}

METHOD_MARKERS = {
    method: SHARED_METHOD_MARKERS[method] for method in METHODS
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    order: int
    group: str
    title: str
    subtitle: str
    axis_label: str
    radar_label: str
    source: str
    description: str
    display: str
    plot_low: float
    plot_high: float
    radar_low: float
    radar_high: float
    values: Mapping[str, float]
    confidence_intervals: Mapping[str, tuple[float, float]] | None = None
    neutral_value: float | None = None


def _load(root: Path, relative: str) -> dict:
    path = root / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _method_values(
    payload: Mapping[str, object],
    field: str,
) -> dict[str, float]:
    return {
        method: float(payload[method][field])  # type: ignore[index]
        for method in METHODS
    }


def load_metric_specs(eval_root: Path) -> tuple[MetricSpec, ...]:
    """Load the twelve selected metrics from their canonical task artifacts."""

    rfve = _load(eval_root, "rfve/training_fidelity.json")["methods"]
    cosine = _load(eval_root, "rfve/reconstruction_cosine.json")["methods"]
    context = _load(
        eval_root,
        "autointerp/autointerp_exact1000/results/summary.json",
    )["methods"]
    high_level_path = eval_root / "semantic_invariance/results.json"
    high_level_payload = (
        _load(eval_root, "semantic_invariance/results.json")
        if high_level_path.exists()
        else None
    )
    high_level_manifest_path = eval_root / "semantic_invariance/manifest.json"
    high_level_manifest = (
        _load(eval_root, "semantic_invariance/manifest.json")
        if high_level_manifest_path.exists()
        else None
    )
    high_level_methods = (
        high_level_payload.get("methods", {})
        if isinstance(high_level_payload, dict)
        else {}
    )
    high_level_complete = (
        isinstance(high_level_payload, dict)
        and (
            high_level_payload.get("complete") is True
            or (
                isinstance(high_level_manifest, dict)
                and high_level_manifest.get("complete") is True
            )
        )
        and all(
            method in high_level_methods
            and int(high_level_methods[method].get("sampled_features", 0)) > 0
            and int(
                high_level_methods[method].get("valid_evidence_features", 0)
            )
            == int(high_level_methods[method].get("sampled_features", 0))
            and int(
                high_level_methods[method].get("complete_vote_features", 0)
            )
            == int(high_level_methods[method].get("sampled_features", 0))
            and "high_level_fraction" in high_level_methods[method]
            for method in METHODS
        )
    )
    if (
        high_level_complete
        and isinstance(high_level_payload, dict)
    ):
        high_level = high_level_payload["methods"]
    else:
        # Keep the summary reproducible while a new semantic-invariance run is
        # replacing its task directory. This snapshot is copied from the last
        # completed result used by the summary.
        high_level = _load(
            eval_root,
            "evaluation_summary/high_level_feature_snapshot.json",
        )["methods"]
    dictionary = _load(
        eval_root,
        "dictionary_utilization/dictionary_utilization.json",
    )["methods"]
    adjacent = _load(
        eval_root,
        "shared/feature_consistency/adjacent_consistency/"
        "adjacent_feature_consistency.json",
    )["methods"]
    temporal = _load(
        eval_root,
        "temporal_robustness/temporal_robustness.json",
    )["methods"]
    probes = _load(
        eval_root,
        "label_efficiency/linear_probe_results.json",
    )["high_level_transfer"]["methods"]
    reasoning = _load(eval_root, "reasoning/results_summary.json")
    steering = _load(eval_root, "steering/steering_summary.json")["methods"]
    linking = _load(
        eval_root,
        "document_linking/document_linking_results.json",
    )["methods"]

    persistence_values: dict[str, float] = {}
    persistence_intervals: dict[str, tuple[float, float]] = {}
    for method in METHODS:
        if method in adjacent:
            record = adjacent[method]["feature_metrics"]
        else:
            # Joint checkpoints were added after the original four-method
            # adjacent-consistency artifact and are stored with Eval 2.
            record = dictionary[method]["feature_persistence_metrics"]
        persistence_values[method] = float(record["mean_persistence_lift"])
        persistence_intervals[method] = tuple(
            float(value) for value in record["mean_persistence_lift_95ci"]
        )

    context_intervals = {
        method: tuple(
            float(value)
            for value in context[method]["supported_selectivity_95ci"]
        )
        for method in METHODS
    }
    high_level_intervals = {}
    for method in METHODS:
        record = high_level[method]
        interval = (
            record["wilson_95ci"]
            if "wilson_95ci" in record
            else record["wilson_95_ci"]
        )
        high_level_intervals[method] = tuple(
            float(value) for value in interval
        )
    steering_intervals = {
        method: tuple(float(value) for value in steering[method]["score_95ci"])
        for method in METHODS
    }

    native_recall = {
        method: float(reasoning["native_recall_mean"][method])
        for method in METHODS
    }
    reasoning_generalization = {
        method: 0.5
        * (
            float(reasoning["cue_free_recall_mean"][method])
            + 1.0
            - float(reasoning["cue_only_false_activation_mean"][method])
        )
        for method in METHODS
    }

    return (
        MetricSpec(
            key="rfve",
            order=1,
            group="intrinsic",
            title="Reference-normalized reconstruction",
            subtitle=(
                "Fraction of task-reference explained variance retained by "
                "the frozen SAE."
            ),
            axis_label="RFVE",
            radar_label="RFVE",
            source="rfve/training_fidelity.json",
            description=(
                "Reference-normalized fraction of variance explained for each "
                "SAE's task-specific reconstruction objective."
            ),
            display="percent",
            plot_low=0.80,
            plot_high=0.95,
            radar_low=0.75,
            radar_high=1.0,
            values=_method_values(rfve, "rfve"),
        ),
        MetricSpec(
            key="reconstruction_cosine",
            order=2,
            group="intrinsic",
            title="Reconstruction cosine similarity",
            subtitle=(
                "Mean cosine similarity between the reconstruction and its "
                "task-specific target."
            ),
            axis_label="Mean cosine similarity",
            radar_label="Reconstruction cosine",
            source="rfve/reconstruction_cosine.json",
            description=(
                "Mean per-sample cosine similarity between the raw target and "
                "the SAE reconstruction."
            ),
            display="decimal3",
            plot_low=0.90,
            plot_high=1.00,
            radar_low=0.85,
            radar_high=1.0,
            values=_method_values(cosine, "mean_cosine_similarity"),
        ),
        MetricSpec(
            key="context_autointerp",
            order=3,
            group="intrinsic",
            title="Focused Interpretability",
            subtitle=(
                "Specificity and narrowness of a frozen rule after requiring "
                "held-out active support."
            ),
            axis_label="Focused Interpretability",
            radar_label="Focused Interpretability",
            source=(
                "autointerp/autointerp_exact1000/results/summary.json"
            ),
            description=(
                "Mean of inactive rejection and one minus active recall among "
                "frozen rules that identify at least one held-out active "
                "context; active support is reported separately."
            ),
            display="percent",
            plot_low=0.50,
            plot_high=0.70,
            radar_low=0.50,
            radar_high=0.70,
            values=_method_values(context, "supported_selectivity"),
            confidence_intervals=context_intervals,
        ),
        MetricSpec(
            key="high_level_feature_fraction",
            order=4,
            group="intrinsic",
            title="High-level feature fraction",
            subtitle=(
                "Percentage of 1,000 uniformly sampled coordinates classified "
                "as high-level; Wilson 95% CI."
            ),
            axis_label="Validated feature fraction",
            radar_label="High-level feature fraction",
            source="semantic_invariance/results.json",
            description=(
                "Fraction of uniformly sampled dictionary coordinates whose "
                "ten typical activations receive a passing semantic/functional "
                "judgment (at least 6/10 matches and surface form insufficient)."
            ),
            display="percent",
            plot_low=0.15,
            plot_high=0.38,
            radar_low=0.0,
            radar_high=0.45,
            values=_method_values(high_level, "high_level_fraction"),
            confidence_intervals=high_level_intervals,
        ),
        MetricSpec(
            key="feature_persistence_lift",
            order=5,
            group="intrinsic",
            title="Feature persistence lift",
            subtitle=(
                "Increase in feature recurrence for true adjacent chunks over "
                "length-matched random partners."
            ),
            axis_label="Mean persistence lift",
            radar_label="Feature persistence lift",
            source=(
                "shared/feature_consistency/adjacent_consistency/"
                "adjacent_feature_consistency.json; Joint extension from "
                "dictionary_utilization/dictionary_utilization.json"
            ),
            description=(
                "Mean feature-level increase in adjacent-chunk activation "
                "probability relative to a shuffled different-document control."
            ),
            display="decimal3",
            plot_low=0.16,
            plot_high=0.32,
            radar_low=0.10,
            radar_high=0.35,
            values=persistence_values,
            confidence_intervals=persistence_intervals,
        ),
        MetricSpec(
            key="dictionary_utilization",
            order=6,
            group="intrinsic",
            title="Dictionary utilization",
            subtitle=(
                "Entropy-equivalent fraction of the sampled alive dictionary "
                "used by held-out activations."
            ),
            axis_label="Effective feature fraction",
            radar_label="Dictionary utilization",
            source="dictionary_utilization/dictionary_utilization.json",
            description=(
                "Effective feature count divided by the uniformly sampled "
                "alive-feature population."
            ),
            display="percent",
            plot_low=0.04,
            plot_high=0.34,
            radar_low=0.0,
            radar_high=0.40,
            values=_method_values(dictionary, "effective_feature_fraction"),
        ),
        MetricSpec(
            key="arxiv_ood_accuracy",
            order=7,
            group="downstream",
            title="OOD accuracy",
            subtitle="Classification accuracy on held-out future-year abstracts.",
            axis_label="OOD accuracy",
            radar_label="OOD accuracy",
            source="temporal_robustness/temporal_robustness.json",
            description=(
                "Eight-domain classification accuracy on the held-out "
                "future-publication-year split."
            ),
            display="percent",
            plot_low=0.78,
            plot_high=0.84,
            radar_low=0.77,
            radar_high=0.86,
            values=_method_values(temporal, "ood_accuracy"),
        ),
        MetricSpec(
            key="arxiv_low_label_auc",
            order=8,
            group="downstream",
            title="Low-label AUC",
            subtitle=(
                "Average linear-probe performance across small labeled-data "
                "budgets."
            ),
            axis_label="Low-label AUC",
            radar_label="Low-label AUC",
            source="label_efficiency/linear_probe_results.json",
            description=(
                "Area under the frozen linear-probe accuracy curve across the "
                "predefined low-label training budgets."
            ),
            display="decimal3",
            plot_low=0.70,
            plot_high=0.735,
            radar_low=0.69,
            radar_high=0.75,
            values=_method_values(probes, "low_label_auc"),
        ),
        MetricSpec(
            key="reasoning_native_recall",
            order=9,
            group="downstream",
            title="Native reasoning recall",
            subtitle=(
                "Activation rate on held-out texts containing the target "
                "reasoning relation (panel A)."
            ),
            axis_label="Native recall",
            radar_label="Native reasoning recall",
            source="reasoning/results_summary.json",
            description=(
                "Mean frozen-feature activation rate on native validation "
                "examples for the five reasoning scenarios."
            ),
            display="percent",
            plot_low=0.42,
            plot_high=0.84,
            radar_low=0.0,
            radar_high=1.0,
            values=native_recall,
        ),
        MetricSpec(
            key="reasoning_generalization",
            order=10,
            group="downstream",
            title="Reasoning generalization",
            subtitle=(
                "Mean of cue-free recall and rejection when surface cues "
                "remain but the relation is removed (panels B and C)."
            ),
            axis_label="Cue-controlled generalization score",
            radar_label="Reasoning generalization",
            source="reasoning/results_summary.json",
            description=(
                "0.5 × [cue-free recall + (1 − cue-only false activation)], "
                "so both relation retention and cue rejection are required."
            ),
            display="percent",
            plot_low=0.25,
            plot_high=0.72,
            radar_low=0.0,
            radar_high=1.0,
            values=reasoning_generalization,
        ),
        MetricSpec(
            key="causal_steering",
            order=11,
            group="downstream",
            title="Positive causal steering",
            subtitle=(
                "Best-of-grid concept-direction score with relative coherence "
                "preservation; 50 is the calibrated null."
            ),
            axis_label="Steering Score",
            radar_label="Causal steering",
            source="steering/steering_summary.json",
            description=(
                "Mean best positive-intervention score over sampled features, "
                "combining movement toward the concept with coherence "
                "preservation."
            ),
            display="score1",
            plot_low=49.0,
            plot_high=71.0,
            radar_low=55.0,
            radar_high=72.0,
            values=_method_values(steering, "mean"),
            confidence_intervals=steering_intervals,
            neutral_value=50.0,
        ),
        MetricSpec(
            key="document_recall_at_5",
            order=12,
            group="downstream",
            title="Document retrieval Recall@5",
            subtitle=(
                "Same-document retrieval from lexically disjoint candidate "
                "chunks."
            ),
            axis_label="Recall@5",
            radar_label="Document Recall@5",
            source="document_linking/document_linking_results.json",
            description=(
                "Fraction of queries whose lexically disjoint same-document "
                "partner is ranked within the top five."
            ),
            display="percent",
            plot_low=0.58,
            plot_high=0.78,
            radar_low=0.50,
            radar_high=0.85,
            values=_method_values(linking, "recall_at_5"),
        ),
    )


def serializable_metric_summary(
    metrics: tuple[MetricSpec, ...],
) -> dict[str, object]:
    """Return the plotted values and definitions in a machine-readable form."""

    return {
        "format": "chunk-saes-evaluation-summary-selected-metrics-v1",
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "reasoning_generalization_formula": (
            "0.5 * (cue_free_recall + 1 - cue_only_false_activation)"
        ),
        "metrics": {
            metric.key: {
                "order": metric.order,
                "group": metric.group,
                "title": metric.title,
                "description": metric.description,
                "source": metric.source,
                "display": metric.display,
                "radar_range": [metric.radar_low, metric.radar_high],
                "values": dict(metric.values),
                "confidence_intervals": (
                    {
                        method: list(interval)
                        for method, interval in metric.confidence_intervals.items()
                    }
                    if metric.confidence_intervals is not None
                    else None
                ),
            }
            for metric in metrics
        },
    }
