#!/usr/bin/env python
"""Collect completed ablation artifacts into the requested results.md.

This is deliberately tolerant of an in-progress run: missing seed/stage
artifacts are listed as pending, while every completed checkpoint contributes
its validation metrics.  The final invocation by the experiment runner is
therefore idempotent and also useful for live progress snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


CROSS_ATTAINABLE_REFERENCE_FVE = 0.6272699794963396


def _read(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _num(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {sd:.4f} (n={len(values)})"


def _cross_metrics(complete: dict[str, Any]) -> dict[str, Any]:
    metrics = complete.get("final_full_validation_metrics") or complete.get(
        "last_periodic_validation_metrics"
    )
    if not isinstance(metrics, dict):
        return {}
    def get(direction: str, key: str) -> Any:
        return metrics.get(f"validation/{direction}/{key}")
    pooled_fve = _num(metrics.get("validation/fve"))
    return {
        "a_to_b_fve": get("a_to_b", "fve"),
        "b_to_a_fve": get("b_to_a", "fve"),
        "pooled_fve": pooled_fve,
        "pooled_nmse": metrics.get("validation/nmse"),
        "attainable_fidelity": (
            pooled_fve / CROSS_ATTAINABLE_REFERENCE_FVE
            if pooled_fve is not None
            else None
        ),
        "attainable_reference_fve": CROSS_ATTAINABLE_REFERENCE_FVE,
        "a_to_b_nmse": get("a_to_b", "nmse"),
        "b_to_a_nmse": get("b_to_a", "nmse"),
        "l0": metrics.get("validation/effective_l0"),
        "samples": complete.get("samples_seen"),
        "alive": complete.get("alive_features"),
        "dead": complete.get("dead_features"),
        "coverage": complete.get("coverage_fraction"),
        "accepted_rows": complete.get("accepted_training_rows"),
        "filtered_rows": complete.get("activation_norm_filtered_rows"),
        "balanced_rows": complete.get("activation_norm_balance_dropped_rows"),
    }


def _sequence_metrics(complete: dict[str, Any]) -> dict[str, Any]:
    metrics = complete.get("validation_metrics") or complete.get("training_metrics")
    if not isinstance(metrics, dict):
        return {}
    partner_fve = _num(metrics.get("partner_fve"))
    self_fve = _num(metrics.get("self_fve"))
    total_fve = _num(metrics.get("total_fve"))
    return {
        "partner_fve": partner_fve,
        "self_fve": self_fve,
        "total_fve": total_fve,
        "partner_nmse": 1.0 - partner_fve if partner_fve is not None else None,
        "self_nmse": 1.0 - self_fve if self_fve is not None else None,
        "total_nmse": 1.0 - total_fve if total_fve is not None else None,
        "partner_mse": metrics.get("partner_mse"),
        "self_mse": metrics.get("self_mse"),
        "l0": metrics.get("effective_l0"),
        "partner_l0": metrics.get("partner_effective_l0"),
        "self_l0": metrics.get("self_effective_l0"),
        "partner_tokens": metrics.get("partner_tokens"),
        "self_tokens": metrics.get("self_tokens"),
        "samples": complete.get("target_occurrences_seen"),
        "alive": complete.get("alive_features"),
        "dead": complete.get("dead_features"),
        "direction_counts": metrics.get("direction_counts"),
        "coverage": complete.get("target_coverage"),
        "exact_coverage": complete.get("exact_target_coverage"),
        "accepted_rows": complete.get("target_occurrences_seen"),
        "padding_rows": complete.get("padding_rows_excluded"),
    }


def _resource_metrics(
    complete: dict[str, Any],
    experiment: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    activation_dim = int(complete.get("activation_dim", 4096))
    dict_size = int(complete.get("dict_size", 65536))
    k = int(complete.get("k", 128))
    # Encoder dense matmul plus sparse K-feature decoder, counting a multiply
    # and add as two FLOPs. Bias/ReLU/top-k bookkeeping is intentionally
    # excluded, so this is a comparable dictionary-forward lower bound.
    dictionary_flops_per_row = 2 * activation_dim * dict_size + 2 * k * activation_dim
    sae_parameters = 2 * activation_dim * dict_size + dict_size + activation_dim
    if experiment in {"E1", "E2"}:
        selected_rows = _num(metrics.get("accepted_rows"))
        parameter_count = sae_parameters
    else:
        validation = complete.get("training_metrics") or {}
        partner_rows = _num(validation.get("partner_tokens")) or 0.0
        self_rows = _num(validation.get("self_tokens")) or 0.0
        selected_rows = partner_rows + self_rows
        context_dim = int(complete.get("context_dim", 256))
        max_length = int(complete.get("max_chunk_length", 512))
        context_parameters = (
            activation_dim * context_dim  # source projection
            + 3 * context_dim * context_dim
            + 3 * context_dim
            + context_dim * context_dim
            + context_dim
            + context_dim * activation_dim
            + 1  # context gate
            + 2 * activation_dim  # query LayerNorm
            + max_length * context_dim
        )
        parameter_count = sae_parameters + context_parameters
    return {
        "parameter_count": parameter_count,
        "dictionary_flops_per_row": dictionary_flops_per_row,
        "dictionary_forward_flops": (
            dictionary_flops_per_row * selected_rows
            if selected_rows is not None
            else None
        ),
    }


def _discover(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Cross trainers keep their completion marker under ``cross/`` while the
    # sequence trainer writes it directly in the seed directory.  Search
    # recursively so both checkpoint layouts are discovered, without pulling
    # in the deliberately prefixed smoke/audit directories.
    for path in sorted(root.glob("E[1-4]/*/seed*/**/complete.json")):
        complete = _read(path)
        if complete is None:
            continue
        parts = path.parts
        try:
            index = parts.index("ablation")
            experiment, stage, seed_name = parts[index + 1 : index + 4]
        except (ValueError, IndexError):
            continue
        if experiment not in {"E1", "E2", "E3", "E4"}:
            continue
        if stage not in {"pilot", "confirm", "full"}:
            continue
        if not seed_name.startswith("seed"):
            continue
        record = {
            "experiment": experiment,
            "stage": stage,
            "seed": seed_name.removeprefix("seed"),
            "path": str(path.parent),
            "complete": complete,
        }
        record["metrics"] = (
            _cross_metrics(complete)
            if experiment in {"E1", "E2"}
            else _sequence_metrics(complete)
        )
        record["metrics"].update(
            _resource_metrics(complete, experiment, record["metrics"])
        )
        stage_budget = {"pilot": 10_016_000, "confirm": 100_000_000, "full": 1_000_000_000}.get(stage)
        seen = record["metrics"].get("samples") or record["metrics"].get("accepted_rows")
        if stage_budget and _num(seen) is not None:
            record["metrics"]["stage_coverage"] = float(seen) / stage_budget
        invariant_path = record["path"]
        # ``path`` is the checkpoint directory for Cross and the seed
        # directory for sequence runs; normalize to the seed directory.
        seed_dir = path.parents[1] if path.parent.name == "cross" else path.parent
        invariant = _read(seed_dir / "invariant_metrics.json")
        if isinstance(invariant, dict):
            adjacent = invariant.get("adjacent") or {}
            linking = invariant.get("linking") or {}
            probes = adjacent.get("probes") or {}
            pair = adjacent.get("pair_metrics") or {}
            feature = adjacent.get("feature_metrics") or {}
            record["metrics"].update(
                {
                    "activation_entropy": adjacent.get("activation_entropy"),
                    "activation_effective_features": adjacent.get("activation_effective_features"),
                    "side_probe_accuracy": (probes.get("side") or {}).get("balanced_accuracy"),
                    "length_probe_accuracy": (probes.get("chunk_length") or {}).get("balanced_accuracy"),
                    "persistence": feature.get("mean_persistence_lift"),
                    "persistence_ci": feature.get("mean_persistence_lift_95ci"),
                    "adjacent_auc": pair.get("adjacent_retrieval_auc"),
                    "link_recall1": linking.get("recall_at_1"),
                    "link_recall5": linking.get("recall_at_5"),
                    "link_mrr": linking.get("mrr"),
                    "invariant_status": adjacent.get("status"),
                }
            )
        records.append(record)
    return records


def _baseline(root: Path) -> dict[str, Any]:
    pipeline = root.parent
    # The formal evaluation is task-oriented.  Keep this optional ablation
    # report pointed at the canonical locations instead of the retired
    # ``evalN``/flat ``shared`` layout.
    eval_root = pipeline / "eval/layer21_w65536_k128_1b"
    fidelity_path = eval_root / "rfve/training_fidelity.json"
    adjacent_path = (
        eval_root
        / "shared/feature_consistency/adjacent_consistency/adjacent_feature_consistency.json"
    )
    abstraction_path = (
        eval_root
        / "shared/high_level_feature_analysis/high_level_feature_results.json"
    )
    linking_path = eval_root / "document_linking/document_linking_results.json"
    fidelity = _read(fidelity_path) or {}
    adjacent = _read(adjacent_path) or {}
    abstraction = _read(abstraction_path) or {}
    linking = _read(linking_path) or {}
    cross = (fidelity.get("methods") or {}).get("cross", {})
    direction = cross.get("directions") if isinstance(cross, dict) else {}
    cross_method = (adjacent.get("methods") or {}).get("cross", {})
    pair_metrics = cross_method.get("pair_metrics", {}) if isinstance(cross_method, dict) else {}
    link = (linking.get("rank_distributions") or {}).get("cross", [])
    abstraction_cross = (abstraction.get("methods") or {}).get("cross", {})
    abstraction_metric = abstraction_cross.get("cross_document_abstraction", {}) if isinstance(abstraction_cross, dict) else {}
    recall1 = sum(1 for rank in link if float(rank) <= 1.0) / len(link) if link else None
    mrr = sum(1.0 / float(rank) for rank in link if float(rank) > 0) / len(link) if link else None
    def direction_fve(name: str) -> Any:
        value = (direction or {}).get(name, {})
        if not isinstance(value, dict):
            return None
        return value.get("raw_fve", value.get("fve"))

    return {
        "checkpoint": str(pipeline / "checkpoints"),
        "pooled_fve": cross.get("raw_fve"),
        "pooled_nmse": cross.get("nmse"),
        "attainable_fidelity": cross.get("attainable_fidelity"),
        "attainable_reference_fve": cross.get("reference_fve"),
        "a_to_b_fve": direction_fve("a_to_b"),
        "b_to_a_fve": direction_fve("b_to_a"),
        "l0": cross.get("effective_l0"),
        "alive": cross.get("alive_features"),
        "persistence": cross_method.get("feature_metrics", {}).get("mean_persistence_lift"),
        "adjacent_auc": pair_metrics.get("adjacent_retrieval_auc"),
        "cross_document_abstraction": abstraction_metric.get("mean"),
        "cross_document_abstraction_95ci": abstraction_metric.get("95ci"),
        # Keep the anchor on the same absolute scale as E1/E2 linking
        # evaluations.  The existing report also publishes Cross-minus-Mean
        # effects; retain those separately so the matrix never mixes an
        # absolute score with a delta.
        "link_recall1": recall1,
        "link_mrr": mrr,
        "link_delta_recall1": ((linking.get("comparisons") or {}).get("cross_minus_mean") or {}).get("recall_at_1", {}).get("point"),
        "link_delta_mrr": ((linking.get("comparisons") or {}).get("cross_minus_mean") or {}).get("mrr", {}).get("point"),
        "link_rank_proxy_recall1": recall1,
        "link_rank_proxy_mrr": mrr,
    }


def _status(root: Path) -> list[str]:
    path = root / "status.tsv"
    if not path.is_file():
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _table(records: list[dict[str, Any]], experiment: str, stage: str) -> str:
    selected = [r for r in records if r["experiment"] == experiment and r["stage"] == stage]
    if experiment in {"E1", "E2"}:
        rows = []
        for key in (
            "a_to_b_fve", "b_to_a_fve", "pooled_fve", "pooled_nmse",
            "attainable_fidelity", "attainable_reference_fve", "a_to_b_nmse",
            "b_to_a_nmse", "l0", "alive", "dead", "stage_coverage", "coverage",
            "accepted_rows", "filtered_rows", "balanced_rows", "parameter_count",
            "dictionary_flops_per_row", "dictionary_forward_flops",
            "activation_entropy", "activation_effective_features",
            "side_probe_accuracy", "length_probe_accuracy",
            "persistence", "adjacent_auc", "link_recall1", "link_recall5",
            "link_mrr",
        ):
            values = [_num(r["metrics"].get(key)) for r in selected]
            rows.append(f"| {key} | {_mean_sd([v for v in values if v is not None])} |")
    else:
        rows = []
        for key in (
            "partner_fve", "self_fve", "total_fve", "partner_nmse", "self_nmse",
            "total_nmse", "partner_mse", "self_mse",
            "l0", "partner_l0", "self_l0", "alive", "dead", "stage_coverage", "coverage", "exact_coverage", "padding_rows", "persistence", "adjacent_auc",
            "partner_tokens", "self_tokens", "accepted_rows", "parameter_count",
            "dictionary_flops_per_row", "dictionary_forward_flops",
            "activation_entropy", "activation_effective_features",
            "side_probe_accuracy", "length_probe_accuracy",
            "link_recall1", "link_recall5", "link_mrr",
        ):
            values = [_num(r["metrics"].get(key)) for r in selected]
            rows.append(f"| {key} | {_mean_sd([v for v in values if v is not None])} |")
    if not selected:
        return "| status | pending |\n"
    return "| metric | mean +/- sd |\n|---|---|\n" + "\n".join(rows) + "\n"


def build(root: Path) -> str:
    records = _discover(root)
    baseline = _baseline(root)
    lines: list[str] = []
    lines.append("# Cross-Chunk SAE Ablation Results")
    lines.append("")
    lines.append("Generated from the checkpoints and validation artifacts under this directory. `E0` is the existing bidirectional Cross anchor; E1/E2 are mean-Cross direction ablations; E3/E4 are the new masked-sequence objectives.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Model: Qwen3.5-9B-Base, layer 21, hidden size 4096, width 65,536, BatchTopK K=128.")
    lines.append("- Cache: activation-cache-v2, independent A/B forwards, positions reset per chunk; train 1,000,000,000 occurrences and validation 10,000,128 occurrences.")
    lines.append("- Pilot/confirm/full budgets: 10,016,000 (2 seeds), 100,000,000 (3 seeds), 1,000,000,000 (2 seeds), as specified in `ablation.md`.")
    lines.append("- E1/E2 keep the same pair/row/BatchTopK budget and replace the removed direction with the selected direction's pair-side means. E3/E4 use the same sequence architecture; E4 adds owning-chunk self loss.")
    lines.append("- Shared invariant audit uses the validation cache's stratified 2,048 pairs, 8,192 sampled features, exact-length shuffled controls, and 10,000 bootstrap draws; sequence partner-query persistence uses the same pair universe with its explicit token-level representation.")
    lines.append("")
    lines.append("## Completion")
    lines.append("")
    lines.append("| experiment | pilot | confirm | full |")
    lines.append("|---|---:|---:|---:|")
    for experiment in ("E1", "E2", "E3", "E4"):
        cells = []
        for stage in ("pilot", "confirm", "full"):
            n = sum(1 for r in records if r["experiment"] == experiment and r["stage"] == stage)
            expected = {"pilot": 2, "confirm": 3, "full": 2}[stage]
            cells.append(f"{n}/{expected}")
        lines.append(f"| {experiment} | {' | '.join(cells)} |")
    lines.append("")
    lines.append("## E0 Anchor")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key, label in (("pooled_fve", "validation pooled raw FVE"), ("pooled_nmse", "validation pooled NMSE"), ("attainable_fidelity", "attainable fidelity"), ("attainable_reference_fve", "attainable reference FVE"), ("a_to_b_fve", "A->B FVE"), ("b_to_a_fve", "B->A FVE"), ("l0", "effective L0"), ("alive", "alive features"), ("persistence", "adjacent persistence lift"), ("adjacent_auc", "adjacent retrieval AUC"), ("cross_document_abstraction", "cross-document abstraction"), ("link_recall1", "lexical-controlled linking Recall@1"), ("link_mrr", "lexical-controlled linking MRR")):
        lines.append(f"| {label} | {_fmt(baseline.get(key))} |")
    lines.append("")
    lines.append("The existing lexical-controlled report gives absolute Cross Recall@1 = 0.5481 (95% CI 0.5279–0.5680) and MRR = 0.6515 (95% CI 0.6343–0.6683), with Cross-minus-Mean effects of +0.1175 and +0.1142 respectively. E0 cross-document abstraction is computed from the shared blinded evidence artifact; new ablation checkpoints do not contain raw text, so their text-linked abstraction/linking values are marked unavailable unless a matching feature artifact exists.")
    lines.append("")
    lines.append("## Direction Ablation")
    lines.append("")
    for experiment, description in (("E1", "A->B only"), ("E2", "B->A only")):
        lines.append(f"### {experiment}: {description}")
        lines.append("")
        for stage in ("pilot", "confirm", "full"):
            lines.append(f"**{stage}**")
            lines.append("")
            lines.append(_table(records, experiment, stage).rstrip())
            lines.append("")
    lines.append("### Full direction matrix")
    lines.append("")
    lines.append("| train model | eval A->B FVE | eval B->A FVE | persistence lift | adjacent AUC | linking Recall@1 | linking MRR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    full_records = {
        experiment: [
            r for r in records if r["experiment"] == experiment and r["stage"] == "full"
        ]
        for experiment in ("E1", "E2")
    }
    def matrix_value(experiment: str, key: str) -> str:
        values = [_num(r["metrics"].get(key)) for r in full_records[experiment]]
        values = [v for v in values if v is not None]
        return _mean_sd(values) if values else "pending"
    lines.append(
        f"| E0 bidirectional | {_fmt(baseline.get('a_to_b_fve'))} | {_fmt(baseline.get('b_to_a_fve'))} | {_fmt(baseline.get('persistence'))} | {_fmt(baseline.get('adjacent_auc'))} | {_fmt(baseline.get('link_recall1'))} | {_fmt(baseline.get('link_mrr'))} |"
    )
    for experiment in ("E1", "E2"):
        lines.append(
            f"| {experiment} | {matrix_value(experiment, 'a_to_b_fve')} | {matrix_value(experiment, 'b_to_a_fve')} | {matrix_value(experiment, 'persistence')} | {matrix_value(experiment, 'adjacent_auc')} | {matrix_value(experiment, 'link_recall1')} | {matrix_value(experiment, 'link_mrr')} |"
        )
    lines.append("")
    lines.append("## Mask Ablation")
    lines.append("")
    for experiment, description in (("E3", "masked partner-only"), ("E4", "masked all-target (self + partner)")):
        lines.append(f"### {experiment}: {description}")
        lines.append("")
        for stage in ("pilot", "confirm", "full"):
            lines.append(f"**{stage}**")
            lines.append("")
            lines.append(_table(records, experiment, stage).rstrip())
            lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append("- Every completed E1/E2 artifact records `samples_seen`, accepted rows, direction policy, alive/dead features, and direction-stratified validation FVE/NMSE. Every E3/E4 artifact records exact partner-token coverage, padding exclusion, effective L0, direction counts, and separate self/partner/total FVE.")
    lines.append("- The invariant evaluator additionally reports entropy/effective feature count and pair-clustered side/length probes on the fixed validation sample. A lexical-overlap probe is marked unavailable when the cache contains no raw text; this is preferable to silently reusing a different corpus.")
    lines.append("- `status.tsv` and per-seed `run.log` files are the live provenance trail. Missing cells above mean the corresponding process has not yet published `complete.json`.")
    lines.append("- Raw Cross FVE is a task-fit metric with an attainable ceiling below one; it is not ranked directly against self-reconstruction FVE without the task ceiling and invariant metrics.")
    lines.append("- Sequence invariant evaluation is kept separate from the mean-only document-linking cache; E3/E4 partner/self FVE, direction counts, padding exclusion, L0, and coverage are reported directly from their checkpoints, while unavailable text-linked metrics remain explicitly pending.")
    sequence_full = [
        r for r in records if r["experiment"] in {"E3", "E4"} and r["stage"] == "full"
    ]
    if sequence_full:
        lines.append("")
        lines.append("### Full sequence diagnostics")
        lines.append("")
        lines.append("| experiment | seed | partner tokens | self tokens | A->B tokens | B->A tokens | effective L0 | alive | dead | exact coverage | padding rows |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for record in sequence_full:
            metrics = record["metrics"]
            counts = metrics.get("direction_counts") or {}
            lines.append(
                f"| {record['experiment']} | {record['seed']} | {_fmt(metrics.get('partner_tokens'), 0)} | {_fmt(metrics.get('self_tokens'), 0)} | {_fmt(counts.get('a_to_b'), 0)} | {_fmt(counts.get('b_to_a'), 0)} | {_fmt(metrics.get('l0'))} | {_fmt(metrics.get('alive'), 0)} | {_fmt(metrics.get('dead'), 0)} | {metrics.get('exact_coverage', 'n/a')} | {_fmt(metrics.get('padding_rows'), 0)} |"
            )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    full = {e: [r for r in records if r["experiment"] == e and r["stage"] == "full"] for e in ("E1", "E2", "E3", "E4")}
    e0_fve = _num(baseline.get("pooled_fve"))
    e1 = [v for r in full["E1"] for v in [_num(r["metrics"].get("a_to_b_fve"))] if v is not None]
    e2 = [v for r in full["E2"] for v in [_num(r["metrics"].get("b_to_a_fve"))] if v is not None]
    e3 = [v for r in full["E3"] for v in [_num(r["metrics"].get("partner_fve"))] if v is not None]
    e4p = [v for r in full["E4"] for v in [_num(r["metrics"].get("partner_fve"))] if v is not None]
    e4s = [v for r in full["E4"] for v in [_num(r["metrics"].get("self_fve"))] if v is not None]
    if e1 and e2 and e0_fve is not None:
        best_single = max(statistics.fmean(e1), statistics.fmean(e2))
        lines.append(f"- Direction conclusion: E0 anchor pooled FVE={e0_fve:.4f}; best completed single-direction primary FVE={best_single:.4f}. Invariant persistence/linking must decide whether any gap is meaningful; FVE alone is not sufficient.")
    else:
        lines.append("- Direction conclusion: pending full E1/E2 checkpoints and shared invariant evaluation.")
    if e3 and e4p:
        delta = statistics.fmean(e4p) - statistics.fmean(e3)
        lines.append(f"- Mask conclusion: E4 partner-FVE minus E3 partner-FVE = {delta:+.4f}; E4 self-FVE is {_fmt(statistics.fmean(e4s) if e4s else None)}. The self term is accepted only if partner and invariant metrics do not regress.")
    else:
        lines.append("- Mask conclusion: pending full E3/E4 partner/self comparison.")
    mask_invariant = {
        e: [
            _num(r["metrics"].get("persistence"))
            for r in records
            if r["experiment"] == e and r["stage"] == "full"
        ]
        for e in ("E3", "E4")
    }
    if all(any(v is not None for v in values) for values in mask_invariant.values()):
        e3_inv = statistics.fmean([v for v in mask_invariant["E3"] if v is not None])
        e4_inv = statistics.fmean([v for v in mask_invariant["E4"] if v is not None])
        verdict = "does not regress" if e4_inv >= e3_inv else "regresses"
        lines.append(f"- Invariant mask check: E4 persistence lift={e4_inv:.4f} vs E3={e3_inv:.4f}; the self term {verdict} on this partner-query diagnostic, but text-linked metrics must still be available before a deployment choice.")
    else:
        lines.append("- Invariant mask check: pending full E3/E4 partner-query persistence artifacts.")
    full_persistence = {
        e: [
            _num(r["metrics"].get("persistence"))
            for r in records
            if r["experiment"] == e and r["stage"] == "full"
        ]
        for e in ("E1", "E2")
    }
    if all(any(v is not None for v in values) for values in full_persistence.values()):
        best = max(
            statistics.fmean([v for v in full_persistence["E1"] if v is not None]),
            statistics.fmean([v for v in full_persistence["E2"] if v is not None]),
        )
        lines.append(f"- Invariant direction check: best single-direction full persistence lift={best:.4f}; E0={_fmt(baseline.get('persistence'))}. The final direction choice requires the paired bootstrap CIs in each `invariant_metrics.json`, not FVE alone.")
    else:
        lines.append("- Invariant direction check: pending the shared full E1/E2 evaluator artifacts.")
    lines.append("")
    lines.append("## Live Status")
    lines.append("")
    status_lines = _status(root)
    if status_lines:
        lines.append("```text")
        lines.extend(status_lines[-20:])
        lines.append("```")
    else:
        lines.append("No status entries yet.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.ablation_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build(root), encoding="utf-8")


if __name__ == "__main__":
    main()
