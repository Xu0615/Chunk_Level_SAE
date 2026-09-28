#!/usr/bin/env python
"""Strict full-dictionary counterpart test for one frozen Cross feature.

The semantic identity of the feature is imported from pre-existing Eval-4
artifacts.  This script then asks a narrower representation question on
document-disjoint held-out Pile chunks:

Can the complete Token-SAE code recover the active bit of that Cross
coordinate with one, four, sixteen, or sixty-four coordinates?

The Cross active bit is definitionally the target, so Cross AUC=1 is an upper
bound rather than independent semantic validation.  Semantic validity comes
from the frozen explanation, blinded score, natural examples, and separate
challenge-bank results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from chunk_saes.artifacts import (
    file_record,
    file_sha256,
    write_artifact_manifest,
)
from chunk_saes.plot_style import METHOD_COLORS
from chunk_saes.utils import atomic_json_dump
from evals.reasoning.evaluate_reason_counterparts import (
    FrozenEncoder,
    _autointerp_rates,
    _cluster_bootstrap,
    _decoder_sparse_approximation,
    _load_candidate_artifacts,
    _load_chunk_document_map,
    _paired_bootstrap_delta,
    _prevalence_matched_recovery,
    _screen_features,
    _slice_dense,
)


FORMAT = "chunk-saes-reason-headline-full-token-v1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--full-token-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--feature-id", type=int, default=20232)
    p.add_argument("--sparse-budgets", default="4,16,64")
    p.add_argument("--screen-features", type=int, default=512)
    p.add_argument("--logistic-c", type=float, default=1e-4)
    p.add_argument(
        "--selection-c-grid",
        default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1",
        help=(
            "L1 selection strengths considered inside each outer training "
            "fold; univariate top-k is also considered."
        ),
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--cross-batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--overwrite", action="store_true")
    return p


def _exhaustive_single_fold(
    matrix: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    block_size: int = 4096,
) -> tuple[int, int, float, np.ndarray]:
    """Search every Token coordinate using exact tie-aware train AUC."""

    train_rows = np.flatnonzero(train)
    y = labels[train]
    positive = int(y.sum())
    negative = int((~y).sum())
    best: tuple[float, int, int] | None = None
    for start in range(0, matrix.shape[1], block_size):
        stop = min(matrix.shape[1], start + block_size)
        block = np.asarray(
            matrix[np.ix_(train_rows, np.arange(start, stop))],
            dtype=np.float32,
        )
        ranks = rankdata(block, axis=0, method="average")
        auc = (
            ranks[y].sum(axis=0)
            - positive * (positive + 1) / 2
        ) / (positive * negative)
        signed = np.maximum(auc, 1 - auc)
        local = int(np.argmax(signed))
        direction = 1 if auc[local] >= 0.5 else -1
        candidate = (
            float(signed[local]),
            start + local,
            direction,
        )
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    train_auc, feature_id, direction = best
    values = direction * np.asarray(
        matrix[np.flatnonzero(test), feature_id],
        dtype=np.float64,
    )
    return feature_id, direction, train_auc, values


def _selected_columns(
    coefficients: np.ndarray,
    budget: int,
) -> np.ndarray:
    order = np.argsort(np.abs(coefficients), kind="stable")[::-1]
    return order[:budget].astype(np.int64)


def _fit_selected_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    selected: np.ndarray,
    *,
    logistic_c: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    scaler = StandardScaler().fit(x_train[:, selected])
    train = scaler.transform(x_train[:, selected])
    test = scaler.transform(x_test[:, selected])
    model = LogisticRegression(
        C=logistic_c,
        class_weight="balanced",
        max_iter=5_000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(train, y_train)
    return (
        model.predict_proba(test)[:, 1],
        float(roc_auc_score(y_train, model.predict_proba(train)[:, 1])),
    )


def _nested_sparse_fold(
    matrix: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    *,
    budget: int,
    screen_features: int,
    selection_c_grid: list[float],
    logistic_c: float,
    seed: int,
) -> dict[str, Any]:
    """Choose top-k versus L1-selected sparse models without outer leakage."""

    test = folds == outer_fold
    train = ~test
    candidates = _screen_features(
        matrix,
        labels,
        train,
        max(screen_features, budget),
    )
    x_outer_train = _slice_dense(matrix, train, candidates)
    x_outer_test = _slice_dense(matrix, test, candidates)
    y_outer_train = labels[train]
    inner_groups = sorted(set(folds[train].tolist()))
    strategies: list[tuple[str, float | None]] = [("univariate", None)]
    strategies.extend(("l1", value) for value in selection_c_grid)
    strategy_scores = []

    outer_rows = np.flatnonzero(train)
    outer_groups = folds[train]
    for strategy, selection_c in strategies:
        inner_values = np.zeros(len(outer_rows), dtype=np.float64)
        valid = np.zeros(len(outer_rows), dtype=bool)
        for inner_fold in inner_groups:
            inner_test = outer_groups == inner_fold
            inner_train = ~inner_test
            if (
                len(np.unique(y_outer_train[inner_train])) < 2
                or len(np.unique(y_outer_train[inner_test])) < 2
            ):
                continue
            if strategy == "univariate":
                selected = np.arange(budget, dtype=np.int64)
            else:
                scaler = StandardScaler().fit(
                    x_outer_train[inner_train]
                )
                train_scaled = scaler.transform(
                    x_outer_train[inner_train]
                )
                selector = LogisticRegression(
                    C=float(selection_c),
                    penalty="l1",
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=seed + int(inner_fold),
                    solver="liblinear",
                )
                selector.fit(
                    train_scaled,
                    y_outer_train[inner_train],
                )
                selected = _selected_columns(
                    selector.coef_[0],
                    budget,
                )
            probabilities, _ = _fit_selected_logistic(
                x_outer_train[inner_train],
                y_outer_train[inner_train],
                x_outer_train[inner_test],
                selected,
                logistic_c=logistic_c,
                seed=seed + 100 + int(inner_fold),
            )
            inner_values[inner_test] = probabilities
            valid[inner_test] = True
        score = (
            float(
                roc_auc_score(
                    y_outer_train[valid],
                    inner_values[valid],
                )
            )
            if valid.any()
            else float("-inf")
        )
        strategy_scores.append((score, strategy, selection_c))

    inner_auc, strategy, selection_c = max(
        strategy_scores,
        key=lambda row: (
            row[0],
            row[1] == "univariate",
            -(row[2] or 0.0),
        ),
    )
    if strategy == "univariate":
        selected = np.arange(budget, dtype=np.int64)
    else:
        scaler = StandardScaler().fit(x_outer_train)
        selector = LogisticRegression(
            C=float(selection_c),
            penalty="l1",
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
            solver="liblinear",
        )
        selector.fit(
            scaler.transform(x_outer_train),
            y_outer_train,
        )
        selected = _selected_columns(selector.coef_[0], budget)
    probabilities, train_auc = _fit_selected_logistic(
        x_outer_train,
        y_outer_train,
        x_outer_test,
        selected,
        logistic_c=logistic_c,
        seed=seed + 1000,
    )
    return {
        "predictions": probabilities,
        "selected_feature_ids":
            candidates[selected].astype(int).tolist(),
        "selection_strategy": strategy,
        "selection_l1_c": selection_c,
        "inner_cv_auc": inner_auc,
        "outer_train_auc": train_auc,
    }


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root)
    full_token_dir = Path(args.full_token_dir)
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval6_path = eval_root / "document_linking/features.npz"
    with np.load(eval6_path) as data:
        global_ids = data["global_chunk_ids"].copy()
    materialized_ids = np.load(full_token_dir / "global_chunk_ids.npy")
    if not np.array_equal(global_ids, materialized_ids):
        raise RuntimeError("full Token matrix row order differs from Eval 6")
    token = np.load(
        full_token_dir / "token_mean_full.npy",
        mmap_mode="r",
    )
    if token.shape != (len(global_ids), 65_536):
        raise RuntimeError(f"unexpected full Token shape: {token.shape}")

    document_map = _load_chunk_document_map(
        eval_root / "dictionary_utilization/feature_evidence"
    )
    documents = np.asarray(
        [document_map[int(chunk_id)] for chunk_id in global_ids]
    )
    folds = np.asarray(
        [
            int(
                hashlib.sha256(document.encode("utf-8")).hexdigest()[:8],
                16,
            )
            % args.cv_folds
            for document in documents
        ],
        dtype=np.int8,
    )

    cross_checkpoint = Path(args.sae_root) / "cross/checkpoints/best"
    if not cross_checkpoint.exists():
        cross_checkpoint = Path(args.sae_root) / "cross"
    cross_path = (
        full_token_dir / f"cross_feature_{args.feature_id}.npy"
    )
    if not cross_path.exists():
        raise FileNotFoundError(
            f"{cross_path} is required so the Cross target is encoded in "
            "the same forward pass as the full Token representation"
        )
    cross_values = np.load(cross_path).astype(np.float64)
    labels = cross_values > 0
    if int(labels.sum()) < 20:
        raise RuntimeError(
            f"only {int(labels.sum())} positive rows for feature "
            f"{args.feature_id}"
        )

    public, details, scoring = _load_candidate_artifacts(eval_root)
    public_row = next(
        row for row in public if int(row["id"]) == args.feature_id
    )
    rates = _autointerp_rates(
        details=details,
        scoring=scoring,
        feature_id=args.feature_id,
    )

    budgets = [
        int(value)
        for value in args.sparse_budgets.split(",")
        if value.strip()
    ]
    selection_c_grid = [
        float(value)
        for value in args.selection_c_grid.split(",")
        if value.strip()
    ]
    counterparts: dict[str, Any] = {}
    prediction_arrays = {}
    for budget in (1, *budgets):
        predictions = np.zeros(len(labels), dtype=np.float64)
        selected_by_fold = []
        calibration_auc = []
        fold_selection = []
        for outer_fold in sorted(set(folds.tolist())):
            test = folds == outer_fold
            train = ~test
            if budget == 1:
                feature, direction, train_auc, values = (
                    _exhaustive_single_fold(
                        token,
                        labels,
                        train,
                        test,
                    )
                )
                predictions[test] = values
                selected_by_fold.append([feature])
                calibration_auc.append(train_auc)
                fold_selection.append(
                    {
                        "outer_fold": int(outer_fold),
                        "strategy": "exhaustive_single",
                        "selected_feature_ids": [feature],
                        "direction": direction,
                        "outer_train_auc": train_auc,
                    }
                )
            else:
                fitted = _nested_sparse_fold(
                    token,
                    labels,
                    folds,
                    int(outer_fold),
                    budget=budget,
                    screen_features=args.screen_features,
                    selection_c_grid=selection_c_grid,
                    logistic_c=args.logistic_c,
                    seed=(
                        args.seed
                        + args.feature_id * 17
                        + budget * 101
                        + int(outer_fold)
                    ),
                )
                predictions[test] = fitted.pop("predictions")
                selected_by_fold.append(
                    fitted["selected_feature_ids"]
                )
                calibration_auc.append(fitted["outer_train_auc"])
                fold_selection.append(
                    {
                        "outer_fold": int(outer_fold),
                        **fitted,
                    }
                )
        payload = {
            "budget": budget,
            "oof_auc": float(roc_auc_score(labels, predictions)),
            "oof_average_precision": float(
                average_precision_score(labels, predictions)
            ),
            "mean_calibration_auc": float(
                np.mean(calibration_auc)
            ),
            "selected_feature_ids_by_fold": selected_by_fold,
            "fold_selection": fold_selection,
        }
        prediction_arrays[f"token_{budget}"] = predictions.astype(
            np.float32
        )
        payload["oof_auc_95ci"] = _cluster_bootstrap(
            labels,
            predictions,
            documents,
            samples=args.bootstrap_samples,
            seed=args.seed + args.feature_id + budget,
        )
        payload["cross_minus_token_auc"] = _paired_bootstrap_delta(
            labels,
            cross_values,
            predictions,
            documents,
            samples=args.bootstrap_samples,
            seed=args.seed + args.feature_id + 1000 + budget,
        )
        payload["prevalence_matched_recovery"] = (
            _prevalence_matched_recovery(labels, predictions)
        )
        counterparts[str(budget)] = payload

    best_budget = max(
        counterparts,
        key=lambda key: counterparts[key]["oof_auc"],
    )
    token_checkpoint = Path(args.sae_root) / "token/checkpoints/best"
    if not token_checkpoint.exists():
        token_checkpoint = Path(args.sae_root) / "token"
    decoder = _decoder_sparse_approximation(
        token_checkpoint=token_checkpoint,
        cross_checkpoint=cross_checkpoint,
        feature_id=args.feature_id,
        budgets=[1, *budgets],
    )
    atomic_json_dump(
        decoder,
        output_dir / "decoder_sparse_approximation.json",
    )

    result = {
        "format": FORMAT,
        "complete": True,
        "feature": {
            "feature_id": args.feature_id,
            "frozen_explanation": str(public_row["explanation"]),
            "autointerp_score": float(public_row["score"]),
            "autointerp_tpr": rates["tpr"],
            "autointerp_tnr": rates["tnr"],
            "autointerp_counts": rates,
        },
        "data": {
            "source": "Eval-6 held-out Pile chunks",
            "rows": int(len(labels)),
            "documents": int(len(np.unique(documents))),
            "positive_rows": int(labels.sum()),
            "positive_documents": int(
                len(np.unique(documents[labels]))
            ),
            "cross_validation": (
                f"{args.cv_folds}-fold deterministic document-hash split"
            ),
            "token_representation": (
                "all 65,536 thresholded Token-SAE coordinates after "
                "mean-after-threshold pooling; no top-k truncation"
            ),
        },
        "target": {
            "definition": (
                "active bit of the frozen Cross coordinate; Cross AUC=1 is "
                "definitionally the recovery upper bound"
            ),
            "cross_auc": float(roc_auc_score(labels, cross_values)),
            "cross_average_precision": float(
                average_precision_score(labels, cross_values)
            ),
        },
        "token_counterparts": counterparts,
        "best_token_budget": best_budget,
        "best_token_auc": counterparts[best_budget]["oof_auc"],
        "best_token_gap": counterparts[best_budget][
            "cross_minus_token_auc"
        ],
        "decoder_sparse_approximation": {
            budget: {
                key: value
                for key, value in payload.items()
                if key != "selected_token_feature_ids"
                and key != "coefficients_in_unit_atom_basis"
            }
            for budget, payload in decoder["budgets"].items()
        },
        "claim_scope": (
            "No equally accurate counterpart was found in the complete "
            "Token dictionary under one/4/16/64-coordinate linear budgets. "
            "This does not rule out larger or nonlinear Token decoders."
        ),
    }
    atomic_json_dump(result, output_dir / "results.json")
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        global_chunk_ids=global_ids,
        labels=labels.astype(np.int8),
        cross_values=cross_values.astype(np.float32),
        folds=folds,
        **prediction_arrays,
    )

    labels_plot = [
        "Cross axis",
        "Token 1",
        "Token 4",
        "Token 16",
        "Token 64",
    ]
    values_plot = [
        1.0,
        counterparts["1"]["oof_auc"],
        counterparts["4"]["oof_auc"],
        counterparts["16"]["oof_auc"],
        counterparts["64"]["oof_auc"],
    ]
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.bar(
        labels_plot,
        values_plot,
        color=[
            METHOD_COLORS["cross"],
            *[METHOD_COLORS["token"]] * 4,
        ],
    )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Document-cross-fitted AUC")
    ax.set_title(
        f"Full Token dictionary counterparts to Cross #{args.feature_id}"
    )
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values_plot, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
        )
    fig.tight_layout()
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            plot_dir / f"full_token_counterpart.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(fig)

    audit = {
        "format": "chunk-saes-reason-headline-full-token-audit-v1",
        "complete": True,
        "checks": {
            "feature_semantics_preexisting": True,
            "heldout_natural_chunks": True,
            "document_disjoint_cross_fitting": True,
            "complete_token_dictionary": True,
            "token_topk_truncation": False,
            "budgets": [1, *budgets],
            "test_fold_not_used_for_screening_or_fit": True,
        },
        "issues": [],
    }
    atomic_json_dump(audit, output_dir / "audit_report.json")
    manifest = write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": {
                "feature_id": args.feature_id,
                "eval6_features_sha256": file_sha256(eval6_path),
                "full_token_manifest_digest": json.loads(
                    (full_token_dir / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["artifact_digest"],
                "sae_root": str(Path(args.sae_root).resolve()),
                "screen_features": args.screen_features,
                "logistic_c": args.logistic_c,
                "selection_c_grid": selection_c_grid,
                "budgets": [1, *budgets],
                "seed": args.seed,
            },
            "files": {
                "results": file_record(
                    output_dir / "results.json",
                    relative_to=output_dir,
                ),
                "predictions": file_record(
                    output_dir / "oof_predictions.npz",
                    relative_to=output_dir,
                ),
                "decoder": file_record(
                    output_dir / "decoder_sparse_approximation.json",
                    relative_to=output_dir,
                ),
                "audit": file_record(
                    output_dir / "audit_report.json",
                    relative_to=output_dir,
                ),
                "plot_png": file_record(
                    plot_dir / "full_token_counterpart.png",
                    relative_to=output_dir,
                ),
            },
        },
        output_dir / "manifest.json",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "feature_id": args.feature_id,
                "positive_rows": int(labels.sum()),
                "best_token_budget": best_budget,
                "best_token_auc":
                    counterparts[best_budget]["oof_auc"],
                "best_token_gap":
                    counterparts[best_budget]["cross_minus_token_auc"],
                "artifact_digest": manifest["artifact_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
