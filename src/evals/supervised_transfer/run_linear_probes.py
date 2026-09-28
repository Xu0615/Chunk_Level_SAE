#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from chunk_saes.artifacts import (
    ensure_reusable_artifact,
    file_record,
    load_artifact_manifest,
    write_artifact_manifest,
)
from chunk_saes.evaluation_protocol import (
    FIXED_CHUNK_REPRESENTATION_PROTOCOL,
    LEGACY_VARIABLE_LENGTH_REPRESENTATION_PROTOCOL,
    TOKEN_AGGREGATION,
)
from chunk_saes.utils import atomic_json_dump


PROBE_RESULT_FORMAT = "chunk-saes-linear-probe-results-v2"
CHANCE_ACCURACY = 1.0 / 8.0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frozen-SAE linear probes and sparsity/label/OOD reports.")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--feature-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feature-budgets", default="16,64,256")
    p.add_argument("--low-label-budgets", default="1,2,4,8,16,64,256")
    p.add_argument("--low-label-seeds", default="0,1,2,3,4")
    p.add_argument("--classifier-c", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=20260815)
    p.add_argument("--dict-size", type=int, default=65536)
    p.add_argument("--token-match-reference", choices=["cross", "mean"], default="cross")
    p.add_argument(
        "--token-match-k",
        type=int,
        default=0,
        help="Fixed token aggregation K_eval; 0 matches the reference chunk SAE's train mean nnz.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(root / f"features-{split}.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def csr_from_arrays(
    indices: np.ndarray,
    values: np.ndarray,
    nnz: np.ndarray,
    dict_size: int,
    max_nnz: int | None = None,
) -> sparse.csr_matrix:
    if max_nnz is not None:
        if max_nnz <= 0:
            raise ValueError("max_nnz must be positive")
        indices = indices[:, :max_nnz]
        values = values[:, :max_nnz]
        nnz = np.minimum(nnz, max_nnz)
    rows = np.repeat(np.arange(len(indices)), nnz.astype(np.int64))
    cols = indices[indices >= 0].astype(np.int64)
    vals = values[indices >= 0].astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(indices), dict_size), dtype=np.float32)


def fit_classifier(x_train, y_train, c: float, max_iter: int, n_jobs: int):
    # L-BFGS supports CSR inputs and is deterministic; using one solver for raw
    # and sparse representations avoids solver-specific comparison effects.
    model = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs", n_jobs=n_jobs, random_state=0)
    model.fit(x_train, y_train)
    if int(model.n_iter_.max()) >= max_iter:
        raise RuntimeError(f"L-BFGS did not converge within max_iter={max_iter}")
    return model


def metrics(model, x, y) -> dict[str, float]:
    pred = model.predict(x)
    return metrics_from_predictions(pred, y)


def metrics_from_predictions(pred, y) -> dict[str, float]:
    classes = np.unique(y)
    per_class = {
        str(label): float(np.mean(pred[y == label] == label))
        for label in classes
    }
    class_values = np.asarray(list(per_class.values()), dtype=np.float64)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_accuracy": per_class,
        "worst_class_accuracy": float(class_values.min()),
        "class_accuracy_std": float(class_values.std()),
    }


def feature_matrix(
    data: dict[str, np.ndarray],
    mode: str,
    aggregate: str,
    dict_size: int,
    token_match_k: int,
):
    if mode == "raw":
        return data["raw"].astype(np.float32)
    suffix = aggregate if mode in {"token", "temporal"} else "direct"
    return csr_from_arrays(
        data[f"{mode}_{suffix}_indices"],
        data[f"{mode}_{suffix}_values"],
        data[f"{mode}_{suffix}_nnz"],
        dict_size,
        max_nnz=token_match_k if mode in {"token", "temporal"} else None,
    )


def select_features(x, y, count: int) -> np.ndarray:
    scores, _ = chi2(x, y)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float64).max)
    return np.argsort(scores)[::-1][: min(count, x.shape[1])]


def evaluate_representation(name, x_train, y_train, x_val, y_val, x_test, y_test, x_ood, y_ood, args, budgets):
    classifier = fit_classifier(x_train, y_train, args.classifier_c, args.max_iter, args.n_jobs)
    test_predictions = classifier.predict(x_test)
    ood_predictions = classifier.predict(x_ood)
    result = {
        "representation": name,
        "full": metrics_from_predictions(test_predictions, y_test),
        "ood": metrics_from_predictions(ood_predictions, y_ood),
        "acc_at": {},
        "low_label": {},
    }
    for budget in budgets:
        if not sparse.issparse(x_train):
            continue
        selected = select_features(x_train, y_train, budget)
        budget_classifier = fit_classifier(x_train[:, selected], y_train, args.classifier_c, args.max_iter, args.n_jobs)
        result["acc_at"][str(budget)] = metrics(budget_classifier, x_test[:, selected], y_test)
    classes = np.unique(y_train)
    low_budgets = [int(value) for value in args.low_label_budgets.split(",") if value]
    low_seeds = [int(value) for value in args.low_label_seeds.split(",") if value]
    for budget in low_budgets:
        scores = []
        for seed in low_seeds:
            rng = np.random.default_rng(seed)
            selected_rows = np.concatenate(
                [rng.choice(np.flatnonzero(y_train == label), size=min(budget, np.sum(y_train == label)), replace=False) for label in classes]
            )
            low_classifier = fit_classifier(x_train[selected_rows], y_train[selected_rows], args.classifier_c, args.max_iter, args.n_jobs)
            scores.append(metrics(low_classifier, x_test, y_test)["accuracy"])
        result["low_label"][str(budget)] = {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "seeds": scores,
        }
    return result, {
        "test": test_predictions,
        "ood": ood_predictions,
    }


def _bootstrap_mean_difference(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    differences = np.asarray(differences, dtype=np.float64).reshape(-1)
    if differences.size == 0:
        raise ValueError("bootstrap differences must be non-empty")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        count = min(512, samples - start)
        indices = rng.integers(
            0,
            differences.size,
            size=(count, differences.size),
        )
        estimates[start : start + count] = differences[indices].mean(axis=1)
    return {
        "point": float(differences.mean()),
        "95ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
    }


def paired_comparisons(
    results: dict,
    predictions: dict[str, dict[str, np.ndarray]],
    *,
    test_labels: np.ndarray,
    ood_labels: np.ndarray,
    chosen_token: str,
    low_label_budgets: list[int],
    samples: int,
    seed: int,
) -> dict:
    names = {
        "token": f"token_sae_{chosen_token}",
        "temporal": "temporal_sae_mean",
        "mean": "mean_chunk_sae",
        "cross": "cross_chunk_sae",
    }
    available_modes = tuple(
        mode
        for mode, name in names.items()
        if name in results.get("representations", {})
    )
    output = {}
    cross_name = names["cross"]
    for reference_index, reference in enumerate(
        mode
        for mode in available_modes
        if mode != "cross"
    ):
        reference_name = names[reference]
        row = {}
        for split_index, (split, labels) in enumerate(
            (("test", test_labels), ("ood", ood_labels))
        ):
            cross_correct = (
                predictions[cross_name][split] == labels
            ).astype(np.float64)
            reference_correct = (
                predictions[reference_name][split] == labels
            ).astype(np.float64)
            row[f"{split}_accuracy"] = _bootstrap_mean_difference(
                cross_correct - reference_correct,
                samples=samples,
                seed=seed + reference_index * 100 + split_index,
            )
        for budget_index, budget in enumerate(low_label_budgets):
            cross_scores = np.asarray(
                results["representations"][cross_name]["low_label"][
                    str(budget)
                ]["seeds"],
                dtype=np.float64,
            )
            reference_scores = np.asarray(
                results["representations"][reference_name]["low_label"][
                    str(budget)
                ]["seeds"],
                dtype=np.float64,
            )
            row[f"low_label_{budget}"] = _bootstrap_mean_difference(
                cross_scores - reference_scores,
                samples=samples,
                seed=seed + 1_000 + reference_index * 100 + budget_index,
            )
        output[f"cross_minus_{reference}"] = row
    return output


def _chance_normalized_accuracy(value: float) -> float:
    return (float(value) - CHANCE_ACCURACY) / (1.0 - CHANCE_ACCURACY)


def high_level_summary(
    results: dict,
    chosen_token: str,
    low_label_budgets: list[int],
) -> dict:
    """Summarize the objective, non-LLM high-level transfer evidence.

    Full in-distribution accuracy is retained as a representation sanity check.
    The primary high-level score averages chance-normalized full, time-OOD, and
    low-label accuracies.  These tasks reward domain information that survives
    changes in examples, label availability, and publication year rather than
    memorization of an individual token.
    """

    names = {
        "token": f"token_sae_{chosen_token}",
        "temporal": "temporal_sae_mean",
        "mean": "mean_chunk_sae",
        "cross": "cross_chunk_sae",
    }
    methods = {}
    for mode, name in names.items():
        if name not in results.get("representations", {}):
            continue
        row = results["representations"][name]
        low_label_curve = np.asarray(
            [
                float(row["low_label"][str(budget)]["mean_accuracy"])
                for budget in low_label_budgets
            ],
            dtype=np.float64,
        )
        log_budgets = np.log2(
            np.asarray(low_label_budgets, dtype=np.float64)
        )
        if len(low_label_budgets) > 1:
            integrate = (
                np.trapezoid
                if hasattr(np, "trapezoid")
                else np.trapz
            )
            low_label_auc = float(
                integrate(low_label_curve, log_budgets)
                / (log_budgets[-1] - log_budgets[0])
            )
        else:
            low_label_auc = float(low_label_curve[0])
        accuracies = {
            "full_accuracy": float(row["full"]["accuracy"]),
            "ood_accuracy": float(row["ood"]["accuracy"]),
            **{
                f"low_label_{budget}": float(
                    row["low_label"][str(budget)]["mean_accuracy"]
                )
                for budget in low_label_budgets
            },
        }
        normalized = {
            key: _chance_normalized_accuracy(value)
            for key, value in accuracies.items()
        }
        methods[mode] = {
            "representation": name,
            **accuracies,
            "chance_normalized": normalized,
            "high_level_transfer_score": float(
                np.mean(list(normalized.values()))
            ),
            "low_label_auc": low_label_auc,
            "chance_normalized_low_label_auc": (
                _chance_normalized_accuracy(low_label_auc)
            ),
            "ood_retention": float(row["ood"]["accuracy"])
            / max(float(row["full"]["accuracy"]), 1e-12),
            "worst_class_ood_accuracy": float(
                row["ood"]["worst_class_accuracy"]
            ),
            "ood_class_accuracy_std": float(
                row["ood"]["class_accuracy_std"]
            ),
        }
    comparisons = {
        f"cross_minus_{reference}": {
            metric: float(
                methods["cross"][metric] - methods[reference][metric]
            )
            for metric in (
                "full_accuracy",
                "ood_accuracy",
                *(
                    f"low_label_{budget}"
                    for budget in low_label_budgets
                ),
                "high_level_transfer_score",
                "low_label_auc",
                "chance_normalized_low_label_auc",
                "ood_retention",
                "worst_class_ood_accuracy",
                "ood_class_accuracy_std",
            )
        }
        for reference in (
            mode for mode in methods if mode != "cross"
        )
    }
    return {
        "definition": (
            "Mean chance-normalized accuracy over full train, all configured "
            "low-label budgets, "
            "examples per class, and first-submission-year OOD evaluation."
        ),
        "chance_accuracy": CHANCE_ACCURACY,
        "low_label_budgets": low_label_budgets,
        "methods": methods,
        "comparisons": comparisons,
    }


def main() -> None:
    args = parser().parse_args()
    root = Path(args.features_dir)
    feature_manifest = load_artifact_manifest(
        args.feature_manifest,
        expected_format="chunk-saes-probe-features-v2",
        verify_files=True,
    )
    if Path(args.feature_manifest).parent.resolve() != root.resolve():
        raise ValueError("--feature-manifest must belong to --features-dir")
    representation_protocol = feature_manifest.get("representation_protocol")
    if representation_protocol is None:
        representation_protocol = (
            feature_manifest.get("identity", {})
            .get("representation_protocol")
        )
    if not isinstance(representation_protocol, dict):
        raise ValueError(
            "probe feature manifest lacks an explicit representation protocol"
        )
    protocol_name = representation_protocol.get("name")
    if protocol_name not in {
        FIXED_CHUNK_REPRESENTATION_PROTOCOL,
        LEGACY_VARIABLE_LENGTH_REPRESENTATION_PROTOCOL,
    }:
        raise ValueError(f"unknown probe feature representation protocol: {protocol_name!r}")
    if representation_protocol.get("token_temporal_aggregation") != TOKEN_AGGREGATION:
        raise ValueError("probe feature protocol must use mean-after-threshold aggregation")
    budgets = [int(value) for value in args.feature_budgets.split(",") if value]
    low_budgets = [int(value) for value in args.low_label_budgets.split(",") if value]
    low_seeds = [int(value) for value in args.low_label_seeds.split(",") if value]
    train, val, test, ood = (
        load_split(root, split)
        for split in ("train", "validation", "test", "ood")
    )
    feature_widths = {
        str(mode): int(width)
        for mode, width in (
            feature_manifest.get("feature_widths")
            or feature_manifest.get("identity", {}).get("feature_widths")
            or {}
        ).items()
    }
    available_modes = tuple(
        mode
        for mode in ("token", "temporal", "mean", "cross")
        if (
            f"{mode}_mean_indices" in train
            if mode in {"token", "temporal"}
            else f"{mode}_direct_indices" in train
        )
    )
    for mode in available_modes:
        feature_widths.setdefault(mode, int(args.dict_size))
    mismatched_widths = {
        mode: width
        for mode, width in feature_widths.items()
        if int(width) != int(args.dict_size)
    }
    if mismatched_widths:
        raise ValueError(
            "probe features do not use the complete common dictionary: "
            f"{mismatched_widths}; rerun extract_probe_features.py"
        )
    identity = {
        "feature_artifact_digest": feature_manifest["artifact_digest"],
        "dict_size": args.dict_size,
        "feature_widths": feature_widths,
        "feature_budgets": budgets,
        "low_label_budgets": low_budgets,
        "low_label_seeds": low_seeds,
        "classifier": {
            "type": "multinomial logistic regression",
            "solver": "lbfgs",
            "c": args.classifier_c,
            "max_iter": args.max_iter,
            "n_jobs": args.n_jobs,
            "random_state": 0,
        },
        "token_match_reference": args.token_match_reference,
        "token_match_k": args.token_match_k,
        "aggregation_protocol": TOKEN_AGGREGATION,
        "representation_protocol": representation_protocol,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    output_path = Path(args.output)
    manifest_path = output_path.with_name("linear_probe_manifest.json")
    if manifest_path.exists() and not args.overwrite:
        existing = ensure_reusable_artifact(
            manifest_path,
            expected_format=PROBE_RESULT_FORMAT,
            expected_identity=identity,
        )
        if existing is not None:
            print(output_path.read_text(encoding="utf-8"), flush=True)
            return
    if output_path.exists() and not args.overwrite:
        raise ValueError(
            f"probe result exists without matching verified provenance: {output_path}; "
            "use --overwrite"
        )
    labels = np.unique(train["labels"])
    # All SAE checkpoints share a dictionary width; infer it from the stored indices.
    dict_size = args.dict_size
    reference_nnz = train[f"{args.token_match_reference}_direct_nnz"]
    token_match_k = args.token_match_k or max(1, round(float(reference_nnz.mean())))
    token_candidate_k = train["token_mean_indices"].shape[1]
    if token_match_k > token_candidate_k:
        raise ValueError(
            f"token_match_k={token_match_k} exceeds stored candidate width={token_candidate_k}"
        )
    matrices = {}
    matrix_specs = [
        ("raw", "direct"),
        ("token", "mean"),
        ("mean", "direct"),
        ("cross", "direct"),
    ]
    if "temporal" in available_modes:
        matrix_specs.insert(2, ("temporal", "mean"))
    for mode, aggregate in matrix_specs:
        matrices[(mode, aggregate)] = tuple(
            feature_matrix(
                data,
                mode,
                aggregate,
                feature_widths.get(mode, dict_size),
                token_match_k,
            )
            for data in (train, val, test, ood)
        )

    # Aggregation is pre-registered before looking at labels.  Validation must
    # not choose whichever aggregation happens to win on this benchmark.
    chosen_token = "mean"
    token_validation = {
        "selection": "predeclared",
        "chosen": chosen_token,
        "aggregation": "mean_after_threshold",
    }
    results = {
        "token_validation_selection": token_validation,
        "chosen_token_aggregation": chosen_token,
        "representations": {},
    }
    prediction_sets: dict[str, dict[str, np.ndarray]] = {}
    ordered = [
        ("raw_mean_hidden", "raw", "direct"),
        ("token_sae_mean", "token", "mean"),
        *(
            (
                ("temporal_sae_mean", "temporal", "mean"),
            )
            if "temporal" in available_modes
            else ()
        ),
        ("mean_chunk_sae", "mean", "direct"),
        ("cross_chunk_sae", "cross", "direct"),
    ]
    for name, mode, aggregate in ordered:
        xtr, xval, xtest, xood = matrices[(mode, aggregate)]
        result, predictions = evaluate_representation(
            name,
            xtr,
            train["labels"],
            xval,
            val["labels"],
            xtest,
            test["labels"],
            xood,
            ood["labels"],
            args,
            budgets,
        )
        results["representations"][name] = result
        prediction_sets[name] = predictions
    results["high_level_transfer"] = high_level_summary(
        results,
        chosen_token,
        low_budgets,
    )
    results["paired_comparisons"] = paired_comparisons(
        results,
        prediction_sets,
        test_labels=test["labels"],
        ood_labels=ood["labels"],
        chosen_token=chosen_token,
        low_label_budgets=low_budgets,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    results["metadata"] = {
        "classifier": "multinomial logistic regression",
        "classifier_max_iter": args.max_iter,
        "classifier_n_jobs": args.n_jobs,
        "frozen": True,
        "dict_size": dict_size,
        "feature_widths": feature_widths,
        "feature_budgets": budgets,
        "low_label_budgets": [int(value) for value in args.low_label_budgets.split(",") if value],
        "low_label_seeds": low_seeds,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "feature_artifact_digest": feature_manifest["artifact_digest"],
        "representation_protocol": representation_protocol,
        "labels": labels.tolist(),
        "ood_split": "ArXiv first-submission year >= configured OOD year",
        "primary_question": (
            "How much linearly readable high-level domain information survives "
            "low-label learning and a publication-year distribution shift?"
        ),
        "sparsity_matching": {
            "token_match_reference": args.token_match_reference,
            "reference_train_mean_nnz": float(reference_nnz.mean()),
            "token_eval_k": token_match_k,
            "selection_split": "train",
        },
        "mean_nnz": {
            f"{mode}_{suffix}": {
                split_name: float(
                    np.minimum(data[f"{mode}_{suffix}_nnz"], token_match_k).mean()
                    if mode in {"token", "temporal"}
                    else data[f"{mode}_{suffix}_nnz"].mean()
                )
                for split_name, data in (("train", train), ("validation", val), ("test", test), ("ood", ood))
            }
            for mode, suffix in (
                ("token", "mean"),
                *(
                    (
                        ("temporal", "mean"),
                    )
                    if "temporal" in available_modes
                    else ()
                ),
                ("mean", "direct"),
                ("cross", "direct"),
            )
        },
    }
    atomic_json_dump(results, output_path)
    write_artifact_manifest(
        {
            "format": PROBE_RESULT_FORMAT,
            "complete": True,
            "identity": identity,
            "chosen_token_aggregation": chosen_token,
            "token_eval_k": token_match_k,
            "representation_protocol": representation_protocol,
            "files": {
                "results": file_record(output_path, relative_to=output_path.parent),
            },
        },
        manifest_path,
    )
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
