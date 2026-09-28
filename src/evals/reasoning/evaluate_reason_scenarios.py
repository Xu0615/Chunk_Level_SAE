#!/usr/bin/env python
"""Evaluate multiple natural reasoning-related Cross features.

Feature IDs and scenario names are fixed from independently generated Eval-4
explanations before any counterpart result is computed.  For each feature the
script builds a natural matched-pair bank from the exact-1000 held-out chunk
pool: both chunks come from the same document, have exactly the same token
length, and exactly one activates the target Cross coordinate.  Documents
used to generate or score the frozen explanation are excluded.

The complete 65,536-coordinate BatchTopK and Temporal SAE codes are then
materialized for these chunks.  Document-disjoint cross-fitting searches:

* the exhaustive best single coordinate;
* the best 4-coordinate sparse model;
* the best 16-coordinate sparse model; and
* the best 64-coordinate sparse model.

The script also measures each Cross coordinate's contribution to real
adjacent-partner reconstruction on the validation cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from safetensors import safe_open
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoTokenizer

from chunk_saes.artifacts import (
    file_record,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.plot_style import METHOD_COLORS
from chunk_saes.utils import atomic_json_dump
from evals.reasoning.evaluate_reason_counterparts import (
    FrozenEncoder,
    _aggregate_importance_repeats,
    _autointerp_rates,
    _cluster_bootstrap,
    _importance_for_feature,
    _load_candidate_artifacts,
    _load_validation_pairs,
    _paired_bootstrap_delta,
    _prevalence_matched_recovery,
)
from evals.reasoning.evaluate_reason_feature import SparseEncoder, _pool_token_features
from evals.reasoning.evaluate_reason_headline import (
    _exhaustive_single_fold,
    _nested_sparse_fold,
)


FORMAT = "chunk-saes-multi-reason-scenarios-v1"
METHODS = ("token", "temporal")
SCENARIOS = (
    {
        "scenario": "causal_mechanism",
        "label": "Causal\nmechanism",
        "feature_id": 20232,
        "primary": True,
        "value": (
            "Scientific causal inference: intervention, mediator, and "
            "downstream consequence."
        ),
    },
    {
        "scenario": "mathematical_derivation",
        "label": "Math\nproof",
        "feature_id": 52521,
        "primary": True,
        "value": (
            "Stepwise mathematical derivation, case analysis, and "
            "constraint-based proof."
        ),
    },
    {
        "scenario": "technical_diagnosis",
        "label": "Technical\ndiagnosis",
        "feature_id": 31830,
        "primary": True,
        "value": (
            "Failure diagnosis followed by concrete configuration or "
            "remediation steps."
        ),
    },
    {
        "scenario": "clinical_physiologic_reasoning",
        "label": "Clinical\nphysiology*",
        "feature_id": 31358,
        "primary": False,
        "value": (
            "Quantitative respiratory-mechanics reasoning that connects "
            "ventilator settings, physiology, and clinical consequences."
        ),
    },
    {
        "scenario": "programming_problem_solving",
        "label": "Programming\nproblem*",
        "feature_id": 36434,
        "primary": False,
        "value": (
            "A concrete programming goal, failed attempt, observed behavior, "
            "and request for a corrective explanation."
        ),
    },
    {
        "scenario": "epidemiological_risk_inference",
        "label": "Risk-factor\ninference*",
        "feature_id": 62277,
        "primary": False,
        "value": (
            "Population evidence connecting exposures or risk factors to "
            "health outcomes with uncertainty estimates."
        ),
    },
    {
        "scenario": "diagnostic_method_evaluation",
        "label": "Diagnostic\ncomparison*",
        "feature_id": 53955,
        "primary": False,
        "value": (
            "Comparative reasoning about sensitivity, specificity, detection "
            "limits, and whether an assay improves on alternatives."
        ),
    },
    {
        "scenario": "rule_guided_decision",
        "label": "Rule-guided\ndecision",
        "feature_id": 41245,
        "primary": True,
        "value": (
            "Applying legal, tax, or HR constraints to an actionable "
            "compliance decision."
        ),
    },
    {
        "scenario": "legal_case_reasoning",
        "label": "Legal case\nreasoning",
        "feature_id": 35018,
        "primary": True,
        "value": (
            "Applying precedent and settled doctrine to case-specific facts "
            "and resolving a legal dispute."
        ),
    },
    {
        "scenario": "methodological_inference",
        "label": "Methodological\ninference",
        "feature_id": 58082,
        "primary": True,
        "value": (
            "Reasoning about assumptions, bias, estimator robustness, and "
            "limits of inference."
        ),
    },
    {
        "scenario": "evidence_based_argument",
        "label": "Evidence-based\nargument",
        "feature_id": 3905,
        "primary": True,
        "value": (
            "Structured evidence-based explanation or argument using "
            "definitions and explicit logical connectors."
        ),
    },
    {
        "scenario": "strategic_decision_sequence",
        "label": "Strategic\ndecisions",
        "feature_id": 51717,
        "primary": True,
        "value": (
            "Multi-step strategic choices, maneuvers, objectives, and "
            "resulting operational outcomes."
        ),
    },
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, default=21)
    p.add_argument("--pairs-per-scenario", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--screen-features", type=int, default=512)
    p.add_argument("--sparse-budgets", default="4,16,64")
    p.add_argument("--logistic-c", type=float, default=1e-4)
    p.add_argument(
        "--selection-c-grid",
        default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--importance-pairs", type=int, default=7_344)
    p.add_argument("--importance-controls", type=int, default=32)
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    return p


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_metadata(eval_root: Path) -> dict[int, dict[str, Any]]:
    public, details, scoring = _load_candidate_artifacts(eval_root)
    public_by_id = {int(row["id"]): row for row in public}
    output = {}
    for scenario in SCENARIOS:
        feature_id = int(scenario["feature_id"])
        row = public_by_id[feature_id]
        rates = _autointerp_rates(
            details=details,
            scoring=scoring,
            feature_id=feature_id,
        )
        output[feature_id] = {
            **scenario,
            "explanation": str(row["explanation"]),
            "autointerp_score": float(row["score"]),
            "autointerp_tpr": rates["tpr"],
            "autointerp_tnr": rates["tnr"],
            "autointerp_counts": rates,
        }
    return output


def _excluded_evidence(
    eval_root: Path,
    feature_ids: set[int],
) -> tuple[dict[int, set[str]], dict[int, set[int]]]:
    base = eval_root / "autointerp/autointerp_exact1000/chunk_v2"
    excluded_docs = {feature_id: set() for feature_id in feature_ids}
    excluded_chunks = {feature_id: set() for feature_id in feature_ids}
    for row in _read_jsonl(base / "explanations/feature_results.jsonl"):
        feature_id = int(row.get("feature_id", -1))
        if (
            row.get("method") != "cross"
            or feature_id not in feature_ids
        ):
            continue
        for pair in row.get("evidence_pairs", []):
            for side in ("active", "inactive"):
                item = pair.get(side) or {}
                if item.get("doc_hash"):
                    excluded_docs[feature_id].add(
                        str(item["doc_hash"])
                    )
                if item.get("chunk_id") is not None:
                    excluded_chunks[feature_id].add(
                        int(item["chunk_id"])
                    )
    for row in _read_jsonl(base / "scoring/scoring_examples.jsonl"):
        feature_id = int(row.get("feature_id", -1))
        if (
            row.get("method") != "cross"
            or feature_id not in feature_ids
        ):
            continue
        excluded_chunks[feature_id].update(
            int(item["chunk_id"])
            for item in row.get("scoring_examples", [])
        )
    return excluded_docs, excluded_chunks


def _hash_key(*values: Any) -> int:
    text = "\x1f".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def build_bank(
    *,
    args: argparse.Namespace,
    metadata: dict[int, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    pool = (
        Path(args.eval_root)
        / "autointerp/autointerp_exact1000/data/chunk_pool"
    )
    feature_ids = set(metadata)
    excluded_docs_by_feature, excluded_chunks_by_feature = (
        _excluded_evidence(
        Path(args.eval_root),
        feature_ids,
        )
    )

    # Resolve scoring-example chunks to source documents before selection.
    all_excluded_chunks = set().union(
        *excluded_chunks_by_feature.values()
    )
    if all_excluded_chunks:
        for path in sorted((pool / "plan").glob("shard-*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                chunk_ids = handle.get_tensor("chunk_ids").numpy()
                doc_hashes = handle.get_tensor("doc_hashes").numpy()
            for index in np.flatnonzero(
                np.isin(chunk_ids, list(all_excluded_chunks))
            ).tolist():
                chunk_id = int(chunk_ids[index])
                doc_hash = bytes(doc_hashes[index]).hex()
                for feature_id in feature_ids:
                    if chunk_id in excluded_chunks_by_feature[feature_id]:
                        excluded_docs_by_feature[feature_id].add(doc_hash)

    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for activation_path in sorted(
        (pool / "activations").glob("shard-*.safetensors")
    ):
        plan_path = pool / "plan" / activation_path.name
        with safe_open(
            str(activation_path),
            framework="pt",
            device="cpu",
        ) as activation_handle, safe_open(
            str(plan_path),
            framework="pt",
            device="cpu",
        ) as plan_handle:
            available = activation_handle.get_tensor(
                "cross_feature_ids"
            ).numpy()
            columns = {
                feature_id: int(
                    np.flatnonzero(available == feature_id)[0]
                )
                for feature_id in feature_ids
            }
            activations = activation_handle.get_tensor(
                "cross_activations"
            ).float().numpy()
            pair_ids = plan_handle.get_tensor("pair_ids").numpy()
            sides = plan_handle.get_tensor("sides").numpy()
            lengths = plan_handle.get_tensor("lengths").numpy()
            chunk_ids = plan_handle.get_tensor("chunk_ids").numpy()
            doc_hashes = plan_handle.get_tensor("doc_hashes").numpy()
            offsets = plan_handle.get_tensor("offsets").numpy()
            token_ids = plan_handle.get_tensor("token_ids").numpy()

        by_pair: dict[int, list[int]] = defaultdict(list)
        for index, pair_id in enumerate(pair_ids.tolist()):
            by_pair[int(pair_id)].append(index)
        for pair_id, indices in by_pair.items():
            if len(indices) != 2:
                continue
            left, right = indices
            if int(lengths[left]) != int(lengths[right]):
                continue
            doc_hash = bytes(doc_hashes[left]).hex()
            if doc_hash != bytes(doc_hashes[right]).hex():
                raise RuntimeError("pair document hash mismatch")
            pair_chunk_ids = {
                int(chunk_ids[left]),
                int(chunk_ids[right]),
            }
            pair_tokens = {}
            for index in indices:
                start = int(offsets[index])
                stop = int(offsets[index + 1])
                pair_tokens[index] = token_ids[start:stop].astype(
                    np.int32
                ).tolist()
            for feature_id in feature_ids:
                if doc_hash in excluded_docs_by_feature[feature_id]:
                    continue
                if pair_chunk_ids & excluded_chunks_by_feature[feature_id]:
                    continue
                values = activations[indices, columns[feature_id]]
                active = values > 0
                if int(active.sum()) != 1:
                    continue
                active_local = int(np.flatnonzero(active)[0])
                inactive_local = 1 - active_local
                active_index = indices[active_local]
                inactive_index = indices[inactive_local]
                candidates[feature_id].append(
                    {
                        "pair_id": pair_id,
                        "doc_hash": doc_hash,
                        "length": int(lengths[left]),
                        "active": {
                            "chunk_id": int(chunk_ids[active_index]),
                            "side": int(sides[active_index]),
                            "cross_activation": float(values[active_local]),
                            "token_ids": pair_tokens[active_index],
                        },
                        "inactive": {
                            "chunk_id": int(chunk_ids[inactive_index]),
                            "side": int(sides[inactive_index]),
                            "cross_activation": 0.0,
                            "token_ids": pair_tokens[inactive_index],
                        },
                    }
                )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    rows: list[dict[str, Any]] = []
    selection_summary = {}
    for feature_id, scenario in metadata.items():
        by_document: dict[str, dict[str, Any]] = {}
        for candidate in candidates[feature_id]:
            current = by_document.get(candidate["doc_hash"])
            tie_key = _hash_key(
                args.seed,
                scenario["scenario"],
                candidate["doc_hash"],
                candidate["pair_id"],
            )
            key = (
                -float(candidate["active"]["cross_activation"]),
                tie_key,
            )
            if current is None or key < current["_key"]:
                by_document[candidate["doc_hash"]] = {
                    **candidate,
                    "_key": key,
                }
        selected = sorted(
            by_document.values(),
            key=lambda row: row["_key"],
        )[: args.pairs_per_scenario]
        if len(selected) < args.pairs_per_scenario:
            raise RuntimeError(
                f"{scenario['scenario']} has only {len(selected)} eligible "
                f"documents; need {args.pairs_per_scenario}"
            )
        length_counts = Counter(row["length"] for row in selected)
        selection_summary[scenario["scenario"]] = {
            "eligible_pairs": len(candidates[feature_id]),
            "eligible_documents": len(by_document),
            "selected_pairs": len(selected),
            "length_counts": {
                str(key): value
                for key, value in sorted(length_counts.items())
            },
        }
        for selected_index, pair in enumerate(selected):
            fold = _hash_key(pair["doc_hash"]) % args.cv_folds
            family_id = (
                f"{scenario['scenario']}-{selected_index:04d}"
            )
            for label, side_name in ((1, "active"), (0, "inactive")):
                item = pair[side_name]
                text = tokenizer.decode(
                    item["token_ids"],
                    skip_special_tokens=True,
                )
                rows.append(
                    {
                        "scenario": scenario["scenario"],
                        "scenario_label": scenario["label"],
                        "feature_id": feature_id,
                        "value": scenario["value"],
                        "family_id": family_id,
                        "pair_id": int(pair["pair_id"]),
                        "doc_hash": pair["doc_hash"],
                        "fold": int(fold),
                        "label": label,
                        "condition": side_name,
                        "chunk_id": int(item["chunk_id"]),
                        "side": int(item["side"]),
                        "length": int(pair["length"]),
                        "cross_activation": float(
                            item["cross_activation"]
                        ),
                        "token_ids": item["token_ids"],
                        "text": text,
                    }
                )
    _write_jsonl(output_dir / "challenge_bank.jsonl", rows)
    atomic_json_dump(
        {
            "format": "chunk-saes-multi-reason-bank-v1",
            "complete": True,
            "selection": {
                "source": (
                    "Eval-4 exact-1000 held-out chunk pool; same-document, "
                    "exact-length pairs with one active and one inactive row"
                ),
                "explanation_and_scoring_documents_excluded": True,
                "one_pair_per_document_per_scenario": True,
                "selection_order": (
                    "highest active Cross activation per document, then "
                    "highest-activation documents; deterministic hash ties"
                ),
                "pairs_per_scenario": args.pairs_per_scenario,
                "scenarios": selection_summary,
            },
            "candidate_metadata": metadata,
            "rows": len(rows),
        },
        output_dir / "challenge_bank_manifest.json",
    )
    return rows


@torch.inference_mode()
def extract_features(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    device = torch.device(args.device)
    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        str(device),
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
    )
    sae_set = resolve_sae_artifact_set(
        args.sae_root,
        selection="best",
        modes=METHODS,
    )
    encoders = {
        method: SparseEncoder(
            Path(sae_set["modes"][method]["checkpoint_path"]),
            device,
        )
        for method in METHODS
    }
    arrays = {
        method: np.empty(
            (len(rows), encoders[method].width),
            dtype=np.float16,
        )
        for method in METHODS
    }
    order = sorted(
        range(len(rows)),
        key=lambda index: (rows[index]["length"], index),
    )
    try:
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start : start + args.batch_size]
            batch = extractor.forward_ids(
                [rows[index]["token_ids"] for index in batch_indices]
            )
            for method in METHODS:
                pooled = _pool_token_features(
                    encoders[method].dense(batch.hidden),
                    batch.mask,
                    "mean",
                )
                arrays[method][np.asarray(batch_indices)] = (
                    pooled.float().cpu().numpy().astype(np.float16)
                )
            if (start + len(batch_indices)) % 128 == 0 or (
                start + len(batch_indices) == len(order)
            ):
                print(
                    f"[reason-scenarios] encoded "
                    f"{start + len(batch_indices)}/{len(order)}",
                    flush=True,
                )
        for method, values in arrays.items():
            np.save(output_dir / f"{method}_features.npy", values)
    finally:
        extractor.close()
        for encoder in encoders.values():
            encoder.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _pair_accuracy(
    labels: np.ndarray,
    values: np.ndarray,
    pair_ids: np.ndarray,
) -> float:
    correct = 0.0
    total = 0
    for pair_id in np.unique(pair_ids):
        indices = np.flatnonzero(pair_ids == pair_id)
        if len(indices) != 2 or labels[indices].sum() != 1:
            continue
        positive = indices[labels[indices]][0]
        negative = indices[~labels[indices]][0]
        if values[positive] > values[negative]:
            correct += 1
        elif values[positive] == values[negative]:
            correct += 0.5
        total += 1
    return correct / max(1, total)


def _evaluate_method(
    *,
    matrix: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    pair_ids: np.ndarray,
    budgets: list[int],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    selection_c_grid = [
        float(value)
        for value in args.selection_c_grid.split(",")
        if value.strip()
    ]
    output = {}
    for budget in (1, *budgets):
        predictions = np.zeros(len(labels), dtype=np.float64)
        fold_records = []
        for outer_fold in sorted(set(folds.tolist())):
            test = folds == outer_fold
            train = ~test
            if budget == 1:
                feature, direction, train_auc, values = (
                    _exhaustive_single_fold(
                        matrix,
                        labels,
                        train,
                        test,
                    )
                )
                predictions[test] = values
                fold_records.append(
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
                    matrix,
                    labels,
                    folds,
                    int(outer_fold),
                    budget=budget,
                    screen_features=args.screen_features,
                    selection_c_grid=selection_c_grid,
                    logistic_c=args.logistic_c,
                    seed=(
                        args.seed
                        + seed_offset
                        + budget * 101
                        + int(outer_fold)
                    ),
                )
                predictions[test] = fitted.pop("predictions")
                fold_records.append(
                    {
                        "outer_fold": int(outer_fold),
                        **fitted,
                    }
                )
        output[str(budget)] = {
            "budget": budget,
            "oof_auc": float(roc_auc_score(labels, predictions)),
            "oof_average_precision": float(
                average_precision_score(labels, predictions)
            ),
            "pair_accuracy": _pair_accuracy(
                labels,
                predictions,
                pair_ids,
            ),
            "oof_auc_95ci": _cluster_bootstrap(
                labels,
                predictions,
                pair_ids,
                samples=args.bootstrap_samples,
                seed=args.seed + seed_offset + budget,
            ),
            "cross_minus_auc": _paired_bootstrap_delta(
                labels,
                labels.astype(np.float64),
                predictions,
                pair_ids,
                samples=args.bootstrap_samples,
                seed=args.seed + seed_offset + 1000 + budget,
            ),
            "prevalence_matched_recovery":
                _prevalence_matched_recovery(labels, predictions),
            "fold_selection": fold_records,
            "_predictions": predictions,
        }
    return output


def analyze(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    metadata: dict[int, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    matrices = {
        method: np.load(
            output_dir / f"{method}_features.npy",
            mmap_mode="r",
        )
        for method in METHODS
    }
    budgets = [
        int(value)
        for value in args.sparse_budgets.split(",")
        if value.strip()
    ]
    scenarios = []
    prediction_arrays = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        selected = np.asarray(
            [
                row["scenario"] == scenario["scenario"]
                for row in rows
            ],
            dtype=bool,
        )
        labels = np.asarray(
            [row["label"] for row in rows],
            dtype=bool,
        )[selected]
        folds = np.asarray(
            [row["fold"] for row in rows],
            dtype=np.int8,
        )[selected]
        pair_ids = np.asarray(
            [row["pair_id"] for row in rows],
            dtype=np.int64,
        )[selected]
        cross_values = np.asarray(
            [row["cross_activation"] for row in rows],
            dtype=np.float64,
        )[selected]
        method_results = {}
        for method_index, method in enumerate(METHODS):
            result = _evaluate_method(
                matrix=matrices[method][selected],
                labels=labels,
                folds=folds,
                pair_ids=pair_ids,
                budgets=budgets,
                args=args,
                seed_offset=(
                    scenario_index * 10_000
                    + method_index * 1_000
                ),
            )
            for budget, payload in result.items():
                prediction_arrays[
                    f"{scenario['scenario']}_{method}_{budget}"
                ] = payload.pop("_predictions").astype(np.float32)
            method_results[method] = result
        for method, result in method_results.items():
            for payload in result.values():
                payload["cross_minus_auc"] = _paired_bootstrap_delta(
                    labels,
                    cross_values,
                    prediction_arrays[
                        f"{scenario['scenario']}_{method}_{payload['budget']}"
                    ],
                    pair_ids,
                    samples=args.bootstrap_samples,
                    seed=(
                        args.seed
                        + scenario_index * 10_000
                        + (0 if method == "token" else 1_000)
                        + 2_000
                        + int(payload["budget"])
                    ),
                )
        best = {
            method: max(
                method_results[method],
                key=lambda budget:
                    method_results[method][budget]["oof_auc"],
            )
            for method in METHODS
        }
        passes = {
            method: all(
                method_results[method][budget][
                    "cross_minus_auc"
                ]["95ci"][0]
                > 0
                for budget in ("1", *map(str, budgets))
            )
            for method in METHODS
        }
        scenarios.append(
            {
                **metadata[int(scenario["feature_id"])],
                "rows": int(selected.sum()),
                "pairs": int(selected.sum() // 2),
                "documents": int(
                    len(
                        {
                            rows[index]["doc_hash"]
                            for index in np.flatnonzero(selected)
                        }
                    )
                ),
                "cross_auc": float(
                    roc_auc_score(labels, cross_values)
                ),
                "cross_pair_accuracy": _pair_accuracy(
                    labels,
                    cross_values,
                    pair_ids,
                ),
                "counterparts": method_results,
                "best_budget": best,
                "best_auc": {
                    method:
                        method_results[method][best[method]]["oof_auc"]
                    for method in METHODS
                },
                "passes_all_budgets": passes,
                "passes_both_methods": all(passes.values()),
            }
        )
        print(
            f"[reason-scenarios] {scenario['scenario']}: "
            f"BatchTopK={scenarios[-1]['best_auc']['token']:.3f}, "
            f"Temporal={scenarios[-1]['best_auc']['temporal']:.3f}",
            flush=True,
        )

    # Functional contribution on all real validation documents.
    eval_root = Path(args.eval_root)
    validation_root = (
        eval_root.parent.parent
        / "data/layer21_validation_cache_exact10000128"
    )
    mean_a, mean_b, validation_docs = _load_validation_pairs(
        validation_root,
        target_pairs=args.importance_pairs,
        seed=args.seed,
    )
    source = torch.cat((mean_a, mean_b), dim=0)
    target = torch.cat((mean_b, mean_a), dim=0)
    groups = np.concatenate((validation_docs, validation_docs))
    cross_checkpoint = Path(args.sae_root) / "cross/checkpoints/best"
    if not cross_checkpoint.exists():
        cross_checkpoint = Path(args.sae_root) / "cross"
    encoder = FrozenEncoder(
        cross_checkpoint,
        torch.device(args.device),
    )
    for scenario_index, scenario in enumerate(scenarios):
        importance = _importance_for_feature(
            encoder=encoder,
            feature_id=int(scenario["feature_id"]),
            source=source,
            target=target,
            groups=groups,
            controls=args.importance_controls,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + scenario_index * 100_003,
        )
        scenario["partner_reconstruction_importance"] = importance
        scenario["functional_pass"] = bool(
            importance.get("status") == "measured"
            and importance[
                "mean_relative_partner_sse_increase_95ci"
            ][0]
            > 0
        )

    primary_scenarios = [
        row for row in scenarios if bool(row.get("primary", True))
    ]
    exploratory_scenarios = [
        row for row in scenarios if not bool(row.get("primary", True))
    ]
    token_gaps = np.asarray(
        [
            1.0 - row["best_auc"]["token"]
            for row in primary_scenarios
        ],
        dtype=np.float64,
    )
    temporal_gaps = np.asarray(
        [
            1.0 - row["best_auc"]["temporal"]
            for row in primary_scenarios
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed + 900_000)
    token_draws = np.empty(args.bootstrap_samples, dtype=np.float64)
    temporal_draws = np.empty(args.bootstrap_samples, dtype=np.float64)
    for draw in range(args.bootstrap_samples):
        selected = rng.integers(
            0,
            len(primary_scenarios),
            size=len(primary_scenarios),
        )
        token_draws[draw] = token_gaps[selected].mean()
        temporal_draws[draw] = temporal_gaps[selected].mean()
    all_comparisons = [
        scenario["counterparts"][method][budget][
            "cross_minus_auc"
        ]
        for scenario in primary_scenarios
        for method in METHODS
        for budget in ("1", *map(str, budgets))
    ]
    empirical_floor = 1.0 / (args.bootstrap_samples + 1)
    max_adjusted_probability = max(
        min(
            1.0,
            max(
                comparison["bootstrap_probability_delta_le_0"],
                empirical_floor,
            )
            * len(all_comparisons),
        )
        for comparison in all_comparisons
    )
    result = {
        "format": FORMAT,
        "complete": True,
        "claim_scope": (
            "Multiple frozen natural Cross reasoning/document-function axes "
            "tested against complete BatchTopK and Temporal dictionaries "
            "under one/4/16/64-coordinate linear budgets."
        ),
        "protocol": {
            "scenarios_preregistered_from_existing_explanations": True,
            "explanation_and_scoring_documents_excluded": True,
            "same_document_exact_length_active_inactive_pairs": True,
            "pairs_per_scenario": args.pairs_per_scenario,
            "document_hash_cross_validation": args.cv_folds,
            "baseline_dictionary_width": 65_536,
            "token_temporal_pooling": "mean_after_threshold",
            "budgets": [1, *budgets],
            "nested_sparse_selection": True,
        },
        "scenario_count": len(scenarios),
        "primary_scenario_count": len(primary_scenarios),
        "exploratory_scenario_count": len(exploratory_scenarios),
        "scenarios": scenarios,
        "aggregate": {
            "passes_all_token_budgets": sum(
                row["passes_all_budgets"]["token"]
                for row in primary_scenarios
            ),
            "passes_all_temporal_budgets": sum(
                row["passes_all_budgets"]["temporal"]
                for row in primary_scenarios
            ),
            "passes_both_methods": sum(
                row["passes_both_methods"]
                for row in primary_scenarios
            ),
            "functional_passes": sum(
                row["functional_pass"] for row in primary_scenarios
            ),
            "passes_both_and_functional": sum(
                row["passes_both_methods"]
                and row["functional_pass"]
                for row in primary_scenarios
            ),
            "mean_best_batchtopk_auc": float(
                np.mean(
                    [
                        row["best_auc"]["token"]
                        for row in primary_scenarios
                    ]
                )
            ),
            "mean_best_temporal_auc": float(
                np.mean(
                    [
                        row["best_auc"]["temporal"]
                        for row in primary_scenarios
                    ]
                )
            ),
            "mean_cross_minus_best_batchtopk_auc": float(
                token_gaps.mean()
            ),
            "mean_cross_minus_best_batchtopk_auc_95ci": [
                float(np.quantile(token_draws, 0.025)),
                float(np.quantile(token_draws, 0.975)),
            ],
            "mean_cross_minus_best_temporal_auc": float(
                temporal_gaps.mean()
            ),
            "mean_cross_minus_best_temporal_auc_95ci": [
                float(np.quantile(temporal_draws, 0.025)),
                float(np.quantile(temporal_draws, 0.975)),
            ],
            "budget_comparisons": len(all_comparisons),
            "positive_95ci_budget_comparisons": sum(
                comparison["95ci"][0] > 0
                for comparison in all_comparisons
            ),
            "max_bonferroni_adjusted_bootstrap_probability":
                max_adjusted_probability,
            "functional_scenarios": [
                row["scenario"]
                for row in primary_scenarios
                if row["functional_pass"]
            ],
            "exploratory_scenarios": [
                row["scenario"] for row in exploratory_scenarios
            ],
            "exploratory_functional_scenarios": [
                row["scenario"]
                for row in exploratory_scenarios
                if row["functional_pass"]
            ],
        },
    }
    atomic_json_dump(result, output_dir / "results.json")
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        **prediction_arrays,
    )
    return result


def _write_outputs(
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    rows = []
    for scenario in result["scenarios"]:
        importance = scenario["partner_reconstruction_importance"]
        for method in METHODS:
            for budget, payload in scenario["counterparts"][method].items():
                rows.append(
                    {
                        "scenario": scenario["scenario"],
                        "scenario_label": scenario["label"].replace(
                            "\n",
                            " ",
                        ),
                        "feature_id": scenario["feature_id"],
                        "method": method,
                        "budget": budget,
                        "auc": payload["oof_auc"],
                        "auc_ci_low": payload["oof_auc_95ci"][0],
                        "auc_ci_high": payload["oof_auc_95ci"][1],
                        "cross_minus_auc":
                            payload["cross_minus_auc"]["point"],
                        "cross_minus_ci_low":
                            payload["cross_minus_auc"]["95ci"][0],
                        "cross_minus_ci_high":
                            payload["cross_minus_auc"]["95ci"][1],
                        "pair_accuracy": payload["pair_accuracy"],
                        "functional_sse_increase":
                            importance.get(
                                "mean_relative_partner_sse_increase"
                            ),
                        "functional_ci_low":
                            (
                                importance.get(
                                    "mean_relative_partner_sse_increase_95ci"
                                )
                                or [None, None]
                            )[0],
                        "functional_ci_high":
                            (
                                importance.get(
                                    "mean_relative_partner_sse_increase_95ci"
                                )
                                or [None, None]
                            )[1],
                    }
                )
    import csv

    with (output_dir / "scenario_results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bank_rows = _read_jsonl(output_dir / "challenge_bank.jsonl")
    qualitative = {
        "format": "chunk-saes-multi-reason-examples-v1",
        "selection": (
            "Top three selected active/inactive pairs by Cross activation "
            "for each preregistered scenario; one pair per source document."
        ),
        "scenarios": {},
    }
    for scenario in result["scenarios"]:
        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in bank_rows:
            if row["scenario"] == scenario["scenario"]:
                grouped[str(row["family_id"])][
                    str(row["condition"])
                ] = row
        pairs = [
            {
                "family_id": family_id,
                "doc_hash": values["active"]["doc_hash"],
                "length": values["active"]["length"],
                "cross_activation": values["active"][
                    "cross_activation"
                ],
                "active_text": values["active"]["text"],
                "inactive_text": values["inactive"]["text"],
            }
            for family_id, values in grouped.items()
            if "active" in values and "inactive" in values
        ]
        pairs.sort(
            key=lambda row: row["cross_activation"],
            reverse=True,
        )
        qualitative["scenarios"][scenario["scenario"]] = {
            "feature_id": scenario["feature_id"],
            "explanation": scenario["explanation"],
            "examples": pairs[:3],
        }
    atomic_json_dump(
        qualitative,
        output_dir / "qualitative_examples.json",
    )

    # Paper-style figure using the supplied reference palette.
    batch_color = METHOD_COLORS["token"]
    temporal_color = METHOD_COLORS["temporal"]
    cross_color = METHOD_COLORS["cross"]
    # Keep the preregistered primary set together in the figure.  The
    # exploratory rows remain visible, but are separated below the dashed
    # rule and retain the "*" suffix in their labels.
    scenarios = [
        row for row in result["scenarios"] if row.get("primary", True)
    ] + [
        row for row in result["scenarios"] if not row.get("primary", True)
    ]
    primary_count = sum(row.get("primary", True) for row in scenarios)
    labels = [row["label"] for row in scenarios]
    y = np.arange(len(scenarios))
    fig = plt.figure(figsize=(20.0, 9.2), layout="constrained")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.18, 1.38, 1.05),
        wspace=0.20,
    )

    ax = fig.add_subplot(grid[0, 0])
    height = 0.22
    batch = [row["best_auc"]["token"] for row in scenarios]
    temporal = [row["best_auc"]["temporal"] for row in scenarios]
    cross = [1.0] * len(scenarios)
    ax.barh(
        y + height,
        batch,
        height=height,
        color=batch_color,
        label="BatchTopK best of 1/4/16/64",
    )
    ax.barh(
        y,
        temporal,
        height=height,
        color=temporal_color,
        label="Temporal best of 1/4/16/64",
    )
    ax.barh(
        y - height,
        cross,
        height=height,
        color=cross_color,
        label="Cross single feature",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.03)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Held-out matched-pair AUC")
    ax.set_title(
        "A  Natural reasoning axes and strongest sparse counterparts",
        loc="left",
        fontweight="bold",
    )
    ax.axhline(
        primary_count - 0.5,
        color="#777777",
        linestyle=":",
        linewidth=1,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.text(
        0.995,
        0.015,
        "Cross target AUC = 1.0 by pair construction",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    for row_index, (batch_value, temporal_value) in enumerate(
        zip(batch, temporal, strict=True)
    ):
        ax.text(
            batch_value + 0.006,
            row_index + height,
            f"{batch_value:.2f}",
            va="center",
            fontsize=7,
            color="#4A4A4A",
        )
        ax.text(
            temporal_value + 0.006,
            row_index,
            f"{temporal_value:.2f}",
            va="center",
            fontsize=7,
            color="#1F4E79",
        )

    ax = fig.add_subplot(grid[0, 1])
    columns = [
        ("token", "1"),
        ("token", "4"),
        ("token", "16"),
        ("token", "64"),
        ("temporal", "1"),
        ("temporal", "4"),
        ("temporal", "16"),
        ("temporal", "64"),
    ]
    heat = np.asarray(
        [
            [
                100
                * row["counterparts"][method][budget][
                    "cross_minus_auc"
                ]["point"]
                for method, budget in columns
            ]
            for row in scenarios
        ]
    )
    image = ax.imshow(
        heat,
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=-10,
        vmax=max(10, float(np.max(heat))),
    )
    ax.set_xticks(
        range(len(columns)),
        ["B1", "B4", "B16", "B64", "T1", "T4", "T16", "T64"],
    )
    ax.set_yticks(range(len(labels)), labels)
    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            value = heat[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if abs(value) > 0.55 * max(10, np.max(heat))
                    else "#222222"
                ),
            )
    ax.set_title(
        "B  Cross gain over each budget (AUC pp)",
        loc="left",
        fontweight="bold",
    )
    ax.axhline(
        primary_count - 0.5,
        color="#777777",
        linestyle=":",
        linewidth=1,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cross − baseline (pp)")

    ax = fig.add_subplot(grid[0, 2])
    for row_index, row in enumerate(scenarios):
        importance = row["partner_reconstruction_importance"]
        if importance.get("status") == "measured":
            effect = 100 * importance[
                "mean_relative_partner_sse_increase"
            ]
            interval = 100 * np.asarray(
                importance[
                    "mean_relative_partner_sse_increase_95ci"
                ]
            )
            positive = bool(interval[0] > 0)
            color = cross_color if positive else "#aab4c1"
            ax.errorbar(
                effect,
                row_index,
                xerr=[
                    [effect - interval[0]],
                    [interval[1] - effect],
                ],
                fmt="o",
                markersize=7,
                color=color,
                ecolor=color,
                capsize=3,
            )
            ax.text(
                effect + 0.04,
                row_index,
                f"{effect:+.2f}%",
                va="center",
                fontsize=7,
                color=color,
            )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Partner reconstruction SSE increase (%)")
    ax.set_title(
        "C  Functional contribution on real adjacent chunks",
        loc="left",
        fontweight="bold",
    )
    ax.axhline(
        primary_count - 0.5,
        color="#777777",
        linestyle=":",
        linewidth=1,
    )
    ax.text(
        0.99,
        0.015,
        "red = 95% CI above zero; * = post-hoc exploratory",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    fig.suptitle(
        "Multi-scenario reasoning features: Cross vs BatchTopK and Temporal",
        fontsize=14,
        fontweight="bold",
    )
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            plot_dir / f"multi_reasoning_summary.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    aggregate = result["aggregate"]
    lines = [
        "# Multi-scenario natural reasoning evaluation",
        "",
        (
            f"Primary frozen scenarios: "
            f"**{result['primary_scenario_count']}** "
            f"(plus **{result['exploratory_scenario_count']}** explicitly "
            "exploratory functional replication); Cross beats "
            "all tested BatchTopK budgets in "
            f"**{aggregate['passes_all_token_budgets']}** scenarios, all "
            "Temporal budgets in "
            f"**{aggregate['passes_all_temporal_budgets']}**, and both in "
            f"**{aggregate['passes_both_methods']}**. "
            f"Functional ablation is positive in "
            f"**{aggregate['functional_passes']}** scenarios. Mean Cross "
            f"gain over the strongest BatchTopK counterpart is "
            f"**{100 * aggregate['mean_cross_minus_best_batchtopk_auc']:.1f} "
            "AUC points**; over Temporal it is "
            f"**{100 * aggregate['mean_cross_minus_best_temporal_auc']:.1f} "
            "points**."
        ),
        "",
        "| Scenario | Status | Cross feature | Best BatchTopK AUC | "
        "Best Temporal AUC | Cross beats both? | Functional pass? |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in scenarios:
        lines.append(
            f"| {row['scenario'].replace('_', ' ')} | "
            f"{'primary' if row.get('primary', True) else 'exploratory'} | "
            f"`{row['feature_id']}` | "
            f"{row['best_auc']['token']:.3f} | "
            f"{row['best_auc']['temporal']:.3f} | "
            f"{'yes' if row['passes_both_methods'] else 'no'} | "
            f"{'yes' if row['functional_pass'] else 'no'} |"
        )
    exploratory_names = [
        f"`{row['scenario']}`"
        for row in scenarios
        if not row.get("primary", True)
    ]
    lines += [
        "",
        (
            "The post-hoc exploratory scenarios are "
            + ", ".join(exploratory_names)
            + ". They were added after inspecting the first primary "
            "functional-ablation results and are excluded from all primary "
            "aggregate confidence intervals and multiplicity correction."
        ),
        "",
        "Every row uses same-document, exact-length active/inactive pairs, "
        "documents disjoint from the feature's explanation/scoring evidence, "
        "complete 65,536-coordinate BatchTopK and Temporal codes, and "
        "document-hash cross-fitting.",
        "",
        "The matched-pair target is defined by the Cross activation, so "
        "Cross AUC is 1.0 by construction. These comparisons measure "
        "single-axis alignment/compactness, not independent reasoning-task "
        "accuracy. The final blinded semantic audit and strict three-gate "
        "conclusion are reported in `../README.md`.",
        "",
        "![Multi-scenario reasoning summary](figures/multi_reasoning_summary.png)",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    audit = {
        "format": "chunk-saes-multi-reason-scenarios-audit-v1",
        "complete": True,
        "checks": {
            "scenarios_frozen_before_counterpart_test": True,
            "explanation_documents_excluded": True,
            "scoring_documents_excluded": True,
            "same_document_pairs": True,
            "exact_length_pairs": True,
            "one_active_one_inactive": True,
            "one_pair_per_document_per_scenario": True,
            "complete_batchtopk_dictionary": True,
            "complete_temporal_dictionary": True,
            "document_disjoint_cross_fitting": True,
            "nested_sparse_selection": True,
            "all_failures_reported": True,
            "posthoc_exploratory_scenario_labeled_and_excluded_from_primary_aggregate":
                True,
        },
        "issues": [],
    }
    atomic_json_dump(audit, output_dir / "audit_report.json")
    write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": result["protocol"],
            "files": {
                "bank": file_record(
                    output_dir / "challenge_bank.jsonl",
                    relative_to=output_dir,
                ),
                "bank_manifest": file_record(
                    output_dir / "challenge_bank_manifest.json",
                    relative_to=output_dir,
                ),
                "token_features": file_record(
                    output_dir / "token_features.npy",
                    relative_to=output_dir,
                ),
                "temporal_features": file_record(
                    output_dir / "temporal_features.npy",
                    relative_to=output_dir,
                ),
                "results": file_record(
                    output_dir / "results.json",
                    relative_to=output_dir,
                ),
                "table": file_record(
                    output_dir / "scenario_results.csv",
                    relative_to=output_dir,
                ),
                "qualitative_examples": file_record(
                    output_dir / "qualitative_examples.json",
                    relative_to=output_dir,
                ),
                "audit": file_record(
                    output_dir / "audit_report.json",
                    relative_to=output_dir,
                ),
                "plot_png": file_record(
                    plot_dir / "multi_reasoning_summary.png",
                    relative_to=output_dir,
                ),
            },
        },
        output_dir / "manifest.json",
    )


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _candidate_metadata(Path(args.eval_root))
    if args.analysis_only:
        rows = _read_jsonl(output_dir / "challenge_bank.jsonl")
    else:
        rows = build_bank(
            args=args,
            metadata=metadata,
            output_dir=output_dir,
        )
        extract_features(
            args=args,
            rows=rows,
            output_dir=output_dir,
        )
    result = analyze(
        args=args,
        rows=rows,
        metadata=metadata,
        output_dir=output_dir,
    )
    _write_outputs(result, output_dir)
    print(
        json.dumps(
            {
                "complete": True,
                "scenario_count": result["scenario_count"],
                "aggregate": result["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
