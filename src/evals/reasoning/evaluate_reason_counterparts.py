#!/usr/bin/env python
"""Search for Token counterparts to natural Cross reasoning features.

This evaluation reuses the independent held-out Pile chunks and top-128 sparse
codes materialized by Eval 6.  It selects candidate Cross features using only
pre-existing, independently generated artifacts:

* the frozen exact-1000 V2 explanation contains a reasoning/discourse term;
* its balanced exact-128 Context AutoInterp score is at least 0.875;
* its active recall in that score is at least 2/4; and
* it activates on at least ``--minimum-positive-rows`` Eval-6 chunks.

For each candidate, document-hash cross-fitting predicts the Cross active bit
from the complete Token-SAE code using:

* the best single Token feature;
* 4 selected Token features;
* 16 selected Token features; and
* 64 selected Token features.

The test therefore asks whether one Cross axis is available as a comparably
simple axis or sparse linear combination in Token SAE.  It does not claim to
rule out arbitrary nonlinear decoders.

Feature importance is measured on real adjacent validation-cache pairs by
ablating the Cross coordinate and measuring the increase in partner target SSE,
with support/activation-matched active Cross coordinates as controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from safetensors import safe_open
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from chunk_saes.artifacts import (
    file_record,
    file_sha256,
    write_artifact_manifest,
)
from chunk_saes.plot_style import METHOD_COLORS
from chunk_saes.sae import DecoderHead, SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.utils import atomic_json_dump


FORMAT = "chunk-saes-natural-reason-counterpart-v1"
REASON_RE = re.compile(
    r"\b(reason|reasoning|argument|proof|derivation|derive|causal|"
    r"counterfactual|hypothetical|premise|conclusion|inference|"
    r"problem.?solution|diagnos|debug|case analys|rebuttal|trade.?off|"
    r"justify|decision|solve|solution)\w*\b",
    re.I,
)
CORE_REASON_RE = re.compile(
    r"\b(reasoning|inference|inferential|premise|conclusion|proof|"
    r"derivation|deduct\w*|induct\w*|causal\w*|counterfactual\w*|"
    r"hypothetical\w*|mechanis(?:m|tic)\w*)\b",
    re.I,
)


class FrozenEncoder:
    """Chunked encoder/decoder view of a frozen SAE."""

    def __init__(
        self,
        checkpoint_dir: Path,
        device: torch.device,
        *,
        decoder_head: DecoderHead | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        config = json.loads(
            (checkpoint_dir / "config.json").read_text(encoding="utf-8")
        )
        with safe_open(
            str(checkpoint_dir / "sae.safetensors"),
            framework="pt",
            device="cpu",
        ) as handle:
            names = set(handle.keys())
            self.weight = handle.get_tensor("encoder_weight").to(device)
            self.bias = handle.get_tensor("encoder_bias").to(device)
            is_joint = (
                int(config.get("decoder_heads", 1)) == 2
                or "decoder_cross_weight" in names
                or "decoder_cross_bias" in names
            )
            has_cross_weight = "decoder_cross_weight" in names
            has_cross_bias = "decoder_cross_bias" in names
            if has_cross_weight != has_cross_bias:
                raise ValueError(
                    f"{checkpoint_dir} stores an incomplete Cross decoder head"
                )
            if int(config.get("decoder_heads", 1)) == 2 and not (
                has_cross_weight and has_cross_bias
            ):
                raise ValueError(
                    f"{checkpoint_dir} declares two decoder heads but lacks "
                    "the Cross head tensors"
                )
            if is_joint:
                if decoder_head not in ("mean", "cross"):
                    raise ValueError(
                        f"{checkpoint_dir} is a Joint checkpoint; pass "
                        "decoder_head='mean' or decoder_head='cross'"
                    )
                weight_name = (
                    "decoder_cross_weight"
                    if decoder_head == "cross"
                    else "decoder_weight"
                )
                bias_name = (
                    "decoder_cross_bias"
                    if decoder_head == "cross"
                    else "decoder_bias"
                )
                if weight_name not in names or bias_name not in names:
                    raise ValueError(
                        f"{checkpoint_dir} lacks requested decoder tensors: "
                        f"{weight_name}, {bias_name}"
                    )
            else:
                if decoder_head not in (None, "mean"):
                    raise ValueError(
                        f"single-head checkpoint {checkpoint_dir} cannot select "
                        f"decoder_head={decoder_head!r}"
                    )
                weight_name = "decoder_weight"
                bias_name = "decoder_bias"
            self.decoder_weight = handle.get_tensor(weight_name).to(device)
            self.decoder_bias = handle.get_tensor(bias_name).to(device)
            # Legacy checkpoints used the primary decoder bias for input
            # centering.  Joint checkpoints normally carry pre_bias explicitly.
            legacy_pre_bias = handle.get_tensor("decoder_bias").to(device)
            self.decoder_head = decoder_head or "mean"
            if "pre_bias" in names:
                self.pre_bias = handle.get_tensor("pre_bias").to(device)
            elif (
                config.get("sae_parameter_schema_version")
                == SAE_PARAMETER_SCHEMA_VERSION
            ):
                raise ValueError(f"{checkpoint_dir} lacks pre_bias")
            else:
                self.pre_bias = legacy_pre_bias
            self.threshold = float(handle.get_tensor("threshold"))
            self.scale = float(handle.get_tensor("activation_scale"))
            self.counts = handle.get_tensor("feature_counts").long()
        self.width = int(self.weight.shape[0])
        self.device = device

    def selected_activations(
        self,
        hidden: torch.Tensor,
        feature_ids: list[int],
    ) -> torch.Tensor:
        ids = torch.tensor(
            feature_ids,
            dtype=torch.long,
            device=self.device,
        )
        weight = self.weight.index_select(0, ids)
        bias = self.bias.index_select(0, ids)
        hidden = hidden.to(self.device, dtype=self.weight.dtype)
        values = F.relu(
            F.linear(
                hidden * self.scale - self.pre_bias,
                weight,
                bias,
            )
        )
        return values * (values > self.threshold)

    def dense_activations(
        self,
        hidden: torch.Tensor,
        *,
        feature_block_size: int,
    ) -> torch.Tensor:
        hidden = hidden.to(self.device, dtype=self.weight.dtype)
        result = torch.empty(
            (hidden.shape[0], self.width),
            dtype=self.weight.dtype,
            device=self.device,
        )
        scaled = hidden * self.scale - self.pre_bias
        for start in range(0, self.width, feature_block_size):
            stop = min(self.width, start + feature_block_size)
            values = F.relu(
                F.linear(
                    scaled,
                    self.weight[start:stop],
                    self.bias[start:stop],
                )
            )
            result[:, start:stop] = values * (values > self.threshold)
        return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate natural Cross reasoning features against Token "
            "single/4/16/64-feature counterparts and real-pair ablations."
        )
    )
    p.add_argument("--eval-root", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--decoder-head",
        choices=("mean", "cross"),
        default=None,
        help=(
            "Read this head when cross_checkpoint is a Joint Chunk SAE. "
            "Leave unset for legacy single-head checkpoints."
        ),
    )
    p.add_argument("--minimum-positive-rows", type=int, default=20)
    p.add_argument("--minimum-autointerp-score", type=float, default=0.875)
    p.add_argument("--minimum-autointerp-tpr", type=float, default=0.5)
    p.add_argument(
        "--headline-feature",
        type=int,
        default=20232,
        help=(
            "Natural Cross feature chosen from the frozen pre-existing "
            "explanations before running the Token-counterpart test."
        ),
    )
    p.add_argument("--screen-features", type=int, default=512)
    p.add_argument("--sparse-budgets", default="4,16,64")
    p.add_argument(
        "--logistic-c",
        type=float,
        default=1e-4,
        help=(
            "Fixed L2 inverse regularization for all sparse Token probes. "
            "The same value is used for every budget and candidate."
        ),
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--importance-pairs", type=int, default=512)
    p.add_argument("--importance-controls", type=int, default=32)
    p.add_argument(
        "--importance-repeats",
        type=int,
        default=1,
        help=(
            "Number of independent document-disjoint validation-pair "
            "subsamples used for the partner-reconstruction ablation."
        ),
    )
    p.add_argument("--feature-block-size", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--overwrite", action="store_true")
    return p


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
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


def _csr(indices: np.ndarray, values: np.ndarray) -> sparse.csr_matrix:
    valid = indices >= 0
    rows, slots = np.nonzero(valid)
    return sparse.csr_matrix(
        (
            values[rows, slots].astype(np.float32),
            (rows, indices[rows, slots].astype(np.int64)),
        ),
        shape=(indices.shape[0], 65_536),
        dtype=np.float32,
    )


def _dense_matrix(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _column_values(
    matrix: np.ndarray | sparse.csr_matrix,
    feature_id: int,
) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix[:, feature_id].toarray().ravel()
    return np.asarray(matrix[:, feature_id]).ravel()


def _slice_dense(
    matrix: np.ndarray | sparse.csr_matrix,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    row_ids = np.flatnonzero(rows)
    if sparse.issparse(matrix):
        return matrix[rows][:, columns].toarray()
    return np.asarray(
        matrix[np.ix_(row_ids, columns)],
        dtype=np.float32,
    )


def _load_chunk_document_map(evidence_dir: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for path in sorted((evidence_dir / "partials").glob("chunks-rank*.jsonl")):
        for row in _read_jsonl(path):
            result[int(row["global_chunk_id"])] = str(row["doc_id"])
    return result


def _load_chunk_rows(evidence_dir: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((evidence_dir / "partials").glob("chunks-rank*.jsonl")):
        for row in _read_jsonl(path):
            result[int(row["global_chunk_id"])] = row
    return result


def _autointerp_rates(
    *,
    details: dict[tuple[str, int], dict[str, Any]],
    feature_id: int,
) -> dict[str, Any]:
    result = details[("cross", feature_id)]
    return {
        "positive_examples": 4,
        "negative_examples": 4,
        "true_positive": int(result["tp"]),
        "true_negative": int(result["tn"]),
        "tpr": float(result["active_recall"]),
        "tnr": float(result["inactive_rejection"]),
    }


def _prevalence_matched_recovery(
    labels: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    positives = int(labels.sum())
    order = np.argsort(values, kind="stable")[::-1]
    predicted = order[:positives]
    recovered = int(labels[predicted].sum())
    rate = recovered / max(1, positives)
    return {
        "positive_rows": positives,
        "recovered_positive_rows_at_same_budget": recovered,
        "precision_at_n_positive": rate,
        "recall_at_n_positive": rate,
    }


def _screen_features(
    matrix: np.ndarray | sparse.csr_matrix,
    labels: np.ndarray,
    train: np.ndarray,
    count: int,
) -> np.ndarray:
    if sparse.issparse(matrix):
        positive = matrix[train][labels[train]]
        negative = matrix[train][~labels[train]]
        positive_mean = np.asarray(positive.mean(axis=0)).ravel()
        negative_mean = np.asarray(negative.mean(axis=0)).ravel()
        positive_rate = np.asarray(
            (positive > 0).mean(axis=0)
        ).ravel()
        negative_rate = np.asarray(
            (negative > 0).mean(axis=0)
        ).ravel()
    else:
        train_rows = np.flatnonzero(train)
        positive_rows = train_rows[labels[train]]
        negative_rows = train_rows[~labels[train]]
        width = matrix.shape[1]
        positive_mean = np.empty(width, dtype=np.float64)
        negative_mean = np.empty(width, dtype=np.float64)
        positive_rate = np.empty(width, dtype=np.float64)
        negative_rate = np.empty(width, dtype=np.float64)
        for start in range(0, width, 4096):
            stop = min(width, start + 4096)
            positive = np.asarray(
                matrix[positive_rows, start:stop],
                dtype=np.float32,
            )
            negative = np.asarray(
                matrix[negative_rows, start:stop],
                dtype=np.float32,
            )
            positive_mean[start:stop] = positive.mean(axis=0)
            negative_mean[start:stop] = negative.mean(axis=0)
            positive_rate[start:stop] = (positive > 0).mean(axis=0)
            negative_rate[start:stop] = (negative > 0).mean(axis=0)
    score = np.abs(positive_rate - negative_rate) * np.log1p(
        positive_mean + negative_mean
    )
    return np.argsort(score)[::-1][:count]


def _best_single(
    matrix: np.ndarray | sparse.csr_matrix,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    candidates: np.ndarray,
) -> tuple[int, int, float, np.ndarray]:
    best: tuple[float, int, int] | None = None
    for feature_id in candidates.tolist():
        values = _column_values(matrix, feature_id)
        auc = float(roc_auc_score(labels[train], values[train]))
        direction = 1
        if auc < 0.5:
            auc = 1 - auc
            direction = -1
        candidate = (auc, int(feature_id), direction)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    calibration_auc, feature_id, direction = best
    values = direction * _column_values(matrix, feature_id)
    return (
        feature_id,
        direction,
        calibration_auc,
        values[test].astype(np.float64),
    )


def _fit_sparse(
    matrix: np.ndarray | sparse.csr_matrix,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    candidates: np.ndarray,
    budget: int,
    logistic_c: float,
    seed: int,
) -> tuple[list[int], float, np.ndarray]:
    selected = candidates[:budget]
    x_train = _slice_dense(matrix, train, selected)
    x_test = _slice_dense(matrix, test, selected)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    model = LogisticRegression(
        # Strong shrinkage is important in the rare-positive, p >> n regime
        # here.  This value was fixed for every budget/candidate rather than
        # tuned on any held-out fold.
        C=logistic_c,
        class_weight="balanced",
        max_iter=5_000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(x_train, labels[train])
    train_values = model.predict_proba(x_train)[:, 1]
    test_values = model.predict_proba(x_test)[:, 1]
    return (
        selected.astype(int).tolist(),
        float(roc_auc_score(labels[train], train_values)),
        test_values,
    )


def _cross_fitted_prediction(
    *,
    token: np.ndarray | sparse.csr_matrix,
    labels: np.ndarray,
    folds: np.ndarray,
    budget: int,
    screen_features: int,
    logistic_c: float,
    seed: int,
) -> dict[str, Any]:
    predictions = np.zeros(len(labels), dtype=np.float64)
    selected_by_fold: list[list[int]] = []
    calibration_auc = []
    for fold in sorted(set(folds.tolist())):
        test = folds == fold
        train = ~test
        candidates = _screen_features(
            token,
            labels,
            train,
            max(screen_features, budget),
        )
        if budget == 1:
            feature_id, direction, auc, values = _best_single(
                token,
                labels,
                train,
                test,
                candidates,
            )
            selected_by_fold.append([feature_id])
            calibration_auc.append(auc)
            predictions[test] = values
        else:
            selected, auc, values = _fit_sparse(
                token,
                labels,
                train,
                test,
                candidates,
                budget,
                logistic_c,
                seed + int(fold),
            )
            selected_by_fold.append(selected)
            calibration_auc.append(auc)
            predictions[test] = values
    return {
        "budget": budget,
        "oof_auc": float(roc_auc_score(labels, predictions)),
        "oof_average_precision": float(
            average_precision_score(labels, predictions)
        ),
        "mean_calibration_auc": float(np.mean(calibration_auc)),
        "selected_feature_ids_by_fold": selected_by_fold,
        "_predictions": predictions,
    }


def _cluster_bootstrap(
    labels: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    draws = _cluster_auc_draws(
        labels,
        (values,),
        groups,
        samples=samples,
        seed=seed,
    )[0]
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _cluster_auc_draws(
    labels: np.ndarray,
    value_vectors: tuple[np.ndarray, ...],
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
    draw_batch_size: int = 256,
) -> list[np.ndarray]:
    """Cluster bootstrap AUCs without materializing resampled row arrays.

    Each bootstrap draw samples source documents with replacement.  The
    resulting document multiplicities become row weights; weighted AUC is
    computed exactly, including ties.
    """

    labels = np.asarray(labels, dtype=np.int8)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    output = [
        np.empty(samples, dtype=np.float64)
        for _ in value_vectors
    ]
    orders_and_bounds = []
    for values in value_vectors:
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        bounds = np.r_[
            0,
            np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1,
        ]
        orders_and_bounds.append((order, bounds))

    probabilities = np.full(n_groups, 1.0 / n_groups)
    for start in range(0, samples, draw_batch_size):
        stop = min(samples, start + draw_batch_size)
        counts = rng.multinomial(
            n_groups,
            probabilities,
            size=stop - start,
        )
        row_weights = counts[:, inverse].astype(np.float64)
        total_positive = (row_weights * labels[None, :]).sum(axis=1)
        total_negative = (
            row_weights * (1 - labels)[None, :]
        ).sum(axis=1)
        denominator = total_positive * total_negative
        for result, (order, bounds) in zip(
            output,
            orders_and_bounds,
            strict=True,
        ):
            sorted_weights = row_weights[:, order]
            sorted_labels = labels[order]
            positive = sorted_weights * sorted_labels[None, :]
            negative = sorted_weights * (1 - sorted_labels)[None, :]
            positive_by_tie = np.add.reduceat(
                positive,
                bounds,
                axis=1,
            )
            negative_by_tie = np.add.reduceat(
                negative,
                bounds,
                axis=1,
            )
            negative_below = (
                np.cumsum(negative_by_tie, axis=1)
                - negative_by_tie
            )
            numerator = (
                positive_by_tie
                * (negative_below + 0.5 * negative_by_tie)
            ).sum(axis=1)
            result[start:stop] = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )
    return output


def _paired_bootstrap_delta(
    labels: np.ndarray,
    cross_values: np.ndarray,
    token_values: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    cross_draws, token_draws = _cluster_auc_draws(
        labels,
        (cross_values, token_values),
        groups,
        samples=samples,
        seed=seed,
    )
    draws = cross_draws - token_draws
    observed = float(
        roc_auc_score(labels, cross_values)
        - roc_auc_score(labels, token_values)
    )
    return {
        "point": observed,
        "95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_probability_delta_le_0": float((draws <= 0).mean()),
    }


def _load_candidate_artifacts(eval_root: Path) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
]:
    auto = eval_root / "autointerp/autointerp_exact1000"
    public = json.loads(
        (auto / "results/cross_features.json").read_text(encoding="utf-8")
    )
    details = {
        (str(row["method"]), int(row["feature_id"])): row
        for row in _read_jsonl(
            auto / "results/feature_results.jsonl"
        )
    }
    return public, details


def _load_validation_pairs(
    cache_root: Path,
    *,
    target_pairs: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    records: list[tuple[bytes, int, int, torch.Tensor, torch.Tensor]] = []
    for path in sorted(cache_root.glob("rank*/shard-*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            mean_a = handle.get_tensor("mean_a")
            mean_b = handle.get_tensor("mean_b")
            doc_hash = np.ascontiguousarray(
                handle.get_tensor("doc_hash").numpy()
            ).view("S32").reshape(-1)
            pair_ids = handle.get_tensor("pair_id").numpy()
            length_a = handle.get_tensor("length_a").numpy()
            length_b = handle.get_tensor("length_b").numpy()
        for index in range(len(pair_ids)):
            digest = bytes(doc_hash[index])
            key = int.from_bytes(
                hashlib.sha256(
                    digest + int(pair_ids[index]).to_bytes(8, "little")
                    + seed.to_bytes(8, "little")
                ).digest()[:8],
                "little",
            )
            records.append(
                (
                    digest,
                    key,
                    int(length_a[index]) + int(length_b[index]),
                    mean_a[index].clone(),
                    mean_b[index].clone(),
                )
            )
    # At most one pair per source document prevents a long document from
    # dominating the importance estimate.
    best_by_doc: dict[bytes, tuple[Any, ...]] = {}
    for record in records:
        current = best_by_doc.get(record[0])
        if current is None or record[1] < current[1]:
            best_by_doc[record[0]] = record
    chosen = sorted(best_by_doc.values(), key=lambda row: row[1])[:target_pairs]
    if len(chosen) < target_pairs:
        raise ValueError(
            f"only {len(chosen)} distinct validation documents, "
            f"need {target_pairs}"
        )
    a = torch.stack([row[3] for row in chosen]).float()
    b = torch.stack([row[4] for row in chosen]).float()
    doc_ids = np.asarray([row[0].hex() for row in chosen])
    return a, b, doc_ids


def _importance_for_feature(
    *,
    encoder: FrozenEncoder,
    feature_id: int,
    source: torch.Tensor,
    target: torch.Tensor,
    groups: np.ndarray,
    controls: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    support = int(encoder.counts[feature_id])
    # First locate every held-out direction on which the target coordinate is
    # active.  Full reconstruction is then restricted to those directions,
    # making it feasible to use thousands of distinct validation documents.
    target_z = encoder.selected_activations(source, [feature_id])[:, 0]
    active = target_z > 0
    active_count = int(active.sum())
    if active_count < 5:
        return {
            "feature_id": feature_id,
            "training_support": support,
            "directions": source.shape[0],
            "active_directions": active_count,
            "status": "insufficient_active_pairs",
        }
    groups = np.asarray(groups)[active.cpu().numpy()]
    source = source[active.cpu()]
    target = target[active.cpu()]

    # Full Cross reconstruction is needed for an exact single-coordinate
    # ablation.  Keep the code on the target-active rows so controls can be
    # matched by training support and decoded contribution energy, then
    # evaluated on exactly the same rows.
    reconstruction = encoder.decoder_bias.float().repeat(source.shape[0], 1)
    source = source.to(encoder.device, dtype=encoder.weight.dtype)
    target = target.to(encoder.device, dtype=encoder.weight.dtype)
    scaled = source * encoder.scale - encoder.pre_bias
    code = torch.empty(
        (source.shape[0], encoder.width),
        dtype=encoder.weight.dtype,
        device=encoder.device,
    )
    for start in range(0, encoder.width, 4096):
        stop = min(encoder.width, start + 4096)
        values = F.relu(
            F.linear(
                scaled,
                encoder.weight[start:stop],
                encoder.bias[start:stop],
            )
        )
        values *= values > encoder.threshold
        code[:, start:stop] = values
        reconstruction += F.linear(
            values,
            encoder.decoder_weight[:, start:stop],
        ).float()
    target_scaled = target.float() * encoder.scale
    residual = reconstruction - target_scaled
    baseline_sse = (residual**2).sum(dim=1)

    decoder_norm_sq = (
        encoder.decoder_weight.float().pow(2).sum(dim=0)
    )
    contribution_energy = (
        code.float().pow(2).mean(dim=0) * decoder_norm_sq
    )
    target_energy = float(contribution_energy[feature_id])
    active_on_context = (code > 0).sum(dim=0)
    minimum_control_active = max(5, math.ceil(active_count * 0.20))
    support_log = torch.log1p(
        encoder.counts.to(
            contribution_energy.device,
            dtype=torch.float32,
        )
    )
    target_support_log = float(support_log[feature_id])
    energy_log = torch.log1p(contribution_energy)
    target_energy_log = float(energy_log[feature_id])
    activity_log = torch.log1p(active_on_context.float())
    target_activity_log = float(activity_log[feature_id])
    match_score = (
        torch.abs(support_log - target_support_log)
        + torch.abs(energy_log - target_energy_log)
        + torch.abs(activity_log - target_activity_log)
    )
    support_ratio = (
        encoder.counts.to(contribution_energy.device).float()
        / max(1, support)
    )
    energy_ratio = contribution_energy / max(target_energy, 1e-12)
    activity_ratio = active_on_context.float() / max(1, active_count)
    eligible = (
        (active_on_context >= minimum_control_active)
        & (support_ratio >= 0.25)
        & (support_ratio <= 4.0)
        & (energy_ratio >= 0.25)
        & (energy_ratio <= 4.0)
        & (activity_ratio >= 0.25)
        & (activity_ratio <= 4.0)
    )
    eligible[feature_id] = False
    match_score[~eligible] = torch.inf
    sorted_ids = torch.argsort(match_score)
    control_ids = sorted_ids[
        torch.isfinite(match_score[sorted_ids])
    ][:controls].tolist()
    if len(control_ids) < min(8, controls):
        # Fall back to nearest active features when the hard ratio window is
        # too narrow, but never pad with inactive/zero-contribution features.
        eligible = (
            (active_on_context >= minimum_control_active)
            & (contribution_energy > 0)
        )
        eligible[feature_id] = False
        match_score = (
            torch.abs(support_log - target_support_log)
            + torch.abs(energy_log - target_energy_log)
            + torch.abs(activity_log - target_activity_log)
        )
        match_score[~eligible] = torch.inf
        sorted_ids = torch.argsort(match_score)
        control_ids = sorted_ids[
            torch.isfinite(match_score[sorted_ids])
        ][:controls].tolist()
    candidate = [feature_id, *map(int, control_ids)]

    effects = []
    effect_vectors = []
    for current_id in candidate:
        coefficient = code[:, current_id]
        delta = (
            coefficient[:, None]
            * encoder.decoder_weight[:, current_id][None, :]
        ).float()
        ablated = ((residual - delta) ** 2).sum(dim=1)
        relative = (
            (ablated - baseline_sse)
            / baseline_sse.clamp_min(1e-12)
        )
        effect_vectors.append(relative.detach().cpu().numpy())
        effects.append(
            {
                "feature_id": current_id,
                "active_pairs": int(
                    active_on_context[current_id].item()
                ),
                "effect": float(relative.mean()),
                "median_effect": float(relative.median()),
                "training_support": int(encoder.counts[current_id]),
                "mean_decoded_contribution_energy": float(
                    contribution_energy[current_id]
                ),
            }
        )
    observed = effects[0]
    control_values = [
        row["effect"] for row in effects[1:] if row["effect"] is not None
    ]
    if observed["effect"] is None:
        return {
            "feature_id": feature_id,
            "training_support": support,
            "directions": source.shape[0],
            "active_directions": observed["active_pairs"],
            "status": "insufficient_active_pairs",
            "control_details": effects[1:],
        }
    if not control_values:
        return {
            "feature_id": feature_id,
            "training_support": support,
            "directions": source.shape[0],
            "active_directions": observed["active_pairs"],
            "status": "insufficient_active_controls",
            "mean_relative_partner_sse_increase": observed["effect"],
            "median_relative_partner_sse_increase":
                observed.get("median_effect"),
            "control_details": effects[1:],
        }
    observed_vector = effect_vectors[0].astype(np.float64)
    control_mean_vector = np.stack(
        effect_vectors[1:],
        axis=1,
    ).mean(axis=1)
    excess_vector = observed_vector - control_mean_vector

    def cluster_mean_interval(
        values: np.ndarray,
        *,
        local_seed: int,
    ) -> tuple[list[float], float]:
        unique_groups = np.unique(groups)
        rows_by_group = [
            np.flatnonzero(groups == group)
            for group in unique_groups
        ]
        rng = np.random.default_rng(local_seed)
        draws = np.empty(bootstrap_samples, dtype=np.float64)
        for draw in range(bootstrap_samples):
            sampled = rng.choice(
                len(rows_by_group),
                size=len(rows_by_group),
                replace=True,
            )
            indices = np.concatenate(
                [rows_by_group[index] for index in sampled]
            )
            draws[draw] = values[indices].mean()
        return (
            [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            float((draws <= 0).mean()),
        )

    observed_ci, observed_p = cluster_mean_interval(
        observed_vector,
        local_seed=seed + feature_id,
    )
    excess_ci, excess_p = cluster_mean_interval(
        excess_vector,
        local_seed=seed + feature_id + 1,
    )
    return {
        "feature_id": feature_id,
        "training_support": support,
        "directions": source.shape[0],
        "active_directions": observed["active_pairs"],
        "status": "measured",
        "control_protocol": (
            "same target-active directions; nearest coordinates by log "
            "training support, log decoded-contribution energy, and active "
            "count; controls must activate on at least "
            f"{minimum_control_active} directions"
        ),
        "mean_decoded_contribution_energy": target_energy,
        "mean_relative_partner_sse_increase": observed["effect"],
        "mean_relative_partner_sse_increase_95ci": observed_ci,
        "bootstrap_probability_mean_effect_le_0": observed_p,
        "median_relative_partner_sse_increase":
            observed.get("median_effect"),
        "matched_control_count": len(control_values),
        "matched_control_mean": float(np.mean(control_values)),
        "matched_control_95pct_range": [
            float(np.quantile(control_values, 0.025)),
            float(np.quantile(control_values, 0.975)),
        ],
        "percentile_among_controls": float(
            np.mean(
                np.asarray(control_values)
                <= float(observed["effect"])
            )
        ),
        "effect_over_control_mean": float(observed["effect"])
        - float(np.mean(control_values)),
        "effect_over_control_mean_95ci": excess_ci,
        "bootstrap_probability_excess_le_0": excess_p,
        "control_details": effects[1:],
    }


def _aggregate_importance_repeats(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    measured = [row for row in rows if row.get("status") == "measured"]
    if not measured:
        return {
            "status": "no_measured_repeat",
            "repeats": rows,
        }
    observed = np.asarray(
        [
            row["mean_relative_partner_sse_increase"]
            for row in measured
        ],
        dtype=np.float64,
    )
    controls = np.asarray(
        [row["matched_control_mean"] for row in measured],
        dtype=np.float64,
    )
    excess = observed - controls
    observed_ci_lows = [
        row["mean_relative_partner_sse_increase_95ci"][0]
        for row in measured
    ]
    excess_ci_lows = [
        row["effect_over_control_mean_95ci"][0]
        for row in measured
    ]
    return {
        "status": "measured",
        "measured_repeats": len(measured),
        "active_directions_total": int(
            sum(row["active_directions"] for row in measured)
        ),
        "mean_relative_partner_sse_increase": float(observed.mean()),
        "mean_relative_partner_sse_repeat_range": [
            float(observed.min()),
            float(observed.max()),
        ],
        "matched_control_mean": float(controls.mean()),
        "effect_over_control_mean": float(excess.mean()),
        "effect_over_control_repeat_range": [
            float(excess.min()),
            float(excess.max()),
        ],
        "positive_effect_repeat_fraction": float((observed > 0).mean()),
        "positive_excess_repeat_fraction": float((excess > 0).mean()),
        "positive_effect_ci_repeat_fraction": float(
            np.mean(np.asarray(observed_ci_lows) > 0)
        ),
        "positive_excess_ci_repeat_fraction": float(
            np.mean(np.asarray(excess_ci_lows) > 0)
        ),
        "repeats": rows,
    }


def _qualitative_examples(
    *,
    feature: dict[str, Any],
    cross: sparse.csr_matrix,
    global_ids: np.ndarray,
    chunk_rows: dict[int, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    feature_id = int(feature["feature_id"])
    values = cross[:, feature_id].toarray().ravel().astype(np.float64)
    order = np.argsort(values)[::-1]
    examples = []
    used_documents: set[str] = set()
    for row_index in order.tolist():
        activation = float(values[row_index])
        if activation <= 0:
            break
        chunk_id = int(global_ids[row_index])
        evidence = chunk_rows.get(chunk_id)
        if evidence is None:
            continue
        document = str(evidence["doc_id"])
        if document in used_documents:
            continue
        used_documents.add(document)
        examples.append(
            {
                "rank": len(examples) + 1,
                "global_chunk_id": chunk_id,
                "doc_id": document,
                "source": evidence.get("source"),
                "length": int(evidence.get("length", 0)),
                "cross_activation": activation,
                "text": str(evidence["text"]),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _counterpart_examples(
    *,
    feature: dict[str, Any],
    cross: sparse.csr_matrix,
    token: np.ndarray | sparse.csr_matrix,
    global_ids: np.ndarray,
    chunk_rows: dict[int, dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    feature_id = int(feature["feature_id"])
    labels = cross[:, feature_id].toarray().ravel() > 0
    selected_by_fold = feature["token_counterparts"]["1"][
        "selected_feature_ids_by_fold"
    ]
    selected = [int(row[0]) for row in selected_by_fold]
    counts = Counter(selected)
    token_feature_id, occurrences = counts.most_common(1)[0]
    token_values = _column_values(
        token,
        token_feature_id,
    ).astype(np.float64)
    positive_budget = int(labels.sum())
    predicted = np.zeros(len(labels), dtype=bool)
    predicted[
        np.argsort(token_values, kind="stable")[::-1][:positive_budget]
    ] = True

    def records(mask: np.ndarray, score: np.ndarray) -> list[dict[str, Any]]:
        output = []
        used_documents: set[str] = set()
        for row_index in np.argsort(score, kind="stable")[::-1].tolist():
            if not mask[row_index]:
                continue
            chunk_id = int(global_ids[row_index])
            evidence = chunk_rows.get(chunk_id)
            if evidence is None:
                continue
            document = str(evidence["doc_id"])
            if document in used_documents:
                continue
            used_documents.add(document)
            output.append(
                {
                    "global_chunk_id": chunk_id,
                    "doc_id": document,
                    "cross_activation": float(
                        cross[row_index, feature_id]
                    ),
                    "token_feature_activation": float(
                        token_values[row_index]
                    ),
                    "text": str(evidence["text"]),
                }
            )
            if len(output) >= limit:
                break
        return output

    return {
        "token_single_feature_id": token_feature_id,
        "selected_in_folds": occurrences,
        "total_folds": len(selected),
        "same_prevalence_budget": positive_budget,
        "true_positive_rows": int((labels & predicted).sum()),
        "missed_cross_rows": int((labels & ~predicted).sum()),
        "token_false_positive_rows": int((~labels & predicted).sum()),
        "shared_examples": records(
            labels & predicted,
            cross[:, feature_id].toarray().ravel(),
        ),
        "missed_cross_examples": records(
            labels & ~predicted,
            cross[:, feature_id].toarray().ravel(),
        ),
        "token_false_positive_examples": records(
            ~labels & predicted,
            token_values,
        ),
    }


def _decoder_sparse_approximation(
    *,
    token_checkpoint: Path,
    cross_checkpoint: Path,
    feature_id: int,
    budgets: list[int],
    decoder_head: DecoderHead | None = None,
) -> dict[str, Any]:
    """Approximate one Cross decoder direction with Token decoder atoms.

    Orthogonal matching pursuit is intentionally generous to Token: atoms may
    receive arbitrary signed coefficients, and selection directly optimizes
    the target Cross direction rather than an external task.
    """

    with safe_open(
        str(token_checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        token_decoder = handle.get_tensor("decoder_weight").float()
    with safe_open(
        str(cross_checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        names = set(handle.keys())
        config = json.loads(
            (cross_checkpoint / "config.json").read_text(encoding="utf-8")
        )
        is_joint = (
            int(config.get("decoder_heads", 1)) == 2
            or "decoder_cross_weight" in names
        )
        if is_joint and decoder_head not in ("mean", "cross"):
            raise ValueError(
                f"{cross_checkpoint} is a Joint checkpoint; pass --decoder-head"
            )
        if not is_joint and decoder_head not in (None, "mean"):
            raise ValueError(
                f"single-head checkpoint {cross_checkpoint} cannot select "
                f"decoder_head={decoder_head!r}"
            )
        target_name = (
            "decoder_cross_weight"
            if is_joint and decoder_head == "cross"
            else "decoder_weight"
        )
        target = handle.get_tensor(target_name)[:, feature_id].float()

    token_norms = token_decoder.norm(dim=0).clamp_min(1e-12)
    dictionary = token_decoder / token_norms
    target = target / target.norm().clamp_min(1e-12)
    selected: list[int] = []
    selected_mask = torch.zeros(
        dictionary.shape[1],
        dtype=torch.bool,
    )
    residual = target.clone()
    output: dict[str, Any] = {}
    budget_set = set(budgets)
    for iteration in range(1, max(budgets) + 1):
        correlations = dictionary.T @ residual
        correlations[selected_mask] = 0
        feature = int(correlations.abs().argmax())
        selected.append(feature)
        selected_mask[feature] = True
        basis = dictionary[:, selected]
        coefficients = torch.linalg.lstsq(basis, target).solution
        reconstruction = basis @ coefficients
        residual = target - reconstruction
        if iteration in budget_set:
            output[str(iteration)] = {
                "budget": iteration,
                "cosine_to_cross_direction": float(
                    F.cosine_similarity(
                        reconstruction,
                        target,
                        dim=0,
                    )
                ),
                "explained_direction_energy": float(
                    1 - residual.square().sum()
                ),
                "residual_norm": float(residual.norm()),
                "selected_token_feature_ids": list(selected),
                "coefficients_in_unit_atom_basis": [
                    float(value) for value in coefficients
                ],
            }
    return {
        "method": (
            "signed orthogonal matching pursuit on unit-normalized decoder "
            "columns; direct access to the target Cross decoder direction"
        ),
        "target_cross_feature_id": feature_id,
        "budgets": output,
    }


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval6 = eval_root / "document_linking"
    with np.load(eval6 / "features.npz") as loaded:
        global_ids = loaded["global_chunk_ids"]
        cross = _csr(
            loaded["cross_indices"],
            loaded["cross_values"],
        )
        token = _csr(
            loaded["token_indices"],
            loaded["token_values"],
        )
    evidence_dir = eval_root / "dictionary_utilization/feature_evidence"
    chunk_rows = _load_chunk_rows(evidence_dir)
    document_map = {
        chunk_id: str(row["doc_id"])
        for chunk_id, row in chunk_rows.items()
    }
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

    public, details = _load_candidate_artifacts(eval_root)
    candidate_rows = []
    for row in public:
        feature_id = int(row["id"])
        if not REASON_RE.search(str(row["explanation"])):
            continue
        if float(row["score"]) < args.minimum_autointerp_score:
            continue
        autointerp_rates = _autointerp_rates(
            details=details,
            feature_id=feature_id,
        )
        if autointerp_rates["tpr"] < args.minimum_autointerp_tpr:
            continue
        positive_rows = int(cross[:, feature_id].getnnz())
        if positive_rows < args.minimum_positive_rows:
            continue
        candidate_rows.append(
            {
                "feature_id": feature_id,
                "explanation": str(row["explanation"]),
                "core_reasoning_semantics": bool(
                    CORE_REASON_RE.search(str(row["explanation"]))
                ),
                "autointerp_score": float(row["score"]),
                "autointerp_tpr": autointerp_rates["tpr"],
                "autointerp_tnr": autointerp_rates["tnr"],
                "autointerp_counts": autointerp_rates,
                "natural_positive_rows": positive_rows,
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            -row["autointerp_score"],
            -row["autointerp_tpr"],
            -row["natural_positive_rows"],
            row["feature_id"],
        )
    )
    if not candidate_rows:
        raise RuntimeError("no natural reasoning candidates passed selection")

    budgets = [
        int(value)
        for value in args.sparse_budgets.split(",")
        if value.strip()
    ]
    evaluated = []
    for candidate_index, row in enumerate(candidate_rows):
        feature_id = int(row["feature_id"])
        labels = cross[:, feature_id].toarray().ravel() > 0
        cross_values = cross[:, feature_id].toarray().ravel().astype(np.float64)
        cross_auc = float(roc_auc_score(labels, cross_values))
        baselines = {}
        for budget in (1, *budgets):
            result = _cross_fitted_prediction(
                token=token,
                labels=labels,
                folds=folds,
                budget=budget,
                screen_features=args.screen_features,
                logistic_c=args.logistic_c,
                seed=args.seed + feature_id * 17 + budget,
            )
            predictions = result.pop("_predictions")
            result["oof_auc_95ci"] = _cluster_bootstrap(
                labels,
                predictions,
                documents,
                samples=args.bootstrap_samples,
                seed=args.seed + feature_id + budget,
            )
            result["cross_minus_token_auc"] = _paired_bootstrap_delta(
                labels,
                cross_values,
                predictions,
                documents,
                samples=args.bootstrap_samples,
                seed=args.seed + feature_id + 1000 + budget,
            )
            result["prevalence_matched_recovery"] = (
                _prevalence_matched_recovery(labels, predictions)
            )
            baselines[str(budget)] = result
        evaluated.append(
            {
                **row,
                "natural_cross_auc": cross_auc,
                "natural_cross_average_precision": float(
                    average_precision_score(labels, cross_values)
                ),
                "token_counterparts": baselines,
                "best_token_budget": max(
                    baselines,
                    key=lambda key: baselines[key]["oof_auc"],
                ),
                "best_token_auc": max(
                    result["oof_auc"] for result in baselines.values()
                ),
            }
        )
        print(
            f"[reason-counterpart] {candidate_index + 1}/"
            f"{len(candidate_rows)} feature={feature_id} "
            f"best-token={evaluated[-1]['best_token_auc']:.3f}",
            flush=True,
        )

    # Candidate selection is external and frozen; choose the largest
    # cross-vs-token gap among all eligible natural reasoning candidates.
    evaluated.sort(
        key=lambda row: (
            row["natural_cross_auc"] - row["best_token_auc"],
            row["autointerp_score"],
        ),
        reverse=True,
    )
    shortlist = evaluated[: min(8, len(evaluated))]
    configured_headline = next(
        (
            row
            for row in evaluated
            if int(row["feature_id"]) == args.headline_feature
        ),
        None,
    )
    if configured_headline is None:
        raise RuntimeError(
            f"configured headline feature {args.headline_feature} did not "
            "pass the frozen candidate filters"
        )
    if not any(
        int(row["feature_id"]) == args.headline_feature
        for row in shortlist
    ):
        shortlist = [
            configured_headline,
            *shortlist[: max(0, 7)],
        ]

    cross_checkpoint = Path(args.sae_root) / "cross/checkpoints/best"
    if not cross_checkpoint.exists():
        cross_checkpoint = Path(args.sae_root) / "cross"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    encoder = FrozenEncoder(
        cross_checkpoint,
        device,
        decoder_head=args.decoder_head,
    )
    validation_root = (
        eval_root.parent.parent
        / "data/layer21_validation_cache_exact10000128"
    )
    if not validation_root.exists():
        validation_root = (
            Path(args.sae_root).parents[1]
            / "data/layer21_validation_cache_exact10000128"
        )
    importance = {}
    importance_rows = [
        row
        for row in shortlist
        if int(row["feature_id"]) == args.headline_feature
    ]
    repeat_results: dict[int, list[dict[str, Any]]] = {
        int(row["feature_id"]): [] for row in importance_rows
    }
    all_mean_a, all_mean_b, all_validation_docs = _load_validation_pairs(
        validation_root,
        target_pairs=args.importance_pairs * args.importance_repeats,
        seed=args.seed,
    )
    validation_documents: set[str] = set(
        all_validation_docs.tolist()
    )
    for repeat in range(args.importance_repeats):
        repeat_seed = args.seed + repeat * 1_000_003
        start = repeat * args.importance_pairs
        stop = start + args.importance_pairs
        mean_a = all_mean_a[start:stop]
        mean_b = all_mean_b[start:stop]
        repeat_docs = all_validation_docs[start:stop]
        source = torch.cat((mean_a, mean_b), dim=0)
        target = torch.cat((mean_b, mean_a), dim=0)
        direction_documents = np.concatenate(
            (repeat_docs, repeat_docs)
        )
        for row in importance_rows:
            feature_id = int(row["feature_id"])
            current = _importance_for_feature(
                encoder=encoder,
                feature_id=feature_id,
                source=source,
                target=target,
                groups=direction_documents,
                controls=args.importance_controls,
                bootstrap_samples=args.bootstrap_samples,
                seed=repeat_seed,
            )
            current["repeat"] = repeat
            current["pair_sample_seed"] = repeat_seed
            repeat_results[feature_id].append(current)
    for row in shortlist:
        feature_id = int(row["feature_id"])
        if feature_id in repeat_results:
            importance[str(feature_id)] = _aggregate_importance_repeats(
                repeat_results[feature_id]
            )
        else:
            importance[str(feature_id)] = {
                "status": "not_evaluated_non_core_screening_candidate",
                "feature_id": feature_id,
            }
        row["partner_reconstruction_importance"] = importance[str(feature_id)]

    qualified = [
        row
        for row in shortlist
        if (
            int(row["feature_id"]) == args.headline_feature
            and
            row["natural_cross_auc"] - row["best_token_auc"] >= 0.05
            and row["token_counterparts"][row["best_token_budget"]][
                "cross_minus_token_auc"
            ]["95ci"][0]
            > 0
            and row["partner_reconstruction_importance"].get("status")
            == "measured"
            and row["partner_reconstruction_importance"][
                "positive_effect_ci_repeat_fraction"
            ]
            >= 0.5
        )
    ]
    headline = next(
        row
        for row in shortlist
        if int(row["feature_id"]) == args.headline_feature
    )
    result = {
        "format": FORMAT,
        "complete": True,
        "cross_decoder_head": args.decoder_head or "legacy_single_head",
        "feature_identity": "(decoder_head, shared_feature_id) for Joint checkpoints",
        "selection": {
            "source": (
                "frozen exact-1000 explanations and balanced exact-128 "
                "Context AutoInterp scores, before this counterpart test"
            ),
            "reason_regex": REASON_RE.pattern,
            "minimum_autointerp_score": args.minimum_autointerp_score,
            "minimum_autointerp_tpr": args.minimum_autointerp_tpr,
            "minimum_natural_positive_rows": args.minimum_positive_rows,
            "eligible_candidates": len(candidate_rows),
            "headline_rule": (
                "Feature ID supplied before this run and justified only by "
                "its frozen, pre-existing explanation; counterpart AUC and "
                "ablation results do not choose the headline."
            ),
            "configured_headline_feature": args.headline_feature,
            "core_reasoning_regex": CORE_REASON_RE.pattern,
        },
        "natural_data": {
            "source": "Eval-6 held-out Pile chunks",
            "rows": int(cross.shape[0]),
            "documents": int(len(np.unique(documents))),
            "cross_validation": (
                f"{args.cv_folds}-fold deterministic document-hash split"
            ),
            "token_representation": (
                "top-128 mean-pooled thresholded Token SAE activations"
            ),
            "importance_validation": {
                "real_adjacent_pairs_per_repeat":
                    args.importance_pairs,
                "directional_examples_per_repeat":
                    2 * args.importance_pairs,
                "repeats": args.importance_repeats,
                "distinct_documents_across_repeats":
                    len(validation_documents),
            },
        },
        "claim_scope": (
            "For the selected natural Cross axis, no equally accurate Token "
            "counterpart was found under best single and 4/16/64-feature "
            "linear budgets on the tested held-out corpus. This is not an "
            "impossibility claim about arbitrary nonlinear decoders."
        ),
        "headline_feature": int(headline["feature_id"]),
        "qualified_features": [
            int(row["feature_id"]) for row in qualified
        ],
        "features": shortlist,
    }
    atomic_json_dump(result, output_dir / "natural_counterpart_results.json")
    _atomic_jsonl(output_dir / "natural_counterpart_features.jsonl", shortlist)
    examples = {
        "format": "chunk-saes-natural-reason-examples-v1",
        "feature_id": int(headline["feature_id"]),
        "decoder_head": args.decoder_head or "legacy_single_head",
        "explanation": headline["explanation"],
        "selection": (
            "Highest activations with at most one held-out chunk per "
            "document; no manual cherry-picking."
        ),
        "examples": _qualitative_examples(
            feature=headline,
            cross=cross,
            global_ids=global_ids,
            chunk_rows=chunk_rows,
            limit=8,
        ),
        "token_single_counterpart_errors": _counterpart_examples(
            feature=headline,
            cross=cross,
            token=token,
            global_ids=global_ids,
            chunk_rows=chunk_rows,
            limit=5,
        ),
    }
    atomic_json_dump(examples, output_dir / "qualitative_examples.json")

    token_checkpoint = Path(args.sae_root) / "token/checkpoints/best"
    if not token_checkpoint.exists():
        token_checkpoint = Path(args.sae_root) / "token"
    decoder_approximation = _decoder_sparse_approximation(
        token_checkpoint=token_checkpoint,
        cross_checkpoint=cross_checkpoint,
        feature_id=int(headline["feature_id"]),
        budgets=[1, *budgets],
        decoder_head=args.decoder_head,
    )
    atomic_json_dump(
        decoder_approximation,
        output_dir / "decoder_sparse_approximation.json",
    )

    feature = headline
    labels = ["Cross axis", "Token single", "Token 4", "Token 16", "Token 64"]
    values = [
        feature["natural_cross_auc"],
        feature["token_counterparts"]["1"]["oof_auc"],
        feature["token_counterparts"]["4"]["oof_auc"],
        feature["token_counterparts"]["16"]["oof_auc"],
        feature["token_counterparts"]["64"]["oof_auc"],
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.bar(
        labels,
        values,
        color=[
            METHOD_COLORS["cross"],
            *[METHOD_COLORS["token"]] * 4,
        ],
    )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Document-cross-fitted AUC")
    ax.set_title(
        f"Natural Cross feature #{feature['feature_id']} and Token counterparts"
    )
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values, strict=True):
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
            plot_dir / f"natural_reason_counterpart.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(fig)

    audit = {
        "format": "chunk-saes-natural-reason-counterpart-audit-v1",
        "complete": True,
        "checks": {
            "candidate_selection_preexisting": True,
            "natural_heldout_chunks": True,
            "document_hash_cross_fitting": True,
            "full_token_dictionary_search": True,
            "budgets": [1, *budgets],
            "real_adjacent_pair_importance": True,
            "importance_repeats": args.importance_repeats,
        },
        "issues": [],
    }
    atomic_json_dump(audit, output_dir / "natural_counterpart_audit.json")
    manifest = write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": {
                "eval6_features_sha256": file_sha256(
                    eval6 / "features.npz"
                ),
                "sae_root": str(Path(args.sae_root).resolve()),
                "seed": args.seed,
                "cv_folds": args.cv_folds,
                "sparse_budgets": [1, *budgets],
                "logistic_c": args.logistic_c,
                "importance_pairs": args.importance_pairs,
                "importance_controls": args.importance_controls,
                "importance_repeats": args.importance_repeats,
            },
            "files": {
                "results": file_record(
                    output_dir / "natural_counterpart_results.json",
                    relative_to=output_dir,
                ),
                "features": file_record(
                    output_dir / "natural_counterpart_features.jsonl",
                    relative_to=output_dir,
                ),
                "audit": file_record(
                    output_dir / "natural_counterpart_audit.json",
                    relative_to=output_dir,
                ),
                "qualitative_examples": file_record(
                    output_dir / "qualitative_examples.json",
                    relative_to=output_dir,
                ),
                "decoder_sparse_approximation": file_record(
                    output_dir / "decoder_sparse_approximation.json",
                    relative_to=output_dir,
                ),
                "plot_png": file_record(
                    plot_dir / "natural_reason_counterpart.png",
                    relative_to=output_dir,
                ),
                "plot_pdf": file_record(
                    plot_dir / "natural_reason_counterpart.pdf",
                    relative_to=output_dir,
                ),
            },
        },
        output_dir / "natural_counterpart_manifest.json",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "headline_feature": result["headline_feature"],
                "qualified_features": result["qualified_features"],
                "artifact_digest": manifest["artifact_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
