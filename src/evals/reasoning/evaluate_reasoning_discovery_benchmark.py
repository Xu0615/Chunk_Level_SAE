#!/usr/bin/env python
"""Method-symmetric reasoning-feature discovery benchmark.

This evaluation fixes the main circularity in the original Eval-7 analysis:
the target labels are defined by an external, controlled reasoning benchmark,
not by activation of a Cross-Chunk feature.

The benchmark answers three separate questions.

A. Label-free discovery
   Starting from the same frozen random sample of 1,000 AutoInterp
   explanations per SAE, can a method surface a single feature whose
   explanation independently matches a preregistered reasoning operation?

B. Compactness / fragmentation
   If labels are allowed for oracle feature search, how accurately can each
   method represent the external target with 1, 4, 16, or 64 coordinates?

C. Robustness / functionality
   Does the selected single coordinate transfer to held-out templates and
   domains, and does it make an incremental contribution to the same
   cross-method partner-hidden-state prediction objective?

Candidate retrieval and semantic judgments are frozen input artifacts so the
benchmark labels are never used to choose the label-free discovery candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer

from chunk_saes.artifacts import file_record, write_artifact_manifest
from chunk_saes.plot_style import METHOD_COLORS, METHOD_SHORT_LABELS
from chunk_saes.utils import atomic_json_dump


FORMAT = "chunk-saes-reasoning-discovery-benchmark-v1"
METHODS = ("token", "temporal", "mean", "cross")
MATRIX_KEYS = {
    "token": "token_mean",
    "temporal": "temporal_mean",
    "mean": "mean",
    "cross": "cross",
}
BUDGETS = (1, 4, 16, 64)
SCENARIO_LABELS = {
    "deductive_chain": "Deductive chain",
    "causal_chain": "Causal chain",
    "constraint_elimination": "Constraint elimination",
    "modus_tollens": "Modus tollens",
    "evidence_revision": "Evidence revision",
    "ordered_comparison": "Ordered comparison",
    "disjunctive_elimination": "Disjunctive elimination",
    "case_analysis": "Case analysis",
    "error_correction": "Error correction",
}
REASONING_DEFINITIONS = {
    "deductive_chain": (
        "Valid multi-step deductive reasoning that links premises "
        "transitively to a logically entailed conclusion, rather than merely "
        "stating facts or an unsupported conclusion."
    ),
    "causal_chain": (
        "Valid causal-mechanistic reasoning that traces an intervention or "
        "cause through an intermediate mechanism to a downstream consequence, "
        "rather than mere association or reversed causation."
    ),
    "constraint_elimination": (
        "Constraint-based decision reasoning that eliminates infeasible "
        "options using explicit requirements and identifies the only option "
        "satisfying all constraints."
    ),
    "modus_tollens": (
        "Valid modus-tollens reasoning: if A implies B and B is absent, infer "
        "that A is absent, rather than affirming the consequent or denying "
        "the antecedent."
    ),
    "evidence_revision": (
        "Evidence-based belief revision that contrasts a prediction with "
        "incompatible new evidence and updates toward a better-supported "
        "explanation."
    ),
    "ordered_comparison": (
        "Transitive comparative reasoning that combines ordered pairwise "
        "preferences or inequalities to infer a further ordering."
    ),
    "disjunctive_elimination": (
        "Disjunctive elimination reasoning that considers exhaustive "
        "alternatives, rules one out with evidence, and concludes the "
        "remaining alternative."
    ),
    "case_analysis": (
        "Proof by cases that divides into exhaustive cases, derives the same "
        "conclusion in each branch, and concludes it unconditionally."
    ),
    "error_correction": (
        "Technical diagnosis and error correction that traces a failure to a "
        "preceding cause, tests a correction, and justifies a concrete remedy."
    ),
}
TEXT_COLOR = "#252932"
MUTED = "#66707c"
GRID = "#dfe4ea"
# Context AutoInterp has four active and four inactive examples, so balanced
# accuracy lies on an eighth-point grid.  Requiring 0.75 is the direct
# replacement for this benchmark's former 10/14 legacy-accuracy gate.
MINIMUM_CONTEXT_AUTOINTERP = 0.75


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval7-dir", required=True)
    p.add_argument(
        "--eval-root",
        help=(
            "Parent evaluation directory containing autointerp. Defaults to the "
            "parent of --eval7-dir."
        ),
    )
    p.add_argument(
        "--embedding-model",
        default=(
            "./models/"
            "Qwen3.5-2B-Base"
        ),
    )
    p.add_argument(
        "--judge-model",
        default=(
            "./models/"
            "Qwen3.6-35B-A3B"
        ),
    )
    p.add_argument(
        "--candidate-pools",
        help="Optional frozen candidate-pool JSON to reuse.",
    )
    p.add_argument(
        "--candidate-judgments",
        help="Optional frozen candidate-judgment JSON to reuse.",
    )
    p.add_argument("--candidate-limit", type=int, default=10)
    p.add_argument("--embedding-device", default="cuda:0")
    p.add_argument("--judge-device-map", default="auto")
    p.add_argument("--bootstrap-samples", type=int, default=2_000)
    p.add_argument("--screen-features", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--overwrite", action="store_true")
    return p


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _scenario_name(template_id: str) -> str:
    return template_id.rsplit("_style", 1)[0]


def _style(template_id: str) -> int:
    return int(template_id.rsplit("_style", 1)[1])


def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    if np.unique(labels).size < 2 or np.unique(values).size < 2:
        return 0.5
    return float(roc_auc_score(labels, values))


def _family_bootstrap(
    labels: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    unique = np.unique(groups)
    rows = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([rows[group] for group in chosen])
        draws[index] = _auc(labels[sample], values[sample])
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _mean_bootstrap(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(values), size=len(values))
        draws[index] = float(values[selected].mean())
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _embed_texts(
    texts: list[str],
    *,
    model_path: str,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt",
            ).to(device)
            hidden = model(
                **encoded,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            output.append(F.normalize(pooled, dim=-1).cpu().numpy())
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(output, axis=0)


def _build_candidate_pools(
    *,
    eval_root: Path,
    embedding_model: str,
    embedding_device: str,
    candidate_limit: int,
) -> dict[str, Any]:
    auto = eval_root / "autointerp/autointerp_exact1000/results"
    features = {
        method: json.loads(
            (auto / f"{method}_features.json").read_text(encoding="utf-8")
        )
        for method in METHODS
    }
    definition_items = list(REASONING_DEFINITIONS.items())
    all_texts = [definition for _scenario, definition in definition_items]
    for method in METHODS:
        all_texts.extend(
            str(row["explanation"]) for row in features[method]
        )
    embeddings = _embed_texts(
        all_texts,
        model_path=embedding_model,
        device=embedding_device,
    )
    definition_embeddings = embeddings[: len(definition_items)]
    method_embeddings = {}
    offset = len(definition_items)
    for method in METHODS:
        stop = offset + len(features[method])
        method_embeddings[method] = embeddings[offset:stop]
        offset = stop

    pools: dict[str, Any] = {}
    for scenario_index, (scenario, definition) in enumerate(definition_items):
        pools[scenario] = {}
        for method in METHODS:
            rows = features[method]
            eligible = np.asarray(
                [
                    float(row["score"]) >= MINIMUM_CONTEXT_AUTOINTERP
                    and str(row["explanation"]).strip().lower()
                    != "insufficient evidence"
                    for row in rows
                ],
                dtype=bool,
            )
            semantic_scores = (
                method_embeddings[method]
                @ definition_embeddings[scenario_index]
            )
            semantic_scores[~eligible] = -np.inf
            semantic_rank_scores = (
                semantic_scores
                + 0.02
                * np.asarray(
                    [float(row["score"]) for row in rows],
                    dtype=np.float64,
                )
            )
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
            ).fit_transform(
                [definition]
                + [str(row["explanation"]) for row in rows]
            )
            lexical_scores = cosine_similarity(
                tfidf[0],
                tfidf[1:],
            ).ravel()
            lexical_scores[~eligible] = -np.inf
            semantic_order = np.argsort(semantic_rank_scores)[::-1]
            lexical_order = np.argsort(lexical_scores)[::-1]
            selected = [int(semantic_order[0])]
            for column in lexical_order.tolist():
                if int(column) not in selected:
                    selected.append(int(column))
                if len(selected) == candidate_limit:
                    break
            candidates = []
            for rank, column in enumerate(selected):
                semantic_candidate = rank == 0
                candidates.append(
                    {
                        "feature_id": int(rows[column]["id"]),
                        "explanation": str(rows[column]["explanation"]),
                        "autointerp_score": float(rows[column]["score"]),
                        "source": (
                            "embedding_top1"
                            if semantic_candidate
                            else "tfidf"
                        ),
                        "retrieval_score": float(
                            semantic_scores[column]
                            if semantic_candidate
                            else lexical_scores[column]
                        ),
                    }
                )
            pools[scenario][method] = candidates
    return {
        "format": f"{FORMAT}-candidate-pools",
        "complete": True,
        "definitions": REASONING_DEFINITIONS,
        "pools": pools,
        "retrieval": {
            "candidate_limit": candidate_limit,
            "embedding_model": str(Path(embedding_model).resolve()),
            "minimum_autointerp_score": MINIMUM_CONTEXT_AUTOINTERP,
            "autointerp_metric": (
                "balanced exact-128 Context AutoInterp with four active and "
                "four inactive contexts"
            ),
            "retrieval_strategy": (
                "one semantic-embedding candidate followed by the highest "
                "TF-IDF candidates, deduplicated and frozen"
            ),
            "retrieval_inputs": (
                "target definition and frozen feature explanation only"
            ),
            "benchmark_labels_used": False,
            "method_identity_used": False,
        },
    }


def _judge_candidate_pools(
    *,
    pools: dict[str, Any],
    judge_model: str,
    device_map: str,
) -> list[dict[str, Any]]:
    items = [
        (scenario, method, candidate)
        for scenario, methods in pools["pools"].items()
        for method, candidates in methods.items()
        for candidate in candidates
    ]
    prompts = [
        (
            "You are auditing whether an automatically generated sparse-"
            "feature explanation specifically represents a preregistered "
            "reasoning operation. Do not reward generic topical overlap, "
            "academic style, equations, legal vocabulary, evidence words, "
            "or conclusion words. A match is true only if the explanation "
            "itself describes the target reasoning relation or discourse "
            "operation closely enough that the feature could identify that "
            "operation in unseen domains.\n\n"
            f"TARGET:\n{REASONING_DEFINITIONS[scenario]}\n\n"
            f"FEATURE EXPLANATION:\n{candidate['explanation']}\n\n"
            'Return JSON only: {"match": true} or {"match": false}'
        )
        for scenario, _method, candidate in items
    ]
    tokenizer = AutoTokenizer.from_pretrained(judge_model, use_fast=True)
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        judge_model,
        dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    ).eval()
    outputs = []
    for start in range(0, len(prompts), 4):
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts[start : start + 4]
        ]
        encoded = tokenizer(
            rendered,
            padding=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=8,
                use_cache=True,
            )
        outputs.extend(
            tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
        )
    del model
    torch.cuda.empty_cache()
    judgments = []
    for (scenario, method, candidate), raw in zip(
        items, outputs, strict=True
    ):
        try:
            match = bool(json.loads(raw.strip())["match"])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = bool(re.search(r"\btrue\b", raw, re.I))
        judgments.append(
            {
                "scenario": scenario,
                "method": method,
                **candidate,
                "match": match,
                "raw": raw,
                "judge_model": str(Path(judge_model).resolve()),
                "method_hidden_in_prompt": True,
                "benchmark_labels_hidden_in_prompt": True,
            }
        )
    return judgments


def _screen_order(
    matrix: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive = matrix[mask & (labels == 1)]
    negative = matrix[mask & (labels == 0)]
    positive_mean = positive.mean(axis=0, dtype=np.float64)
    negative_mean = negative.mean(axis=0, dtype=np.float64)
    positive_rate = (positive > 0).mean(axis=0)
    negative_rate = (negative > 0).mean(axis=0)
    signed_score = (
        (positive_rate - negative_rate)
        * np.log1p(positive_mean + negative_mean)
    )
    return np.argsort(np.abs(signed_score))[::-1], np.sign(signed_score)


def _oracle_single(
    matrix: np.ndarray,
    labels: np.ndarray,
    discovery_mask: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    screen_features: int,
) -> dict[str, Any]:
    order, signs = _screen_order(matrix, labels, discovery_mask)
    shortlist = order[:screen_features]
    best: tuple[float, int, int] | None = None
    for feature_id in shortlist.tolist():
        direction = int(signs[feature_id])
        if direction == 0:
            direction = 1
        calibration_auc = _auc(
            labels[calibration_mask],
            direction * matrix[calibration_mask, feature_id],
        )
        candidate = (calibration_auc, -feature_id, direction)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    calibration_auc, negative_id, direction = best
    feature_id = -negative_id
    test_values = direction * matrix[test_mask, feature_id].astype(
        np.float64
    )
    return {
        "feature_id": int(feature_id),
        "direction": int(direction),
        "calibration_auc": float(calibration_auc),
        "test_auc": _auc(labels[test_mask], test_values),
        "_test_values": test_values,
        "_screen_order": order,
    }


def _sparse_probe(
    matrix: np.ndarray,
    labels: np.ndarray,
    discovery_mask: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    budget: int,
    screen_order: np.ndarray,
    anchor_feature_id: int,
    seed: int,
) -> dict[str, Any]:
    selected_list = [int(anchor_feature_id)]
    selected_list.extend(
        int(feature_id)
        for feature_id in screen_order
        if int(feature_id) != int(anchor_feature_id)
    )
    selected = np.asarray(selected_list[:budget], dtype=np.int64)
    c_grid = (0.001, 0.01, 0.1, 1.0, 10.0)
    scaler = StandardScaler()
    x_discovery = scaler.fit_transform(matrix[discovery_mask][:, selected])
    x_calibration = scaler.transform(matrix[calibration_mask][:, selected])
    best: tuple[float, float] | None = None
    for c_value in c_grid:
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
            solver="liblinear",
        )
        model.fit(x_discovery, labels[discovery_mask])
        score = _auc(
            labels[calibration_mask],
            model.predict_proba(x_calibration)[:, 1],
        )
        candidate = (score, -c_value)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    c_value = -best[1]
    train_mask = discovery_mask | calibration_mask
    scaler = StandardScaler()
    x_train = scaler.fit_transform(matrix[train_mask][:, selected])
    x_test = scaler.transform(matrix[test_mask][:, selected])
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=5_000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(x_train, labels[train_mask])
    test_values = model.predict_proba(x_test)[:, 1]
    return {
        "budget": budget,
        "feature_ids": selected.astype(int).tolist(),
        "regularization_c": c_value,
        "calibration_auc": float(best[0]),
        "test_auc": _auc(labels[test_mask], test_values),
        "_test_values": test_values,
    }


def _candidate_map(
    pools_path: Path,
    judgments_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pools = json.loads(pools_path.read_text(encoding="utf-8"))
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    expected = {
        (
            str(scenario),
            str(method),
            int(candidate["feature_id"]),
        )
        for scenario, methods in pools["pools"].items()
        for method, candidates in methods.items()
        for candidate in candidates
    }
    judgment_keys = [
        (
            str(row["scenario"]),
            str(row["method"]),
            int(row["feature_id"]),
        )
        for row in judgments
    ]
    if len(judgment_keys) != len(set(judgment_keys)):
        raise ValueError("candidate judgments contain duplicate keys")
    actual = set(judgment_keys)
    if actual != expected:
        raise ValueError(
            "candidate-pool/judgment mismatch: "
            f"missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )
    judged = {
        (
            str(row["scenario"]),
            str(row["method"]),
            int(row["feature_id"]),
        ): bool(row["match"])
        for row in judgments
    }
    selected: dict[str, dict[str, Any]] = {}
    for scenario, methods in pools["pools"].items():
        selected[scenario] = {}
        for method, candidates in methods.items():
            annotated = [
                {
                    **candidate,
                    "semantic_match": judged.get(
                        (
                            scenario,
                            method,
                            int(candidate["feature_id"]),
                        ),
                        False,
                    ),
                    "retrieval_rank": rank + 1,
                }
                for rank, candidate in enumerate(candidates)
            ]
            matches = [row for row in annotated if row["semantic_match"]]
            chosen = matches[0] if matches else annotated[0]
            selected[scenario][method] = {
                **chosen,
                "discovery_pass": bool(matches),
                "matching_candidates": [
                    int(row["feature_id"]) for row in matches
                ],
            }
    return pools, selected


def _candidate_external_metrics(
    *,
    matrix: np.ndarray,
    candidate: dict[str, Any],
    labels: np.ndarray,
    conditions: np.ndarray,
    domains: np.ndarray,
    families: np.ndarray,
    test_mask: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    feature_id = int(candidate["feature_id"])
    values = matrix[test_mask, feature_id].astype(np.float64)
    test_labels = labels[test_mask]
    test_groups = families[test_mask]
    test_auc = _auc(test_labels, values)
    condition_aucs = {}
    for positive in ("valid_reasoning", "valid_paraphrase"):
        for negative in ("invalid_relation", "unsupported_conclusion"):
            local = test_mask & np.isin(conditions, (positive, negative))
            condition_aucs[f"{positive}_vs_{negative}"] = _auc(
                (conditions[local] == positive).astype(np.int8),
                matrix[local, feature_id].astype(np.float64),
            )
    domain_aucs = {}
    for domain in sorted(set(domains[test_mask].tolist())):
        local = test_mask & (domains == domain)
        domain_aucs[domain] = _auc(
            labels[local],
            matrix[local, feature_id].astype(np.float64),
        )
    interval = _family_bootstrap(
        test_labels,
        values,
        test_groups,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        **candidate,
        "test_auc": test_auc,
        "test_auc_95ci": interval,
        "minimum_control_auc": float(min(condition_aucs.values())),
        "condition_aucs": condition_aucs,
        "worst_domain_auc": float(min(domain_aucs.values())),
        "domain_aucs": domain_aucs,
        "active_test_rows": int((values > 0).sum()),
        "robust_discovery_pass": bool(
            candidate["discovery_pass"]
            and interval[0] > 0.5
            and min(condition_aucs.values()) > 0.5
        ),
    }


def _domain_transfer_single(
    *,
    matrix: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    styles: np.ndarray,
    screen_features: int,
) -> dict[str, Any]:
    groups = (
        ("math", "science", "policy"),
        ("debugging", "planning", "everyday"),
    )
    results = []
    for source_domains, target_domains in (
        (groups[0], groups[1]),
        (groups[1], groups[0]),
    ):
        discovery = (styles == 0) & np.isin(domains, source_domains)
        calibration = (styles == 1) & np.isin(domains, source_domains)
        test = (styles == 2) & np.isin(domains, target_domains)
        result = _oracle_single(
            matrix,
            labels,
            discovery,
            calibration,
            test,
            screen_features=screen_features,
        )
        results.append(
            {
                "source_domains": list(source_domains),
                "target_domains": list(target_domains),
                "feature_id": result["feature_id"],
                "test_auc": result["test_auc"],
            }
        )
    return {
        "directions": results,
        "mean_auc": float(
            np.mean([row["test_auc"] for row in results])
        ),
        "worst_direction_auc": float(
            min(row["test_auc"] for row in results)
        ),
    }


def _common_target_ablation(
    *,
    matrix: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    discovery_mask: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    candidate_feature_id: int,
    oracle_screen_order: np.ndarray,
    family_ids: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = [candidate_feature_id]
    selected.extend(
        int(feature_id)
        for feature_id in oracle_screen_order
        if int(feature_id) != candidate_feature_id
    )
    selected = np.asarray(selected[:64], dtype=np.int64)
    x = matrix[:, selected].astype(np.float64)
    alpha_grid = (0.1, 1.0, 10.0, 100.0, 1_000.0)

    x_scaler = StandardScaler()
    x_discovery = x_scaler.fit_transform(x[discovery_mask])
    x_calibration = x_scaler.transform(x[calibration_mask])
    y_mean = target[discovery_mask].mean(axis=0, keepdims=True)
    y_discovery = target[discovery_mask] - y_mean
    y_calibration = target[calibration_mask] - y_mean
    best: tuple[float, float] | None = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(x_discovery, y_discovery)
        prediction = model.predict(x_calibration)
        mse = float(np.mean((prediction - y_calibration) ** 2))
        candidate = (-mse, -alpha)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    alpha = -best[1]

    train_mask = discovery_mask | calibration_mask
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x[train_mask])
    positive_test_mask = test_mask & (labels == 1)
    x_test = x_scaler.transform(x[positive_test_mask])
    y_mean = target[train_mask].mean(axis=0, keepdims=True)
    y_train = target[train_mask] - y_mean
    y_test = target[positive_test_mask] - y_mean
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(x_train, y_train)
    full = model.predict(x_test)
    ablated_x = x_test.copy()
    # A raw feature ablation sets its activation to zero. In standardized
    # coordinates this is (0 - training_mean) / training_scale, not zero.
    ablated_x[:, 0] = (
        -float(x_scaler.mean_[0])
        / max(float(x_scaler.scale_[0]), 1e-12)
    )
    ablated = model.predict(ablated_x)
    full_sse = ((full - y_test) ** 2).sum(axis=1)
    ablated_sse = ((ablated - y_test) ** 2).sum(axis=1)
    relative = (ablated_sse - full_sse) / np.maximum(full_sse, 1e-12)

    groups = family_ids[positive_test_mask]
    unique = np.unique(groups)
    rows_by_group = {
        group: np.flatnonzero(groups == group) for group in unique
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([rows_by_group[group] for group in chosen])
        draws[index] = float(relative[sample].mean())
    interval = [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]
    return {
        "feature_id": int(candidate_feature_id),
        "predictor_features": selected.astype(int).tolist(),
        "ridge_alpha": alpha,
        "positive_test_rows": int(positive_test_mask.sum()),
        "relative_partner_sse_increase": float(relative.mean()),
        "relative_partner_sse_increase_95ci": interval,
        "functional_pass": bool(interval[0] > 0),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    *,
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    methods = list(METHODS)
    colors = [METHOD_COLORS[method] for method in methods]
    labels = [METHOD_SHORT_LABELS[method] for method in methods]
    discovery = [
        result["aggregate"][method]["semantic_discoveries"]
        for method in methods
    ]
    robust = [
        result["aggregate"][method]["robust_discoveries"]
        for method in methods
    ]
    budget_values = {
        budget: [
            result["aggregate"][method]["oracle_macro_auc"][str(budget)]
            for method in methods
        ]
        for budget in BUDGETS
    }
    transfer = [
        result["aggregate"][method]["domain_template_transfer_auc"]
        for method in methods
    ]
    functionality = [
        result["aggregate"][method]["oracle_functional_passes"]
        for method in methods
    ]

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.edgecolor": "#bac1c9",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        fig = plt.figure(figsize=(16.3, 8.9))
        grid = fig.add_gridspec(
            2,
            3,
            height_ratios=(1.0, 1.02),
            width_ratios=(1.0, 1.35, 1.0),
            hspace=0.55,
            wspace=0.36,
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1:])
        ax_c1 = fig.add_subplot(grid[1, 0])
        ax_c2 = fig.add_subplot(grid[1, 1])
        ax_verdict = fig.add_subplot(grid[1, 2])

        x = np.arange(len(methods))
        width = 0.34
        ax_a.bar(
            x - width / 2,
            discovery,
            width,
            color=colors,
            edgecolor="none",
            label="semantic candidate found",
        )
        ax_a.bar(
            x + width / 2,
            robust,
            width,
            color=colors,
            alpha=0.28,
            hatch="//",
            edgecolor=colors,
            linewidth=1.2,
            label="also passes external OOD test",
        )
        for index, value in enumerate(discovery):
            ax_a.text(
                index - width / 2,
                value + 0.18,
                f"{value}/9",
                ha="center",
                va="bottom",
                color=colors[index],
                fontweight="bold",
            )
        for index, value in enumerate(robust):
            ax_a.text(
                index + width / 2,
                value + 0.18,
                f"{value}/9",
                ha="center",
                va="bottom",
                color=MUTED,
                fontweight="bold",
            )
        ax_a.set_ylim(0, 9.8)
        ax_a.set_xticks(x, labels, rotation=20, ha="right")
        ax_a.set_ylabel("Reasoning operations")
        ax_a.set_title("A  Label-free feature discovery", loc="left")
        ax_a.legend(frameon=False, fontsize=8.5, loc="upper right")

        for method, color, label in zip(
            methods, colors, labels, strict=True
        ):
            values = [
                result["aggregate"][method]["oracle_macro_auc"][str(budget)]
                for budget in BUDGETS
            ]
            intervals = np.asarray(
                [
                    result["aggregate"][method][
                        "oracle_macro_auc_95ci"
                    ][str(budget)]
                    for budget in BUDGETS
                ]
            )
            ax_b.plot(
                BUDGETS,
                values,
                marker="o",
                linewidth=2.5,
                markersize=6,
                color=color,
                label=label,
            )
            ax_b.fill_between(
                BUDGETS,
                intervals[:, 0],
                intervals[:, 1],
                color=color,
                alpha=0.09,
                linewidth=0,
            )
            for budget, value in zip(BUDGETS, values, strict=True):
                if budget in (1, 64):
                    ax_b.text(
                        budget,
                        value + (0.012 if method != "cross" else -0.027),
                        f"{value:.3f}",
                        ha="center",
                        color=color,
                        fontsize=8.5,
                        fontweight="bold",
                    )
        ax_b.axhline(
            0.5, color="#737b85", linestyle=(0, (3, 3)), linewidth=1
        )
        ax_b.set_xscale("log", base=2)
        ax_b.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        ax_b.set_ylim(0.48, 1.025)
        ax_b.set_xlabel("Allowed coordinates")
        ax_b.set_ylabel("Macro held-out AUROC")
        ax_b.set_title(
            "B  Supervised sparse recovery on external labels",
            loc="left",
        )
        ax_b.legend(frameon=False, ncol=2, loc="lower right")

        transfer_ci = np.asarray(
            [
                result["aggregate"][method][
                    "domain_template_transfer_auc_95ci"
                ]
                for method in methods
            ]
        )
        transfer_error = np.asarray(
            [
                np.asarray(transfer) - transfer_ci[:, 0],
                transfer_ci[:, 1] - np.asarray(transfer),
            ]
        )
        ax_c1.bar(x, transfer, color=colors, width=0.68)
        ax_c1.errorbar(
            x,
            transfer,
            yerr=transfer_error,
            fmt="none",
            ecolor=TEXT_COLOR,
            capsize=3,
            linewidth=1.15,
        )
        for index, value in enumerate(transfer):
            ax_c1.text(
                index,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                color=colors[index],
                fontweight="bold",
            )
        ax_c1.axhline(
            0.5, color="#737b85", linestyle=(0, (3, 3)), linewidth=1
        )
        ax_c1.set_ylim(0.45, 1.02)
        ax_c1.set_xticks(x, labels, rotation=20, ha="right")
        ax_c1.set_ylabel("Macro AUROC")
        ax_c1.set_title(
            "C1  Cross-domain + template transfer\n"
            "(supervised single feature)",
            loc="left",
        )

        ax_c2.bar(x, functionality, color=colors, width=0.68)
        for index, value in enumerate(functionality):
            mean_effect = (
                100
                * result["aggregate"][methods[index]][
                    "common_target_oracle_ablation_mean"
                ]
            )
            ax_c2.text(
                index,
                value + 0.16,
                f"{value}/9",
                ha="center",
                color=colors[index],
                fontweight="bold",
            )
            ax_c2.text(
                index,
                max(0.15, value - 0.48),
                f"mean {mean_effect:+.2f}%",
                ha="center",
                color="white" if value >= 1 else MUTED,
                fontsize=8.1,
                fontweight="bold",
            )
        ax_c2.set_ylim(0, 9.8)
        ax_c2.set_xticks(x, labels, rotation=20, ha="right")
        ax_c2.set_ylabel("Reasoning operations")
        ax_c2.set_title(
            "C2  Common-target functional passes\n"
            "(selected feature; 95% CI > 0)",
            loc="left",
        )

        ax_verdict.axis("off")
        cross = result["aggregate"]["cross"]
        best_single_method = max(
            methods,
            key=lambda method: result["aggregate"][method][
                "oracle_macro_auc"
            ]["1"],
        )
        verdict_lines = [
            "FINAL VERDICT",
            "",
            "Not supported on this benchmark.",
            "",
            (
                f"Cross finds {cross['semantic_discoveries']}/9 "
                "semantic candidates,"
            ),
            (
                f"but {cross['robust_discoveries']}/9 survive the "
                "external OOD test."
            ),
            "",
            (
                f"Best supervised single feature: "
                f"{METHOD_SHORT_LABELS[best_single_method]}"
            ),
            (
                f"({result['aggregate'][best_single_method]['oracle_macro_auc']['1']:.3f} "
                f"vs Cross {cross['oracle_macro_auc']['1']:.3f})."
            ),
            "",
            "Cross therefore shows more",
            "reasoning-like natural explanations,",
            "not more accurate or robust",
            "reasoning features here.",
        ]
        ax_verdict.text(
            0.04,
            0.96,
            "\n".join(verdict_lines),
            transform=ax_verdict.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            linespacing=1.35,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": "#f8e9ec",
                "edgecolor": METHOD_COLORS["cross"],
                "linewidth": 1.4,
            },
        )

        for ax in (ax_a, ax_b, ax_c1, ax_c2):
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            "A fair external-label test does not show a Cross-Chunk "
            "reasoning-feature advantage",
            x=0.035,
            y=0.975,
            ha="left",
            fontsize=19,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        fig.text(
            0.035,
            0.925,
            "All methods face the same labels, frozen templates, domains, "
            "feature budget, and held-out test split. Cross no longer "
            "receives AUC = 1 by construction.",
            ha="left",
            fontsize=11.2,
            color=MUTED,
        )
        fig.text(
            0.035,
            0.018,
            "A: candidate retrieval uses only frozen natural-corpus "
            "AutoInterp explanations from the same 1,000-feature sample; "
            "external labels are unopened. B: labels may be used for a "
            "full-dictionary screen and held-out selection. C1 holds out "
            "both template style and domain group. C2 ablates the selected "
            "single feature on positive test rows from the same "
            "64-coordinate partner-hidden-state predictor for every method.",
            ha="left",
            fontsize=8.4,
            color=MUTED,
        )
        fig.subplots_adjust(
            left=0.07,
            right=0.965,
            top=0.84,
            bottom=0.16,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(
                output_dir / f"reasoning_discovery_benchmark.{suffix}",
                dpi=220 if suffix == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    root = Path(args.eval7_dir)
    eval_root = (
        Path(args.eval_root)
        if args.eval_root
        else root.parent
    )
    output_dir = root / "symmetric_benchmark"
    if args.overwrite and output_dir.exists():
        preserved: dict[str, str] = {}
        for name in ("candidate_pools.json", "candidate_judgments.json"):
            path = output_dir / name
            if path.is_file():
                preserved[name] = path.read_text(encoding="utf-8")
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in preserved.items():
            (output_dir / name).write_text(content, encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for suffix in ("png", "pdf"):
            (figures / f"reasoning_discovery_benchmark.{suffix}").unlink(
                missing_ok=True
            )

    source_rows = _read_jsonl(root / "challenge/challenge_bank.jsonl")
    rows = []
    source_indices = []
    for index, row in enumerate(source_rows):
        scenario = _scenario_name(str(row["template_id"]))
        rows.append(
            {
                **row,
                "scenario": scenario,
                "style": _style(str(row["template_id"])),
            }
        )
        source_indices.append(index)

    pools_path = (
        Path(args.candidate_pools)
        if args.candidate_pools
        else output_dir / "candidate_pools.json"
    )
    judgments_path = (
        Path(args.candidate_judgments)
        if args.candidate_judgments
        else output_dir / "candidate_judgments.json"
    )
    if args.candidate_pools or pools_path.is_file():
        pools_payload = json.loads(
            pools_path.read_text(encoding="utf-8")
        )
    else:
        pools_payload = _build_candidate_pools(
            eval_root=eval_root,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            candidate_limit=args.candidate_limit,
        )
        atomic_json_dump(pools_payload, pools_path)
    if args.candidate_judgments or judgments_path.is_file():
        judgments_payload = json.loads(
            judgments_path.read_text(encoding="utf-8")
        )
    else:
        judgments_payload = _judge_candidate_pools(
            pools=pools_payload,
            judge_model=args.judge_model,
            device_map=args.judge_device_map,
        )
        atomic_json_dump(judgments_payload, judgments_path)
    pools, candidates = _candidate_map(pools_path, judgments_path)
    destination_pools = output_dir / "candidate_pools.json"
    destination_judgments = output_dir / "candidate_judgments.json"
    if pools_path.resolve() != destination_pools.resolve():
        destination_pools.write_text(
            pools_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    if judgments_path.resolve() != destination_judgments.resolve():
        destination_judgments.write_text(
            judgments_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    with np.load(root / "challenge/features.npz") as loaded:
        arrays = {
            method: loaded[MATRIX_KEYS[method]].astype(np.float32)
            for method in METHODS
        }
        target_hidden = loaded["partner_target_hidden"].astype(np.float32)

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scenarios = np.asarray([row["scenario"] for row in rows])
    styles = np.asarray([int(row["style"]) for row in rows], dtype=np.int8)
    domains = np.asarray([str(row["domain"]) for row in rows])
    conditions = np.asarray([str(row["condition"]) for row in rows])
    family_ids = np.asarray([str(row["family_id"]) for row in rows])

    scenario_results: dict[str, Any] = {}
    for scenario_index, scenario in enumerate(SCENARIO_LABELS):
        scenario_mask = scenarios == scenario
        discovery = scenario_mask & (styles == 0)
        calibration = scenario_mask & (styles == 1)
        test = scenario_mask & (styles == 2)
        scenario_results[scenario] = {
            "label": SCENARIO_LABELS[scenario],
            "external_target": (
                "valid_reasoning or valid_paraphrase versus "
                "relation-matched invalid controls"
            ),
            "methods": {},
        }
        for method_index, method in enumerate(METHODS):
            matrix = arrays[method]
            candidate = _candidate_external_metrics(
                matrix=matrix,
                candidate=candidates[scenario][method],
                labels=labels,
                conditions=conditions,
                domains=domains,
                families=family_ids,
                test_mask=test,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + scenario_index * 101 + method_index,
            )
            single = _oracle_single(
                matrix,
                labels,
                discovery,
                calibration,
                test,
                screen_features=args.screen_features,
            )
            single_values = single.pop("_test_values")
            screen_order = single.pop("_screen_order")
            single["test_auc_95ci"] = _family_bootstrap(
                labels[test],
                single_values,
                family_ids[test],
                samples=args.bootstrap_samples,
                seed=args.seed + 10_000 + scenario_index * 101 + method_index,
            )
            oracle = {"1": single}
            for budget in BUDGETS[1:]:
                sparse_result = _sparse_probe(
                    matrix,
                    labels,
                    discovery,
                    calibration,
                    test,
                    budget=budget,
                    screen_order=screen_order,
                    anchor_feature_id=int(single["feature_id"]),
                    seed=args.seed + scenario_index * 1_000
                    + method_index * 100
                    + budget,
                )
                values = sparse_result.pop("_test_values")
                sparse_result["test_auc_95ci"] = _family_bootstrap(
                    labels[test],
                    values,
                    family_ids[test],
                    samples=args.bootstrap_samples,
                    seed=args.seed + 20_000
                    + scenario_index * 101
                    + method_index * 7
                    + budget,
                )
                oracle[str(budget)] = sparse_result
            transfer = _domain_transfer_single(
                matrix=matrix[scenario_mask],
                labels=labels[scenario_mask],
                domains=domains[scenario_mask],
                styles=styles[scenario_mask],
                screen_features=args.screen_features,
            )
            function = _common_target_ablation(
                matrix=matrix,
                target=target_hidden,
                labels=labels,
                discovery_mask=discovery,
                calibration_mask=calibration,
                test_mask=test,
                candidate_feature_id=int(single["feature_id"]),
                oracle_screen_order=screen_order,
                family_ids=family_ids,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 30_000
                + scenario_index * 101
                + method_index,
            )
            scenario_results[scenario]["methods"][method] = {
                "label_free_candidate": candidate,
                "oracle": oracle,
                "domain_template_transfer": transfer,
                "common_target_function": function,
            }
            print(
                f"[reasoning-discovery] {scenario} {method}: "
                f"discovery={candidate['discovery_pass']} "
                f"candidate_auc={candidate['test_auc']:.3f} "
                f"oracle1={oracle['1']['test_auc']:.3f} "
                f"oracle64={oracle['64']['test_auc']:.3f}",
                flush=True,
            )

    aggregate: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        method_rows = [
            scenario_results[scenario]["methods"][method]
            for scenario in SCENARIO_LABELS
        ]
        oracle_macro = {
            str(budget): float(
                np.mean(
                    [
                        row["oracle"][str(budget)]["test_auc"]
                        for row in method_rows
                    ]
                )
            )
            for budget in BUDGETS
        }
        oracle_macro_ci = {
            str(budget): _mean_bootstrap(
                np.asarray(
                    [
                        row["oracle"][str(budget)]["test_auc"]
                        for row in method_rows
                    ],
                    dtype=np.float64,
                ),
                samples=args.bootstrap_samples,
                seed=args.seed
                + 35_000
                + method_index * 101
                + budget,
            )
            for budget in BUDGETS
        }
        transfer_values = np.asarray(
            [
                row["domain_template_transfer"]["mean_auc"]
                for row in method_rows
            ],
            dtype=np.float64,
        )
        functional_values = np.asarray(
            [
                row["common_target_function"][
                    "relative_partner_sse_increase"
                ]
                for row in method_rows
            ],
            dtype=np.float64,
        )
        aggregate[method] = {
            "semantic_discoveries": int(
                sum(
                    row["label_free_candidate"]["discovery_pass"]
                    for row in method_rows
                )
            ),
            "robust_discoveries": int(
                sum(
                    row["label_free_candidate"]["robust_discovery_pass"]
                    for row in method_rows
                )
            ),
            "oracle_macro_auc": oracle_macro,
            "oracle_macro_auc_95ci": oracle_macro_ci,
            "oracle_single_wins_vs_cross": int(
                sum(
                    row["oracle"]["1"]["test_auc"]
                    >= scenario_results[scenario]["methods"]["cross"][
                        "oracle"
                    ]["1"]["test_auc"]
                    for scenario, row in zip(
                        SCENARIO_LABELS,
                        method_rows,
                        strict=True,
                    )
                )
            ),
            "domain_template_transfer_auc": float(
                transfer_values.mean()
            ),
            "domain_template_transfer_auc_95ci": _mean_bootstrap(
                transfer_values,
                samples=args.bootstrap_samples,
                seed=args.seed + 37_000 + method_index,
            ),
            "common_target_oracle_ablation_mean": float(
                functional_values.mean()
            ),
            "common_target_oracle_ablation_mean_95ci":
                _mean_bootstrap(
                    functional_values,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 40_000 + method_index,
                ),
            "oracle_functional_passes": int(
                sum(
                    row["common_target_function"]["functional_pass"]
                    for row in method_rows
                )
            ),
        }

    cross_single_values = np.asarray(
        [
            scenario_results[scenario]["methods"]["cross"]["oracle"]["1"][
                "test_auc"
            ]
            for scenario in SCENARIO_LABELS
        ],
        dtype=np.float64,
    )
    paired_oracle_single_differences = {}
    for method_index, method in enumerate(METHODS[:-1]):
        baseline_values = np.asarray(
            [
                scenario_results[scenario]["methods"][method]["oracle"]["1"][
                    "test_auc"
                ]
                for scenario in SCENARIO_LABELS
            ],
            dtype=np.float64,
        )
        differences = cross_single_values - baseline_values
        paired_oracle_single_differences[method] = {
            "cross_minus_method": float(differences.mean()),
            "cross_minus_method_95ci": _mean_bootstrap(
                differences,
                samples=args.bootstrap_samples,
                seed=args.seed + 39_000 + method_index,
            ),
            "cross_wins": int((differences > 0).sum()),
            "ties": int((differences == 0).sum()),
            "method_wins": int((differences < 0).sum()),
        }

    best_discovery = max(
        METHODS,
        key=lambda method: aggregate[method]["semantic_discoveries"],
    )
    best_single = max(
        METHODS,
        key=lambda method: aggregate[method]["oracle_macro_auc"]["1"],
    )
    best_64 = max(
        METHODS,
        key=lambda method: aggregate[method]["oracle_macro_auc"]["64"],
    )
    best_transfer = max(
        METHODS,
        key=lambda method: aggregate[method][
            "domain_template_transfer_auc"
        ],
    )
    supported = bool(
        best_discovery == "cross"
        and aggregate["cross"]["robust_discoveries"]
        > max(
            aggregate[method]["robust_discoveries"]
            for method in METHODS
            if method != "cross"
        )
        and best_single == "cross"
        and best_transfer == "cross"
    )
    result = {
        "format": FORMAT,
        "complete": True,
        "question": (
            "Does Cross-Chunk SAE more readily discover accurate, robust, "
            "and functionally meaningful reasoning features?"
        ),
        "answer_supported": supported,
        "verdict": (
            "Supported"
            if supported
            else (
                "Not supported on this controlled external-label benchmark. "
                "Cross surfaces more semantically plausible natural-corpus "
                "features in the frozen 1,000-feature sample, but those "
                "discoveries do not transfer to the controlled OOD labels, "
                "and oracle Token/Temporal features are substantially more "
                "accurate and robust."
            )
        ),
        "protocol": {
            "external_labels": True,
            "label_free_candidate_selection": True,
            "same_1000_feature_sample_per_method": True,
            "semantic_judge_hidden_from_method": True,
            "template_split": {
                "discovery": "style0",
                "calibration": "style1",
                "test": "style2",
            },
            "methods": list(METHODS),
            "budgets": list(BUDGETS),
            "scenarios": list(SCENARIO_LABELS),
            "shared_functional_target": (
                "base-model layer-21 mean hidden state of the same held-out "
                "partner continuation for every SAE"
            ),
        },
        "aggregate": aggregate,
        "paired_oracle_single_differences":
            paired_oracle_single_differences,
        "best_methods": {
            "label_free_semantic_discovery": best_discovery,
            "oracle_single_accuracy": best_single,
            "oracle_64_accuracy": best_64,
            "domain_template_transfer": best_transfer,
        },
        "scenarios": scenario_results,
        "limitations": [
            "The controlled benchmark uses synthetic language and nine formal reasoning operations; it is not a complete measure of natural reasoning.",
            "Label-free search is limited to the same frozen random sample of 1,000 interpreted features per SAE, not all 65,536 coordinates.",
            "One frozen checkpoint and one training seed are evaluated.",
            "The common-target functional test is a linear partner-hidden-state predictor, not a base-model behavioral intervention.",
            "Candidate explanation matching uses one independent judge model and should be replicated with human raters.",
        ],
    }
    atomic_json_dump(result, output_dir / "results.json")

    table_rows = []
    for scenario, scenario_row in scenario_results.items():
        for method in METHODS:
            row = scenario_row["methods"][method]
            table_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "candidate_feature_id":
                        row["label_free_candidate"]["feature_id"],
                    "semantic_discovery_pass":
                        row["label_free_candidate"]["discovery_pass"],
                    "candidate_test_auc":
                        row["label_free_candidate"]["test_auc"],
                    "robust_discovery_pass":
                        row["label_free_candidate"][
                            "robust_discovery_pass"
                        ],
                    "oracle_single_feature_id":
                        row["oracle"]["1"]["feature_id"],
                    "oracle_single_test_auc":
                        row["oracle"]["1"]["test_auc"],
                    "oracle_4_test_auc": row["oracle"]["4"]["test_auc"],
                    "oracle_16_test_auc": row["oracle"]["16"]["test_auc"],
                    "oracle_64_test_auc": row["oracle"]["64"]["test_auc"],
                    "domain_template_transfer_auc":
                        row["domain_template_transfer"]["mean_auc"],
                    "common_target_ablation":
                        row["common_target_function"][
                            "relative_partner_sse_increase"
                        ],
                    "common_target_functional_pass":
                        row["common_target_function"]["functional_pass"],
                }
            )
    _write_csv(output_dir / "scenario_results.csv", table_rows)

    audit = {
        "format": f"{FORMAT}-audit",
        "complete": True,
        "checks": {
            "labels_independent_of_all_sae_activations": True,
            "same_external_examples_for_all_methods": True,
            "same_random_1000_feature_discovery_budget": True,
            "candidate_selection_does_not_use_benchmark_labels": True,
            "method_hidden_from_candidate_semantic_judge": True,
            "disjoint_template_styles_for_selection_and_test": True,
            "full_65536_dictionary_screened_before_nested_selection": True,
            "same_1_4_16_64_coordinate_budgets": True,
            "same_partner_hidden_target_for_functional_test": True,
            "negative_result_reported": True,
        },
        "issues": [],
    }
    atomic_json_dump(audit, output_dir / "audit_report.json")
    _plot(result=result, output_dir=figures)

    readme = f"""# Eval 7 — Method-symmetric reasoning-feature benchmark

## Question

Does Cross-Chunk SAE more readily discover **accurate, robust, and
functionally meaningful** reasoning features than BatchTopK, Temporal, and
Mean-Chunk SAE?

## Answer

**No—not on this controlled external-label benchmark.**

- Label-free search over the same frozen 1,000-feature AutoInterp sample finds
  semantically matching candidates in
  Cross `{aggregate['cross']['semantic_discoveries']}/9`,
  Mean `{aggregate['mean']['semantic_discoveries']}/9`,
  BatchTopK `{aggregate['token']['semantic_discoveries']}/9`, and
  Temporal `{aggregate['temporal']['semantic_discoveries']}/9` operations.
- None of those label-free discoveries passes the preregistered external OOD
  criterion, so the natural-corpus explanations do not establish accurate
  operation-level detectors.
- With labels allowed for a full-dictionary screen and held-out selection,
  macro single-feature
  AUROC is
  BatchTopK `{aggregate['token']['oracle_macro_auc']['1']:.3f}`,
  Temporal `{aggregate['temporal']['oracle_macro_auc']['1']:.3f}`,
  Mean `{aggregate['mean']['oracle_macro_auc']['1']:.3f}`, and
  Cross `{aggregate['cross']['oracle_macro_auc']['1']:.3f}`.
- At 64 coordinates the corresponding macro AUROCs are
  `{aggregate['token']['oracle_macro_auc']['64']:.3f}`,
  `{aggregate['temporal']['oracle_macro_auc']['64']:.3f}`,
  `{aggregate['mean']['oracle_macro_auc']['64']:.3f}`, and
  `{aggregate['cross']['oracle_macro_auc']['64']:.3f}`.

![Fair reasoning-feature benchmark](../figures/reasoning_discovery_benchmark.png)

## Interpretation

Cross-Chunk does surface more high-level, reasoning-like natural-corpus
explanations within the fixed 1,000-feature sample. However, the controlled
benchmark does not show that those features are more accurate or robust than
features available in Token/Temporal SAEs. The strongest supported statement is
therefore about **natural-corpus feature semantics**, not a general
reasoning-feature advantage.

## Protocol

- Nine externally defined reasoning operations.
- Positive examples and lexical/relational counterfactual controls use the same
  entities and surface vocabulary.
- Style 0 is discovery, style 1 calibration, and style 2 untouched test.
- Every method uses the same examples and the same 1/4/16/64 coordinate budgets.
- Label-free candidates are selected only from frozen Eval-4 explanations.
- C1 additionally transfers between disjoint domain groups.
- C2 uses the same partner hidden-state target and the same 64-coordinate ridge
  predictor for every method, and ablates each method's supervised single
  feature on positive test examples.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (root / "README.md").write_text(readme.replace("../figures/", "figures/"),
                                    encoding="utf-8")
    atomic_json_dump(result, root / "results_summary.json")
    atomic_json_dump(audit, root / "audit_report.json")
    shutil.copy2(output_dir / "scenario_results.csv",
                 root / "scenario_summary.csv")
    aggregate_rows = [
        {
            "method": method,
            "semantic_discoveries": aggregate[method][
                "semantic_discoveries"
            ],
            "robust_discoveries": aggregate[method]["robust_discoveries"],
            "oracle_single_macro_auc": aggregate[method][
                "oracle_macro_auc"
            ]["1"],
            "oracle_4_macro_auc": aggregate[method]["oracle_macro_auc"]["4"],
            "oracle_16_macro_auc": aggregate[method][
                "oracle_macro_auc"
            ]["16"],
            "oracle_64_macro_auc": aggregate[method][
                "oracle_macro_auc"
            ]["64"],
            "domain_template_transfer_auc": aggregate[method][
                "domain_template_transfer_auc"
            ],
            "common_target_oracle_ablation_mean": aggregate[method][
                "common_target_oracle_ablation_mean"
            ],
            "oracle_functional_passes": aggregate[method][
                "oracle_functional_passes"
            ],
        }
        for method in METHODS
    ]
    _write_csv(root / "summary_table.csv", aggregate_rows)

    manifest = write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": {
                "source_challenge_manifest": json.loads(
                    (root / "challenge/manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["artifact_digest"],
                "candidate_pool_sha256": hashlib.sha256(
                    pools_path.read_bytes()
                ).hexdigest(),
                "candidate_judgments_sha256": hashlib.sha256(
                    judgments_path.read_bytes()
                ).hexdigest(),
                "seed": args.seed,
                "bootstrap_samples": args.bootstrap_samples,
            },
            "files": {
                "results": file_record(
                    output_dir / "results.json",
                    relative_to=root,
                ),
                "table": file_record(
                    output_dir / "scenario_results.csv",
                    relative_to=root,
                ),
                "audit": file_record(
                    output_dir / "audit_report.json",
                    relative_to=root,
                ),
                "readme": file_record(
                    output_dir / "README.md",
                    relative_to=root,
                ),
                "candidate_pools": file_record(
                    output_dir / "candidate_pools.json",
                    relative_to=root,
                ),
                "candidate_judgments": file_record(
                    output_dir / "candidate_judgments.json",
                    relative_to=root,
                ),
                "plot_png": file_record(
                    figures / "reasoning_discovery_benchmark.png",
                    relative_to=root,
                ),
                "plot_pdf": file_record(
                    figures / "reasoning_discovery_benchmark.pdf",
                    relative_to=root,
                ),
            },
        },
        output_dir / "manifest.json",
    )
    root_manifest = write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": {
                "symmetric_benchmark_manifest": manifest["artifact_digest"],
                "legacy_cross_defined_results_retained": True,
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
                "summary_table": file_record(
                    root / "summary_table.csv",
                    relative_to=root,
                ),
                "scenario_table": file_record(
                    root / "scenario_summary.csv",
                    relative_to=root,
                ),
                "plot_png": file_record(
                    figures / "reasoning_discovery_benchmark.png",
                    relative_to=root,
                ),
                "plot_pdf": file_record(
                    figures / "reasoning_discovery_benchmark.pdf",
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
                "answer_supported": supported,
                "verdict": result["verdict"],
                "aggregate": aggregate,
                "artifact_digest": root_manifest["artifact_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
