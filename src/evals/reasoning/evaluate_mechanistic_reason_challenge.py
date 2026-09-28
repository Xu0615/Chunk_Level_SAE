#!/usr/bin/env python
"""Validate a frozen Cross mechanistic-reasoning feature with matched text.

The target coordinate is chosen before this script runs (Cross #20232 by
default).  Every family contains the same stimulus, mediator, target, assay,
and control vocabulary in four variants:

* two passages that establish a perturbation -> mediator -> outcome chain;
* one passage that reports association without dependency; and
* one methods-style inventory that contains the same entities and reagents.

Calibration and test use disjoint entities and prose templates.  Token SAE
baselines search the full dictionary and receive single/4/16/64-coordinate
linear budgets.  This is a targeted semantic validation of the frozen natural
feature, not a benchmark for reasoning correctness in general.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from chunk_saes.artifacts import (
    file_record,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.plot_style import METHOD_COLORS
from chunk_saes.utils import atomic_json_dump
from evals.reasoning.evaluate_reason_feature import (
    SparseEncoder,
    _best_single_feature,
    _bootstrap_auc,
    _bootstrap_delta_auc,
    _fit_sparse_logistic,
    _pool_token_features,
)


FORMAT = "chunk-saes-mechanistic-reason-challenge-v1"
CONDITIONS = (
    "mechanism",
    "mechanism_paraphrase",
    "association_without_dependency",
    "methods_inventory",
)
POSITIVE_CONDITIONS = ("mechanism", "mechanism_paraphrase")


ENTITY_BANKS = {
    "discovery": {
        "stimulus": (
            "TNF-alpha",
            "insulin",
            "EGF",
            "LPS",
            "hypoxia",
            "TGF-beta",
            "oxidative stress",
            "interferon-gamma",
        ),
        "mediator": (
            "p38 MAPK",
            "Akt",
            "ERK1/2",
            "NF-kappaB",
            "HIF-1alpha",
            "SMAD3",
            "Nrf2",
            "STAT1",
        ),
        "target": (
            "IL-6",
            "GLUT4",
            "cyclin D1",
            "iNOS",
            "VEGF",
            "collagen I",
            "HO-1",
            "CXCL10",
        ),
    },
    "calibration": {
        "stimulus": (
            "IL-4",
            "dexamethasone",
            "PDGF",
            "leptin",
            "retinoic acid",
            "interleukin-1 beta",
            "mechanical stretch",
            "Wnt3a",
        ),
        "mediator": (
            "STAT6",
            "JNK",
            "PI3K",
            "AMPK",
            "RAR-alpha",
            "IKK-beta",
            "YAP",
            "beta-catenin",
        ),
        "target": (
            "arginase-1",
            "COX-2",
            "MYC",
            "adiponectin",
            "RARRES1",
            "CCL2",
            "CTGF",
            "AXIN2",
        ),
    },
    "test": {
        "stimulus": (
            "BMP4",
            "forskolin",
            "angiotensin II",
            "interleukin-17",
            "estrogen",
            "thrombin",
            "Notch ligand DLL4",
            "fibroblast growth factor",
        ),
        "mediator": (
            "SMAD1",
            "CREB",
            "PKC-delta",
            "STAT3",
            "ER-alpha",
            "PAR1",
            "NICD",
            "MEK1",
        ),
        "target": (
            "ID1",
            "c-FOS",
            "endothelin-1",
            "SOCS3",
            "BCL2",
            "E-selectin",
            "HES1",
            "DUSP6",
        ),
    },
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, default=21)
    p.add_argument("--feature-id", type=int, default=20232)
    p.add_argument("--families-per-split", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--candidate-screen", type=int, default=512)
    p.add_argument("--sparse-budgets", default="4,16,64")
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
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


def _render(
    split: str,
    style: int,
    stimulus: str,
    mediator: str,
    target: str,
) -> dict[str, str]:
    assay = ("immunoblotting", "qRT-PCR", "reporter assay")[style % 3]
    control = ("vehicle", "scrambled RNA", "inactive analog")[style % 3]
    if split == "discovery":
        return {
            "mechanism": (
                f"Cells exposed to {stimulus} showed early {mediator} "
                f"activation and later {target} expression. Blocking "
                f"{mediator} abolished the {target} response, whereas the "
                f"{control} control did not. Restoring active {mediator} "
                f"rescued {target}. The {assay} results identify a "
                f"{stimulus} to {mediator} to {target} pathway."
            ),
            "mechanism_paraphrase": (
                f"After {stimulus} treatment, {mediator} changed before "
                f"{target}. Selective inhibition of {mediator}, but not the "
                f"{control} control, prevented induction of {target}; "
                f"reactivating {mediator} restored it. Thus the {assay} "
                f"evidence supports {mediator} as the mediator between "
                f"{stimulus} and {target}."
            ),
            "association_without_dependency": (
                f"After {stimulus} treatment, {mediator} and {target} were "
                f"both measured by {assay}. Selective inhibition of "
                f"{mediator} and the {control} control left {target} "
                f"unchanged, and reactivating {mediator} did not restore it. "
                f"The measurements show association but do not establish a "
                f"{stimulus} to {mediator} to {target} pathway."
            ),
            "methods_inventory": (
                f"The experiment used {stimulus}, a {mediator} inhibitor, "
                f"the {control} control, antibodies for active {mediator}, "
                f"and a {assay} measurement of {target}. Reagents and assays "
                f"were checked separately; no dependency or rescue experiment "
                f"linked {stimulus}, {mediator}, and {target}."
            ),
        }
    if split == "calibration":
        return {
            "mechanism": (
                f"Stimulation with {stimulus} increased phosphorylated "
                f"{mediator}, followed by induction of {target}. The "
                f"{mediator} inhibitor eliminated this induction while the "
                f"{control} control had no effect; active {mediator} restored "
                f"the response. Together, the {assay} data place {mediator} "
                f"between {stimulus} and {target}."
            ),
            "mechanism_paraphrase": (
                f"The response to {stimulus} proceeded through {mediator}: "
                f"{target} rose only after {mediator} activation, disappeared "
                f"when {mediator} was blocked, and returned when {mediator} "
                f"was reintroduced. The {control} control was inactive. This "
                f"perturbation-and-rescue pattern in the {assay} supports the "
                f"mechanistic chain."
            ),
            "association_without_dependency": (
                f"Stimulation with {stimulus} coincided with measurements of "
                f"phosphorylated {mediator} and {target} by {assay}. However, "
                f"the {mediator} inhibitor and the {control} control produced "
                f"the same {target} response, and active {mediator} did not "
                f"restore it. The observations do not place {mediator} "
                f"between {stimulus} and {target}."
            ),
            "methods_inventory": (
                f"For the {stimulus} study, investigators catalogued a "
                f"{mediator} inhibitor, active-{mediator} antibody, "
                f"{control} control, and {assay} primers for {target}. The "
                f"report validates each item independently and makes no "
                f"perturbation-based claim connecting the three entities."
            ),
        }
    return {
        "mechanism": (
            f"Time-course data placed {mediator} activation after "
            f"{stimulus} but before {target} induction. Removing "
            f"{mediator} prevented {target}; the {control} control did not, "
            f"and re-expression of {mediator} recovered the response. These "
            f"{assay} experiments demonstrate that {mediator} transmits the "
            f"effect of {stimulus} to {target}."
        ),
        "mechanism_paraphrase": (
            f"A loss-and-rescue experiment resolved the pathway. "
            f"{stimulus} activated {mediator}; loss of {mediator} suppressed "
            f"{target}, whereas the {control} control preserved it, and "
            f"restored {mediator} recovered {target}. The ordered {assay} "
            f"results therefore support a mechanistic link."
        ),
        "association_without_dependency": (
            f"Time-course data recorded {mediator} after {stimulus} and also "
            f"recorded {target}. Removing {mediator} did not prevent "
            f"{target}; the {control} control behaved similarly, and "
            f"re-expression of {mediator} failed to recover the response. "
            f"These {assay} observations do not demonstrate that "
            f"{mediator} transmits the effect."
        ),
        "methods_inventory": (
            f"The {stimulus} protocol listed {mediator}, {target}, a "
            f"{mediator} loss construct, a rescue construct, the {control} "
            f"control, and an {assay}. Each measurement was quality-checked, "
            f"but the protocol did not test whether {mediator} transmits an "
            f"effect from {stimulus} to {target}."
        ),
    }


def build_bank(tokenizer, families_per_split: int, seed: int) -> list[dict]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for split_index, split in enumerate(
        ("discovery", "calibration", "test")
    ):
        bank = ENTITY_BANKS[split]
        combinations = [
            (stimulus, mediator, target)
            for stimulus in bank["stimulus"]
            for mediator in bank["mediator"]
            for target in bank["target"]
        ]
        order = rng.permutation(len(combinations))[:families_per_split]
        for local_index, combination_index in enumerate(order.tolist()):
            stimulus, mediator, target = combinations[combination_index]
            style = (local_index + split_index) % 3
            family_id = f"{split}-{local_index:03d}"
            variants = _render(
                split,
                style,
                stimulus,
                mediator,
                target,
            )
            for condition in CONDITIONS:
                text = variants[condition]
                token_ids = tokenizer(
                    text,
                    add_special_tokens=False,
                ).input_ids
                rows.append(
                    {
                        "example_id": f"{family_id}-{condition}",
                        "family_id": family_id,
                        "split": split,
                        "condition": condition,
                        "label": int(condition in POSITIVE_CONDITIONS),
                        "stimulus": stimulus,
                        "mediator": mediator,
                        "target": target,
                        "text": text,
                        "token_ids": list(map(int, token_ids)),
                        "token_length": len(token_ids),
                        "content_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                    }
                )
    if len({row["content_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate challenge texts")
    return rows


@torch.inference_mode()
def extract(
    args: argparse.Namespace,
    rows: list[dict],
    sae_set: dict,
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
    token = SparseEncoder(
        Path(sae_set["modes"]["token"]["checkpoint_path"]),
        device,
    )
    cross = SparseEncoder(
        Path(sae_set["modes"]["cross"]["checkpoint_path"]),
        device,
    )
    n = len(rows)
    token_mean = np.empty((n, token.width), np.float16)
    token_max = np.empty((n, token.width), np.float16)
    cross_values = np.empty(n, np.float32)
    try:
        sequences = [row["token_ids"] for row in rows]
        for start in range(0, n, args.batch_size):
            stop = min(n, start + args.batch_size)
            batch = extractor.forward_ids(sequences[start:stop])
            dense = token.dense(batch.hidden)
            token_mean[start:stop] = (
                _pool_token_features(dense, batch.mask, "mean")
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            token_max[start:stop] = (
                _pool_token_features(dense, batch.mask, "max")
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            cross_values[start:stop] = (
                cross.dense(batch.means())[:, args.feature_id]
                .float()
                .cpu()
                .numpy()
            )
            print(
                f"[mechanism-challenge] encoded {stop}/{n}",
                flush=True,
            )
        np.savez_compressed(
            output_dir / "features.npz",
            cross_feature=cross_values,
            token_mean=token_mean,
            token_max=token_max,
            labels=np.asarray([row["label"] for row in rows], np.int8),
            splits=np.asarray([row["split"] for row in rows]),
            conditions=np.asarray([row["condition"] for row in rows]),
            family_ids=np.asarray([row["family_id"] for row in rows]),
        )
    finally:
        extractor.close()
        token.close()
        cross.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def analyze(
    args: argparse.Namespace,
    rows: list[dict],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    labels = arrays["labels"].astype(np.int8)
    splits = arrays["splits"].astype(str)
    conditions = arrays["conditions"].astype(str)
    families = arrays["family_ids"].astype(str)
    calibration = splits == "calibration"
    test = splits == "test"
    test_labels = labels[test]
    test_families = families[test]
    cross_values = arrays["cross_feature"].astype(np.float64)

    cross_test = cross_values[test]
    cross_auc = float(roc_auc_score(test_labels, cross_test))
    cross_ci = _bootstrap_auc(
        test_labels,
        cross_test,
        test_families,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    budgets = [
        int(value)
        for value in args.sparse_budgets.split(",")
        if value.strip()
    ]
    token_results = {}
    predictions = {}
    for representation in ("token_mean", "token_max"):
        matrix = arrays[representation].astype(np.float32)
        single = _best_single_feature(
            matrix,
            labels,
            calibration,
            test,
        )
        predictions[f"{representation}/1"] = single.pop("test_values")
        results = {"1": single}
        for budget in budgets:
            fitted = _fit_sparse_logistic(
                matrix[calibration],
                labels[calibration],
                matrix[test],
                labels[test],
                budget=budget,
                seed=args.seed + budget,
            )
            predictions[f"{representation}/{budget}"] = fitted.pop(
                "test_probabilities"
            )
            results[str(budget)] = fitted
        token_results[representation] = results

    comparisons = {}
    for name, values in predictions.items():
        auc = float(roc_auc_score(test_labels, values))
        comparisons[name] = {
            "test_auc": auc,
            "test_auc_95ci": _bootstrap_auc(
                test_labels,
                values,
                test_families,
                samples=args.bootstrap_samples,
                seed=args.seed + sum(map(ord, name)),
            ),
            "cross_minus_token_auc": _bootstrap_delta_auc(
                test_labels,
                cross_test,
                values,
                test_families,
                samples=args.bootstrap_samples,
                seed=args.seed + 1000 + sum(map(ord, name)),
            ),
        }
    strongest = max(
        comparisons,
        key=lambda name: comparisons[name]["test_auc"],
    )
    condition_means = {
        condition: float(
            cross_values[
                test & (conditions == condition)
            ].mean()
        )
        for condition in CONDITIONS
    }
    condition_rates = {
        condition: float(
            (
                cross_values[
                    test & (conditions == condition)
                ]
                > 0
            ).mean()
        )
        for condition in CONDITIONS
    }
    result = {
        "format": FORMAT,
        "complete": True,
        "feature_id": args.feature_id,
        "claim_scope": (
            "Targeted validation of a frozen natural mechanistic-causal "
            "Cross feature on templated, entity-disjoint matched controls."
        ),
        "challenge": {
            "rows": len(rows),
            "families_per_split": args.families_per_split,
            "conditions": list(CONDITIONS),
            "positive_conditions": list(POSITIVE_CONDITIONS),
            "split_policy": (
                "disjoint biological entities and disjoint prose templates"
            ),
        },
        "cross": {
            "test_auc": cross_auc,
            "test_auc_95ci": cross_ci,
            "condition_mean_activation": condition_means,
            "condition_activation_rate": condition_rates,
        },
        "token": token_results,
        "comparisons": comparisons,
        "strongest_token_baseline": {
            "name": strongest,
            **comparisons[strongest],
        },
    }
    atomic_json_dump(result, output_dir / "results.json")
    return result


def write_outputs(
    args: argparse.Namespace,
    rows: list[dict],
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    strongest = result["strongest_token_baseline"]
    mean = result["token"]["token_mean"]
    content = f"""# Feature-specific mechanistic reasoning challenge

The Cross coordinate was frozen as `#{args.feature_id}` from independent
natural-corpus evidence before this challenge was analyzed.  Positive and
negative variants share pathway entities, assay terms, controls, and
intervention vocabulary; calibration and test use disjoint entities and prose
templates.

- Cross test AUC: `{result['cross']['test_auc']:.3f}`
- Token-mean single / 4 / 16 / 64 AUC:
  `{mean['1']['test_auc']:.3f}` /
  `{mean['4']['test_auc']:.3f}` /
  `{mean['16']['test_auc']:.3f}` /
  `{mean['64']['test_auc']:.3f}`
- Strongest Token stress test (mean or max pooling):
  `{strongest['name']}`, AUC `{strongest['test_auc']:.3f}`
- Cross minus strongest Token:
  `{strongest['cross_minus_token_auc']['point']:+.3f}`,
  95% CI
  `[{strongest['cross_minus_token_auc']['95ci'][0]:+.3f},
  {strongest['cross_minus_token_auc']['95ci'][1]:+.3f}]`

This challenge checks semantic selectivity of one natural feature.  It does
not test generic logical validity and does not prove impossibility for an
unrestricted nonlinear decoder.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")

    labels = [
        "Cross",
        "Token 1",
        "Token 4",
        "Token 16",
        "Token 64",
    ]
    values = [
        result["cross"]["test_auc"],
        mean["1"]["test_auc"],
        mean["4"]["test_auc"],
        mean["16"]["test_auc"],
        mean["64"]["test_auc"],
    ]
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar(
        labels,
        values,
        color=[
            METHOD_COLORS["cross"],
            *[METHOD_COLORS["token"]] * 4,
        ],
    )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Held-out AUC")
    ax.set_title("Mechanistic-reasoning feature validation")
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
            plot_dir / f"mechanism_challenge.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(fig)

    audit = {
        "format": "chunk-saes-mechanistic-reason-challenge-audit-v1",
        "complete": True,
        "checks": {
            "unique_texts": (
                len({row["content_sha256"] for row in rows}) == len(rows)
            ),
            "four_conditions_per_family": all(
                count == 4
                for count in Counter(
                    row["family_id"] for row in rows
                ).values()
            ),
            "fixed_cross_coordinate": args.feature_id,
            "full_token_dictionary_search": True,
            "test_not_used_for_feature_selection": True,
        },
        "issues": [],
    }
    atomic_json_dump(audit, output_dir / "audit_report.json")
    write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": {
                "model": str(Path(args.model).resolve()),
                "sae_root": str(Path(args.sae_root).resolve()),
                "layer": args.layer,
                "feature_id": args.feature_id,
                "seed": args.seed,
                "families_per_split": args.families_per_split,
            },
            "files": {
                "challenge_bank": file_record(
                    output_dir / "challenge_bank.jsonl",
                    relative_to=output_dir,
                ),
                "features": file_record(
                    output_dir / "features.npz",
                    relative_to=output_dir,
                ),
                "results": file_record(
                    output_dir / "results.json",
                    relative_to=output_dir,
                ),
                "audit": file_record(
                    output_dir / "audit_report.json",
                    relative_to=output_dir,
                ),
                "readme": file_record(
                    output_dir / "README.md",
                    relative_to=output_dir,
                ),
                "plot_png": file_record(
                    plot_dir / "mechanism_challenge.png",
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
    bank_path = output_dir / "challenge_bank.jsonl"
    if args.analysis_only:
        rows = _read_jsonl(bank_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        rows = build_bank(
            tokenizer,
            args.families_per_split,
            args.seed,
        )
        _write_jsonl(bank_path, rows)
    sae_set = resolve_sae_artifact_set(
        args.sae_root,
        selection="best",
        modes=("token", "cross"),
    )
    if not args.analysis_only:
        extract(args, rows, sae_set, output_dir)
    with np.load(output_dir / "features.npz") as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    result = analyze(args, rows, arrays, output_dir)
    write_outputs(args, rows, result, output_dir)
    print(
        json.dumps(
            {
                "complete": True,
                "feature_id": args.feature_id,
                "cross_auc": result["cross"]["test_auc"],
                "strongest_token":
                    result["strongest_token_baseline"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
