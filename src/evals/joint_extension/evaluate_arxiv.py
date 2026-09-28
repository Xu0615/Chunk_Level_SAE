#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from evals.joint_extension.common import (
    FrozenJointEncoder,
    JointSpec,
    add_joint_root_arguments,
    joint_specs_from_args,
    merge_sidecar,
)
from evals.supervised_transfer import analyze_representation_geometry as geometry_eval
from evals.supervised_transfer import run_linear_probes as probe_eval


SPLITS = ("train", "validation", "test", "ood")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Encode the existing ArXiv raw hidden archive with four nested "
            "Joint-Chunk SAEs and evaluate probes/geometry without rerunning Qwen."
        )
    )
    p.add_argument("--eval-root", required=True)
    add_joint_root_arguments(p)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--storage-k", type=int, default=2048)
    p.add_argument("--feature-budgets", default="16,64,256")
    p.add_argument("--low-label-budgets", default="1,2,4,8,16,64,256")
    p.add_argument("--low-label-seeds", default="0,1,2,3,4")
    p.add_argument("--classifier-c", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=72)
    p.add_argument("--svd-components", type=int, default=50)
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--tsne-perplexity", type=float, default=35.0)
    p.add_argument("--tsne-iterations", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument(
        "--reuse-encoded",
        action="store_true",
        help=(
            "Reuse existing Joint sparse split archives after verifying that "
            "their row identity matches the canonical raw-feature archives."
        ),
    )
    return p


def _specs(args: argparse.Namespace) -> list[JointSpec]:
    return joint_specs_from_args(args)


def _csr(
    indices: np.ndarray,
    values: np.ndarray,
    nnz: np.ndarray,
    width: int,
    *,
    max_nnz: int | None = None,
) -> sparse.csr_matrix:
    return probe_eval.csr_from_arrays(
        indices,
        values,
        nnz,
        width,
        max_nnz=max_nnz,
    )


def _encode_split(
    spec: JointSpec,
    *,
    source: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
    storage_k: int,
) -> dict[str, Any]:
    with np.load(source, allow_pickle=True) as handle:
        raw = handle["raw"]
        labels = handle["labels"]
        years = handle["years"]
        ids = handle["ids"]
    encoder = FrozenJointEncoder(spec.checkpoint, device)
    indices_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    nnz_parts: list[np.ndarray] = []
    try:
        for start in range(0, int(raw.shape[0]), int(batch_size)):
            stop = min(int(raw.shape[0]), start + int(batch_size))
            hidden = torch.from_numpy(raw[start:stop].astype(np.float32))
            indices, values, nnz = encoder.topk(hidden, storage_k)
            indices_np = indices.cpu().numpy().astype(np.int32)
            values_np = values.float().cpu().numpy().astype(np.float16)
            values_valid = values_np > 0
            indices_np[~values_valid] = -1
            indices_parts.append(indices_np)
            values_parts.append(values_np)
            nnz_parts.append(nnz.cpu().numpy().astype(np.int16))
    finally:
        encoder.close()
    indices = np.concatenate(indices_parts)
    values = np.concatenate(values_parts)
    nnz = np.concatenate(nnz_parts)
    np.savez_compressed(
        output,
        labels=labels,
        years=years,
        ids=ids,
        indices=indices,
        values=values,
        nnz=nnz,
    )
    return {
        "rows": int(indices.shape[0]),
        "mean_nnz": float(nnz.mean()),
        "zero_rows": int(np.sum(nnz == 0)),
        "file": output.name,
    }


def _load_encoded(root: Path, spec: JointSpec, split: str) -> dict[str, np.ndarray]:
    with np.load(root / f"{spec.key}-{split}.npz", allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def _encoded_summary(
    *,
    encoded: Path,
    source: Path,
    storage_k: int,
) -> dict[str, Any] | None:
    """Validate an encoded archive against the canonical split identity."""

    if not encoded.is_file():
        return None
    try:
        with np.load(source, allow_pickle=True) as raw_handle:
            source_labels = raw_handle["labels"]
            source_years = raw_handle["years"]
            source_ids = raw_handle["ids"]
        with np.load(encoded, allow_pickle=True) as handle:
            required = {"labels", "years", "ids", "indices", "values", "nnz"}
            if not required.issubset(handle.files):
                return None
            labels = handle["labels"]
            years = handle["years"]
            ids = handle["ids"]
            indices = handle["indices"]
            values = handle["values"]
            nnz = handle["nnz"]
        rows = len(source_ids)
        valid = (
            np.array_equal(labels, source_labels)
            and np.array_equal(years, source_years)
            and np.array_equal(ids, source_ids)
            and indices.shape == (rows, int(storage_k))
            and values.shape == indices.shape
            and nnz.shape == (rows,)
            and np.all(nnz >= 0)
            and np.all(nnz <= int(storage_k))
        )
        if not valid:
            return None
        # A positive final stored value proves that Top-storage_k truncated an
        # active code and would invalidate exact sparse downstream evaluation.
        if np.any(values[:, -1] > 0):
            raise RuntimeError(
                f"{encoded} truncates active Joint features at storage_k="
                f"{storage_k}"
            )
        return {
            "rows": int(rows),
            "mean_nnz": float(np.asarray(nnz, dtype=np.float64).mean()),
            "zero_rows": int(np.sum(nnz == 0)),
            "file": encoded.name,
            "reused": True,
        }
    except RuntimeError:
        raise
    except Exception:
        return None


def _high_level_summary(
    representations: dict[str, dict[str, Any]],
    low_label_budgets: list[int],
) -> dict[str, Any]:
    methods = {}
    for key, row in representations.items():
        curve = np.asarray(
            [
                float(row["low_label"][str(budget)]["mean_accuracy"])
                for budget in low_label_budgets
            ],
            dtype=np.float64,
        )
        log_budgets = np.log2(
            np.asarray(low_label_budgets, dtype=np.float64)
        )
        integrate = (
            np.trapezoid
            if hasattr(np, "trapezoid")
            else np.trapz
        )
        auc = float(
            integrate(curve, log_budgets)
            / (log_budgets[-1] - log_budgets[0])
        )
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
            name: (value - probe_eval.CHANCE_ACCURACY)
            / (1.0 - probe_eval.CHANCE_ACCURACY)
            for name, value in accuracies.items()
        }
        methods[key] = {
            "representation": key,
            **accuracies,
            "chance_normalized": normalized,
            "high_level_transfer_score": float(np.mean(list(normalized.values()))),
            "low_label_auc": auc,
            "chance_normalized_low_label_auc": (
                auc - probe_eval.CHANCE_ACCURACY
            )
            / (1.0 - probe_eval.CHANCE_ACCURACY),
            "ood_retention": float(row["ood"]["accuracy"])
            / max(float(row["full"]["accuracy"]), 1e-12),
            "worst_class_ood_accuracy": float(
                row["ood"]["worst_class_accuracy"]
            ),
            "ood_class_accuracy_std": float(row["ood"]["class_accuracy_std"]),
        }
    return {
        "definition": (
            "Mean chance-normalized accuracy over full train, all configured "
            "low-label budgets, and first-submission-year OOD evaluation."
        ),
        "chance_accuracy": probe_eval.CHANCE_ACCURACY,
        "low_label_budgets": low_label_budgets,
        "methods": methods,
    }


def _geometry(
    *,
    spec: JointSpec,
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    ood: dict[str, np.ndarray],
    probe_row: dict[str, Any],
    common_utilization_k: int,
    args: argparse.Namespace,
    mode_index: int,
) -> tuple[dict[str, Any], np.ndarray]:
    matrices = {
        "train": _csr(
            train["indices"],
            train["values"],
            train["nnz"],
            spec.dictionary_width,
        ),
        "test": _csr(
            test["indices"],
            test["values"],
            test["nnz"],
            spec.dictionary_width,
        ),
        "ood": _csr(
            ood["indices"],
            ood["values"],
            ood["nnz"],
            spec.dictionary_width,
        ),
    }
    utilization_matrix = _csr(
        train["indices"],
        train["values"],
        train["nnz"],
        spec.dictionary_width,
        max_nnz=common_utilization_k,
    )
    counts = np.asarray((utilization_matrix > 0).sum(axis=0)).reshape(-1)
    utilization = geometry_eval._distribution_metrics(
        counts,
        spec.dictionary_width,
    )
    nmi = geometry_eval._normalized_mutual_information(
        utilization_matrix,
        train["labels"],
    )
    combined = sparse.vstack((matrices["test"], matrices["ood"]), format="csr")
    combined = normalize(combined, norm="l2", copy=False)
    components = min(
        int(args.svd_components),
        combined.shape[0] - 1,
        combined.shape[1] - 1,
    )
    svd = TruncatedSVD(
        n_components=components,
        n_iter=7,
        random_state=int(args.seed) + mode_index,
    )
    reduced = svd.fit_transform(combined)
    test_reduced = reduced[: len(test["labels"])]
    ood_reduced = reduced[len(test["labels"]) :]
    geometry = {
        "silhouette": float(
            silhouette_score(test_reduced, test["labels"])
        ),
        **geometry_eval._leave_one_out_neighbor_metrics(
            test_reduced,
            test["labels"],
            int(args.neighbors),
        ),
        "explained_variance_50d": float(
            svd.explained_variance_ratio_.sum()
        ),
    }
    cross_time = geometry_eval._cross_time_neighbor_metrics(
        test_reduced,
        test["labels"],
        ood_reduced,
        ood["labels"],
        int(args.neighbors),
    )
    xy = TSNE(
        n_components=2,
        perplexity=float(args.tsne_perplexity),
        learning_rate="auto",
        init="pca",
        max_iter=int(args.tsne_iterations),
        random_state=int(args.seed) + 100 + mode_index,
    ).fit_transform(test_reduced).astype(np.float32)
    return {
        "label": spec.label,
        "alpha": spec.alpha,
        "representation": spec.key,
        "geometry": geometry,
        "cross_time_neighbors": cross_time,
        "feature_utilization": utilization,
        "semantic_information_nmi": nmi,
        "test_class_balance": geometry_eval._classwise_summary(
            probe_row["full"]
        ),
        "ood_class_balance": geometry_eval._classwise_summary(
            probe_row["ood"]
        ),
        "ood_retention": float(probe_row["ood"]["accuracy"])
        / max(float(probe_row["full"]["accuracy"]), 1e-12),
    }, xy


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root).resolve()
    source_root = eval_root / "shared/downstream_transfer/probe_features"
    output_root = source_root / "joint_extension"
    output_root.mkdir(parents=True, exist_ok=True)
    specs = _specs(args)
    device = torch.device(args.device)

    encoding_summary: dict[str, Any] = {}
    for spec in specs:
        encoding_summary[spec.key] = {}
        for split in SPLITS:
            print(f"[joint-arxiv] {spec.label} split={split}", flush=True)
            source = source_root / f"features-{split}.npz"
            encoded = output_root / f"{spec.key}-{split}.npz"
            reused = (
                _encoded_summary(
                    encoded=encoded,
                    source=source,
                    storage_k=int(args.storage_k),
                )
                if args.reuse_encoded
                else None
            )
            if reused is not None:
                encoding_summary[spec.key][split] = reused
                continue
            encoding_summary[spec.key][split] = _encode_split(
                spec,
                source=source,
                output=encoded,
                device=device,
                batch_size=int(args.batch_size),
                storage_k=int(args.storage_k),
            )

    feature_budgets = [
        int(value) for value in args.feature_budgets.split(",") if value
    ]
    low_label_budgets = [
        int(value) for value in args.low_label_budgets.split(",") if value
    ]
    low_label_seeds = [
        int(value) for value in args.low_label_seeds.split(",") if value
    ]
    probe_args = SimpleNamespace(
        classifier_c=float(args.classifier_c),
        max_iter=int(args.max_iter),
        n_jobs=int(args.n_jobs),
        low_label_budgets=",".join(map(str, low_label_budgets)),
        low_label_seeds=",".join(map(str, low_label_seeds)),
    )
    representations: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    split_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for spec in specs:
        split_data[spec.key] = {
            split: _load_encoded(output_root, spec, split)
            for split in SPLITS
        }
        matrices = tuple(
            _csr(
                split_data[spec.key][split]["indices"],
                split_data[spec.key][split]["values"],
                split_data[spec.key][split]["nnz"],
                spec.dictionary_width,
            )
            for split in SPLITS
        )
        row, pred = probe_eval.evaluate_representation(
            spec.key,
            matrices[0],
            split_data[spec.key]["train"]["labels"],
            matrices[1],
            split_data[spec.key]["validation"]["labels"],
            matrices[2],
            split_data[spec.key]["test"]["labels"],
            matrices[3],
            split_data[spec.key]["ood"]["labels"],
            probe_args,
            feature_budgets,
        )
        row["label"] = spec.label
        row["alpha"] = spec.alpha
        representations[spec.key] = row
        predictions[spec.key] = pred

    probe_summary = {
        "representations": representations,
        "high_level_transfer": _high_level_summary(
            representations,
            low_label_budgets,
        ),
        "metadata": {
            "feature_budgets": feature_budgets,
            "low_label_budgets": low_label_budgets,
            "low_label_seeds": low_label_seeds,
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "labels": np.unique(
                split_data[specs[0].key]["train"]["labels"]
            ).tolist(),
            "mean_nnz": {
                spec.key: {
                    split: float(
                        split_data[spec.key][split]["nnz"].mean()
                    )
                    for split in SPLITS
                }
                for spec in specs
            },
        },
    }

    base_probe = json.loads(
        (eval_root / "label_efficiency/linear_probe_results.json")
        .read_text(encoding="utf-8")
    )
    base_geometry = json.loads(
        (
            eval_root
            / "semantic_geometry/representation_geometry/"
            "representation_geometry.json"
        ).read_text(encoding="utf-8")
    )
    common_utilization_k = int(
        base_geometry["identity"]["common_utilization_k"]
    )
    geometry_methods: dict[str, Any] = {}
    embedding_payload = {
        "labels": split_data[specs[0].key]["test"]["labels"],
    }
    for index, spec in enumerate(specs):
        row, xy = _geometry(
            spec=spec,
            train=split_data[spec.key]["train"],
            test=split_data[spec.key]["test"],
            ood=split_data[spec.key]["ood"],
            probe_row=representations[spec.key],
            common_utilization_k=common_utilization_k,
            args=args,
            mode_index=index,
        )
        geometry_methods[spec.key] = row
        embedding_payload[f"{spec.key}_xy"] = xy
    embedding_path = output_root / "representation_embeddings.npz"
    np.savez_compressed(embedding_path, **embedding_payload)

    methods = {
        spec.key: {
            "label": spec.label,
            "alpha": spec.alpha,
            "encoding": encoding_summary[spec.key],
            "probe": representations[spec.key],
            "probe_summary": probe_summary["high_level_transfer"]["methods"][
                spec.key
            ],
            "geometry": geometry_methods[spec.key],
        }
        for spec in specs
    }
    output = output_root / "joint_extension.json"
    merge_sidecar(
        output,
        task="arxiv_transfer",
        methods=methods,
        specs=specs,
        protocol={
            "qwen_forward_reused": True,
            "source_raw_hidden": "features-{split}.npz:raw",
            "storage_k": int(args.storage_k),
            "feature_budgets": feature_budgets,
            "low_label_budgets": low_label_budgets,
            "low_label_seeds": low_label_seeds,
            "classifier": "multinomial logistic regression",
            "common_geometry_utilization_k": common_utilization_k,
            "svd_components": int(args.svd_components),
            "neighbors": int(args.neighbors),
            "tsne_perplexity": float(args.tsne_perplexity),
            "tsne_iterations": int(args.tsne_iterations),
        },
        files={
            "embeddings": embedding_path.name,
            "encoded_splits": [
                f"{spec.key}-{split}.npz"
                for spec in specs
                for split in SPLITS
            ],
        },
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
