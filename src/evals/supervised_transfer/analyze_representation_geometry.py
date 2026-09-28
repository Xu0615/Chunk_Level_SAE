#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from chunk_saes.artifacts import (
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

from evals.supervised_transfer.run_linear_probes import feature_matrix, load_split


RESULT_FORMAT = "chunk-saes-representation-geometry-v1"
METHODS = ("token", "temporal", "mean", "cross")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Measure semantic geometry, feature utilization, and cross-time "
            "nearest-neighbor transfer for frozen Chunk-SAE representations."
        )
    )
    p.add_argument("--features-dir", required=True)
    p.add_argument("--feature-manifest", required=True)
    p.add_argument("--probe-results", required=True)
    p.add_argument("--probe-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dict-size", type=int, default=65_536)
    p.add_argument("--svd-components", type=int, default=50)
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--tsne-perplexity", type=float, default=35.0)
    p.add_argument("--tsne-iterations", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=20260816)
    return p


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 0]
    if not values.size:
        return float("nan")
    ordered = np.sort(values)
    n = ordered.size
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float(weights @ ordered / (n * ordered.sum()))


def _distribution_metrics(values: np.ndarray, width: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0]
    probabilities = positive / positive.sum()
    entropy = float(
        -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum()
    )
    effective = float(math.exp(entropy))
    top_count = max(1, round(width * 0.01))
    top_share = float(np.sort(values)[::-1][:top_count].sum() / values.sum())
    ordered = np.sort(values)
    cumulative = np.cumsum(ordered)
    cumulative = np.concatenate(([0.0], cumulative / cumulative[-1]))
    population = np.linspace(0.0, 1.0, cumulative.size)
    sample_points = np.linspace(0.0, 1.0, 257)
    lorenz = np.interp(sample_points, population, cumulative)
    return {
        "active_features": int(positive.size),
        "active_fraction": float(positive.size / width),
        "effective_features": effective,
        "effective_feature_fraction": float(effective / width),
        "gini": _gini(values),
        "top_1pct_activity_share": top_share,
        "lorenz_population": sample_points.tolist(),
        "lorenz_activity": lorenz.tolist(),
    }


def _normalized_mutual_information(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
) -> float:
    classes, inverse = np.unique(labels, return_inverse=True)
    one_hot = sparse.csr_matrix(
        (
            np.ones(inverse.size, dtype=np.float64),
            (np.arange(inverse.size), inverse),
        ),
        shape=(inverse.size, classes.size),
    )
    joint = (matrix.T @ one_hot).toarray().astype(np.float64)
    total = joint.sum()
    if total <= 0:
        return float("nan")
    probabilities = joint / total
    feature_mass = probabilities.sum(axis=1, keepdims=True)
    label_mass = probabilities.sum(axis=0, keepdims=True)
    independent = feature_mass @ label_mass
    valid = probabilities > 0
    mutual_information = float(
        (
            probabilities[valid]
            * np.log(probabilities[valid] / independent[valid])
        ).sum()
    )
    label_probabilities = label_mass.reshape(-1)
    label_entropy = float(
        -(
            label_probabilities
            * np.log(np.maximum(label_probabilities, 1e-300))
        ).sum()
    )
    return mutual_information / max(label_entropy, 1e-12)


def _leave_one_out_neighbor_metrics(
    representation: np.ndarray,
    labels: np.ndarray,
    neighbors: int,
) -> dict[str, float]:
    index = NearestNeighbors(
        n_neighbors=neighbors + 1,
        metric="cosine",
    ).fit(representation)
    indices = index.kneighbors(return_distance=False)[:, 1:]
    neighbor_labels = labels[indices]
    purity = float((neighbor_labels == labels[:, None]).mean())
    predictions = []
    for row in neighbor_labels:
        values, counts = np.unique(row, return_counts=True)
        predictions.append(values[np.argmax(counts)])
    accuracy = float((np.asarray(predictions) == labels).mean())
    return {
        "neighbor_purity": purity,
        "knn_accuracy": accuracy,
    }


def _cross_time_neighbor_metrics(
    reference: np.ndarray,
    reference_labels: np.ndarray,
    query: np.ndarray,
    query_labels: np.ndarray,
    neighbors: int,
) -> dict[str, float]:
    index = NearestNeighbors(
        n_neighbors=neighbors,
        metric="cosine",
    ).fit(reference)
    indices = index.kneighbors(query, return_distance=False)
    neighbor_labels = reference_labels[indices]
    purity = float((neighbor_labels == query_labels[:, None]).mean())
    predictions = []
    for row in neighbor_labels:
        values, counts = np.unique(row, return_counts=True)
        predictions.append(values[np.argmax(counts)])
    accuracy = float((np.asarray(predictions) == query_labels).mean())
    return {
        "neighbor_purity": purity,
        "knn_accuracy": accuracy,
    }


def _classwise_summary(row: dict) -> dict:
    per_class = {
        str(key): float(value)
        for key, value in row["per_class_accuracy"].items()
    }
    values = np.asarray(list(per_class.values()), dtype=np.float64)
    return {
        "per_class_accuracy": per_class,
        "worst_class_accuracy": float(values.min()),
        "class_accuracy_std": float(values.std()),
        "class_accuracy_range": float(values.max() - values.min()),
    }


def main() -> None:
    args = parser().parse_args()
    feature_root = Path(args.features_dir).resolve()
    feature_manifest = load_artifact_manifest(
        args.feature_manifest,
        expected_format="chunk-saes-probe-features-v2",
        verify_files=True,
    )
    if Path(args.feature_manifest).parent.resolve() != feature_root:
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
    if representation_protocol.get("name") not in {
        FIXED_CHUNK_REPRESENTATION_PROTOCOL,
        LEGACY_VARIABLE_LENGTH_REPRESENTATION_PROTOCOL,
    }:
        raise ValueError("unknown probe feature representation protocol")
    if representation_protocol.get("token_temporal_aggregation") != TOKEN_AGGREGATION:
        raise ValueError("geometry requires mean-after-threshold feature aggregation")
    probes = _read(Path(args.probe_results))
    probe_manifest = load_artifact_manifest(
        args.probe_manifest,
        expected_format="chunk-saes-linear-probe-results-v2",
        verify_files=True,
    )
    probe_protocol = probe_manifest.get("representation_protocol")
    if probe_protocol is None:
        probe_protocol = probe_manifest.get("identity", {}).get(
            "representation_protocol"
        )
    if probe_protocol != representation_protocol:
        raise ValueError(
            "probe manifest and feature manifest use different representation protocols"
        )
    if (
        probe_manifest["identity"]["feature_artifact_digest"]
        != feature_manifest["artifact_digest"]
    ):
        raise ValueError(
            "probe results do not derive from the supplied feature artifact"
        )
    train, test, ood = (
        load_split(feature_root, split)
        for split in ("train", "test", "ood")
    )
    token_aggregation = str(probes["chosen_token_aggregation"])
    if token_aggregation != "mean":
        raise ValueError(
            "representation geometry requires the predeclared mean token "
            "aggregation"
        )
    feature_widths = {
        str(mode): int(width)
        for mode, width in (
            feature_manifest.get("feature_widths")
            or feature_manifest.get("identity", {}).get("feature_widths")
            or probes.get("metadata", {}).get("feature_widths")
            or {}
        ).items()
    }
    token_match_k = int(
        probes["metadata"]["sparsity_matching"]["token_eval_k"]
    )
    representation_specs = {
        "token": ("token", token_aggregation),
        "temporal": ("temporal", "mean"),
        "mean": ("mean", "direct"),
        "cross": ("cross", "direct"),
    }
    representation_names = {
        "token": f"token_sae_{token_aggregation}",
        "temporal": "temporal_sae_mean",
        "mean": "mean_chunk_sae",
        "cross": "cross_chunk_sae",
    }
    available_modes = tuple(
        mode
        for mode in METHODS
        if (
            f"{representation_specs[mode][0]}_"
            f"{representation_specs[mode][1] if representation_specs[mode][0] in {'token', 'temporal'} else 'direct'}"
            "_indices"
        )
        in test
        and representation_names[mode]
        in probes.get("representations", {})
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
            "representation geometry requires the complete common dictionary: "
            f"{mismatched_widths}"
        )
    required_modes = {"token", "mean", "cross"}
    missing_required = required_modes - set(available_modes)
    if missing_required:
        raise ValueError(
            "geometry analysis lacks required representations "
            f"{sorted(missing_required)}"
        )
    mean_nnz = probes["metadata"]["mean_nnz"]
    utilization_budgets = [token_match_k]
    for mode in available_modes:
        source_mode, aggregate = representation_specs[mode]
        suffix = (
            aggregate
            if source_mode in {"token", "temporal"}
            else "direct"
        )
        utilization_budgets.append(
            max(
                1,
                round(
                    float(
                        mean_nnz[f"{source_mode}_{suffix}"]["train"]
                    )
                ),
            )
        )
    common_utilization_k = min(utilization_budgets)

    methods = {}
    embedding_payload: dict[str, np.ndarray] = {
        "labels": test["labels"],
    }
    for mode_index, mode in enumerate(available_modes):
        source_mode, aggregate = representation_specs[mode]
        matrices = {
            split: feature_matrix(
                data,
                source_mode,
                aggregate,
                feature_widths[mode],
                token_match_k,
            )
            for split, data in (
                ("train", train),
                ("test", test),
                ("ood", ood),
            )
        }
        if not sparse.issparse(matrices["train"]):
            raise ValueError("geometry analysis requires sparse SAE features")

        # Utilization is compared at one common per-sample activity budget.
        suffix = (
            aggregate
            if source_mode in {"token", "temporal"}
            else "direct"
        )
        utilization_matrix = feature_matrix(
            train,
            source_mode,
            suffix,
            feature_widths[mode],
            common_utilization_k,
        )
        utilization_counts = np.asarray(
            (utilization_matrix > 0).sum(axis=0)
        ).reshape(-1)
        utilization = _distribution_metrics(
            utilization_counts,
            feature_widths[mode],
        )
        semantic_information = _normalized_mutual_information(
            utilization_matrix,
            train["labels"],
        )

        combined = sparse.vstack(
            (matrices["test"], matrices["ood"]),
            format="csr",
        )
        combined = normalize(combined, norm="l2", copy=False)
        components = min(
            args.svd_components,
            combined.shape[0] - 1,
            combined.shape[1] - 1,
        )
        svd = TruncatedSVD(
            n_components=components,
            n_iter=7,
            random_state=args.seed + mode_index,
        )
        reduced = svd.fit_transform(combined)
        test_reduced = reduced[: len(test["labels"])]
        ood_reduced = reduced[len(test["labels"]) :]
        geometry = {
            "silhouette": float(
                silhouette_score(test_reduced, test["labels"])
            ),
            **_leave_one_out_neighbor_metrics(
                test_reduced,
                test["labels"],
                args.neighbors,
            ),
            "explained_variance_50d": float(
                svd.explained_variance_ratio_.sum()
            ),
        }
        cross_time = _cross_time_neighbor_metrics(
            test_reduced,
            test["labels"],
            ood_reduced,
            ood["labels"],
            args.neighbors,
        )
        tsne = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=args.tsne_iterations,
            random_state=args.seed + 100 + mode_index,
        )
        embedding_payload[f"{mode}_xy"] = tsne.fit_transform(
            test_reduced
        ).astype(np.float32)
        probe_row = probes["representations"][representation_names[mode]]
        methods[mode] = {
            "representation": representation_names[mode],
            "geometry": geometry,
            "cross_time_neighbors": cross_time,
            "feature_utilization": utilization,
            "semantic_information_nmi": semantic_information,
            "test_class_balance": _classwise_summary(probe_row["full"]),
            "ood_class_balance": _classwise_summary(probe_row["ood"]),
            "ood_retention": float(probe_row["ood"]["accuracy"])
            / max(float(probe_row["full"]["accuracy"]), 1e-12),
        }

    comparisons = {
        f"cross_minus_{reference}": {
            "silhouette": (
                methods["cross"]["geometry"]["silhouette"]
                - methods[reference]["geometry"]["silhouette"]
            ),
            "test_neighbor_purity": (
                methods["cross"]["geometry"]["neighbor_purity"]
                - methods[reference]["geometry"]["neighbor_purity"]
            ),
            "cross_time_knn_accuracy": (
                methods["cross"]["cross_time_neighbors"]["knn_accuracy"]
                - methods[reference]["cross_time_neighbors"]["knn_accuracy"]
            ),
            "effective_feature_fraction": (
                methods["cross"]["feature_utilization"][
                    "effective_feature_fraction"
                ]
                - methods[reference]["feature_utilization"][
                    "effective_feature_fraction"
                ]
            ),
            "semantic_information_nmi": (
                methods["cross"]["semantic_information_nmi"]
                - methods[reference]["semantic_information_nmi"]
            ),
            "ood_retention": (
                methods["cross"]["ood_retention"]
                - methods[reference]["ood_retention"]
            ),
            "worst_class_ood_accuracy": (
                methods["cross"]["ood_class_balance"][
                    "worst_class_accuracy"
                ]
                - methods[reference]["ood_class_balance"][
                    "worst_class_accuracy"
                ]
            ),
            "ood_class_accuracy_std_reduction": (
                methods[reference]["ood_class_balance"][
                    "class_accuracy_std"
                ]
                - methods["cross"]["ood_class_balance"][
                    "class_accuracy_std"
                ]
            ),
        }
        for reference in (
            mode for mode in available_modes if mode != "cross"
        )
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "representation_geometry.json"
    embeddings_path = output_dir / "representation_embeddings.npz"
    identity = {
        "feature_artifact_digest": feature_manifest["artifact_digest"],
        "probe_artifact_digest": probe_manifest["artifact_digest"],
        "dict_size": args.dict_size,
        "feature_widths": feature_widths,
        "token_aggregation": token_aggregation,
        "token_match_k": token_match_k,
        "common_utilization_k": common_utilization_k,
        "svd_components": args.svd_components,
        "neighbors": args.neighbors,
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_iterations": args.tsne_iterations,
        "seed": args.seed,
        "representation_protocol": representation_protocol,
    }
    atomic_json_dump(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": identity,
            "methods": methods,
            "comparisons": comparisons,
            "interpretation": {
                "geometry": (
                    "Higher silhouette and neighbor purity indicate cleaner "
                    "high-level semantic organization."
                ),
                "cross_time_neighbors": (
                    "OOD abstracts query the in-distribution test manifold; "
                    "higher accuracy indicates stable concepts across years."
                ),
                "feature_utilization": (
                    "Higher effective feature fraction and lower Gini/top-1% "
                    "share indicate a more uniformly used dictionary."
                ),
                "semantic_information_nmi": (
                    "Normalized mutual information between sparse activation "
                    "mass and ArXiv domain labels."
                ),
            },
        },
        results_path,
    )
    np.savez_compressed(embeddings_path, **embedding_payload)
    write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": identity,
            "files": {
                "results": file_record(
                    results_path,
                    relative_to=output_dir,
                ),
                "embeddings": file_record(
                    embeddings_path,
                    relative_to=output_dir,
                ),
            },
        },
        output_dir / "representation_geometry_manifest.json",
    )
    print(results_path, flush=True)


if __name__ == "__main__":
    main()
