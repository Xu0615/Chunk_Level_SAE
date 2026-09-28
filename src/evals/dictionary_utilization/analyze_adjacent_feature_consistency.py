#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from sklearn.metrics import roc_auc_score

from chunk_saes.artifacts import (
    file_record,
    file_sha256,
    json_digest,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.evaluation_protocol import (
    fixed_chunk_protocol_metadata,
    full_dictionary_feature_widths,
    mean_after_threshold,
)
from chunk_saes.sae import SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.utils import atomic_json_dump


RESULT_FORMAT = "chunk-saes-adjacent-feature-consistency-v1"
DICTIONARY_UTILIZATION_FORMAT = "chunk-saes-dictionary-utilization-v1"
METHODS = ("token", "temporal", "mean", "cross")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Measure whether frozen SAE codes retain information shared by two "
            "genuinely adjacent chunks, against length-matched shuffled partners. "
            "This evaluation uses no ArXiv labels or downstream supervision."
        )
    )
    p.add_argument("--activation-cache-dir", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--dictionary-utilization-output",
        default=None,
        help=(
            "Optional standalone Effective feature fraction summary. The full "
            "adjacent-consistency artifact remains in --output-dir for internal "
            "provenance, while Eval 2 publishes only this dictionary-level result."
        ),
    )
    p.add_argument(
        "--checkpoint-selection",
        choices=("best", "final"),
        default="best",
    )
    p.add_argument("--sample-pairs", type=int, default=2048)
    p.add_argument("--feature-sample-size", type=int, default=8192)
    p.add_argument("--min-feature-support", type=int, default=8)
    p.add_argument(
        "--utilization-k",
        type=int,
        default=8,
        help=(
            "Common top-K active-feature budget per chunk used only for the "
            "label-free dictionary-utilization comparison."
        ),
    )
    p.add_argument("--token-batch-size", type=int, default=256)
    p.add_argument("--mean-batch-size", type=int, default=64)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit device such as cuda:0",
    )
    return p


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        return torch.device("cuda:0")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _cache_identity(cache_dir: Path, manifest: dict) -> dict:
    if manifest.get("format") != "chunk-saes-activation-cache-v2":
        raise ValueError("adjacent consistency requires activation cache v2")
    if manifest.get("complete") is not True:
        raise ValueError("activation cache is incomplete")
    if manifest.get("independent_forwards") is not True:
        raise ValueError("A/B chunks must come from independent model forwards")
    coverage = manifest.get("coverage") or {}
    if not all(
        bool(coverage.get(key))
        for key in (
            "pair_ids_complete",
            "pair_ids_unique",
            "occurrence_ranges_complete",
            "token_hidden_rows_equal_occurrences",
            "plan_row_digest_matches",
        )
    ):
        raise ValueError("activation cache lacks complete pair/occurrence coverage")
    return {
        "path": str(cache_dir),
        "format": manifest["format"],
        "activation_digest": manifest["activation_digest"],
        "plan_digest": manifest["plan_digest"],
        "pairs": int(manifest["pairs"]),
        "layer": int(manifest["layer"]),
        "hidden_size": int(manifest["hidden_size"]),
        "independent_forwards": True,
        "position_policy": manifest["position_policy"],
    }


def _shard_paths(cache_dir: Path, manifest: dict) -> list[Path]:
    paths = []
    for rank in manifest.get("ranks") or []:
        for shard in rank.get("shards") or []:
            path = cache_dir / str(shard["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    if not paths:
        raise ValueError("activation cache manifest contains no shards")
    return sorted(paths)


def _scan_pair_metadata(
    paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    document_hashes = []
    for shard_index, path in enumerate(paths):
        with safe_open(
            str(path),
            framework="pt",
            device="cpu",
        ) as handle:
            pair_ids = handle.get_tensor("pair_id").numpy()
            lengths_a = handle.get_tensor("length_a").numpy()
            lengths_b = handle.get_tensor("length_b").numpy()
            doc_hash = handle.get_tensor("doc_hash").numpy()
        for local_index, (pair_id, length_a, length_b) in enumerate(
            zip(pair_ids, lengths_a, lengths_b, strict=True)
        ):
            rows.append(
                (
                    shard_index,
                    local_index,
                    int(pair_id),
                    int(length_a),
                    int(length_b),
                )
            )
            document_hashes.append(bytes(doc_hash[local_index].tolist()))
    metadata = np.asarray(rows, dtype=np.int64)
    if metadata.ndim != 2 or metadata.shape[1] != 5:
        raise ValueError("invalid activation-cache pair metadata")
    if np.unique(metadata[:, 2]).size != metadata.shape[0]:
        raise ValueError("pair IDs are not unique")
    return metadata, np.asarray(document_hashes, dtype="S32")


def _stratified_sample(
    metadata: np.ndarray,
    *,
    sample_pairs: int,
    seed: int,
) -> np.ndarray:
    if sample_pairs <= 0:
        raise ValueError("--sample-pairs must be positive")
    if sample_pairs > metadata.shape[0]:
        raise ValueError(
            f"requested {sample_pairs} pairs but cache contains "
            f"{metadata.shape[0]}"
        )
    rng = np.random.default_rng(seed)
    cells = sorted(
        {
            (int(length_a), int(length_b))
            for length_a, length_b in metadata[:, 3:5]
        }
    )
    base, remainder = divmod(sample_pairs, len(cells))
    chosen: list[int] = []
    for cell_index, (length_a, length_b) in enumerate(cells):
        candidates = np.flatnonzero(
            (metadata[:, 3] == length_a)
            & (metadata[:, 4] == length_b)
        )
        target = base + int(cell_index < remainder)
        if candidates.size < target:
            raise ValueError(
                f"cell {(length_a, length_b)} contains {candidates.size} "
                f"pairs but stratified sample requests {target}"
            )
        chosen.extend(
            rng.choice(candidates, size=target, replace=False).tolist()
        )
    selected = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(selected)
    if selected.size != sample_pairs:
        raise AssertionError("stratified pair sample has the wrong size")
    return selected


def _load_selected_pairs(
    paths: list[Path],
    metadata: np.ndarray,
    selected: np.ndarray,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    positions_by_shard: dict[int, list[int]] = defaultdict(list)
    for selected_position, global_row in enumerate(selected):
        shard_index = int(metadata[global_row, 0])
        positions_by_shard[shard_index].append(selected_position)

    loaded: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    ] = [None] * selected.size
    for shard_index, selected_positions in positions_by_shard.items():
        path = paths[shard_index]
        with safe_open(
            str(path),
            framework="pt",
            device="cpu",
        ) as handle:
            mean_a = handle.get_tensor("mean_a")
            mean_b = handle.get_tensor("mean_b")
            token_hidden = handle.get_tensor("token_hidden")
            offsets = handle.get_tensor("chunk_offsets").long()
            for selected_position in selected_positions:
                global_row = int(selected[selected_position])
                local_index = int(metadata[global_row, 1])
                start = int(offsets[2 * local_index])
                split = int(offsets[2 * local_index + 1])
                stop = int(offsets[2 * local_index + 2])
                if not start < split < stop:
                    raise ValueError(f"invalid A/B offsets in {path}")
                loaded[selected_position] = (
                    mean_a[local_index].float(),
                    mean_b[local_index].float(),
                    token_hidden[start:split].float(),
                    token_hidden[split:stop].float(),
                )

    if any(row is None for row in loaded):
        raise AssertionError("some selected cache pairs were not loaded")
    complete = [row for row in loaded if row is not None]
    means_a = torch.stack([row[0] for row in complete], dim=0)
    means_b = torch.stack([row[1] for row in complete], dim=0)
    tokens_a = [row[2] for row in complete]
    tokens_b = [row[3] for row in complete]
    pair_ids = metadata[selected, 2].copy()
    lengths_a = metadata[selected, 3].copy()
    lengths_b = metadata[selected, 4].copy()
    if not all(
        chunk.shape[0] == int(length)
        for chunk, length in zip(tokens_a, lengths_a, strict=True)
    ):
        raise ValueError("A token-hidden lengths do not match cache metadata")
    if not all(
        chunk.shape[0] == int(length)
        for chunk, length in zip(tokens_b, lengths_b, strict=True)
    ):
        raise ValueError("B token-hidden lengths do not match cache metadata")
    return (
        means_a,
        means_b,
        tokens_a,
        tokens_b,
        pair_ids,
        lengths_a,
        lengths_b,
    )


def _matched_derangement(
    lengths_a: np.ndarray,
    lengths_b: np.ndarray,
    document_hashes: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = np.empty(lengths_a.size, dtype=np.int64)
    cells = sorted(
        {
            (int(length_a), int(length_b))
            for length_a, length_b in zip(
                lengths_a,
                lengths_b,
                strict=True,
            )
        }
    )
    for cell in cells:
        indices = np.flatnonzero(
            (lengths_a == cell[0]) & (lengths_b == cell[1])
        )
        if indices.size < 2:
            raise ValueError(
                f"length cell {cell} needs at least two sampled pairs "
                "for a shuffled control"
            )
        for _attempt in range(10_000):
            candidate = rng.permutation(indices)
            if np.any(candidate == indices):
                continue
            if np.any(document_hashes[candidate] == document_hashes[indices]):
                continue
            permutation[indices] = candidate
            break
        else:
            raise RuntimeError(
                f"could not construct a different-document shuffled "
                f"control for length cell {cell}"
            )
    if np.any(permutation == np.arange(permutation.size)):
        raise AssertionError("shuffled control contains a fixed point")
    if not np.array_equal(
        lengths_b[permutation],
        lengths_b,
    ):
        raise AssertionError("shuffled partners are not length matched")
    if np.any(document_hashes[permutation] == document_hashes):
        raise AssertionError("shuffled partners reuse the source document")
    return permutation


class SampledEncoder:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        feature_sample_size: int,
        seed: int,
        device: torch.device,
    ) -> None:
        config = _read(checkpoint_dir / "config.json")
        with safe_open(
            str(checkpoint_dir / "sae.safetensors"),
            framework="pt",
            device="cpu",
        ) as handle:
            tensor_names = set(handle.keys())
            counts = handle.get_tensor("feature_counts")
            self.dictionary_width = int(counts.numel())
            encoder_shape = handle.get_slice("encoder_weight").get_shape()
            if int(encoder_shape[0]) != self.dictionary_width:
                raise ValueError(
                    f"{checkpoint_dir} encoder width {encoder_shape[0]} does not "
                    f"match full dictionary width {self.dictionary_width}"
                )
            alive = torch.nonzero(
                counts > 0,
                as_tuple=False,
            ).flatten()
            if alive.numel() == 0:
                raise ValueError(f"no alive features in {checkpoint_dir}")
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(
                alive.numel(),
                generator=generator,
            )
            self.feature_ids = (
                alive[
                    permutation[
                        : min(feature_sample_size, int(alive.numel()))
                    ]
                ]
                .sort()
                .values
            )
            self.weight = handle.get_tensor("encoder_weight")[
                self.feature_ids
            ].to(device)
            self.bias = handle.get_tensor("encoder_bias")[
                self.feature_ids
            ].to(device)
            if "pre_bias" in tensor_names:
                self.pre_bias = handle.get_tensor("pre_bias").to(device)
            elif (
                config.get("sae_parameter_schema_version")
                == SAE_PARAMETER_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{checkpoint_dir} declares "
                    f"{SAE_PARAMETER_SCHEMA_VERSION} but lacks pre_bias"
                )
            else:
                self.pre_bias = handle.get_tensor("decoder_bias").to(
                    device
                )
            self.threshold = float(handle.get_tensor("threshold"))
            self.scale = float(handle.get_tensor("activation_scale"))
        self.device = device

    @torch.inference_mode()
    def _scores(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.device, dtype=self.weight.dtype)
        pre = F.relu(
            F.linear(
                hidden * self.scale - self.pre_bias,
                self.weight,
                self.bias,
            )
        )
        return pre * (pre > self.threshold)

    @torch.inference_mode()
    def encode_means(
        self,
        hidden: torch.Tensor,
        *,
        batch_size: int,
    ) -> np.ndarray:
        outputs = []
        for start in range(0, hidden.shape[0], batch_size):
            outputs.append(
                self._scores(hidden[start : start + batch_size])
                .float()
                .cpu()
                .numpy()
            )
        return np.concatenate(outputs, axis=0)

    @torch.inference_mode()
    def encode_token_mean(
        self,
        chunks: list[torch.Tensor],
        *,
        token_batch_size: int,
    ) -> np.ndarray:
        outputs = []
        for chunk in chunks:
            total = torch.zeros(
                self.feature_ids.numel(),
                device=self.device,
                dtype=torch.float32,
            )
            for start in range(0, chunk.shape[0], token_batch_size):
                piece = self._scores(chunk[start : start + token_batch_size])
                total.add_(
                    mean_after_threshold(piece) * piece.shape[0]
                )
            outputs.append((total / chunk.shape[0]).cpu().numpy())
        return np.stack(outputs, axis=0)

    def close(self) -> None:
        del self.weight, self.bias, self.pre_bias
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(512, samples))
    for start in range(0, samples, chunk):
        current = min(chunk, samples - start)
        indices = rng.integers(
            0,
            values.size,
            size=(current, values.size),
        )
        draws[start : start + current] = values[indices].mean(axis=1)
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _bootstrap_auc(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.shape != negative.shape:
        raise ValueError("positive and negative similarity arrays must match")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    labels = np.r_[
        np.ones(positive.size, dtype=np.int8),
        np.zeros(negative.size, dtype=np.int8),
    ]
    for draw in range(samples):
        indices = rng.integers(0, positive.size, size=positive.size)
        scores = np.r_[positive[indices], negative[indices]]
        estimates[draw] = roc_auc_score(labels, scores)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size == 0 or right.size == 0:
        raise ValueError("cannot bootstrap a difference with empty arrays")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(256, samples))
    for start in range(0, samples, chunk):
        current = min(chunk, samples - start)
        left_indices = rng.integers(
            0,
            left.size,
            size=(current, left.size),
        )
        right_indices = rng.integers(
            0,
            right.size,
            size=(current, right.size),
        )
        estimates[start : start + current] = (
            left[left_indices].mean(axis=1)
            - right[right_indices].mean(axis=1)
        )
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    return np.divide(
        np.einsum("ij,ij->i", left, right),
        denominator,
        out=np.zeros(left.shape[0], dtype=np.float64),
        where=denominator > 0,
    )


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("Gini values must be non-negative")
    if not np.any(values > 0):
        return float("nan")
    ordered = np.sort(values)
    n = ordered.size
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float(weights @ ordered / (n * ordered.sum()))


def _dictionary_utilization(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    *,
    utilization_k: int,
) -> dict:
    if not 0 < utilization_k <= codes_a.shape[1]:
        raise ValueError("utilization_k is outside the sampled dictionary")
    counts = np.zeros(codes_a.shape[1], dtype=np.float64)
    for codes in (codes_a, codes_b):
        indices = np.argpartition(
            codes,
            kth=codes.shape[1] - utilization_k,
            axis=1,
        )[:, -utilization_k:]
        values = np.take_along_axis(codes, indices, axis=1)
        valid = values > 0
        counts += np.bincount(
            indices[valid],
            minlength=codes.shape[1],
        )
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("sampled dictionary has no held-out activations")
    probabilities = counts[counts > 0] / total
    entropy = float(
        -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum()
    )
    effective = float(math.exp(entropy))
    top_count = max(1, round(counts.size * 0.01))
    ordered = np.sort(counts)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(ordered) / total)
    )
    population = np.linspace(0.0, 1.0, cumulative.size)
    sample_points = np.linspace(0.0, 1.0, 257)
    return {
        "sampled_features": int(counts.size),
        "common_activity_budget": utilization_k,
        "active_features": int(np.sum(counts > 0)),
        "active_fraction": float(np.mean(counts > 0)),
        "effective_features": effective,
        "effective_feature_fraction": float(effective / counts.size),
        "gini": _gini(counts),
        "top_1pct_activity_share": float(
            np.sort(counts)[::-1][:top_count].sum() / total
        ),
        "lorenz_population": sample_points.tolist(),
        "lorenz_activity": np.interp(
            sample_points,
            population,
            cumulative,
        ).tolist(),
    }


def _analyze_codes(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    shuffled: np.ndarray,
    *,
    min_feature_support: int,
    utilization_k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    true_cosine = _cosine_rows(codes_a, codes_b)
    shuffled_cosine = _cosine_rows(codes_a, codes_b[shuffled])
    separation = true_cosine - shuffled_cosine
    labels = np.r_[
        np.ones(true_cosine.size, dtype=np.int8),
        np.zeros(shuffled_cosine.size, dtype=np.int8),
    ]
    auc = float(
        roc_auc_score(
            labels,
            np.r_[true_cosine, shuffled_cosine],
        )
    )
    pairwise_accuracy = float(
        np.mean(true_cosine > shuffled_cosine)
        + 0.5 * np.mean(true_cosine == shuffled_cosine)
    )

    active_a = codes_a > 0
    active_b = codes_b > 0
    true_intersection = (active_a & active_b).sum(axis=1)
    true_union = (active_a | active_b).sum(axis=1)
    shuffled_intersection = (active_a & active_b[shuffled]).sum(axis=1)
    shuffled_union = (active_a | active_b[shuffled]).sum(axis=1)
    true_jaccard = true_intersection / np.maximum(true_union, 1)
    shuffled_jaccard = shuffled_intersection / np.maximum(
        shuffled_union,
        1,
    )

    support = active_a.sum(axis=0)
    supported = support >= min_feature_support
    true_feature_persistence = (active_a & active_b).sum(axis=0) / np.maximum(
        support,
        1,
    )
    shuffled_feature_persistence = (
        active_a & active_b[shuffled]
    ).sum(axis=0) / np.maximum(support, 1)
    feature_lift = true_feature_persistence - shuffled_feature_persistence
    supported_lift = feature_lift[supported]
    if supported_lift.size == 0:
        raise ValueError(
            "no sampled features meet --min-feature-support; "
            "increase pairs or reduce support"
        )

    return {
        "pair_metrics": {
            "true_cosine_mean": float(true_cosine.mean()),
            "shuffled_cosine_mean": float(shuffled_cosine.mean()),
            "cosine_separation_mean": float(separation.mean()),
            "cosine_separation_95ci": _bootstrap_mean(
                separation,
                samples=bootstrap_samples,
                seed=seed,
            ),
            "adjacent_retrieval_auc": auc,
            "adjacent_retrieval_auc_95ci": _bootstrap_auc(
                true_cosine,
                shuffled_cosine,
                samples=bootstrap_samples,
                seed=seed + 1,
            ),
            "paired_retrieval_accuracy": pairwise_accuracy,
            "true_jaccard_mean": float(true_jaccard.mean()),
            "shuffled_jaccard_mean": float(shuffled_jaccard.mean()),
            "jaccard_separation_mean": float(
                (true_jaccard - shuffled_jaccard).mean()
            ),
        },
        "feature_metrics": {
            "eligible_features": int(supported.sum()),
            "min_support": min_feature_support,
            "mean_persistence_lift": float(supported_lift.mean()),
            "median_persistence_lift": float(
                np.median(supported_lift)
            ),
            "positive_lift_fraction": float(
                np.mean(supported_lift > 0)
            ),
            "mean_persistence_lift_95ci": _bootstrap_mean(
                supported_lift,
                samples=bootstrap_samples,
                seed=seed + 2,
            ),
        },
        "dictionary_utilization": _dictionary_utilization(
            codes_a,
            codes_b,
            utilization_k=utilization_k,
        ),
        "distributions": {
            "true_cosine": true_cosine.tolist(),
            "shuffled_cosine": shuffled_cosine.tolist(),
            "cosine_separation": separation.tolist(),
            "feature_persistence_lift": supported_lift.tolist(),
        },
    }


def main() -> None:
    args = parser().parse_args()
    if args.feature_sample_size <= 0:
        raise ValueError("--feature-sample-size must be positive")
    if args.min_feature_support <= 0:
        raise ValueError("--min-feature-support must be positive")
    if args.utilization_k <= 0:
        raise ValueError("--utilization-k must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    device = _device(args.device)
    cache_dir = Path(args.activation_cache_dir).resolve()
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = _read(cache_manifest_path)
    cache_identity = _cache_identity(cache_dir, cache_manifest)
    sae_set = resolve_sae_artifact_set(
        args.sae_root,
        selection=args.checkpoint_selection,
    )
    if int(sae_set["common"]["layer"]) != cache_identity["layer"]:
        raise ValueError("SAE layer does not match activation cache")
    if int(sae_set["common"]["activation_dim"]) != cache_identity[
        "hidden_size"
    ]:
        raise ValueError("SAE activation dimension does not match cache")
    validation_digest = sae_set["common"].get(
        "validation_cache_digest"
    )
    if validation_digest != cache_identity["activation_digest"]:
        raise ValueError(
            "SAEs do not derive from the supplied validation cache"
        )

    shard_paths = _shard_paths(cache_dir, cache_manifest)
    metadata, document_hashes = _scan_pair_metadata(shard_paths)
    selected = _stratified_sample(
        metadata,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
    )
    (
        means_a,
        means_b,
        tokens_a,
        tokens_b,
        pair_ids,
        lengths_a,
        lengths_b,
    ) = _load_selected_pairs(shard_paths, metadata, selected)
    shuffled = _matched_derangement(
        lengths_a,
        lengths_b,
        document_hashes[selected],
        seed=args.seed + 1,
    )

    modes = tuple(sae_set["modes"])
    feature_widths = full_dictionary_feature_widths(sae_set, modes=modes)
    methods = {}
    for mode_index, mode in enumerate(modes):
        checkpoint_dir = Path(
            sae_set["modes"][mode]["checkpoint_path"]
        )
        encoder = SampledEncoder(
            checkpoint_dir,
            feature_sample_size=args.feature_sample_size,
            seed=args.seed + 100 + mode_index,
            device=device,
        )
        if encoder.dictionary_width != feature_widths[mode]:
            raise ValueError(
                f"{mode} checkpoint width {encoder.dictionary_width} does not "
                f"match full dictionary width {feature_widths[mode]}"
            )
        try:
            if mode in {"token", "temporal"}:
                codes_a = encoder.encode_token_mean(
                    tokens_a,
                    token_batch_size=args.token_batch_size,
                )
                codes_b = encoder.encode_token_mean(
                    tokens_b,
                    token_batch_size=args.token_batch_size,
                )
                representation = (
                    "mean of thresholded token-level feature activations"
                    if mode == "token"
                    else (
                        "mean of thresholded Temporal SAE token feature "
                        "activations over the complete dictionary"
                    )
                )
            else:
                codes_a = encoder.encode_means(
                    means_a,
                    batch_size=args.mean_batch_size,
                )
                codes_b = encoder.encode_means(
                    means_b,
                    batch_size=args.mean_batch_size,
                )
                representation = (
                    "thresholded feature activations of the chunk mean"
                )
            metrics = _analyze_codes(
                codes_a,
                codes_b,
                shuffled,
                min_feature_support=args.min_feature_support,
                utilization_k=args.utilization_k,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 1000 * (mode_index + 1),
            )
            methods[mode] = {
                "representation": representation,
                "sampled_feature_ids_digest": json_digest(
                    encoder.feature_ids.tolist()
                ),
                "sampled_features": int(
                    encoder.feature_ids.numel()
                ),
                "mean_active_features_per_chunk": float(
                    np.mean((codes_a > 0).sum(axis=1))
                ),
                **metrics,
            }
        finally:
            encoder.close()

    comparisons = {}
    for reference in (
        mode for mode in modes if mode != "cross"
    ):
        cross_lift = np.asarray(
            methods["cross"]["distributions"][
                "feature_persistence_lift"
            ],
            dtype=np.float64,
        )
        reference_lift = np.asarray(
            methods[reference]["distributions"][
                "feature_persistence_lift"
            ],
            dtype=np.float64,
        )
        comparisons[f"cross_minus_{reference}"] = {
            "cosine_separation": (
                methods["cross"]["pair_metrics"][
                    "cosine_separation_mean"
                ]
                - methods[reference]["pair_metrics"][
                    "cosine_separation_mean"
                ]
            ),
            "adjacent_retrieval_auc": (
                methods["cross"]["pair_metrics"][
                    "adjacent_retrieval_auc"
                ]
                - methods[reference]["pair_metrics"][
                    "adjacent_retrieval_auc"
                ]
            ),
            "feature_persistence_lift": (
                methods["cross"]["feature_metrics"][
                    "mean_persistence_lift"
                ]
                - methods[reference]["feature_metrics"][
                    "mean_persistence_lift"
                ]
            ),
            "feature_persistence_lift_95ci": _bootstrap_difference(
                cross_lift,
                reference_lift,
                samples=args.bootstrap_samples,
                seed=args.seed
                + 10_000
                + (0 if reference == "token" else 1),
            ),
            "effective_feature_fraction": (
                methods["cross"]["dictionary_utilization"][
                    "effective_feature_fraction"
                ]
                - methods[reference]["dictionary_utilization"][
                    "effective_feature_fraction"
                ]
            ),
            "gini_reduction": (
                methods[reference]["dictionary_utilization"]["gini"]
                - methods["cross"]["dictionary_utilization"]["gini"]
            ),
        }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "adjacent_feature_consistency.json"
    identity = {
        "sae_set_digest": sae_set["artifact_digest"],
        "checkpoint_selection": args.checkpoint_selection,
        "activation_cache": cache_identity,
        "activation_cache_manifest_sha256": file_sha256(
            cache_manifest_path
        ),
        "sample_pairs": args.sample_pairs,
        "feature_sample_size": args.feature_sample_size,
        "min_feature_support": args.min_feature_support,
        "utilization_k": args.utilization_k,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "feature_widths": feature_widths,
        "representation_protocol": fixed_chunk_protocol_metadata(
            feature_widths=feature_widths,
            chunk_lengths=sorted(
                set(lengths_a.tolist()) | set(lengths_b.tolist())
            ),
        ),
        "negative_control": (
            "B chunk shuffled only within identical "
            "(length_a, length_b) cells, with no fixed points and no "
            "same-document matches"
        ),
    }
    atomic_json_dump(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": identity,
            "sample": {
                "pairs": args.sample_pairs,
                "pair_ids_digest": json_digest(pair_ids.tolist()),
                "document_hashes_digest": json_digest(
                    [
                        value.hex()
                        for value in document_hashes[selected].tolist()
                    ]
                ),
                "length_cells": {
                    f"{length_a}x{length_b}": int(
                        np.sum(
                            (lengths_a == length_a)
                            & (lengths_b == length_b)
                        )
                    )
                    for length_a, length_b in sorted(
                        set(
                            zip(
                                lengths_a.tolist(),
                                lengths_b.tolist(),
                            )
                        )
                    )
                },
                "shuffled_pair_ids_digest": json_digest(
                    pair_ids[shuffled].tolist()
                ),
            },
            "methods": methods,
            "comparisons": comparisons,
            "primary_metric": (
                "mean per-feature persistence lift: "
                "P(feature active in true adjacent B | active in A) minus "
                "the same probability for a length-matched, "
                "different-document shuffled B"
            ),
            "interpretation": (
                "A strong high-level representation should make the two "
                "genuinely adjacent chunks more similar than a length-matched "
                "random partner. This test is unsupervised and does not use "
                "ArXiv categories or downstream labels."
            ),
        },
        results_path,
    )
    write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": identity,
            "files": {
                "results": file_record(
                    results_path,
                    relative_to=output_dir,
                )
            },
        },
        output_dir / "adjacent_feature_consistency_manifest.json",
    )
    if args.dictionary_utilization_output:
        dictionary_output = Path(args.dictionary_utilization_output).resolve()
        dictionary_output.parent.mkdir(parents=True, exist_ok=True)
        dictionary_identity = {
            "sae_set_digest": sae_set["artifact_digest"],
            "source_adjacent_results_sha256": file_sha256(results_path),
            "full_dictionary_widths": feature_widths,
            "activation_budget_k": 128,
            "utilization_k_used_for_estimate": args.utilization_k,
            "sampled_alive_features": {
                mode: int(
                    methods[mode]["dictionary_utilization"][
                        "sampled_features"
                    ]
                )
                for mode in METHODS
            },
            "definition": (
                "exp(entropy of the dictionary-level activation-mass "
                "distribution) divided by the sampled alive-feature count"
            ),
        }
        atomic_json_dump(
            {
                "format": DICTIONARY_UTILIZATION_FORMAT,
                "complete": True,
                "identity": dictionary_identity,
                "methods": {
                    mode: {
                        "effective_feature_fraction": float(
                            methods[mode]["dictionary_utilization"][
                                "effective_feature_fraction"
                            ]
                        ),
                        "effective_features": float(
                            methods[mode]["dictionary_utilization"][
                                "effective_features"
                            ]
                        ),
                        "sampled_alive_features": int(
                            methods[mode]["dictionary_utilization"][
                                "sampled_features"
                            ]
                        ),
                        "full_dictionary_width": int(feature_widths[mode]),
                        "k128_equivalent_slots": float(
                            128
                            * methods[mode]["dictionary_utilization"][
                                "effective_feature_fraction"
                            ]
                        ),
                    }
                    for mode in METHODS
                },
                "interpretation": (
                    "Higher effective fraction means activation mass is spread "
                    "across more dictionary features. k128_equivalent_slots is "
                    "a visualization of capacity, not a fixed per-example "
                    "activation count."
                ),
            },
            dictionary_output,
        )
        write_artifact_manifest(
            {
                "format": DICTIONARY_UTILIZATION_FORMAT,
                "complete": True,
                "identity": dictionary_identity,
                "files": {
                    "results": file_record(
                        dictionary_output,
                        relative_to=dictionary_output.parent,
                    )
                },
            },
            dictionary_output.with_name("dictionary_utilization_manifest.json"),
        )
        print(dictionary_output, flush=True)
    print(results_path, flush=True)


if __name__ == "__main__":
    main()
