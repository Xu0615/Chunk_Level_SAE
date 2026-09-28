#!/usr/bin/env python
"""Evaluate frozen ablation checkpoints on the shared held-out protocols.

The mean-Cross direction ablations can use the existing activation-cache-v2
pair protocol directly.  This script computes adjacent persistence/retrieval
and, when the document-linking feature cache is available, the same exact
length-controlled lexical linking metrics used by the baseline evaluation.
Sequence checkpoints are intentionally recorded as ``not_available`` here:
their token-level context representation cannot be reconstructed from the
mean-only document-linking artifact without introducing a different corpus or
forward protocol.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file


def _linear_probes(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    lengths_a: np.ndarray,
    lengths_b: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit held-out linear diagnostics with pair-clustered splits."""

    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.preprocessing import normalize
    from scipy import sparse

    pair_count = int(codes_a.shape[0])
    if pair_count < 20 or codes_b.shape[0] != pair_count:
        return {"status": "not_available", "reason": "too few paired rows"}
    rng = np.random.default_rng(seed)
    order = rng.permutation(pair_count)
    split = min(pair_count - 1, max(1, round(0.7 * pair_count)))
    train_pairs, test_pairs = order[:split], order[split:]
    values = np.concatenate((codes_a, codes_b), axis=0).astype(np.float32, copy=False)
    matrix = sparse.csr_matrix(values)
    matrix = normalize(matrix, norm="l2", copy=False)
    train_rows = np.concatenate((train_pairs, train_pairs + pair_count))
    test_rows = np.concatenate((test_pairs, test_pairs + pair_count))

    def fit(labels: np.ndarray, name: str) -> dict[str, Any]:
        y_train = labels[train_rows]
        y_test = labels[test_rows]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            return {"status": "not_available", "reason": f"{name} lacks classes"}
        classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-4,
            random_state=seed,
            class_weight="balanced",
            average=True,
        )
        classifier.fit(matrix[train_rows], y_train)
        predicted = classifier.predict(matrix[test_rows])
        counts = np.bincount(y_test.astype(np.int64))
        return {
            "status": "complete",
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "majority_baseline": float(counts.max() / counts.sum()),
            "classes": int(np.unique(labels).size),
            "train_pairs": int(train_pairs.size),
            "test_pairs": int(test_pairs.size),
        }

    side = np.concatenate(
        (np.zeros(pair_count, dtype=np.int64), np.ones(pair_count, dtype=np.int64))
    )
    all_lengths = np.concatenate((lengths_a, lengths_b)).astype(np.int64)
    length_values = sorted(np.unique(all_lengths).tolist())
    length_to_class = {value: index for index, value in enumerate(length_values)}
    length_labels = np.asarray(
        [length_to_class[int(value)] for value in all_lengths], dtype=np.int64
    )
    return {
        "status": "complete",
        "split_policy": "70/30 pair-clustered deterministic split",
        "side": fit(side, "side"),
        "chunk_length": {
            **fit(length_labels, "chunk length"),
            "length_values": length_values,
        },
    }


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _complete_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for marker in sorted(root.glob("E[1-4]/*/seed*/**/complete.json")):
        complete = _read(marker)
        parts = marker.parts
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
        records.append(
            {
                "experiment": experiment,
                "stage": stage,
                "seed": seed_name.removeprefix("seed"),
                "seed_dir": marker.parents[1] if marker.parent.name == "cross" else marker.parent,
                "artifact_dir": marker.parent,
                "complete": complete,
            }
        )
    return records


def _baseline_record(root: Path) -> dict[str, Any]:
    candidates = []
    for marker in sorted(root.parent.glob("checkpoints/*/cross/complete.json")):
        complete = _read(marker)
        if (
            complete.get("complete") is True
            and int(complete.get("samples_seen", 0)) == 1_000_000_000
        ):
            candidates.append((marker, complete))
    if len(candidates) != 1:
        raise ValueError(
            "expected one completed 1B E0 Cross checkpoint, found "
            f"{len(candidates)}"
        )
    marker, complete = candidates[0]
    return {
        "experiment": "E0",
        "stage": "full",
        "seed": str(complete.get("seed", 42)),
        "seed_dir": root / "E0",
        "artifact_dir": marker.parent,
        "complete": complete,
    }


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _linking_metrics(checkpoint: Path, feature_dir: Path, device: torch.device) -> dict[str, Any]:
    """Run the existing document-linking rank calculation for one checkpoint."""

    # These imports are intentionally lazy: importing the corpus/evaluation
    # module pulls in transformers and is unnecessary for cache-only metrics.
    from evals.document_linking.evaluate_lexical_controlled_document_linking import (
        SparseEncoder,
        _csr_from_fixed_topk,
        _metrics_from_ranks,
        retrieval_ranks,
    )

    import scipy.sparse as sparse

    arrays_path = feature_dir / "features.npz"
    pairs_path = feature_dir / "selected_pairs.jsonl"
    if not arrays_path.is_file() or not pairs_path.is_file():
        return {"status": "not_available", "reason": "document-linking feature cache missing"}
    with np.load(arrays_path) as data:
        arrays = {key: data[key] for key in data.files}
    pairs = [
        json.loads(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row_by_global = {
        int(global_id): index
        for index, global_id in enumerate(arrays["global_chunk_ids"])
    }
    left_rows = np.asarray(
        [row_by_global[int(row["left_global_chunk_id"])] for row in pairs],
        dtype=np.int64,
    )
    right_rows = np.asarray(
        [row_by_global[int(row["right_global_chunk_id"])] for row in pairs],
        dtype=np.int64,
    )
    lengths = arrays["retokenized_lengths"].astype(np.int16)
    left_lengths = lengths[left_rows]
    right_lengths = lengths[right_rows]
    encoder = SparseEncoder(checkpoint, device)
    try:
        # Top-k inference is batched to bound the dense [rows, width] matrix.
        indices_parts: list[np.ndarray] = []
        values_parts: list[np.ndarray] = []
        hidden = arrays["raw_mean_hidden"]
        for start in range(0, hidden.shape[0], 32):
            indices, values = encoder.topk(
                torch.from_numpy(hidden[start : start + 32]),
                128,
            )
            indices_parts.append(indices.cpu().numpy().astype(np.int32))
            values_parts.append(values.cpu().numpy().astype(np.float32))
        indices = np.concatenate(indices_parts, axis=0)
        values = np.concatenate(values_parts, axis=0)
        representation = _csr_from_fixed_topk(
            indices,
            values,
            width=encoder.dictionary_width,
        )
        forward = retrieval_ranks(
            representation,
            query_indices=left_rows,
            gallery_indices=right_rows,
            gallery_lengths=right_lengths,
        )
        backward = retrieval_ranks(
            representation,
            query_indices=right_rows,
            gallery_indices=left_rows,
            gallery_lengths=left_lengths,
        )
        ranks = np.concatenate([forward, backward])
        metrics = _metrics_from_ranks(ranks)
        metrics.update(
            {
                "directions": 2,
                "documents": len(pairs),
                "representation": "thresholded SAE activation of cached chunk mean, top-128 cosine",
            }
        )
        return metrics
    finally:
        encoder.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _adjacent_metrics(
    checkpoint: Path,
    validation_cache: Path,
    *,
    device: torch.device,
    sample_pairs: int,
    feature_sample_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    from evals.dictionary_utilization.analyze_adjacent_feature_consistency import (
        _analyze_codes,
        _cache_identity,
        _load_selected_pairs,
        _matched_derangement,
        _read as read_json,
        _scan_pair_metadata,
        _shard_paths,
        _stratified_sample,
        SampledEncoder,
    )

    manifest = read_json(validation_cache / "manifest.json")
    _cache_identity(validation_cache, manifest)
    paths = _shard_paths(validation_cache, manifest)
    metadata, document_hashes = _scan_pair_metadata(paths)
    selected = _stratified_sample(metadata, sample_pairs=sample_pairs, seed=seed)
    means_a, means_b, _tokens_a, _tokens_b, _pair_ids, lengths_a, lengths_b = _load_selected_pairs(
        paths, metadata, selected
    )
    shuffled = _matched_derangement(
        lengths_a,
        lengths_b,
        document_hashes[selected],
        seed=seed + 1,
    )
    config = _read(checkpoint / "config.json")
    encoder = SampledEncoder(
        checkpoint,
        feature_sample_size=feature_sample_size,
        seed=seed + 17,
        device=device,
    )
    try:
        codes_a = encoder.encode_means(means_a, batch_size=64)
        codes_b = encoder.encode_means(means_b, batch_size=64)
        metrics = _analyze_codes(
            codes_a,
            codes_b,
            shuffled,
            min_feature_support=8,
            utilization_k=8,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 2,
        )
        combined = np.concatenate((codes_a, codes_b), axis=0)
        activity = (combined > 0).sum(axis=0).astype(np.float64)
        probability = activity[activity > 0] / max(float(activity.sum()), 1.0)
        activation_entropy = float(
            -(probability * np.log(np.maximum(probability, 1e-300))).sum()
        ) if probability.size else float("nan")
        return {
            "status": "complete",
            "checkpoint": str(checkpoint),
            "direction_policy": config.get("direction_policy"),
            "sample_pairs": sample_pairs,
            "feature_sample_size": int(encoder.feature_ids.numel()),
            "activation_entropy": activation_entropy,
            "activation_effective_features": float(np.exp(activation_entropy))
            if math.isfinite(activation_entropy)
            else None,
            "probes": _linear_probes(
                codes_a,
                codes_b,
                lengths_a,
                lengths_b,
                seed=seed + 31,
            ),
            **metrics,
        }
    finally:
        encoder.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _sequence_adjacent_metrics(
    checkpoint: Path,
    train_cache: Path,
    validation_cache: Path,
    *,
    loss_mask: str,
    device: torch.device,
    sample_pairs: int,
    feature_sample_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Measure adjacent code persistence for a frozen sequence checkpoint.

    The sequence model's partner-query features are aggregated over each
    predicted target chunk.  This is deliberately a separate representation
    from mean-Cross: it answers whether the masked sequence objective retains
    adjacent information, while document-text metrics remain unavailable.
    """

    from evals.dictionary_utilization.analyze_adjacent_feature_consistency import (
        _analyze_codes,
        _cache_identity,
        _load_selected_pairs,
        _matched_derangement,
        _read as read_json,
        _scan_pair_metadata,
        _shard_paths,
        _stratified_sample,
    )
    from evals.experiments.train_sequence_cross_ablation import (
        MaskedSequenceSAE,
        _estimate_norm,
        _token_mean,
    )

    validation_manifest = read_json(validation_cache / "manifest.json")
    _cache_identity(validation_cache, validation_manifest)
    paths = _shard_paths(validation_cache, validation_manifest)
    metadata, document_hashes = _scan_pair_metadata(paths)
    selected = _stratified_sample(metadata, sample_pairs=sample_pairs, seed=seed)
    means_a, means_b, tokens_a, tokens_b, _pair_ids, lengths_a, lengths_b = _load_selected_pairs(
        paths, metadata, selected
    )
    shuffled = _matched_derangement(
        lengths_a,
        lengths_b,
        document_hashes[selected],
        seed=seed + 1,
    )
    config = read_json(checkpoint.parent.parent / "run_config.json")
    train_manifest = read_json(train_cache / "manifest.json")
    hidden_size = int(train_manifest["hidden_size"])
    norm = _estimate_norm(train_cache, train_manifest, 0)
    token_mean = _token_mean(train_manifest, hidden_size, device)
    sentinel = token_mean * (norm / token_mean.norm().clamp_min(1e-6))
    model = MaskedSequenceSAE(
        hidden_size,
        int(config["dict_size"]),
        int(config["k"]),
        int(config["max_chunk_length"]),
        loss_mask=loss_mask,
        sentinel=sentinel,
        self_mean=token_mean,
        context_dim=int(config["context_dim"]),
        context_heads=int(config["context_heads"]),
        decoder_backend="dense",
        candidate_multiplier=2.0,
        bf16_histogram=False,
    ).to(device)
    state = load_file(str(checkpoint / "model.safetensors"), device=str(device))
    model.load_state_dict(state, strict=False)
    model.eval()

    def pack(sources: list[torch.Tensor], targets: list[torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[int]]:
        max_len = max(max(int(x.shape[0]) for x in sources), max(int(x.shape[0]) for x in targets))
        batch_size = len(sources)
        source = torch.zeros((batch_size, max_len, hidden_size), dtype=torch.bfloat16)
        target = torch.zeros_like(source)
        source_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        partner_mask = torch.zeros_like(source_mask)
        target_lengths: list[int] = []
        for row, (src, tgt) in enumerate(zip(sources, targets, strict=True)):
            sl, tl = int(src.shape[0]), int(tgt.shape[0])
            source[row, :sl] = src.to(torch.bfloat16)
            target[row, :tl] = tgt.to(torch.bfloat16)
            source_mask[row, :sl] = True
            partner_mask[row, :tl] = True
            target_lengths.append(tl)
        return {
            "source": source,
            "target": target,
            "source_mask": source_mask,
            "partner_mask": partner_mask,
            "side_b": torch.zeros((batch_size,), dtype=torch.bool),
        }, target_lengths

    # Build two directions per pair so target A and target B codes are emitted
    # by the same BatchTopK selection context.
    code_a: list[np.ndarray] = []
    code_b: list[np.ndarray] = []
    batch_pairs = 4
    for start in range(0, sample_pairs, batch_pairs):
        stop = min(sample_pairs, start + batch_pairs)
        sources: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for index in range(start, stop):
            sources.extend((tokens_a[index], tokens_b[index]))
            targets.extend((tokens_b[index], tokens_a[index]))
        batch, target_lengths = pack(sources, targets)
        moved = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _reconstructed, _target, features, selected_kind, _activity = model(
                moved,
                distributed=False,
            )
        # The partner rows are contiguous in target-example order.  Split by
        # target length and aggregate feature activations over each chunk.
        partner_features = features[selected_kind] if loss_mask == "all" else features
        offset = 0
        for local in range(stop - start):
            b_len = target_lengths[2 * local]
            a_len = target_lengths[2 * local + 1]
            code_b.append(partner_features[offset : offset + b_len].float().mean(0).cpu().numpy())
            offset += b_len
            code_a.append(partner_features[offset : offset + a_len].float().mean(0).cpu().numpy())
            offset += a_len
    codes_a = np.stack(code_a, axis=0)
    codes_b = np.stack(code_b, axis=0)
    if not np.any(codes_a > 0) or not np.any(codes_b > 0):
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "degenerate",
            "reason": "partner-query branch produced no active dictionary features on the held-out sample",
            "sample_pairs": sample_pairs,
            "feature_sample_size": int(codes_a.shape[1]),
        }
    if feature_sample_size < codes_a.shape[1]:
        rng = np.random.default_rng(seed + 5)
        support = (codes_a > 0).sum(axis=0) + (codes_b > 0).sum(axis=0)
        candidates = np.flatnonzero(support > 0)
        if candidates.size > feature_sample_size:
            candidates = rng.choice(candidates, size=feature_sample_size, replace=False)
        candidates.sort()
        codes_a = codes_a[:, candidates]
        codes_b = codes_b[:, candidates]
    combined = np.concatenate((codes_a, codes_b), axis=0)
    activity = (combined > 0).sum(axis=0).astype(np.float64)
    probability = activity[activity > 0] / max(float(activity.sum()), 1.0)
    activation_entropy = float(
        -(probability * np.log(np.maximum(probability, 1e-300))).sum()
    ) if probability.size else float("nan")
    try:
        metrics = _analyze_codes(
            codes_a,
            codes_b,
            shuffled,
            min_feature_support=2,
            utilization_k=min(8, codes_a.shape[1]),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 7,
        )
    except ValueError as error:
        if "no sampled features meet" not in str(error):
            raise
        metrics = {
            "status": "degenerate",
            "reason": str(error),
            "sample_pairs": sample_pairs,
            "feature_sample_size": int(codes_a.shape[1]),
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": "complete",
        "representation": "mean partner-query feature activation over each predicted target chunk",
        "loss_mask": loss_mask,
        "sample_pairs": sample_pairs,
        "feature_sample_size": int(codes_a.shape[1]),
        "activation_entropy": activation_entropy,
        "activation_effective_features": float(np.exp(activation_entropy))
        if math.isfinite(activation_entropy)
        else None,
        "probes": _linear_probes(
            codes_a,
            codes_b,
            lengths_a,
            lengths_b,
            seed=seed + 31,
        ),
        **metrics,
    }


def evaluate_record(
    record: dict[str, Any],
    *,
    validation_cache: Path,
    train_cache: Path,
    feature_dir: Path | None,
    device: torch.device,
    sample_pairs: int,
    feature_sample_size: int,
    bootstrap_samples: int,
    seed: int,
    sequence_sample_pairs: int,
) -> dict[str, Any]:
    experiment = record["experiment"]
    complete = record["complete"]
    result: dict[str, Any] = {
        "experiment": experiment,
        "stage": record["stage"],
        "seed": record["seed"],
    }
    if experiment in {"E0", "E1", "E2"}:
        checkpoint = record["artifact_dir"] / "checkpoints" / "best"
        result["adjacent"] = _adjacent_metrics(
            checkpoint,
            validation_cache,
            device=device,
            sample_pairs=sample_pairs,
            feature_sample_size=feature_sample_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        result["linking"] = (
            _linking_metrics(checkpoint, feature_dir, device)
            if feature_dir is not None
            else {"status": "not_available"}
        )
    else:
        checkpoint = record["artifact_dir"] / "checkpoints" / "best"
        result["adjacent"] = _sequence_adjacent_metrics(
            checkpoint,
            train_cache,
            validation_cache,
            loss_mask=("all" if experiment == "E4" else "partner-only"),
            device=device,
            sample_pairs=sequence_sample_pairs,
            feature_sample_size=feature_sample_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        result["linking"] = {
            "status": "not_available",
            "reason": "mean-only document-linking cache is not a valid sequence provenance source",
        }
    result["complete_artifact"] = {
        "partner_fve": (complete.get("validation_metrics") or {}).get("partner_fve"),
        "self_fve": (complete.get("validation_metrics") or {}).get("self_fve"),
        "total_fve": (complete.get("validation_metrics") or {}).get("total_fve"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--validation-cache-dir", required=True)
    parser.add_argument("--train-cache-dir", required=True)
    parser.add_argument("--feature-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-pairs", type=int, default=2048)
    parser.add_argument("--feature-sample-size", type=int, default=8192)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--sequence-sample-pairs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    root = Path(args.ablation_root).resolve()
    validation_cache = Path(args.validation_cache_dir).resolve()
    train_cache = Path(args.train_cache_dir).resolve()
    feature_dir = Path(args.feature_dir).resolve() if args.feature_dir else None
    device = _device(args.device)
    records = [_baseline_record(root), *_complete_records(root)]
    evaluations = []
    for index, record in enumerate(records):
        try:
            record_bootstrap = (
                args.bootstrap_samples
                if record["stage"] == "full"
                else min(args.bootstrap_samples, 1000)
            )
            output = evaluate_record(
                record,
                validation_cache=validation_cache,
                train_cache=train_cache,
                feature_dir=feature_dir,
                device=device,
                sample_pairs=args.sample_pairs,
                feature_sample_size=args.feature_sample_size,
                bootstrap_samples=record_bootstrap,
                # Use one fixed pair/sample seed for every checkpoint.  This
                # makes the invariant rows genuinely paired across E0-E4;
                # only the bootstrap draw count varies by stage.
                seed=args.seed,
                sequence_sample_pairs=args.sequence_sample_pairs,
            )
        except Exception as error:  # keep a single bad artifact from hiding all results
            output = {
                "experiment": record["experiment"],
                "stage": record["stage"],
                "seed": record["seed"],
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[ablation-eval] failed {record['experiment']} {record['stage']} seed{record['seed']}: {error}", flush=True)
        evaluations.append(output)
        target = record["seed_dir"] / "invariant_metrics.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[ablation-eval] {record['experiment']} {record['stage']} seed{record['seed']} -> {target}", flush=True)
    payload = {
        "format": "chunk-saes-ablation-invariant-evaluation-v1",
        "complete": True,
        "validation_cache": str(validation_cache),
        "feature_dir": str(feature_dir) if feature_dir else None,
        "device": str(device),
        "protocol": {
            "sample_pairs": args.sample_pairs,
            "feature_sample_size": args.feature_sample_size,
            "bootstrap_samples": args.bootstrap_samples,
            "non_full_bootstrap_samples": min(args.bootstrap_samples, 1000),
        },
        "evaluations": evaluations,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
