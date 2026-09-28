#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from chunk_saes.artifacts import (
    file_record,
    load_artifact_manifest,
    write_artifact_manifest,
)
from chunk_saes.evaluation_protocol import FIXED_CHUNK_REPRESENTATION_PROTOCOL
from chunk_saes.utils import atomic_json_dump


RESULT_FORMAT = "chunk-saes-high-level-feature-analysis-v1"
METHODS = ("token", "temporal", "mean", "cross")
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9']{2,}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute non-LLM cross-document abstraction statistics from matched "
            "top-activating feature evidence."
        )
    )
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--evidence-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260815)
    return p


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def _mean_pairwise_jaccard(token_sets: list[set[str]]) -> float:
    values = [
        _jaccard(token_sets[left], token_sets[right])
        for left in range(len(token_sets))
        for right in range(left + 1, len(token_sets))
    ]
    return float(np.mean(values)) if values else 0.0


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        count = min(512, samples - start)
        indices = rng.integers(
            0,
            values.size,
            size=(count, values.size),
        )
        means[start : start + count] = values[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        count = min(512, samples - start)
        left_indices = rng.integers(
            0,
            left.size,
            size=(count, left.size),
        )
        right_indices = rng.integers(
            0,
            right.size,
            size=(count, right.size),
        )
        differences[start : start + count] = (
            left[left_indices].mean(axis=1)
            - right[right_indices].mean(axis=1)
        )
    return {
        "point": float(left.mean() - right.mean()),
        "95ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
    }


def _load_chunks(root: Path) -> dict[int, dict]:
    chunks: dict[int, dict] = {}
    for path in sorted((root / "partials").glob("chunks-rank*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                chunks[int(row["global_chunk_id"])] = row
    if not chunks:
        raise ValueError(f"no evidence chunks found below {root / 'partials'}")
    return chunks


def _load_global_top_indices(root: Path, mode: str, top_n: int) -> torch.Tensor:
    rank_indices = []
    rank_scores = []
    for path in sorted((root / "partials").glob("candidates-rank*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            rank_indices.append(handle.get_tensor(f"{mode}_indices"))
            rank_scores.append(handle.get_tensor(f"{mode}_scores"))
    if not rank_indices:
        raise ValueError(f"no candidate tensors found for mode={mode}")
    indices = torch.cat(rank_indices, dim=1)
    scores = torch.cat(rank_scores, dim=1)
    top = scores.topk(min(top_n, scores.shape[1]), dim=1)
    return indices.gather(1, top.indices)


def _mode_rows(
    chunks: Mapping[int, Mapping[str, object]],
    indices: torch.Tensor,
) -> list[dict[str, float]]:
    result = []
    for row in indices.tolist():
        evidence = [chunks[int(chunk_id)] for chunk_id in row]
        documents = {
            str(item["doc_id"])
            for item in evidence
        }
        sources = {
            str(item["source"])
            for item in evidence
        }
        token_sets = [_tokens(str(item["text"])) for item in evidence]
        document_diversity = len(documents) / max(1, len(evidence))
        lexical_overlap = _mean_pairwise_jaccard(token_sets)
        result.append(
            {
                "document_diversity": float(document_diversity),
                "source_diversity": float(
                    len(sources) / max(1, len(evidence))
                ),
                "lexical_overlap": lexical_overlap,
                # A high score requires activation across documents without
                # merely repeating the same surface words.
                "cross_document_abstraction": float(
                    document_diversity * (1.0 - lexical_overlap)
                ),
            }
        )
    return result


def main() -> None:
    args = parser().parse_args()
    evidence_root = Path(args.evidence_dir).resolve()
    evidence_manifest = load_artifact_manifest(
        args.evidence_manifest,
        expected_format="chunk-saes-feature-evidence-v2",
        verify_files=True,
    )
    if Path(args.evidence_manifest).parent.resolve() != evidence_root:
        raise ValueError("--evidence-manifest must belong to --evidence-dir")
    representation_protocol = evidence_manifest.get("representation_protocol")
    if (
        not isinstance(representation_protocol, dict)
        or representation_protocol.get("name") != FIXED_CHUNK_REPRESENTATION_PROTOCOL
        or representation_protocol.get("token_temporal_aggregation")
        != "mean_after_threshold"
    ):
        raise ValueError(
            "high-level feature analysis requires mean-after-threshold evidence"
        )
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    chunks = _load_chunks(evidence_root)
    top_n = int(evidence_manifest["top_n"])
    modes = tuple(
        mode
        for mode in METHODS
        if mode in evidence_manifest["features_per_method"]
    )
    per_mode_rows = {
        mode: _mode_rows(
            chunks,
            _load_global_top_indices(evidence_root, mode, top_n),
        )
        for mode in modes
    }

    metric_names = (
        "document_diversity",
        "lexical_overlap",
        "cross_document_abstraction",
    )
    methods = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for mode_index, mode in enumerate(modes):
        arrays[mode] = {
            metric: np.asarray(
                [row[metric] for row in per_mode_rows[mode]],
                dtype=np.float64,
            )
            for metric in metric_names
        }
        methods[mode] = {
            "n_features": len(per_mode_rows[mode]),
            "feature_values": {
                metric: values.tolist()
                for metric, values in arrays[mode].items()
            },
            **{
                metric: {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "95ci": _bootstrap_mean_interval(
                        values,
                        samples=args.bootstrap_samples,
                        seed=args.seed + mode_index * 100 + metric_index,
                    ),
                }
                for metric_index, (metric, values) in enumerate(
                    arrays[mode].items()
                )
            },
        }

    comparisons = {}
    for reference_index, reference in enumerate(
        mode for mode in modes if mode != "cross"
    ):
        comparisons[f"cross_minus_{reference}"] = {
            metric: _bootstrap_difference(
                arrays["cross"][metric],
                arrays[reference][metric],
                samples=args.bootstrap_samples,
                seed=args.seed + 1_000 + reference_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metric_names)
        }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": RESULT_FORMAT,
        "complete": True,
        "metric_definition": {
            "document_diversity": (
                "unique documents among a feature's top activating chunks / top_n"
            ),
            "lexical_overlap": (
                "mean pairwise Jaccard overlap of lower-cased alphanumeric word "
                "sets among top activating chunks"
            ),
            "cross_document_abstraction": (
                "document_diversity * (1 - lexical_overlap)"
            ),
            "interpretation": (
                "Higher cross-document abstraction indicates a feature fires "
                "across more independent documents without depending on repeated "
                "surface words; no LLM labels are used."
            ),
        },
        "top_n": top_n,
        "methods": methods,
        "comparisons": comparisons,
        "identity": {
            "evidence_artifact_digest": evidence_manifest["artifact_digest"],
            "representation_protocol": representation_protocol,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
    }
    atomic_json_dump(payload, output)
    manifest_path = output.with_name("high_level_feature_manifest.json")
    write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": payload["identity"],
            "files": {
                "results": file_record(output, relative_to=output.parent),
            },
        },
        manifest_path,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
