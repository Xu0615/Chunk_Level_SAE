#!/usr/bin/env python
"""Evaluate high-level document linking under a strict lexical-overlap control.

This is intentionally a standalone downstream evaluation.  It consumes the
held-out Pile chunks already materialized by ``extract_feature_evidence.py``
and never changes the existing Eval 1--5 pipeline.

Task
----
For each held-out document, choose the pair of sampled chunks with the lowest
word-set Jaccard overlap.  Keep only pairs below a pre-registered overlap
ceiling.  Given one chunk, retrieve its partner from *different documents*
whose target chunks have exactly the same token length.  We evaluate both
directions.

The exact-length gallery and low-overlap pair construction remove two cheap
shortcuts:

* lexical copying cannot solve the task; and
* chunk length cannot identify the target.

The task is useful for semantic routing, RAG indexing, document clustering,
and duplicate/thread linking: success requires recognizing a document-level
topic or discourse thread after its vocabulary changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Wedge
from matplotlib.path import Path as MplPath
from safetensors import safe_open
from scipy import sparse

from chunk_saes.artifacts import (
    ensure_reusable_artifact,
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
    validate_full_dictionary_width,
)
from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.plot_style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_SHORT_LABELS,
    METHODS,
    style_figure_text,
)
from chunk_saes.sae import SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.utils import atomic_json_dump


RESULT_FORMAT = "chunk-saes-lexical-controlled-document-linking-v1"
FEATURE_FORMAT = "chunk-saes-document-linking-features-v1"
DISPLAY_NAMES = METHOD_LABELS
COLORS = METHOD_COLORS
RETRIEVAL_BENCHMARK_KEYS = ("word_jaccard", "raw_mean_hidden", *METHODS)
RETRIEVAL_BENCHMARK_NAMES = (
    "Word Jaccard",
    "Raw mean hidden",
    *(DISPLAY_NAMES[mode] for mode in METHODS),
)
RETRIEVAL_BENCHMARK_COLORS = (
    "#B9B9B9",
    "#A9826B",
    *(COLORS[mode] for mode in METHODS),
)
WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Pair:
    doc_id: str
    left_index: int
    right_index: int
    lexical_jaccard: float


class SparseEncoder:
    """Encoder-only view of one SAE checkpoint."""

    def __init__(self, checkpoint_dir: Path, device: torch.device) -> None:
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
            if "pre_bias" in names:
                self.pre_bias = handle.get_tensor("pre_bias").to(device)
            elif (
                config.get("sae_parameter_schema_version")
                == SAE_PARAMETER_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{checkpoint_dir} declares {SAE_PARAMETER_SCHEMA_VERSION} "
                    "but lacks pre_bias"
                )
            else:
                self.pre_bias = handle.get_tensor("decoder_bias").to(device)
            self.threshold = handle.get_tensor("threshold").to(device)
            self.scale = handle.get_tensor("activation_scale").to(device)
        self.dictionary_width = int(self.weight.shape[0])
        declared_width = int(config.get("dict_size", self.dictionary_width))
        if self.dictionary_width != declared_width:
            raise ValueError(
                f"{checkpoint_dir} encoder width {self.dictionary_width} does "
                f"not match declared dictionary width {declared_width}"
            )
        self.device = device

    @torch.inference_mode()
    def dense(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.device, dtype=self.weight.dtype)
        pre = F.relu(
            F.linear(
                hidden * self.scale.to(self.weight.dtype) - self.pre_bias,
                self.weight,
                self.bias,
            )
        )
        return pre * (pre > self.threshold.to(pre.dtype))

    @torch.inference_mode()
    def topk(self, hidden: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        dense = self.dense(hidden)
        values, indices = dense.topk(min(k, dense.shape[-1]), dim=-1)
        return indices, values

    def close(self) -> None:
        del self.weight, self.bias, self.pre_bias, self.threshold, self.scale


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Lexically controlled same-document partner retrieval using "
            "frozen BatchTopK/Temporal/Mean-Chunk/Cross-Chunk SAE features."
        )
    )
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--checkpoint-selection", choices=("best", "final"), default="best")
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--evidence-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lexical-jaccard-max", type=float, default=0.10)
    p.add_argument("--min-word-length", type=int, default=3)
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--permutation-samples", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--example-count", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--analysis-only",
        action="store_true",
        help="Reuse verified extracted features and rerun only analysis/figures.",
    )
    return p


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _word_set(text: str, *, min_length: int) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) >= min_length
    )


def lexical_jaccard(left: str, right: str, *, min_length: int = 3) -> float:
    a = _word_set(left, min_length=min_length)
    b = _word_set(right, min_length=min_length)
    return len(a & b) / max(1, len(a | b))


def _load_chunks(evidence_dir: Path, evidence_manifest: dict) -> list[dict[str, Any]]:
    chunks: dict[int, dict[str, Any]] = {}
    files = evidence_manifest.get("files") or {}
    partial_records = [
        record
        for key, record in sorted(files.items())
        if key.startswith("partial_")
        and str(record.get("path", "")).endswith(".jsonl")
    ]
    if not partial_records:
        paths = sorted((evidence_dir / "partials").glob("chunks-rank*.jsonl"))
    else:
        paths = [evidence_dir / str(record["path"]) for record in partial_records]
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                global_id = int(row["global_chunk_id"])
                if global_id in chunks:
                    raise ValueError(f"duplicate global_chunk_id={global_id}")
                chunks[global_id] = row
    rows = [chunks[key] for key in sorted(chunks)]
    expected = int(evidence_manifest["evidence_chunks"])
    if len(rows) != expected:
        raise ValueError(f"loaded {len(rows)} evidence chunks, expected {expected}")
    for row in rows:
        input_ids = row.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) != int(row["length"]):
            raise ValueError(
                "evidence chunks must contain the original fixed-length input_ids"
            )
    candidate_pool = evidence_manifest.get("candidate_pool") or {}
    expected_digest = candidate_pool.get("global_chunk_ids_digest")
    actual_digest = json_digest([int(row["global_chunk_id"]) for row in rows])
    if expected_digest is not None and expected_digest != actual_digest:
        raise ValueError("evidence candidate-pool chunk ID digest does not match")
    return rows


def build_pairs(
    chunks: list[dict[str, Any]],
    *,
    lexical_jaccard_max: float,
    min_word_length: int,
) -> list[Pair]:
    """Choose one most lexically disjoint non-identical pair per document."""

    if not 0.0 <= lexical_jaccard_max <= 1.0:
        raise ValueError("--lexical-jaccard-max must be in [0, 1]")
    if min_word_length <= 0:
        raise ValueError("--min-word-length must be positive")
    by_document: dict[str, list[int]] = defaultdict(list)
    token_sets = []
    for index, row in enumerate(chunks):
        by_document[str(row["doc_id"])].append(index)
        token_sets.append(
            _word_set(str(row["text"]), min_length=min_word_length)
        )
    pairs: list[Pair] = []
    for doc_id, indices in sorted(by_document.items()):
        best: tuple[tuple[Any, ...], int, int, float] | None = None
        for position, left_index in enumerate(indices):
            for right_index in indices[position + 1 :]:
                left = chunks[left_index]
                right = chunks[right_index]
                if left["text"] == right["text"]:
                    continue
                overlap = len(
                    token_sets[left_index] & token_sets[right_index]
                ) / max(
                    1,
                    len(token_sets[left_index] | token_sets[right_index]),
                )
                key = (
                    overlap,
                    -min(int(left["length"]), int(right["length"])),
                    -max(int(left["length"]), int(right["length"])),
                    int(left["global_chunk_id"]),
                    int(right["global_chunk_id"]),
                )
                if best is None or key < best[0]:
                    best = (key, left_index, right_index, overlap)
        if best is None or best[3] > lexical_jaccard_max:
            continue
        left_index, right_index = best[1], best[2]
        # Deterministic orientation prevents a consistent shorter/earlier
        # chunk from always serving as the query.  The evaluation nevertheless
        # runs both directions, so orientation cannot change aggregate scores.
        if int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) & 1:
            left_index, right_index = right_index, left_index
        pairs.append(
            Pair(
                doc_id=doc_id,
                left_index=left_index,
                right_index=right_index,
                lexical_jaccard=float(best[3]),
            )
        )
    if not pairs:
        raise ValueError("no documents passed the lexical-overlap control")
    return pairs


def filter_pairs_for_gallery_support(
    pairs: list[Pair],
    chunks: list[dict[str, Any]],
    *,
    retokenized_length_by_chunk: dict[int, int],
    min_gallery_size: int = 2,
) -> tuple[list[Pair], dict[str, Any]]:
    """Remove rare target-length cells until both directions are valid.

    Decoding and re-tokenizing a chunk is almost always length preserving, but
    byte-level normalization can occasionally change one token.  Such singleton
    lengths would create a trivial one-item gallery.  We therefore iteratively
    retain only pairs whose left and right retokenized target lengths each have
    at least ``min_gallery_size`` documents.
    """

    if min_gallery_size < 2:
        raise ValueError("min_gallery_size must be at least 2")
    kept = list(pairs)
    original = len(kept)
    while True:
        left_counts = Counter(
            retokenized_length_by_chunk[pair.left_index] for pair in kept
        )
        right_counts = Counter(
            retokenized_length_by_chunk[pair.right_index] for pair in kept
        )
        filtered = [
            pair
            for pair in kept
            if left_counts[retokenized_length_by_chunk[pair.left_index]]
            >= min_gallery_size
            and right_counts[retokenized_length_by_chunk[pair.right_index]]
            >= min_gallery_size
        ]
        if len(filtered) == len(kept):
            break
        kept = filtered
    if not kept:
        raise ValueError("gallery support filtering removed every pair")
    return kept, {
        "minimum_gallery_size": min_gallery_size,
        "pairs_before": original,
        "pairs_after": len(kept),
        "pairs_removed": original - len(kept),
    }


def _feature_identity(
    *,
    args: argparse.Namespace,
    evidence_manifest: dict,
    sae_set: dict,
    pair_digest: str,
    unique_chunk_digest: str,
    chunk_lengths: list[int],
) -> dict[str, Any]:
    feature_widths = full_dictionary_feature_widths(sae_set)
    representation_descriptions = {
        "token": "mean of thresholded per-token SAE activations, then top-k",
        "temporal": (
            "mean of thresholded per-token Temporal SAE activations over "
            "the complete dictionary, then top-k"
        ),
        "mean": "thresholded SAE activation of chunk-mean hidden state, then top-k",
        "cross": "thresholded SAE activation of chunk-mean hidden state, then top-k",
    }
    return {
        "evidence_artifact_digest": evidence_manifest["artifact_digest"],
        "sae_set_digest": sae_set["artifact_digest"],
        "checkpoint_selection": args.checkpoint_selection,
        "model": str(Path(args.model).resolve()),
        "layer": args.layer,
        "lexical_jaccard_max": args.lexical_jaccard_max,
        "min_word_length": args.min_word_length,
        "top_k": args.top_k,
        "max_length": args.max_length,
        "feature_widths": feature_widths,
        "representation": {
            mode: representation_descriptions[mode]
            for mode in feature_widths
        },
        "representation_protocol": fixed_chunk_protocol_metadata(
            feature_widths=feature_widths,
            chunk_lengths=chunk_lengths,
        ),
        "pair_digest": pair_digest,
        "unique_chunk_digest": unique_chunk_digest,
    }


def _pair_payload(
    chunks: list[dict[str, Any]],
    pairs: list[Pair],
) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": pair.doc_id,
            "left_global_chunk_id": int(
                chunks[pair.left_index]["global_chunk_id"]
            ),
            "right_global_chunk_id": int(
                chunks[pair.right_index]["global_chunk_id"]
            ),
            "left_length": int(chunks[pair.left_index]["length"]),
            "right_length": int(chunks[pair.right_index]["length"]),
            "lexical_jaccard": pair.lexical_jaccard,
        }
        for pair in pairs
    ]


def _tokenize_selected_chunks(
    extractor: TargetLayerExtractor,
    chunks: list[dict[str, Any]],
    unique_indices: list[int],
    *,
    max_length: int,
) -> tuple[list[list[int]], np.ndarray]:
    sequences: list[list[int]] = []
    retokenized_lengths = []
    for chunk_index in unique_indices:
        row = chunks[chunk_index]
        stored_ids = row.get("input_ids")
        if stored_ids is not None:
            ids = [int(value) for value in stored_ids]
            declared_length = int(row["length"])
            if len(ids) != declared_length:
                raise ValueError(
                    "stored input_ids do not match declared fixed length for "
                    f"global_chunk_id={row['global_chunk_id']}: "
                    f"{len(ids)} != {declared_length}"
                )
            if len(ids) > max_length:
                raise ValueError(
                    f"fixed chunk length {len(ids)} exceeds --max-length="
                    f"{max_length}; increase --max-length without truncating"
                )
        else:
            ids = extractor.tokenizer(
                str(row["text"]),
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            ).input_ids
        if not ids:
            raise ValueError(
                f"retokenized chunk is empty: global_chunk_id={row['global_chunk_id']}"
            )
        sequences.append(ids)
        retokenized_lengths.append(len(ids))
    return sequences, np.asarray(retokenized_lengths, dtype=np.int16)


@torch.inference_mode()
def _extract_features(
    *,
    args: argparse.Namespace,
    chunks: list[dict[str, Any]],
    unique_indices: list[int],
    sae_set: dict,
    output_dir: Path,
    feature_identity: dict[str, Any],
) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        str(device),
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
    )
    modes = tuple(sae_set["modes"])
    encoders = {
        mode: SparseEncoder(
            Path(sae_set["modes"][mode]["checkpoint_path"]),
            device,
        )
        for mode in modes
    }
    for mode, encoder in encoders.items():
        validate_full_dictionary_width(
            mode,
            encoder.dictionary_width,
            int(feature_identity["feature_widths"][mode]),
        )
    try:
        sequences, retokenized_lengths = _tokenize_selected_chunks(
            extractor,
            chunks,
            unique_indices,
            max_length=args.max_length,
        )
        n = len(unique_indices)
        feature_indices = {
            mode: np.full((n, args.top_k), -1, dtype=np.int32)
            for mode in modes
        }
        feature_values = {
            mode: np.zeros((n, args.top_k), dtype=np.float16)
            for mode in modes
        }
        feature_nnz = {
            mode: np.zeros(n, dtype=np.int16)
            for mode in modes
        }
        raw_means = np.empty(
            (n, int(sae_set["common"]["activation_dim"])),
            dtype=np.float16,
        )

        for start in range(0, n, args.batch_size):
            stop = min(n, start + args.batch_size)
            layer_batch = extractor.forward_ids(sequences[start:stop])
            means = layer_batch.means()
            raw_means[start:stop] = means.float().cpu().numpy().astype(
                np.float16
            )
            for mode in (
                mode for mode in ("mean", "cross") if mode in encoders
            ):
                indices, values = encoders[mode].topk(means, args.top_k)
                take = indices.shape[1]
                valid = values > 0
                feature_indices[mode][start:stop, :take] = (
                    indices.cpu().numpy().astype(np.int32)
                )
                feature_values[mode][start:stop, :take] = (
                    values.float().cpu().numpy().astype(np.float16)
                )
                feature_nnz[mode][start:stop] = (
                    valid.sum(1).cpu().numpy().astype(np.int16)
                )
                feature_indices[mode][start:stop, :take][
                    ~valid.cpu().numpy()
                ] = -1

            token_encoder = encoders["token"]
            temporal_encoder = encoders.get("temporal")
            for row_index, ids in enumerate(sequences[start:stop]):
                hidden = layer_batch.hidden[row_index, : len(ids)]
                aggregate = mean_after_threshold(token_encoder.dense(hidden))
                values, indices = aggregate.topk(args.top_k)
                valid = values > 0
                target = start + row_index
                feature_indices["token"][target] = (
                    indices.cpu().numpy().astype(np.int32)
                )
                feature_values["token"][target] = (
                    values.cpu().numpy().astype(np.float16)
                )
                feature_nnz["token"][target] = int(valid.sum().item())
                feature_indices["token"][target][~valid.cpu().numpy()] = -1
                if temporal_encoder is not None:
                    temporal_aggregate = mean_after_threshold(
                        temporal_encoder.dense(hidden)
                    )
                    temporal_values, temporal_indices = temporal_aggregate.topk(
                        min(args.top_k, temporal_aggregate.shape[0])
                    )
                    temporal_valid = temporal_values > 0
                    take = temporal_indices.numel()
                    feature_indices["temporal"][target, :take] = (
                        temporal_indices.cpu().numpy().astype(np.int32)
                    )
                    feature_values["temporal"][target, :take] = (
                        temporal_values.cpu().numpy().astype(np.float16)
                    )
                    feature_nnz["temporal"][target] = int(
                        temporal_valid.sum().item()
                    )
                    feature_indices["temporal"][target, :take][
                        ~temporal_valid.cpu().numpy()
                    ] = -1
            del layer_batch, means
            if stop % 256 == 0 or stop == n:
                print(f"[document-linking] extracted {stop}/{n} chunks", flush=True)

        feature_path = output_dir / "features.npz"
        np.savez_compressed(
            feature_path,
            global_chunk_ids=np.asarray(
                [
                    int(chunks[index]["global_chunk_id"])
                    for index in unique_indices
                ],
                dtype=np.int64,
            ),
            lengths=np.asarray(
                [int(chunks[index]["length"]) for index in unique_indices],
                dtype=np.int16,
            ),
            retokenized_lengths=retokenized_lengths,
            raw_mean_hidden=raw_means,
            **{
                f"{mode}_indices": feature_indices[mode]
                for mode in modes
            },
            **{
                f"{mode}_values": feature_values[mode]
                for mode in modes
            },
            **{
                f"{mode}_nnz": feature_nnz[mode]
                for mode in modes
            },
        )
        feature_manifest = write_artifact_manifest(
            {
                "format": FEATURE_FORMAT,
                "complete": True,
                "identity": feature_identity,
                "chunks": n,
                "files": {
                    "features": file_record(
                        feature_path,
                        relative_to=output_dir,
                    )
                },
            },
            output_dir / "feature_manifest.json",
        )
        return feature_manifest
    finally:
        extractor.close()
        for encoder in encoders.values():
            encoder.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _csr_from_fixed_topk(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    width: int,
) -> sparse.csr_matrix:
    valid = indices >= 0
    rows, positions = np.nonzero(valid)
    matrix = sparse.csr_matrix(
        (
            values[rows, positions].astype(np.float32),
            (rows, indices[rows, positions].astype(np.int64)),
        ),
        shape=(indices.shape[0], width),
        dtype=np.float32,
    )
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    return matrix.multiply(1.0 / np.maximum(norms, 1e-12)[:, None]).tocsr()


def _row_normalize_dense(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _tie_aware_ranks(scores: np.ndarray) -> np.ndarray:
    target = np.diag(scores)
    greater = (scores > target[:, None]).sum(axis=1)
    equal_other = (scores == target[:, None]).sum(axis=1) - 1
    return 1.0 + greater + 0.5 * np.maximum(equal_other, 0)


def retrieval_ranks(
    representation: np.ndarray | sparse.csr_matrix,
    *,
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    gallery_lengths: np.ndarray,
) -> np.ndarray:
    """Return exact-length-controlled ranks for paired query/gallery rows."""

    if not (
        len(query_indices) == len(gallery_indices) == len(gallery_lengths)
    ):
        raise ValueError("query/gallery arrays must have equal length")
    ranks = np.empty(len(query_indices), dtype=np.float64)
    for length in sorted(set(gallery_lengths.tolist())):
        selected = np.flatnonzero(gallery_lengths == length)
        if selected.size < 2:
            raise ValueError(
                f"target length={length} has fewer than two gallery items"
            )
        if sparse.issparse(representation):
            scores = (
                representation[query_indices[selected]]
                @ representation[gallery_indices[selected]].T
            ).toarray()
        else:
            scores = (
                representation[query_indices[selected]]
                @ representation[gallery_indices[selected]].T
            )
        ranks[selected] = _tie_aware_ranks(np.asarray(scores))
    return ranks


def _metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "recall_at_1": float(np.mean(ranks <= 1.0)),
        "recall_at_5": float(np.mean(ranks <= 5.0)),
        "recall_at_10": float(np.mean(ranks <= 10.0)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def _bootstrap_method(
    ranks: np.ndarray,
    pair_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique_pairs = np.unique(pair_ids)
    reciprocal = 1.0 / ranks
    r1 = (ranks <= 1).astype(np.float64)
    estimates_mrr = np.empty(samples, dtype=np.float64)
    estimates_r1 = np.empty(samples, dtype=np.float64)
    pair_rows = [np.flatnonzero(pair_ids == pair) for pair in unique_pairs]
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        draws = rng.integers(
            0,
            len(unique_pairs),
            size=(count, len(unique_pairs)),
        )
        for local, sampled_pairs in enumerate(draws):
            rows = np.concatenate([pair_rows[index] for index in sampled_pairs])
            estimates_mrr[start + local] = reciprocal[rows].mean()
            estimates_r1[start + local] = r1[rows].mean()
    return {
        "mrr_95ci": [
            float(np.quantile(estimates_mrr, 0.025)),
            float(np.quantile(estimates_mrr, 0.975)),
        ],
        "recall_at_1_95ci": [
            float(np.quantile(estimates_r1, 0.025)),
            float(np.quantile(estimates_r1, 0.975)),
        ],
    }


def _paired_bootstrap_difference(
    left_ranks: np.ndarray,
    right_ranks: np.ndarray,
    pair_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_pairs = np.unique(pair_ids)
    pair_rows = [np.flatnonzero(pair_ids == pair) for pair in unique_pairs]
    mrr_delta = 1.0 / left_ranks - 1.0 / right_ranks
    r1_delta = (left_ranks <= 1).astype(float) - (
        right_ranks <= 1
    ).astype(float)
    estimates_mrr = np.empty(samples, dtype=np.float64)
    estimates_r1 = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        draws = rng.integers(
            0,
            len(unique_pairs),
            size=(count, len(unique_pairs)),
        )
        for local, sampled_pairs in enumerate(draws):
            rows = np.concatenate([pair_rows[index] for index in sampled_pairs])
            estimates_mrr[start + local] = mrr_delta[rows].mean()
            estimates_r1[start + local] = r1_delta[rows].mean()
    return {
        "mrr": {
            "point": float(mrr_delta.mean()),
            "95ci": [
                float(np.quantile(estimates_mrr, 0.025)),
                float(np.quantile(estimates_mrr, 0.975)),
            ],
        },
        "recall_at_1": {
            "point": float(r1_delta.mean()),
            "95ci": [
                float(np.quantile(estimates_r1, 0.025)),
                float(np.quantile(estimates_r1, 0.975)),
            ],
        },
        "direction_level_win_rate": float(np.mean(mrr_delta > 0)),
        "direction_level_loss_rate": float(np.mean(mrr_delta < 0)),
        "direction_level_tie_rate": float(np.mean(mrr_delta == 0)),
    }


def _paired_sign_flip_test(
    left_ranks: np.ndarray,
    right_ranks: np.ndarray,
    pair_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    """Two-sided paired randomization test, clustered by document pair."""

    if samples <= 0:
        raise ValueError("--permutation-samples must be positive")
    unique_pairs = np.unique(pair_ids)
    delta = 1.0 / left_ranks - 1.0 / right_ranks
    clustered = np.asarray(
        [delta[pair_ids == pair].mean() for pair in unique_pairs],
        dtype=np.float64,
    )
    observed = abs(float(clustered.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    for start in range(0, samples, 1024):
        count = min(1024, samples - start)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(count, clustered.size),
        )
        estimates = np.abs((signs * clustered).mean(axis=1))
        exceed += int(np.sum(estimates >= observed))
        completed += count
    return float((exceed + 1) / (completed + 1))


def _chance_metrics(gallery_lengths: np.ndarray) -> dict[str, float]:
    sizes = Counter(gallery_lengths.tolist())
    n = len(gallery_lengths)
    r1 = sum(count * (1.0 / count) for count in sizes.values()) / n
    r5 = sum(count * min(1.0, 5.0 / count) for count in sizes.values()) / n
    mrr = (
        sum(
            count
            * (sum(1.0 / rank for rank in range(1, count + 1)) / count)
            for count in sizes.values()
        )
        / n
    )
    return {
        "recall_at_1": float(r1),
        "recall_at_5": float(r5),
        "mrr": float(mrr),
        "gallery_sizes_by_target_length": {
            str(key): int(value) for key, value in sorted(sizes.items())
        },
    }


def _lexical_baseline_ranks(
    chunks: list[dict[str, Any]],
    pairs: list[Pair],
    *,
    min_word_length: int,
    left_lengths: np.ndarray | None = None,
    right_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    token_sets = [
        _word_set(str(row["text"]), min_length=min_word_length)
        for row in chunks
    ]
    ranks = []
    pair_ids = []
    for reverse in (False, True):
        query = [
            pair.right_index if reverse else pair.left_index
            for pair in pairs
        ]
        gallery = [
            pair.left_index if reverse else pair.right_index
            for pair in pairs
        ]
        if left_lengths is None or right_lengths is None:
            target_lengths = np.asarray(
                [int(chunks[index]["length"]) for index in gallery],
                dtype=np.int16,
            )
        else:
            target_lengths = left_lengths if reverse else right_lengths
        current = np.empty(len(pairs), dtype=np.float64)
        for length in sorted(set(target_lengths.tolist())):
            selected = np.flatnonzero(target_lengths == length)
            scores = np.empty((selected.size, selected.size), dtype=np.float64)
            for row_position, pair_position in enumerate(selected):
                q = token_sets[query[pair_position]]
                for column_position, candidate_position in enumerate(selected):
                    g = token_sets[gallery[candidate_position]]
                    scores[row_position, column_position] = len(q & g) / max(
                        1, len(q | g)
                    )
            current[selected] = _tie_aware_ranks(scores)
        ranks.append(current)
        pair_ids.append(np.arange(len(pairs), dtype=np.int32))
    return np.concatenate(ranks), np.concatenate(pair_ids)


def _select_examples(
    chunks: list[dict[str, Any]],
    pairs: list[Pair],
    *,
    cross_ranks: np.ndarray,
    baseline_ranks: dict[str, np.ndarray],
    count: int,
) -> list[dict[str, Any]]:
    # Use forward direction for readable examples; rank by Cross's reciprocal
    # rank gain over the best baseline, then lexical disjointness.
    n = len(pairs)
    best_baseline = np.maximum.reduce(
        [
            1.0 / ranks[:n]
            for ranks in baseline_ranks.values()
        ]
    )
    gains = 1.0 / cross_ranks[:n] - best_baseline
    order = sorted(
        range(n),
        key=lambda index: (
            -gains[index],
            pairs[index].lexical_jaccard,
            pairs[index].doc_id,
        ),
    )
    examples = []
    for index in order:
        if gains[index] <= 0:
            continue
        pair = pairs[index]
        examples.append(
            {
                "doc_id": pair.doc_id,
                "lexical_jaccard": pair.lexical_jaccard,
                "query": str(chunks[pair.left_index]["text"]),
                "correct_partner": str(chunks[pair.right_index]["text"]),
                "query_length": int(chunks[pair.left_index]["length"]),
                "partner_length": int(chunks[pair.right_index]["length"]),
                "ranks": {
                    **{
                        mode: float(ranks[index])
                        for mode, ranks in baseline_ranks.items()
                    },
                    "cross": float(cross_ranks[index]),
                },
                "cross_reciprocal_rank_gain_over_best_baseline": float(
                    gains[index]
                ),
            }
        )
        if len(examples) >= count:
            break
    return examples


def _metric_value(
    results: dict[str, Any],
    key: str,
    metric: str,
) -> float:
    if key in METHODS:
        return float(results["methods"][key][metric])
    return float(results["controls"][key][metric])


def _choose_primary_recall(results: dict[str, Any]) -> dict[str, Any]:
    """Select R@1 or R@5 by the largest Cross-Chunk-vs-Token-SAE gap.

    BatchTopK SAE is the central baseline in the research question.  The selected
    metric is used only for the compact overview panel; the report continues
    to expose both R@1 and R@5.
    """

    candidates = []
    for metric, label in (("recall_at_1", "Recall@1"), ("recall_at_5", "Recall@5")):
        cross = _metric_value(results, "cross", metric)
        token = _metric_value(results, "token", metric)
        candidates.append(
            {
                "metric": metric,
                "label": label,
                "cross": cross,
                "token": token,
                "gap": cross - token,
            }
        )
    return max(candidates, key=lambda row: (row["gap"], row["metric"] == "recall_at_1"))


def _blend(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    return tuple(rgb * (1.0 - amount) + amount)


def _truncate(text: str, width: int) -> str:
    return textwrap.shorten(
        " ".join(str(text).split()),
        width=width,
        placeholder="…",
    )


def _feature_vector(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
    row: int,
) -> tuple[dict[int, float], float]:
    indices = arrays[f"{mode}_indices"][row]
    values = arrays[f"{mode}_values"][row].astype(np.float64)
    result = {
        int(index): float(value)
        for index, value in zip(indices, values, strict=True)
        if index >= 0 and value > 0
    }
    norm = math.sqrt(sum(value * value for value in result.values()))
    return result, max(norm, 1e-12)


def _shared_feature_profile(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
    query_row: int,
    target_row: int,
    gallery_rows: np.ndarray,
    top_n: int = 8,
) -> dict[str, Any]:
    query, query_norm = _feature_vector(arrays, mode=mode, row=query_row)
    target, target_norm = _feature_vector(arrays, mode=mode, row=target_row)
    shared = []
    gallery_indices = arrays[f"{mode}_indices"][gallery_rows]
    for feature_id in query.keys() & target.keys():
        contribution = (
            query[feature_id]
            * target[feature_id]
            / (query_norm * target_norm)
        )
        gallery_frequency = float(
            np.mean(np.any(gallery_indices == feature_id, axis=1))
        )
        shared.append(
            {
                "feature_id": feature_id,
                "contribution": contribution,
                "query_activation": query[feature_id] / query_norm,
                "target_activation": target[feature_id] / target_norm,
                "gallery_frequency": gallery_frequency,
            }
        )
    shared.sort(key=lambda row: row["contribution"], reverse=True)
    cosine = float(sum(row["contribution"] for row in shared))
    if cosine > 0:
        weighted_gallery_frequency = float(
            sum(
                row["contribution"] * row["gallery_frequency"]
                for row in shared
            )
            / cosine
        )
        top_share = float(
            sum(row["contribution"] for row in shared[:3]) / cosine
        )
    else:
        weighted_gallery_frequency = 0.0
        top_share = 0.0
    return {
        "mode": mode,
        "query_nnz": len(query),
        "target_nnz": len(target),
        "shared_count": len(shared),
        "cosine": cosine,
        "weighted_gallery_frequency": weighted_gallery_frequency,
        "top3_similarity_share": top_share,
        "features": shared[:top_n],
    }


def _prepare_feature_story(
    results: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    feature_path = output_dir / "features.npz"
    pair_path = output_dir / "selected_pairs.jsonl"
    if not feature_path.is_file() or not pair_path.is_file():
        return None
    with np.load(feature_path) as data:
        arrays = {key: data[key] for key in data.files}
    pairs = [
        json.loads(line)
        for line in pair_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pair_by_doc = {str(row["doc_id"]): row for row in pairs}
    row_by_global = {
        int(global_id): index
        for index, global_id in enumerate(arrays["global_chunk_ids"])
    }
    length_by_global = {
        int(global_id): int(length)
        for global_id, length in zip(
            arrays["global_chunk_ids"],
            arrays["retokenized_lengths"],
            strict=True,
        )
    }
    examples = [
        row
        for row in results.get("examples", [])
        if str(row.get("doc_id")) in pair_by_doc
    ]
    if not examples:
        return None
    example = max(
        examples,
        key=lambda row: (
            float(row["ranks"]["token"])
            + float(row["ranks"]["mean"])
            - 2.0 * float(row["ranks"]["cross"]),
            -float(row["lexical_jaccard"]),
        ),
    )
    pair = pair_by_doc[str(example["doc_id"])]
    query_global = int(pair["left_global_chunk_id"])
    target_global = int(pair["right_global_chunk_id"])
    query_row = row_by_global[query_global]
    target_row = row_by_global[target_global]
    target_length = length_by_global[target_global]
    gallery_rows = np.asarray(
        [
            row_by_global[int(row["right_global_chunk_id"])]
            for row in pairs
            if length_by_global[int(row["right_global_chunk_id"])]
            == target_length
        ],
        dtype=np.int64,
    )
    story_modes = tuple(
        mode
        for mode in METHODS
        if f"{mode}_indices" in arrays
        and mode in example.get("ranks", {})
    )
    profiles = {
        mode: _shared_feature_profile(
            arrays,
            mode=mode,
            query_row=query_row,
            target_row=target_row,
            gallery_rows=gallery_rows,
        )
        for mode in story_modes
    }
    return {
        "doc_id": str(example["doc_id"]),
        "query": str(example["query"]),
        "target": str(example["correct_partner"]),
        "lexical_jaccard": float(example["lexical_jaccard"]),
        "gallery_size": int(gallery_rows.size),
        "ranks": {
            mode: float(example["ranks"][mode]) for mode in story_modes
        },
        "profiles": profiles,
        "modes": list(story_modes),
    }


def _draw_semicircle(
    ax,
    center: tuple[float, float],
    radius: float,
    *,
    color: str,
    side: str,
    alpha: float = 1.0,
) -> None:
    if side == "left":
        theta1, theta2 = 90, 270
    elif side == "right":
        theta1, theta2 = -90, 90
    else:
        raise ValueError("side must be left or right")
    ax.add_patch(
        Wedge(
            center,
            radius,
            theta1,
            theta2,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=alpha,
        )
    )


def _draw_bezier(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    bend: float = 0.24,
    zorder: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    vertices = [
        (x0, y0),
        (x0 + bend * dx, y0),
        (x1 - bend * dx, y1),
        (x1, y1),
    ]
    path = MplPath(
        vertices,
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            capstyle="round",
            zorder=zorder,
        )
    )


def _plot_feature_bridge(ax, story: dict[str, Any]) -> None:
    """Alluvial sparse-feature bridge for one real low-overlap pair."""

    profile = story["profiles"]["cross"]
    features = profile["features"][:5]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "D. How Cross-Chunk SAE features bridge two lexically disjoint chunks",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    query_box = FancyBboxPatch(
        (0.015, 0.68),
        0.255,
        0.22,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        facecolor="#F6F6F6",
        edgecolor="#D5D5D5",
        linewidth=1.1,
        zorder=0,
    )
    target_box = FancyBboxPatch(
        (0.73, 0.68),
        0.255,
        0.22,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        facecolor="#FFF5F4",
        edgecolor=_blend(COLORS["cross"], 0.45),
        linewidth=1.1,
        zorder=0,
    )
    ax.add_patch(query_box)
    ax.add_patch(target_box)
    ax.text(
        0.03,
        0.872,
        "QUERY CHUNK",
        color="#555555",
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.76,
        0.872,
        "CORRECT PARTNER",
        color=COLORS["cross"],
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.03,
        0.83,
        textwrap.fill(_truncate(story["query"], 126), width=37),
        fontsize=8.6,
        va="top",
        linespacing=1.25,
    )
    ax.text(
        0.76,
        0.83,
        textwrap.fill(_truncate(story["target"], 126), width=37),
        fontsize=8.6,
        va="top",
        linespacing=1.25,
    )
    ax.text(
        0.5,
        0.935,
        (
            f"word-set Jaccard = {story['lexical_jaccard']:.3f}"
            f"   •   {story['gallery_size']} exact-length candidates"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    if not features:
        ax.text(0.5, 0.45, "No shared active features", ha="center")
        return
    max_contribution = max(row["contribution"] for row in features)
    y_positions = np.linspace(0.57, 0.25, len(features))
    for index, (row, y) in enumerate(zip(features, y_positions, strict=True)):
        strength = row["contribution"] / max(max_contribution, 1e-12)
        frequency = row["gallery_frequency"]
        radius = 0.018 + 0.014 * math.sqrt(strength)
        left = (0.315, float(y))
        right = (0.685, float(y))
        center = (0.5, float(y))
        line_color = mcolors.to_hex(
            np.asarray(mcolors.to_rgb(COLORS["cross"])) * (0.60 + 0.40 * strength)
        )
        _draw_bezier(
            ax,
            left,
            center,
            color=line_color,
            linewidth=0.9 + 7.0 * strength,
            alpha=0.22 + 0.62 * strength,
            bend=0.36,
        )
        _draw_bezier(
            ax,
            center,
            right,
            color=line_color,
            linewidth=0.9 + 7.0 * strength,
            alpha=0.22 + 0.62 * strength,
            bend=0.36,
        )
        _draw_semicircle(
            ax,
            center,
            radius,
            color="#606060",
            side="left",
        )
        _draw_semicircle(
            ax,
            center,
            radius,
            color=COLORS["cross"],
            side="right",
        )
        ax.add_patch(
            Circle(
                center,
                radius,
                fill=False,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        )
        ax.text(
            0.5,
            y - radius - 0.010,
            f"f{row['feature_id']}",
            ha="center",
            va="top",
            fontsize=7.4,
            color="#666666",
        )
        ax.text(
            0.272,
            y,
            f"{row['query_activation']:.2f}",
            ha="right",
            va="center",
            fontsize=7.5,
            color="#666666",
        )
        ax.text(
            0.728,
            y,
            f"{row['target_activation']:.2f}",
            ha="left",
            va="center",
            fontsize=7.5,
            color=COLORS["cross"],
        )
        ax.text(
            0.53,
            y + radius + 0.002,
            f"seen in {100 * frequency:.1f}% of gallery",
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#777777",
        )
    ax.text(
        0.5,
        0.105,
        (
            f"{profile['shared_count']} shared Cross-Chunk SAE features explain cosine "
            f"{profile['cosine']:.3f}; top 3 carry "
            f"{100 * profile['top3_similarity_share']:.0f}% of the link"
        ),
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        0.5,
        0.070,
        "Edge width = feature contribution to cosine similarity",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#777777",
    )
    modes = tuple(story.get("modes") or story["profiles"])
    rank_x = np.linspace(0.22, 0.78, len(modes))
    for x, mode in zip(rank_x, modes, strict=True):
        rank = story["ranks"][mode]
        selected = mode == "cross"
        ax.text(
            x,
            0.025,
            f"{METHOD_SHORT_LABELS[mode]}  r={rank:.0f}",
            ha="center",
            va="center",
            fontsize=8.0,
            color="white" if selected else "#252932",
            fontweight="bold" if selected else "normal",
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": COLORS[mode] if selected else _blend(COLORS[mode], 0.90),
                "edgecolor": COLORS[mode],
                "linewidth": 1.0,
            },
        )


def _plot_sparse_fingerprint(ax, story: dict[str, Any]) -> None:
    """Concentration/selectivity view of the feature mechanism."""

    modes = list(story.get("modes") or story["profiles"])
    offsets = {
        "token": (-12, 14),
        "temporal": (-16, -24),
        "mean": (14, -22),
        "cross": (14, 13),
    }
    ax.set_title(
        "E. Pair fingerprint — concentrated, gallery-selective overlap",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    for mode in modes:
        profile = story["profiles"][mode]
        concentration = min(1.0, profile["top3_similarity_share"])
        selectivity = 1.0 - min(
            1.0,
            profile["weighted_gallery_frequency"],
        )
        cosine = max(0.0, profile["cosine"])
        size = 180 + 900 * cosine
        ax.scatter(
            [selectivity],
            [concentration],
            s=size,
            color=COLORS[mode],
            alpha=0.92,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.text(
            selectivity,
            concentration,
            f"{cosine:.2f}",
            ha="center",
            va="center",
            fontsize=8.2,
            color=(
                "#252932"
                if mode in {"token", "mean"}
                else "white"
            ),
            fontweight="bold",
            zorder=4,
        )
        ax.annotate(
            METHOD_SHORT_LABELS[mode],
            xy=(selectivity, concentration),
            xytext=offsets[mode],
            textcoords="offset points",
            ha="left" if offsets[mode][0] > 0 else "right",
            va="bottom" if offsets[mode][1] > 0 else "top",
            fontsize=8.5,
            fontweight="bold" if mode == "cross" else "normal",
            color="#252932",
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS[mode],
                "lw": 0.8,
            },
        )
    ax.set_xlim(0.15, 0.90)
    ax.set_ylim(0.35, 0.95)
    ax.set_xlabel("Gallery selectivity  (1 − activation frequency)  →")
    ax.set_ylabel("Top-3 similarity share  →")
    ax.grid(color="#DADDE3", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plot_results(results: dict[str, Any], output_dir: Path) -> dict[str, dict[str, Any]]:
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    methods = [
        mode
        for mode in METHODS
        if mode in results.get("methods", {})
    ]
    metrics = results["methods"]
    controls = results["controls"]
    comparisons = results["comparisons"]
    primary = _choose_primary_recall(results)
    story = _prepare_feature_story(results, output_dir)

    fig = plt.figure(figsize=(16.2, 9.6))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(0.86, 1.14),
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.965,
        hspace=0.38,
        wspace=0.34,
    )
    ax_primary = fig.add_subplot(grid[0, 0])
    ax_mrr = fig.add_subplot(grid[0, 1])
    ax_effect = fig.add_subplot(grid[0, 2])
    ax_bridge = fig.add_subplot(grid[1, :2])
    ax_fingerprint = fig.add_subplot(grid[1, 2])

    benchmark_keys = ("word_jaccard", "raw_mean_hidden", *methods)
    benchmark_names = (
        "Word Jaccard",
        "Raw mean hidden",
        *(METHOD_SHORT_LABELS[mode] for mode in methods),
    )
    benchmark_colors = (
        "#B9B9B9",
        "#A9826B",
        *(COLORS[mode] for mode in methods),
    )
    primary_values = [
        _metric_value(results, key, primary["metric"])
        for key in benchmark_keys
    ]
    y = np.arange(len(benchmark_keys))
    ax_primary.hlines(
        y,
        0,
        primary_values,
        color=[_blend(color, 0.55) for color in benchmark_colors],
        linewidth=5,
        zorder=1,
    )
    ax_primary.scatter(
        primary_values,
        y,
        s=[
            165 if key == "cross" else 105 if key == "raw_mean_hidden" else 110
            for key in benchmark_keys
        ],
        color=benchmark_colors,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    ax_primary.set_yticks(y, benchmark_names)
    ax_primary.invert_yaxis()
    ax_primary.set_xlim(0, min(1.0, max(primary_values) + 0.12))
    ax_primary.set_xlabel(primary["label"])
    ax_primary.set_title(
        f"A. Unified retrieval benchmark — {primary['label']}",
        loc="left",
        fontweight="bold",
        fontsize=10.8,
        pad=8,
    )
    ax_primary.grid(axis="x", alpha=0.18)
    for yi, value, key in zip(y, primary_values, benchmark_keys, strict=True):
        ax_primary.text(
            value + 0.015,
            yi,
            f"{100 * value:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold" if key == "cross" else "normal",
            color=COLORS["cross"] if key == "cross" else "#333333",
        )
    for side in ("top", "right", "left"):
        ax_primary.spines[side].set_visible(False)

    names = [METHOD_SHORT_LABELS[mode] for mode in methods]
    mrr = [metrics[mode]["mrr"] for mode in methods]
    mrr_low = [
        value - metrics[mode]["mrr_95ci"][0]
        for mode, value in zip(methods, mrr, strict=True)
    ]
    mrr_high = [
        metrics[mode]["mrr_95ci"][1] - value
        for mode, value in zip(methods, mrr, strict=True)
    ]
    bars = ax_mrr.bar(
        names,
        mrr,
        color=[COLORS[mode] for mode in methods],
        width=0.68,
    )
    ax_mrr.errorbar(
        range(len(methods)),
        mrr,
        yerr=[mrr_low, mrr_high],
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.2,
    )
    ax_mrr.set_ylim(0, min(0.85, max(mrr) + 0.14))
    ax_mrr.set_ylabel("Mean reciprocal rank")
    ax_mrr.set_title(
        "B. Mean reciprocal rank (cluster bootstrap 95% CI)",
        loc="left",
        fontweight="bold",
        fontsize=10.8,
        pad=8,
    )
    ax_mrr.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, mrr, strict=True):
        ax_mrr.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    references = (
        "raw_mean_hidden",
        *(
            mode
            for mode in ("token", "temporal", "mean")
            if f"cross_minus_{mode}" in comparisons
        ),
    )
    labels = tuple(
        "vs raw hidden"
        if reference == "raw_mean_hidden"
        else f"vs {METHOD_SHORT_LABELS[reference]}"
        for reference in references
    )
    deltas = [
        comparisons[f"cross_minus_{reference}"]["mrr"]["point"]
        for reference in references
    ]
    lows = [
        value
        - comparisons[f"cross_minus_{reference}"]["mrr"]["95ci"][0]
        for reference, value in zip(references, deltas, strict=True)
    ]
    highs = [
        comparisons[f"cross_minus_{reference}"]["mrr"]["95ci"][1]
        - value
        for reference, value in zip(references, deltas, strict=True)
    ]
    y_effect = np.arange(len(references))
    ax_effect.errorbar(
        deltas,
        y_effect,
        xerr=[lows, highs],
        fmt="o",
        color=COLORS["cross"],
        ecolor="#333333",
        markersize=8,
        capsize=4,
    )
    ax_effect.axvline(0, color="#888888", linewidth=1)
    ax_effect.set_yticks(y_effect, labels)
    ax_effect.invert_yaxis()
    ax_effect.set_xlabel("Cross-Chunk SAE MRR advantage")
    ax_effect.set_title(
        "C. Paired document-level effect sizes",
        loc="left",
        fontweight="bold",
        fontsize=10.8,
        pad=8,
    )
    ax_effect.grid(axis="x", alpha=0.18)
    for yi, value, reference in zip(
        y_effect,
        deltas,
        references,
        strict=True,
    ):
        p_value = comparisons[f"cross_minus_{reference}"][
            "paired_randomization_p"
        ]
        ax_effect.text(
            value,
            yi + 0.18,
            f"+{value:.3f}; p={p_value:.1e}",
            va="top",
            ha="center",
            fontsize=8.5,
        )
    ax_effect.set_ylim(len(references) - 0.55, -0.35)

    if story is None:
        ax_bridge.axis("off")
        ax_fingerprint.axis("off")
        ax_bridge.text(
            0.5,
            0.5,
            "Feature-level story unavailable: features.npz or pair artifact missing.",
            ha="center",
        )
    else:
        _plot_feature_bridge(ax_bridge, story)
        _plot_sparse_fingerprint(ax_fingerprint, story)

    base = output_dir / "figures" / "lexical_controlled_document_linking"
    base.parent.mkdir(parents=True, exist_ok=True)
    style_figure_text(fig, minimum_tick_size=8.5)
    files = {}
    for suffix in ("png", "pdf"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight")
        files[suffix] = file_record(path, relative_to=output_dir)
    plt.close(fig)
    return files


def _summary_markdown(results: dict[str, Any]) -> str:
    rows = []
    methods = [
        mode
        for mode in METHODS
        if mode in results.get("methods", {})
    ]
    for mode in methods:
        value = results["methods"][mode]
        rows.append(
            f"| {DISPLAY_NAMES[mode]} | {value['recall_at_1']:.3f} "
            f"[{value['recall_at_1_95ci'][0]:.3f}, {value['recall_at_1_95ci'][1]:.3f}] "
            f"| {value['recall_at_5']:.3f} | {value['mrr']:.3f} "
            f"[{value['mrr_95ci'][0]:.3f}, {value['mrr_95ci'][1]:.3f}] "
            f"| {value['median_rank']:.1f} |"
        )
    comparisons = results["comparisons"]
    comparison_lines = []
    for reference in ("token", "temporal", "mean"):
        key = f"cross_minus_{reference}"
        if key not in comparisons:
            continue
        label = DISPLAY_NAMES[reference].removesuffix(" SAE")
        prefix = "+" if reference in {"token", "mean"} else ""
        comparison_lines.append(
            f"- Cross − {label}：Recall@1 "
            f"`{prefix}{comparisons[key]['recall_at_1']['point']:.3f}`，"
            f"MRR `{prefix}{comparisons[key]['mrr']['point']:.3f}`，"
            f"paired randomization "
            f"`p={comparisons[key]['paired_randomization_p']:.3g}`；"
        )
    return "\n".join(
        [
            "# Eval 6 — Lexically Disjoint Document Linking",
            "",
            "给定一个 held-out Pile chunk，从**完全不同文档**的候选中找回同一原始文档的",
            "另一个 chunk。正例强制 word-set Jaccard ≤ "
            f"`{results['sample']['lexical_jaccard_max']:.2f}`，候选池按目标 chunk 的**精确 token 长度**匹配；",
            "因此词面复制与长度都不能解决任务。该任务对应 RAG semantic routing、文档聚类、",
            "thread/duplicate linking 等真实下游用途。",
            "",
            "| Method | Recall@1 (95% CI) | Recall@5 | MRR (95% CI) | Median rank |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            *comparison_lines,
            f"- Lexical Jaccard baseline Recall@1：`{results['controls']['word_jaccard']['recall_at_1']:.3f}`；",
            f"- Raw layer-21 mean hidden Recall@1：`{results['controls']['raw_mean_hidden']['recall_at_1']:.3f}`。",
            "",
            "![Lexically controlled document linking](figures/lexical_controlled_document_linking.png)",
            "",
        ]
    )


def _summary_csv(results: dict[str, Any]) -> str:
    lines = [
        "mode,method,recall_at_1,recall_at_1_ci_low,recall_at_1_ci_high,"
        "recall_at_5,mrr,mrr_ci_low,mrr_ci_high,median_rank"
    ]
    for mode in (
        mode
        for mode in METHODS
        if mode in results.get("methods", {})
    ):
        value = results["methods"][mode]
        lines.append(
            ",".join(
                [
                    mode,
                    DISPLAY_NAMES[mode],
                    str(value["recall_at_1"]),
                    str(value["recall_at_1_95ci"][0]),
                    str(value["recall_at_1_95ci"][1]),
                    str(value["recall_at_5"]),
                    str(value["mrr"]),
                    str(value["mrr_95ci"][0]),
                    str(value["mrr_95ci"][1]),
                    str(value["median_rank"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    *,
    pair_path: Path,
    feature_manifest_path: Path,
) -> dict[str, Any]:
    """Publish all derived tables/figures and a verified top-level manifest."""

    result_path = output_dir / "document_linking_results.json"
    atomic_json_dump(result, result_path)
    plot_files = _plot_results(result, output_dir)
    readme_path = output_dir / "README.md"
    readme_path.write_text(_summary_markdown(result), encoding="utf-8")
    csv_path = output_dir / "summary_table.csv"
    csv_path.write_text(_summary_csv(result), encoding="utf-8")
    examples_path = output_dir / "qualitative_examples.json"
    atomic_json_dump({"examples": result["examples"]}, examples_path)
    return write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": result["identity"],
            "files": {
                "results": file_record(result_path, relative_to=output_dir),
                "pairs": file_record(pair_path, relative_to=output_dir),
                "feature_manifest": file_record(
                    feature_manifest_path,
                    relative_to=output_dir,
                ),
                "features": file_record(
                    output_dir / "features.npz",
                    relative_to=output_dir,
                ),
                "readme": file_record(readme_path, relative_to=output_dir),
                "summary_csv": file_record(csv_path, relative_to=output_dir),
                "examples": file_record(
                    examples_path,
                    relative_to=output_dir,
                ),
                **{f"plot_{key}": value for key, value in plot_files.items()},
            },
        },
        output_dir / "manifest.json",
    )


def main() -> None:
    args = parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    output_dir = Path(args.output_dir).resolve()
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_manifest_path = Path(args.evidence_manifest).resolve()
    if evidence_manifest_path.parent != evidence_dir:
        raise ValueError("--evidence-manifest must belong to --evidence-dir")
    evidence_manifest = _read_json(evidence_manifest_path)
    if evidence_manifest.get("format") != "chunk-saes-feature-evidence-v2":
        raise ValueError("expected chunk-saes-feature-evidence-v2")
    if evidence_manifest.get("complete") is not True:
        raise ValueError("feature evidence is incomplete")
    evidence_protocol = evidence_manifest.get("representation_protocol")
    if not isinstance(evidence_protocol, dict) or evidence_protocol.get(
        "name"
    ) != "fixed-independent-chunk-v1":
        raise ValueError(
            "feature evidence was produced with an older representation "
            "protocol; rerun extract_feature_evidence.py"
        )
    if evidence_protocol.get("token_temporal_aggregation") != "mean_after_threshold":
        raise ValueError("feature evidence is not mean-after-threshold")
    candidate_pool = evidence_manifest.get("candidate_pool")
    if not isinstance(candidate_pool, dict) or not candidate_pool.get(
        "shared_across_methods"
    ):
        raise ValueError("feature evidence lacks a shared chunk candidate pool")
    chunks = _load_chunks(evidence_dir, evidence_manifest)
    pairs = build_pairs(
        chunks,
        lexical_jaccard_max=args.lexical_jaccard_max,
        min_word_length=args.min_word_length,
    )
    pair_payload = _pair_payload(chunks, pairs)
    pair_digest = json_digest(pair_payload)
    unique_indices = sorted(
        {index for pair in pairs for index in (pair.left_index, pair.right_index)},
        key=lambda index: int(chunks[index]["global_chunk_id"]),
    )
    unique_chunk_digest = json_digest(
        [int(chunks[index]["global_chunk_id"]) for index in unique_indices]
    )
    sae_set = resolve_sae_artifact_set(
        args.sae_root,
        selection=args.checkpoint_selection,
    )
    if int(sae_set["common"]["layer"]) != args.layer:
        raise ValueError("SAE layer does not match --layer")
    if Path(str(sae_set["common"]["model"])).resolve() != Path(args.model).resolve():
        raise ValueError("SAE model does not match --model")
    feature_identity = _feature_identity(
        args=args,
        evidence_manifest=evidence_manifest,
        sae_set=sae_set,
        pair_digest=pair_digest,
        unique_chunk_digest=unique_chunk_digest,
        chunk_lengths=sorted(
            {
                int(chunks[index]["length"])
                for index in unique_indices
            }
        ),
    )

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "selected_pairs.jsonl"
    with pair_path.open("w", encoding="utf-8") as handle:
        for row in pair_payload:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    feature_manifest_path = output_dir / "feature_manifest.json"
    if feature_manifest_path.exists():
        feature_manifest = ensure_reusable_artifact(
            feature_manifest_path,
            expected_format=FEATURE_FORMAT,
            expected_identity=feature_identity,
        )
    else:
        if args.analysis_only:
            raise FileNotFoundError(
                "--analysis-only requested but no verified feature artifact exists"
            )
        feature_manifest = _extract_features(
            args=args,
            chunks=chunks,
            unique_indices=unique_indices,
            sae_set=sae_set,
            output_dir=output_dir,
            feature_identity=feature_identity,
        )
    assert feature_manifest is not None
    with np.load(output_dir / "features.npz") as data:
        arrays = {key: data[key] for key in data.files}
    global_to_row = {
        int(global_id): index
        for index, global_id in enumerate(arrays["global_chunk_ids"])
    }
    chunk_index_by_global_id = {
        int(row["global_chunk_id"]): index
        for index, row in enumerate(chunks)
    }
    retokenized_length_by_chunk = {
        chunk_index_by_global_id[int(global_id)]: int(length)
        for global_id, length in zip(
            arrays["global_chunk_ids"],
            arrays["retokenized_lengths"],
            strict=True,
        )
    }
    pairs, gallery_filter = filter_pairs_for_gallery_support(
        pairs,
        chunks,
        retokenized_length_by_chunk=retokenized_length_by_chunk,
        min_gallery_size=2,
    )
    pair_payload = _pair_payload(chunks, pairs)
    with pair_path.open("w", encoding="utf-8") as handle:
        for row in pair_payload:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    left_rows = np.asarray(
        [
            global_to_row[int(chunks[pair.left_index]["global_chunk_id"])]
            for pair in pairs
        ],
        dtype=np.int64,
    )
    right_rows = np.asarray(
        [
            global_to_row[int(chunks[pair.right_index]["global_chunk_id"])]
            for pair in pairs
        ],
        dtype=np.int64,
    )
    retokenized_lengths = arrays["retokenized_lengths"].astype(np.int16)
    left_lengths = retokenized_lengths[left_rows]
    right_lengths = retokenized_lengths[right_rows]
    pair_ids = np.concatenate(
        [
            np.arange(len(pairs), dtype=np.int32),
            np.arange(len(pairs), dtype=np.int32),
        ]
    )
    modes = tuple(
        mode
        for mode in METHODS
        if f"{mode}_indices" in arrays
        and mode in feature_identity["feature_widths"]
    )
    representations: dict[str, np.ndarray | sparse.csr_matrix] = {
        mode: _csr_from_fixed_topk(
            arrays[f"{mode}_indices"],
            arrays[f"{mode}_values"],
            width=int(feature_identity["feature_widths"][mode]),
        )
        for mode in modes
    }
    representations["raw_mean_hidden"] = _row_normalize_dense(
        arrays["raw_mean_hidden"]
    )

    rank_distributions: dict[str, np.ndarray] = {}
    for name, representation in representations.items():
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
        rank_distributions[name] = np.concatenate([forward, backward])
    lexical_ranks, lexical_pair_ids = _lexical_baseline_ranks(
        chunks,
        pairs,
        min_word_length=args.min_word_length,
        left_lengths=left_lengths,
        right_lengths=right_lengths,
    )
    if not np.array_equal(lexical_pair_ids, pair_ids):
        raise AssertionError("lexical baseline pair order mismatch")

    methods = {}
    for mode_index, mode in enumerate(modes):
        ranks = rank_distributions[mode]
        methods[mode] = {
            **_metrics_from_ranks(ranks),
            **_bootstrap_method(
                ranks,
                pair_ids,
                samples=args.bootstrap_samples,
                seed=args.seed + mode_index,
            ),
            "mean_nonzero_features": float(arrays[f"{mode}_nnz"].mean()),
            "representation": feature_identity["representation"][mode],
        }
    raw_metrics = {
        **_metrics_from_ranks(rank_distributions["raw_mean_hidden"]),
        **_bootstrap_method(
            rank_distributions["raw_mean_hidden"],
            pair_ids,
            samples=args.bootstrap_samples,
            seed=args.seed + 10,
        ),
    }
    lexical_metrics = {
        **_metrics_from_ranks(lexical_ranks),
        **_bootstrap_method(
            lexical_ranks,
            pair_ids,
            samples=args.bootstrap_samples,
            seed=args.seed + 11,
        ),
    }
    all_reference_ranks = {
        **{
            mode: rank_distributions[mode]
            for mode in modes
            if mode != "cross"
        },
        "raw_mean_hidden": rank_distributions["raw_mean_hidden"],
        "word_jaccard": lexical_ranks,
    }
    comparisons = {}
    for reference_index, (reference, reference_ranks) in enumerate(
        all_reference_ranks.items()
    ):
        comparison = _paired_bootstrap_difference(
            rank_distributions["cross"],
            reference_ranks,
            pair_ids,
            samples=args.bootstrap_samples,
            seed=args.seed + 100 + reference_index,
        )
        comparison["paired_randomization_p"] = _paired_sign_flip_test(
            rank_distributions["cross"],
            reference_ranks,
            pair_ids,
            samples=args.permutation_samples,
            seed=args.seed + 200 + reference_index,
        )
        comparisons[f"cross_minus_{reference}"] = comparison

    all_target_lengths = np.concatenate([right_lengths, left_lengths])
    identity = {
        **feature_identity,
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "seed": args.seed,
        "retrieval_protocol": (
            "bidirectional paired retrieval; each gallery contains one target "
            "chunk per document and is restricted to the query's exact target "
            "token length; cosine similarity; tie-aware average rank"
        ),
    }
    result = {
        "format": RESULT_FORMAT,
        "complete": True,
        "identity": identity,
        "task": {
            "name": "Lexically Disjoint Document Linking",
            "downstream_value": [
                "semantic routing for retrieval-augmented generation",
                "document/thread clustering",
                "near-duplicate and provenance linking after paraphrase",
                "cross-section navigation within long documents",
            ],
            "primary_metric": "bidirectional exact-length-controlled Recall@1",
            "secondary_metric": "mean reciprocal rank",
            "positive_pair": (
                "two independently sampled chunks from the same held-out Pile "
                "document with minimal word-set Jaccard overlap"
            ),
            "negative_pool": (
                "one target chunk from every other selected document with the "
                "same exact target token length"
            ),
        },
        "sample": {
            "documents": len(pairs),
            "directions": 2 * len(pairs),
            "unique_chunks": len(unique_indices),
            "lexical_jaccard_max": args.lexical_jaccard_max,
            "lexical_jaccard_mean": float(
                np.mean([pair.lexical_jaccard for pair in pairs])
            ),
            "lexical_jaccard_median": float(
                np.median([pair.lexical_jaccard for pair in pairs])
            ),
            "lexical_jaccard_95pct": float(
                np.quantile([pair.lexical_jaccard for pair in pairs], 0.95)
            ),
            "target_length_counts": {
                str(key): int(value)
                for key, value in sorted(Counter(all_target_lengths.tolist()).items())
            },
            "retokenization_length_mismatches": int(
                np.sum(
                    arrays["lengths"].astype(np.int16)
                    != arrays["retokenized_lengths"].astype(np.int16)
                )
            ),
            "gallery_support_filter": gallery_filter,
            "pair_digest": pair_digest,
        },
        "methods": methods,
        "controls": {
            "word_jaccard": lexical_metrics,
            "raw_mean_hidden": raw_metrics,
            "chance": _chance_metrics(all_target_lengths),
        },
        "comparisons": comparisons,
        "rank_distributions": {
            mode: rank_distributions[mode].tolist() for mode in modes
        },
        "examples": _select_examples(
            chunks,
            pairs,
            cross_ranks=rank_distributions["cross"],
            baseline_ranks={
                mode: rank_distributions[mode]
                for mode in modes
                if mode != "cross"
            },
            count=args.example_count,
        ),
        "interpretation": (
            "Cross-Chunk SAE wins when the correct partner must be recognized "
            "through shared document-level semantics rather than overlapping "
            "tokens or length. This is task-level evidence that the joint "
            "Cross-Chunk sparse code preserves reusable document-level signal; "
            "by itself it does not establish a larger fraction of individually "
            "monosemantic high-level features."
        ),
    }
    manifest = write_outputs(
        result,
        output_dir,
        pair_path=pair_path,
        feature_manifest_path=feature_manifest_path,
    )
    print(
        json.dumps(
            {
                "result": str(output_dir / "document_linking_results.json"),
                "manifest_digest": manifest["artifact_digest"],
                "documents": len(pairs),
                "methods": {
                    mode: {
                        "recall_at_1": methods[mode]["recall_at_1"],
                        "mrr": methods[mode]["mrr"],
                    }
                    for mode in modes
                },
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
