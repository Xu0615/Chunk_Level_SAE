#!/usr/bin/env python
"""Diagnose the granularity mismatch of a token-level sparse autoencoder.

This script intentionally evaluates one model only: the frozen Token SAE.  It
does not compare against, or report results from, any chunk-level SAE.  The
experiment asks whether a dictionary learned under token-wise reconstruction
is a natural dictionary for passage-level concepts.

Two intrinsic measurements are assembled into one camera-ready infographic:

1. Exact center-token concentration in a fixed random sample of 1,000 alive
   Token-SAE features.
2. Support growth when checkpoint-thresholded token codes are aggregated
   across held-out 128-token passages.

The final part of the visual is a design target, not a downstream evaluation:
the kinds of passage-scale coordinates we want the dictionary to contain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patheffects
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)
from motivation.render_granularity_figure import (
    render_infographic as render_granularity_infographic,
)
from safetensors import safe_open


RESULT_FORMAT = "token-sae-granularity-diagnosis-v2"
TOKEN_HIGHLIGHT_PATTERN = re.compile(r"<<(.*?)>>", flags=re.DOTALL)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")

TOKEN_BLUE = "#B8D7EA"
TOKEN_BLUE_DARK = "#3F789C"
TOKEN_BLUE_DEEP = "#245675"
TOKEN_PALE = "#F3F8FB"
INK = "#252932"
MUTED = "#657080"
FAINT = "#A5AFBD"
WHITE = "#FFFFFF"
PAPER = "#FBFCFE"
WARNING = "#BE223D"
WARNING_PALE = "#F8E9EC"
GOLD = "#E6A43A"
GOLD_PALE = "#FFF5DD"
PURPLE = "#7257A8"
PURPLE_PALE = "#F2EEFA"
GREEN = "#2B806C"
GREEN_PALE = "#E7F4F0"

FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "to",
        "of",
        "in",
        "on",
        "for",
        "from",
        "by",
        "with",
        "as",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "not",
        "no",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "will",
        "may",
        "might",
        "must",
        "all",
        "some",
        "there",
        "here",
        "when",
        "where",
        "which",
        "who",
        "than",
        "then",
        "so",
        "up",
        "out",
        "into",
        "over",
        "after",
        "before",
    }
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path(
            "qwen/qwen35-9b-base/pile_proportional_exact_occurrence/"
            "eval/layer21_w65536_k128_1b"
        ),
    )
    p.add_argument(
        "--activation-cache-dir",
        type=Path,
        default=Path(
            "qwen/qwen35-9b-base/pile_proportional_exact_occurrence/"
            "data/layer21_validation_cache_exact10000128"
        ),
    )
    p.add_argument(
        "--token-checkpoint",
        type=Path,
        default=Path(
            "qwen/qwen35-9b-base/pile_proportional_exact_occurrence/checkpoints/"
            "qwen35-9b-base_layer21_w65536_k128_train1000000000_seed42_"
            "pile_proportional_exact_occurrence_saeimpl_batchtopk_"
            "separate_centers_preemptive_auxk_lr_floor_v3_latest/"
            "token/checkpoints/best"
        ),
    )
    p.add_argument("--output-dir", type=Path, default=Path("motivation"))
    p.add_argument("--passage-length", type=int, default=128)
    p.add_argument("--passage-samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--device", default="auto")
    p.add_argument("--token-batch-size", type=int, default=64)
    p.add_argument("--dpi", type=int, default=320)
    p.add_argument(
        "--reuse-results",
        action="store_true",
        help="Redraw/report from output-dir/results.json without loading the SAE.",
    )
    return p


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _normalise_trigger(value: str) -> str:
    value = value.replace("↵", "\n").strip().lower()
    return re.sub(r"\s+", " ", value)


def _trigger_category(value: str) -> str:
    if not value:
        return "blank_or_whitespace"
    if all(not character.isalnum() for character in value):
        return "punctuation_or_markup"
    if value.isdigit() or re.fullmatch(r"[\d.]+", value):
        return "number"
    if re.fullmatch(r"[a-z]+", value):
        return "function_word" if value in FUNCTION_WORDS else "content_word"
    return "subword_code_or_mixed"


def _center_token_id(example: Mapping[str, Any]) -> int:
    token_ids = example.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) % 2 != 1:
        raise ValueError("expected an odd token-centered context window")
    return int(token_ids[len(token_ids) // 2])


def _highlighted_surface(example: Mapping[str, Any]) -> str:
    match = TOKEN_HIGHLIGHT_PATTERN.search(str(example.get("text", "")))
    if match is None:
        raise ValueError("top example does not contain <<center token>>")
    return _normalise_trigger(match.group(1))


def _context_word_set(example: Mapping[str, Any]) -> frozenset[str]:
    text = TOKEN_HIGHLIGHT_PATTERN.sub(" ", str(example.get("text", "")))
    return frozenset(
        token.lower()
        for token in WORD_PATTERN.findall(text)
        if len(token) >= 3
    )


def _mean_pairwise_jaccard(sets: Sequence[frozenset[str]]) -> float:
    values: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            left = sets[left_index]
            right = sets[right_index]
            values.append(len(left & right) / max(1, len(left | right)))
    return float(np.mean(values)) if values else 0.0


def _display_context(text: str, *, width: int = 62) -> str:
    text = text.replace("↵", " ")
    text = TOKEN_HIGHLIGHT_PATTERN.sub(
        lambda match: f"‹{match.group(1).strip()}›",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return textwrap.shorten(text, width=width, placeholder="…")


def analyze_token_triggers(prepared_path: Path) -> dict[str, Any]:
    """Measure exact center-token anchoring in a random alive-feature sample."""

    records: list[dict[str, Any]] = []
    for row in _read_jsonl(prepared_path):
        if row.get("method") != "token":
            continue
        if row.get("status") not in ("prepared", "complete", "ok", None):
            continue
        examples = [
            example
            for example in row.get("generation_examples", [])
            if example.get("kind") == "top" and example.get("is_active", True)
        ]
        if not examples:
            continue
        token_ids = [_center_token_id(example) for example in examples]
        surfaces = [_highlighted_surface(example) for example in examples]
        token_counts = Counter(token_ids)
        dominant_id, dominant_count = token_counts.most_common(1)[0]
        dominant_surface = Counter(
            surface
            for surface, token_id in zip(surfaces, token_ids, strict=True)
            if token_id == dominant_id
        ).most_common(1)[0][0]
        records.append(
            {
                "feature_id": int(row["feature_id"]),
                "top_examples": len(examples),
                "dominant_center_token_id": dominant_id,
                "dominant_surface": dominant_surface,
                "dominant_category": _trigger_category(dominant_surface),
                "dominant_count": dominant_count,
                "center_token_concentration": dominant_count / len(examples),
                "mean_context_word_jaccard": _mean_pairwise_jaccard(
                    [_context_word_set(example) for example in examples]
                ),
                "examples": [
                    {
                        "text": _display_context(str(example["text"])),
                        "center_activation": float(example["center_activation"]),
                    }
                    for example in examples
                ],
            }
        )

    if len(records) != 1_000:
        raise ValueError(
            f"expected exactly 1,000 sampled Token-SAE features, got {len(records)}"
        )
    top_examples = {record["top_examples"] for record in records}
    if top_examples != {10}:
        raise ValueError(f"expected ten top examples per feature, got {top_examples}")

    concentrations = np.asarray(
        [record["center_token_concentration"] for record in records],
        dtype=np.float64,
    )
    exact = [record for record in records if record["dominant_count"] == 10]
    exact_categories = Counter(record["dominant_category"] for record in exact)
    local_categories = {
        "blank_or_whitespace",
        "punctuation_or_markup",
        "number",
        "function_word",
        "subword_code_or_mixed",
    }

    requested_cards = (21417, 26062, 53225)
    by_id = {record["feature_id"]: record for record in records}
    cards = []
    for feature_id in requested_cards:
        if feature_id not in by_id:
            raise ValueError(f"illustrative feature {feature_id} is not sampled")
        record = by_id[feature_id]
        if record["dominant_count"] != 10:
            raise ValueError(f"feature {feature_id} is not an exact token trigger")
        cards.append(
            {
                key: record[key]
                for key in (
                    "feature_id",
                    "dominant_center_token_id",
                    "dominant_surface",
                    "dominant_category",
                    "dominant_count",
                    "top_examples",
                    "mean_context_word_jaccard",
                    "examples",
                )
            }
        )

    return {
        "protocol": {
            "population": "complete training-alive Token-SAE dictionary",
            "selection": "shared random sample fixed before interpretation",
            "selection_seed": 42,
            "sampled_features": len(records),
            "top_examples_per_feature": 10,
            "trigger_test": "exact Qwen tokenizer ID at the centered activation position",
            "context_overlap": (
                "mean pairwise Jaccard over non-highlighted alphanumeric words "
                "of length >= 3"
            ),
        },
        "sampled_features": len(records),
        "median_center_token_concentration": float(np.median(concentrations)),
        "dominant_token_at_least_5_of_10": int(np.sum(concentrations >= 0.5)),
        "dominant_token_at_least_8_of_10": int(np.sum(concentrations >= 0.8)),
        "same_token_all_10": len(exact),
        "same_token_all_10_fraction": len(exact) / len(records),
        "same_token_all_10_context_jaccard_median": float(
            np.median(
                [record["mean_context_word_jaccard"] for record in exact]
            )
        ),
        "same_token_all_10_categories": dict(sorted(exact_categories.items())),
        "same_token_all_10_surface_local": int(
            sum(
                count
                for category, count in exact_categories.items()
                if category in local_categories
            )
        ),
        "feature_cards": cards,
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _cache_shard_paths(cache_dir: Path, manifest: Mapping[str, Any]) -> list[Path]:
    paths = [
        cache_dir / str(shard["path"])
        for rank in manifest.get("ranks", [])
        for shard in rank.get("shards", [])
    ]
    if not paths:
        raise ValueError("activation-cache manifest contains no shards")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return sorted(paths)


def _sample_fixed_length_passages(
    cache_dir: Path,
    *,
    passage_length: int,
    sample_size: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Reservoir-sample held-out passage token activations without loading cache."""

    manifest = _read_json(cache_dir / "manifest.json")
    required_coverage = {
        "pair_ids_complete",
        "pair_ids_unique",
        "occurrence_ranges_complete",
        "token_hidden_rows_equal_occurrences",
        "plan_row_digest_matches",
    }
    coverage = manifest.get("coverage", {})
    if (
        manifest.get("format") != "chunk-saes-activation-cache-v2"
        or manifest.get("complete") is not True
        or manifest.get("independent_forwards") is not True
        or not all(bool(coverage.get(key)) for key in required_coverage)
    ):
        raise ValueError("activation cache does not meet the held-out protocol")

    paths = _cache_shard_paths(cache_dir, manifest)
    rng = random.Random(seed)
    reservoirs: list[tuple[Path, int, int, int, str, int]] = []
    eligible = 0
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            lengths_a = handle.get_tensor("length_a")
            lengths_b = handle.get_tensor("length_b")
            pair_ids = handle.get_tensor("pair_id")
        for local_index in range(lengths_a.numel()):
            for side, length in (
                ("a", int(lengths_a[local_index])),
                ("b", int(lengths_b[local_index])),
            ):
                if length != passage_length:
                    continue
                item = (
                    path,
                    local_index,
                    int(pair_ids[local_index]),
                    length,
                    side,
                    eligible,
                )
                eligible += 1
                if len(reservoirs) < sample_size:
                    reservoirs.append(item)
                else:
                    replacement = rng.randrange(eligible)
                    if replacement < sample_size:
                        reservoirs[replacement] = item
    if eligible < sample_size:
        raise ValueError(
            f"cache has {eligible} passages of length {passage_length}, "
            f"but {sample_size} were requested"
        )

    by_path: dict[Path, list[tuple[int, tuple[Path, int, int, int, str, int]]]] = {}
    for position, item in enumerate(reservoirs):
        by_path.setdefault(item[0], []).append((position, item))
    loaded: list[torch.Tensor | None] = [None] * sample_size
    metadata: list[dict[str, Any] | None] = [None] * sample_size
    for path, selections in by_path.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            hidden = handle.get_tensor("token_hidden")
            offsets = handle.get_tensor("chunk_offsets").long()
            for position, item in selections:
                _, local_index, pair_id, length, side, stream_index = item
                if side == "a":
                    start = int(offsets[2 * local_index])
                    stop = int(offsets[2 * local_index + 1])
                else:
                    start = int(offsets[2 * local_index + 1])
                    stop = int(offsets[2 * local_index + 2])
                chunk = hidden[start:stop].float().clone()
                if chunk.shape != (passage_length, int(manifest["hidden_size"])):
                    raise ValueError("cached passage shape disagrees with metadata")
                loaded[position] = chunk
                metadata[position] = {
                    "pair_id": pair_id,
                    "side": side,
                    "length": length,
                    "cache_shard": str(path.relative_to(cache_dir)),
                    "stream_index": stream_index,
                }
    if any(chunk is None for chunk in loaded):
        raise AssertionError("failed to load every sampled passage")
    return (
        [chunk for chunk in loaded if chunk is not None],
        [row for row in metadata if row is not None],
        {
            "format": manifest["format"],
            "activation_digest": manifest["activation_digest"],
            "pairs": int(manifest["pairs"]),
            "layer": int(manifest["layer"]),
            "hidden_size": int(manifest["hidden_size"]),
            "independent_forwards": True,
            "eligible_passages": eligible,
        },
    )


class FrozenTokenEncoder:
    def __init__(self, checkpoint_dir: Path, device: torch.device) -> None:
        config = _read_json(checkpoint_dir / "config.json")
        if config.get("mode") != "token":
            raise ValueError("motivation experiment requires a Token-SAE checkpoint")
        if int(config.get("k", -1)) != 128:
            raise ValueError("motivation experiment requires training K=128")
        with safe_open(
            str(checkpoint_dir / "sae.safetensors"),
            framework="pt",
            device="cpu",
        ) as handle:
            self.weight = handle.get_tensor("encoder_weight").to(device)
            self.bias = handle.get_tensor("encoder_bias").to(device)
            self.pre_bias = handle.get_tensor("pre_bias").to(device)
            self.scale = float(handle.get_tensor("activation_scale"))
            self.threshold = float(handle.get_tensor("threshold"))
            self.feature_counts = handle.get_tensor("feature_counts")
        self.device = device
        self.width = int(self.weight.shape[0])
        self.hidden_size = int(self.weight.shape[1])
        self.training_k = int(config["k"])
        self.config = config

    @torch.inference_mode()
    def encode_pretopk(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.device, dtype=self.weight.dtype)
        return F.relu(
            F.linear(
                hidden * self.scale - self.pre_bias,
                self.weight,
                self.bias,
            )
        )

    def close(self) -> None:
        del self.weight, self.bias, self.pre_bias
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _bootstrap_median(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        draws = rng.integers(0, array.size, size=(count, array.size))
        estimates[start : start + count] = np.median(array[draws], axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def analyze_passage_support(
    cache_dir: Path,
    checkpoint_dir: Path,
    *,
    passage_length: int,
    sample_size: int,
    seed: int,
    device: torch.device,
    token_batch_size: int,
) -> dict[str, Any]:
    """Aggregate checkpoint-thresholded token codes and quantify support growth."""

    passages, sample_metadata, cache_identity = _sample_fixed_length_passages(
        cache_dir,
        passage_length=passage_length,
        sample_size=sample_size,
        seed=seed,
    )
    encoder = FrozenTokenEncoder(checkpoint_dir, device)
    if encoder.hidden_size != cache_identity["hidden_size"]:
        raise ValueError("SAE/checkpoint hidden sizes do not match")

    rows: list[dict[str, Any]] = []
    try:
        for sample_index, (passage, metadata) in enumerate(
            zip(passages, sample_metadata, strict=True)
        ):
            pooled = torch.zeros(
                encoder.width,
                dtype=torch.float32,
                device=device,
            )
            seen = torch.zeros(
                encoder.width,
                dtype=torch.bool,
                device=device,
            )
            for start in range(0, passage.shape[0], token_batch_size):
                scores = encoder.encode_pretopk(
                    passage[start : start + token_batch_size]
                )
                activations = scores * (scores > encoder.threshold)
                active_rows, active_features = torch.nonzero(
                    activations > 0,
                    as_tuple=True,
                )
                del active_rows
                if active_features.numel() > 0:
                    seen[active_features] = True
                    pooled.add_(activations.float().sum(dim=0))
            positive = pooled[pooled > 0]
            ordered = torch.sort(positive, descending=True).values
            cumulative = torch.cumsum(ordered, dim=0) / ordered.sum()
            k90 = (
                int(
                    torch.searchsorted(
                        cumulative,
                        torch.tensor(0.9, device=device),
                    ).item()
                )
                + 1
            )
            retained = float(
                torch.topk(
                    positive,
                    k=min(encoder.training_k, int(positive.numel())),
                ).values.sum()
                / positive.sum()
            )
            rows.append(
                {
                    "sample_index": sample_index,
                    **metadata,
                    "unique_features": int(seen.sum().item()),
                    "features_for_90pct_activation_mass": k90,
                    "activation_mass_retained_by_top128": retained,
                }
            )
    finally:
        encoder.close()

    unique = [row["unique_features"] for row in rows]
    k90 = [row["features_for_90pct_activation_mass"] for row in rows]
    retained = [row["activation_mass_retained_by_top128"] for row in rows]
    return {
        "protocol": {
            "sample": (
                f"deterministic reservoir sample of {sample_size} held-out, "
                f"independently forwarded {passage_length}-token passages"
            ),
            "sample_seed": seed,
            "token_code": (
                "checkpoint-thresholded per-token code from the frozen "
                "BatchTopK Token SAE"
            ),
            "pooling": "sum of nonnegative per-token codes",
            "dictionary_width": encoder.width,
            "training_k_budget": encoder.training_k,
            "checkpoint_threshold": encoder.threshold,
            "bootstrap_samples": 10_000,
        },
        "cache_identity": cache_identity,
        "passage_length": passage_length,
        "sampled_passages": sample_size,
        "median_unique_features": float(np.median(unique)),
        "median_unique_features_95ci": _bootstrap_median(
            unique,
            samples=10_000,
            seed=seed + 1,
        ),
        "median_features_for_90pct_activation_mass": float(np.median(k90)),
        "median_features_for_90pct_activation_mass_95ci": _bootstrap_median(
            k90,
            samples=10_000,
            seed=seed + 2,
        ),
        "median_activation_mass_retained_by_top128": float(
            np.median(retained)
        ),
        "median_activation_mass_retained_by_top128_95ci": _bootstrap_median(
            retained,
            samples=10_000,
            seed=seed + 3,
        ),
        "per_passage": rows,
    }


def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = "none",
    linewidth: float = 1.0,
    radius: float = 0.018,
    zorder: float = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.5,
    mutation_scale: float = 13,
    connectionstyle: str = "arc3",
    zorder: float = 2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def _draw_token(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    width: float,
    emphasized: bool = False,
) -> None:
    _rounded_box(
        ax,
        (x, y),
        width,
        0.055,
        facecolor=WARNING_PALE if emphasized else WHITE,
        edgecolor=WARNING if emphasized else "#CAD6E0",
        linewidth=1.25 if emphasized else 0.9,
        radius=0.012,
        zorder=5,
    )
    ax.text(
        x + width / 2,
        y + 0.0275,
        text,
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="bold" if emphasized else "normal",
        color=WARNING if emphasized else INK,
        zorder=6,
    )


def _draw_feature_spark(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    *,
    angle: float,
    color: str = TOKEN_BLUE_DARK,
) -> None:
    radius = 0.017
    dx = math.cos(angle) * 0.03
    dy = math.sin(angle) * 0.026
    ax.plot(
        [x, x + dx],
        [y, y + dy],
        color=color,
        linewidth=0.7,
        alpha=0.65,
        zorder=3,
    )
    circle = Circle(
        (x + dx, y + dy),
        radius,
        facecolor=TOKEN_PALE,
        edgecolor=color,
        linewidth=0.7,
        zorder=4,
    )
    ax.add_patch(circle)
    ax.text(
        x + dx,
        y + dy,
        label,
        ha="center",
        va="center",
        fontsize=4.2,
        color=TOKEN_BLUE_DEEP,
        zorder=5,
    )


def _draw_microscope(ax: plt.Axes, center: tuple[float, float]) -> None:
    x, y = center
    ax.add_patch(
        Circle(
            (x - 0.006, y + 0.018),
            0.032,
            facecolor=TOKEN_PALE,
            edgecolor=TOKEN_BLUE_DARK,
            linewidth=2.3,
            zorder=4,
        )
    )
    ax.plot(
        [x + 0.016, x + 0.054],
        [y - 0.004, y - 0.046],
        color=TOKEN_BLUE_DARK,
        linewidth=4.5,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        [x + 0.03, x + 0.063],
        [y - 0.055, y - 0.055],
        color=TOKEN_BLUE_DARK,
        linewidth=3.5,
        solid_capstyle="round",
        zorder=4,
    )


def _draw_warning(ax: plt.Axes, x: float, y: float, size: float = 0.035) -> None:
    triangle = Polygon(
        [(x, y + size), (x - size * 0.9, y - size), (x + size * 0.9, y - size)],
        closed=True,
        facecolor=WARNING,
        edgecolor="none",
        zorder=5,
    )
    ax.add_patch(triangle)
    ax.text(
        x,
        y - 0.006,
        "!",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=WHITE,
        zorder=6,
    )


def _draw_cloud(
    ax: plt.Axes,
    center: tuple[float, float],
    *,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    zorder: float = 2,
) -> None:
    x, y = center
    circles = (
        (x - width * 0.25, y, height * 0.25),
        (x - width * 0.08, y + height * 0.11, height * 0.34),
        (x + width * 0.12, y + height * 0.08, height * 0.29),
        (x + width * 0.28, y - height * 0.01, height * 0.22),
        (x + width * 0.02, y - height * 0.08, height * 0.31),
    )
    for cx, cy, radius in circles:
        ax.add_patch(
            Circle(
                (cx, cy),
                radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.0,
                zorder=zorder,
            )
        )


def _draw_semantic_object(
    ax: plt.Axes,
    center: tuple[float, float],
    scale: float,
) -> None:
    x, y = center
    ax.add_patch(
        Circle(
            (x, y),
            0.25 * scale,
            facecolor=PURPLE_PALE,
            edgecolor=PURPLE,
            linewidth=1.7,
            zorder=3,
        )
    )
    node_offsets = (
        (-0.13, 0.10),
        (0.02, 0.15),
        (0.15, 0.06),
        (0.11, -0.11),
        (-0.06, -0.14),
        (-0.16, -0.02),
        (0.00, 0.00),
    )
    connections = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (0, 6),
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
    )
    for left, right in connections:
        ax.plot(
            [
                x + node_offsets[left][0] * scale,
                x + node_offsets[right][0] * scale,
            ],
            [
                y + node_offsets[left][1] * scale,
                y + node_offsets[right][1] * scale,
            ],
            color=PURPLE,
            linewidth=0.9,
            alpha=0.7,
            zorder=4,
        )
    for index, (dx, dy) in enumerate(node_offsets):
        ax.add_patch(
            Circle(
                (x + dx * scale, y + dy * scale),
                (0.026 if index == 6 else 0.018) * scale,
                facecolor=GOLD if index == 6 else WHITE,
                edgecolor=PURPLE,
                linewidth=0.9,
                zorder=5,
            )
        )


def _micro_context(text: str, *, flank: int = 15) -> str:
    start = text.find("‹")
    end = text.find("›", start + 1)
    if start < 0 or end < 0:
        return textwrap.shorten(text, width=2 * flank + 8, placeholder="…")
    left = text[max(0, start - flank) : start].strip()
    right = text[end + 1 : end + 1 + flank].strip()
    if start > flank:
        left = "…" + left
    if end + 1 + flank < len(text):
        right = right + "…"
    return f"{left} {text[start:end + 1]} {right}".strip()


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def render_infographic(results: Mapping[str, Any], output_dir: Path, dpi: int) -> dict[str, str]:
    """Render a sparse, single-flow editorial infographic."""

    trigger = results["token_trigger_concentration"]
    support = results["passage_support_expansion"]
    target = results["semantic_target"]
    cards = trigger["feature_cards"]
    retained = 100 * support["median_activation_mass_retained_by_top128"]
    discarded = 100.0 - retained

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(15.2, 5.9), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # One quiet canvas; whitespace, not panel borders, separates the story.
    ax.add_patch(
        FancyBboxPatch(
            (0.022, 0.170),
            0.956,
            0.760,
            boxstyle="round,pad=0.0,rounding_size=0.026",
            facecolor=WHITE,
            edgecolor="#E1E7EE",
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.add_patch(
        Circle(
            (0.46, 0.57),
            0.24,
            facecolor=TOKEN_PALE,
            edgecolor="none",
            alpha=0.34,
            zorder=0,
        )
    )
    ax.add_patch(
        Circle(
            (0.84, 0.48),
            0.22,
            facecolor=PURPLE_PALE,
            edgecolor="none",
            alpha=0.30,
            zorder=0,
        )
    )

    ax.text(
        0.052,
        0.876,
        "Token-wise sparsity  ≠  semantic sparsity",
        fontsize=22,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )
    ax.plot([0.052, 0.948], [0.816, 0.816], color="#E7EBF1", linewidth=1.0)

    header_y = 0.775
    ax.text(
        0.052,
        header_y,
        "TOKEN OBJECTIVE",
        fontsize=8.1,
        fontweight="bold",
        color=TOKEN_BLUE_DEEP,
        va="center",
    )
    ax.text(
        0.387,
        header_y,
        "POST-HOC POOLING",
        fontsize=8.1,
        fontweight="bold",
        color=TOKEN_BLUE_DEEP,
        va="center",
    )
    ax.text(
        0.735,
        header_y,
        "DESIRED DICTIONARY",
        fontsize=8.1,
        fontweight="bold",
        color=PURPLE,
        va="center",
    )

    # Left: the objective is shown operating on individual token boxes.
    _rounded_box(
        ax,
        (0.052, 0.680),
        0.272,
        0.064,
        facecolor="#F8FAFC",
        edgecolor="#D8E0E8",
        linewidth=0.9,
        radius=0.012,
    )
    ax.text(
        0.188,
        0.712,
        "“The volcano erupted again, forcing villages to evacuate.”",
        fontsize=8.3,
        color=INK,
        ha="center",
        va="center",
    )

    token_specs = (
        ("volcano", 0.059),
        ("erupted", 0.056),
        ("again", 0.043),
        (",", 0.024),
        ("evacuate", 0.061),
    )
    token_x = 0.052
    token_centers: list[float] = []
    for token, width in token_specs:
        _draw_token(
            ax,
            token_x,
            0.585,
            token,
            width=width,
            emphasized=token == ",",
        )
        token_centers.append(token_x + width / 2)
        token_x += width + 0.006
    for feature_index, center_x in enumerate(token_centers):
        angles = (-1.86, -1.28) if feature_index % 2 == 0 else (-1.80, -1.34)
        for spark_index, angle in enumerate(angles):
            _draw_feature_spark(
                ax,
                center_x,
                0.584,
                f"f{feature_index * 2 + spark_index + 1}",
                angle=angle,
            )

    ax.plot(
        [0.188, 0.188],
        [0.535, 0.505],
        color="#AFC6D5",
        linewidth=1.0,
    )
    ax.plot([0.095, 0.281], [0.505, 0.505], color="#AFC6D5", linewidth=1.0)

    # Compact real-feature chips: no excerpts, just exact observed triggers.
    chip_x = (0.052, 0.144, 0.236)
    chip_colors = (WARNING, GOLD, TOKEN_BLUE_DARK)
    for x, row, color in zip(chip_x, cards, chip_colors, strict=True):
        _rounded_box(
            ax,
            (x, 0.415),
            0.080,
            0.072,
            facecolor=mcolors.to_hex(
                np.asarray(mcolors.to_rgb(color)) * 0.045 + 0.955
            ),
            edgecolor=mcolors.to_hex(
                np.asarray(mcolors.to_rgb(color)) * 0.55 + 0.45
            ),
            linewidth=0.9,
            radius=0.012,
        )
        ax.text(
            x + 0.010,
            0.469,
            f"f{row['feature_id']:,}",
            fontsize=6.2,
            fontweight="bold",
            color=MUTED,
            ha="left",
            va="center",
        )
        ax.text(
            x + 0.040,
            0.441,
            f"‹{row['dominant_surface']}›",
            fontsize=11.4,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
        ax.text(
            x + 0.070,
            0.469,
            "10/10",
            fontsize=6.2,
            fontweight="bold",
            color=INK,
            ha="right",
            va="center",
        )

    ax.text(
        0.188,
        0.314,
        f"{trigger['same_token_all_10']:,} / {trigger['sampled_features']:,}",
        fontsize=25,
        fontweight="bold",
        color=WARNING,
        ha="center",
        va="center",
    )
    ax.text(
        0.188,
        0.270,
        "exact-token anchors",
        fontsize=8.4,
        fontweight="bold",
        color=INK,
        ha="center",
        va="center",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.064, 0.228),
            0.248,
            0.012,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor="#EDF0F4",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.064, 0.228),
            0.248 * trigger["same_token_all_10_fraction"],
            0.012,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=WARNING,
            edgecolor="none",
            zorder=2,
        )
    )

    # A single directional flow leads from token codes to an oversized union.
    _arrow(
        ax,
        (0.335, 0.530),
        (0.382, 0.570),
        color=TOKEN_BLUE_DARK,
        linewidth=2.1,
        connectionstyle="arc3,rad=-0.12",
    )
    _draw_cloud(
        ax,
        (0.463, 0.590),
        width=0.205,
        height=0.150,
        facecolor=TOKEN_PALE,
        edgecolor="#9FC5DC",
        zorder=1,
    )
    rng = np.random.default_rng(17)
    points = rng.normal(size=(72, 2))
    points[:, 0] = 0.463 + 0.073 * points[:, 0]
    points[:, 1] = 0.590 + 0.043 * points[:, 1]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=rng.uniform(5, 18, size=points.shape[0]),
        c=rng.choice(
            [TOKEN_BLUE_DARK, "#6FA7C7", "#94BDD4", WARNING],
            size=points.shape[0],
            p=[0.40, 0.33, 0.22, 0.05],
        ),
        alpha=0.78,
        linewidths=0,
        zorder=3,
    )
    ax.text(
        0.463,
        0.608,
        f"{support['median_unique_features']:,.0f}",
        fontsize=25,
        fontweight="bold",
        color=INK,
        ha="center",
        va="center",
        zorder=5,
        path_effects=[patheffects.withStroke(linewidth=5, foreground=TOKEN_PALE)],
    )
    ax.text(
        0.463,
        0.562,
        "unique coordinates",
        fontsize=7.8,
        fontweight="bold",
        color=MUTED,
        ha="center",
        va="center",
        zorder=5,
    )
    ax.text(
        0.463,
        0.516,
        "128-token passage",
        fontsize=7.2,
        color=FAINT,
        ha="center",
        va="center",
    )

    # The upper path preserves evidence but destroys passage sparsity.
    _arrow(
        ax,
        (0.505, 0.670),
        (0.600, 0.711),
        color=WARNING,
        linewidth=1.2,
        connectionstyle="arc3,rad=-0.18",
    )
    _rounded_box(
        ax,
        (0.585, 0.685),
        0.094,
        0.052,
        facecolor=WARNING_PALE,
        edgecolor="#DDA2AE",
        linewidth=0.9,
        radius=0.012,
    )
    ax.text(
        0.632,
        0.711,
        "KEEP ALL  →  DENSE",
        fontsize=7.0,
        fontweight="bold",
        color=WARNING,
        ha="center",
        va="center",
    )

    # The lower path restores a fixed budget through a literal funnel.
    funnel = Polygon(
        [
            (0.536, 0.645),
            (0.612, 0.610),
            (0.612, 0.570),
            (0.536, 0.535),
        ],
        closed=True,
        facecolor=GOLD_PALE,
        edgecolor=GOLD,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(funnel)
    for x, y in (
        (0.548, 0.619),
        (0.557, 0.599),
        (0.565, 0.625),
        (0.573, 0.575),
        (0.582, 0.603),
        (0.594, 0.585),
    ):
        ax.add_patch(
            Circle(
                (x, y),
                0.004,
                facecolor=TOKEN_BLUE_DARK,
                edgecolor="none",
                alpha=0.75,
                zorder=3,
            )
        )
    _rounded_box(
        ax,
        (0.621, 0.565),
        0.056,
        0.050,
        facecolor=TOKEN_BLUE_DEEP,
        edgecolor="none",
        radius=0.012,
        zorder=3,
    )
    ax.text(
        0.649,
        0.590,
        "128",
        fontsize=12.0,
        fontweight="bold",
        color=WHITE,
        ha="center",
        va="center",
        zorder=4,
    )
    ax.text(
        0.606,
        0.458,
        f"{discarded:.1f}%",
        fontsize=22,
        fontweight="bold",
        color=WARNING,
        ha="center",
        va="center",
    )
    ax.text(
        0.606,
        0.418,
        "activation mass discarded",
        fontsize=7.7,
        fontweight="bold",
        color=INK,
        ha="center",
        va="center",
    )
    ax.text(
        0.527,
        0.294,
        "POOLING  ≠  ABSTRACTION",
        fontsize=12.0,
        fontweight="bold",
        color=WARNING,
        ha="center",
        va="center",
    )
    # The untrained passage-level unit is a coherent, deliberately sparse
    # constellation rather than another quantitative panel.
    ax.text(
        0.696,
        0.565,
        "≠",
        fontsize=25,
        fontweight="bold",
        color=WARNING,
        ha="center",
        va="center",
    )
    center = (0.844, 0.565)
    outer = Circle(
        center,
        0.071,
        facecolor=PURPLE_PALE,
        edgecolor=PURPLE,
        linewidth=1.4,
        zorder=2,
    )
    ax.add_patch(outer)
    angles = np.linspace(0, 2 * math.pi, 6, endpoint=False) + math.pi / 6
    vertices = np.column_stack(
        (
            center[0] + 0.047 * np.cos(angles),
            center[1] + 0.047 * np.sin(angles),
        )
    )
    ax.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=GOLD_PALE,
            edgecolor=GOLD,
            linewidth=1.2,
            zorder=3,
        )
    )
    ax.text(
        center[0],
        center[1],
        "VOLCANIC\nHAZARD",
        fontsize=8.5,
        fontweight="bold",
        color=PURPLE,
        ha="center",
        va="center",
        linespacing=1.0,
        zorder=4,
    )

    concept_specs = (
        ("TOPIC", target["concepts"]["topic"], 0.754, 0.640),
        ("INTENT", target["concepts"]["intent"], 0.934, 0.640),
        ("REASONING", target["concepts"]["reasoning"], 0.754, 0.480),
        ("DISCOURSE", target["concepts"]["discourse"], 0.934, 0.480),
    )
    for label, value, x, y in concept_specs:
        ax.plot(
            [center[0], x],
            [center[1], y],
            color="#A998CF",
            linewidth=0.9,
            zorder=1,
        )
        _rounded_box(
            ax,
            (x - 0.049, y - 0.035),
            0.098,
            0.070,
            facecolor=PURPLE_PALE,
            edgecolor="#A998CF",
            linewidth=0.9,
            radius=0.012,
            zorder=3,
        )
        ax.text(
            x,
            y + 0.012,
            label,
            fontsize=6.3,
            fontweight="bold",
            color=PURPLE,
            ha="center",
            va="center",
            zorder=4,
        )
        ax.text(
            x,
            y - 0.013,
            value,
            fontsize=7.0,
            color=INK,
            ha="center",
            va="center",
            zorder=4,
        )

    ax.text(
        0.844,
        0.348,
        "ONE FEATURE  ≈  ONE HIGH-LEVEL IDEA",
        fontsize=11.0,
        fontweight="bold",
        color=PURPLE,
        ha="center",
        va="center",
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": figures / "token_sae_granularity_gap.png",
        "pdf": figures / "token_sae_granularity_gap.pdf",
        "svg": figures / "token_sae_granularity_gap.svg",
    }
    fig.savefig(
        paths["png"],
        dpi=dpi,
        facecolor=PAPER,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        paths["pdf"],
        facecolor=PAPER,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        paths["svg"],
        facecolor=PAPER,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return {
        key: str(path.relative_to(output_dir)) for key, path in paths.items()
    }


def _caption(results: Mapping[str, Any]) -> str:
    trigger = results["token_trigger_concentration"]
    support = results["passage_support_expansion"]
    return (
        "**Token-wise sparsity does not yield passage-level concepts.** "
        "In a fixed random sample of "
        f"{trigger['sampled_features']:,} alive Token-SAE features, "
        f"{trigger['same_token_all_10']:,} fire on the identical tokenizer ID "
        "in all ten highest-activation contexts, directly exposing the "
        "token-anchored unit selected by token reconstruction. Aggregating "
        "checkpoint-thresholded token codes over a held-out 128-token passage "
        f"activates a median of {support['median_unique_features']:,.0f} "
        "distinct coordinates; truncating the pooled vector back to 128 keeps "
        f"only {100 * support['median_activation_mass_retained_by_top128']:.1f}% "
        "of its activation mass. Thus post-hoc pooling is forced either to "
        "abandon sparsity or discard most local evidence, but neither operation "
        "changes what the dictionary learned to represent. The right-hand "
        "illustration makes the design target explicit: if dictionary "
        "coordinates should denote topic, intent, discourse, or reasoning, "
        "the sparse objective must encounter the whole semantic object as its "
        "native unit.\n"
    )


def _report(results: Mapping[str, Any]) -> str:
    trigger = results["token_trigger_concentration"]
    support = results["passage_support_expansion"]
    categories = trigger["same_token_all_10_categories"]
    support_ci = support["median_unique_features_95ci"]
    retained_ci = support["median_activation_mass_retained_by_top128_95ci"]
    return f"""# Token-Level SAE 的粒度错配

## 结论

问题不在于 Token SAE 能否重构 token hidden state，而在于它把稀疏竞争和重构误差都
定义在**单个 token**上。因此，字典最直接的容量收益来自词、标点、数字、格式和固定
模板等局部触发器。事后把 token codes 做 pooling，只能汇总已经学成的 token-level
坐标，不能把这些坐标重新训练成主题、意图、篇章语义或推理步骤。

## 两个 Token-SAE-only 内生诊断

1. **字典被精确 token 锚定。** 在解释前固定、从完整 alive dictionary 随机抽取的
   {trigger['sampled_features']:,} 个 feature 中，有
   **{trigger['same_token_all_10']:,} 个（{trigger['same_token_all_10_fraction']:.1%}）**
   在十个最高激活上下文中都落在完全相同的 tokenizer ID。这里不依赖 LLM 标签。
   其中 {categories.get('punctuation_or_markup', 0)} 个是标点/markup，
   {categories.get('blank_or_whitespace', 0)} 个是空白，
   {categories.get('number', 0)} 个是数字，
   {categories.get('function_word', 0)} 个是功能词。

2. **token 稀疏性在 passage 上不闭合。** 使用冻结 checkpoint 的阈值化 token
   code，在 {support['sampled_passages']} 个独立 forward 的 held-out
   128-token passage 中，
   feature union 的中位数为 **{support['median_unique_features']:,.0f}**
   （bootstrap 95% CI [{support_ci[0]:,.0f}, {support_ci[1]:,.0f}]）。
   若 passage 表示重新截断到 128 维，只保留
   **{100 * support['median_activation_mass_retained_by_top128']:.1f}%**
   激活总量（95% CI [{100 * retained_ci[0]:.1f}%, {100 * retained_ci[1]:.1f}%]）；
   若不截断，则表示不再是紧凑的 sparse concept code。

## 审稿人问题的直接回答

Token-Level SAE 是 token-local attribution 的合适工具，但它没有机制要求一个
feature 概括整个 passage。对于希望字典坐标直接对应主题、语义、意图、推理和篇章
结构的目标，必须把 sparse bottleneck 的原生 observation unit 提升到相应的
high-level semantic unit；否则 pooling 只能在表示生成之后补救，无法改变字典形成时
的归纳偏置。

主图：`figures/token_sae_granularity_gap.png`。
"""


def _readme(results: Mapping[str, Any]) -> str:
    trigger = results["token_trigger_concentration"]
    support = results["passage_support_expansion"]
    return f"""# Motivation: Token-SAE granularity diagnosis

This directory contains a **Token-SAE-only** motivation experiment. No
Chunk-SAE result or model comparison appears in the experiment or figure.

The figure combines two intrinsic diagnostics:

- exact center-token anchoring across the top ten contexts of a fixed random
  sample of {trigger['sampled_features']:,} alive features;
- support expansion after pooling checkpoint-thresholded token codes over
  {support['sampled_passages']} held-out 128-token passages;

The right side is a conceptual design target—not a downstream or comparative
experiment—showing the passage-scale concepts the dictionary should natively
name.

Main figure:

```text
figures/token_sae_granularity_gap.png
```

Reproduce:

```bash
PYTHONPATH=src .venv/bin/python \\
  src/motivation/run_motivation_experiment.py \\
  --output-dir motivation
```

Redraw without loading the 2.1 GB checkpoint:

```bash
PYTHONPATH=src .venv/bin/python \\
  src/motivation/run_motivation_experiment.py \\
  --output-dir motivation --reuse-results
```
"""


def _build_audit(
    results: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rendered_text = " ".join(
        [
            json.dumps(results, ensure_ascii=False),
            (output_dir / "CAPTION.md").read_text(encoding="utf-8"),
            (output_dir / "REPORT.md").read_text(encoding="utf-8"),
        ]
    ).lower()
    forbidden_result_terms = (
        "cross-chunk",
        "mean-chunk",
        "joint-chunk",
        "cross_minus",
        "chunk_vs_token",
    )
    figures = sorted((output_dir / "figures").glob("*"))
    checks = {
        "token_sae_is_only_evaluated_model": not any(
            term in rendered_text for term in forbidden_result_terms
        ),
        "random_feature_sample_has_1000_rows": (
            results["token_trigger_concentration"]["sampled_features"] == 1_000
        ),
        "ten_top_examples_per_feature": (
            results["token_trigger_concentration"]["protocol"][
                "top_examples_per_feature"
            ]
            == 10
        ),
        "held_out_passage_sample_complete": (
            len(results["passage_support_expansion"]["per_passage"])
            == results["passage_support_expansion"]["sampled_passages"]
            and results["passage_support_expansion"]["cache_identity"][
                "independent_forwards"
            ]
            is True
        ),
        "no_downstream_evaluation_artifact": (
            "document_linking" not in rendered_text
            and "retrieval" not in rendered_text
            and "recall@" not in rendered_text
        ),
        "only_one_figure_stem": (
            {path.stem for path in figures} == {"token_sae_granularity_gap"}
        ),
        "png_pdf_svg_written": (
            {path.suffix for path in figures} == {".png", ".pdf", ".svg"}
        ),
        "plots_directory_removed": not (output_dir / "plots").exists(),
    }
    return {
        "format": f"{RESULT_FORMAT}-audit",
        "complete": all(checks.values()),
        "checks": checks,
        "primary_numbers": {
            "same_token_all_10": results["token_trigger_concentration"][
                "same_token_all_10"
            ],
            "median_unique_features_per_128_token_passage": results[
                "passage_support_expansion"
            ]["median_unique_features"],
            "top128_activation_mass_retained": results[
                "passage_support_expansion"
            ]["median_activation_mass_retained_by_top128"],
        },
    }


def _clean_output(output_dir: Path) -> None:
    plots = output_dir / "plots"
    if plots.exists():
        shutil.rmtree(plots)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for path in figures.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    obsolete = output_dir / "method_metrics.csv"
    if obsolete.exists():
        obsolete.unlink()


def _manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files[str(path.relative_to(output_dir))] = _file_record(
            path,
            relative_to=output_dir,
        )
    return {
        "format": f"{RESULT_FORMAT}-manifest",
        "complete": True,
        "identity": {
            "eval_root": str(args.eval_root.resolve()),
            "activation_cache_dir": str(args.activation_cache_dir.resolve()),
            "token_checkpoint": str(args.token_checkpoint.resolve()),
            "passage_length": args.passage_length,
            "passage_samples": args.passage_samples,
            "seed": args.seed,
            "sources": sources,
        },
        "files": files,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_output(output_dir)
    results_path = output_dir / "results.json"

    prepared_path = (
        args.eval_root
        / "autointerp/autointerp_exact1000/token_temporal/prepared_examples.jsonl"
    )
    data_manifest_path = (
        args.eval_root
        / "autointerp/autointerp_exact1000/data/data_manifest.json"
    )
    cache_manifest_path = args.activation_cache_dir / "manifest.json"
    checkpoint_manifest_path = args.token_checkpoint / "checkpoint_manifest.json"
    source_paths = (
        prepared_path,
        data_manifest_path,
        cache_manifest_path,
        checkpoint_manifest_path,
    )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.reuse_results:
        if not results_path.is_file():
            raise FileNotFoundError(
                f"--reuse-results requested but {results_path} does not exist"
            )
        results = _read_json(results_path)
        if results.get("format") != RESULT_FORMAT:
            raise ValueError("stored motivation results use an obsolete format")
    else:
        trigger = analyze_token_triggers(prepared_path)
        support = analyze_passage_support(
            args.activation_cache_dir,
            args.token_checkpoint,
            passage_length=args.passage_length,
            sample_size=args.passage_samples,
            seed=args.seed,
            device=_device(args.device),
            token_batch_size=args.token_batch_size,
        )
        sources = {
            str(path): _file_record(path) for path in source_paths
        }
        results = {
            "format": RESULT_FORMAT,
            "complete": True,
            "scope": {
                "evaluated_model": "Token SAE only",
                "chunk_sae_results_included": False,
                "model": "Qwen3.5-9B-Base",
                "layer": 21,
                "dictionary_width": 65_536,
                "training_k": 128,
            },
            "claim": (
                "A token-reconstruction dictionary is rewarded for local "
                "token identity. Pooling its codes later either destroys "
                "passage sparsity or discards most local evidence, but cannot "
                "retroactively make its coordinates native passage concepts."
            ),
            "token_trigger_concentration": trigger,
            "passage_support_expansion": support,
            "semantic_target": {
                "status": "design target; not a model evaluation",
                "passage": (
                    "The volcano erupted again, forcing nearby villages "
                    "to evacuate."
                ),
                "concepts": {
                    "topic": "eruption",
                    "intent": "warning",
                    "reasoning": "eruption → risk",
                    "discourse": "cause → action",
                },
            },
            "sources": sources,
        }
        _atomic_json(results_path, results)

    _atomic_text(output_dir / "CAPTION.md", _caption(results))
    _atomic_text(output_dir / "REPORT.md", _report(results))
    _atomic_text(output_dir / "README.md", _readme(results))
    plots = render_granularity_infographic(results, output_dir, args.dpi)
    results = dict(results)
    results["figure"] = plots
    _atomic_json(results_path, results)

    audit = _build_audit(results, output_dir)
    _atomic_json(output_dir / "audit_report.json", audit)
    if not audit["complete"]:
        raise RuntimeError(f"motivation audit failed: {audit['checks']}")

    sources = results.get("sources", {})
    manifest = _manifest(output_dir, args=args, sources=sources)
    _atomic_json(output_dir / "manifest.json", manifest)
    return results


def main() -> None:
    args = parser().parse_args()
    results = run(args)
    print(
        json.dumps(
            {
                "format": results["format"],
                "complete": results["complete"],
                "figure": results["figure"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
