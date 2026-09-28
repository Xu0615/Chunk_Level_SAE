#!/usr/bin/env python
"""Render the camera-ready sequence-activation and semantic-manifold figure.

Panel A follows the model-symmetric sequence protocol used in the T-SAE paper:
all methods see identical text, each method independently ranks its own full
dictionary, and no method is used as a feature-selection anchor.  To avoid a
resolution artifact, the displayed observation unit is also identical: one
independently forwarded 32-token chunk.  Token/Temporal activations are encoded
per token and averaged within that chunk, whereas Mean/Cross encode the hidden
state mean of the same chunk.  Every displayed feature therefore has exactly
the same number of plotted observations.

Panel B displays the shared ArXiv benchmark geometry.  Its numerical labels are
computed in the formal 50-D representation with leave-one-out 10-NN; t-SNE is
used only to draw the point clouds.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)
from matplotlib.textpath import TextPath
from safetensors import safe_open
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from chunk_saes.plot_style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_PALE_COLORS,
    METHOD_TEXT_COLORS,
    METHODS,
    style_figure_text,
)
from chunk_saes.sae import DecoderHead, SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.utils import parse_int_csv


DOMAIN_COLORS = {
    "CS": "#4C78A8",
    "Economics": "#F58518",
    "Electrical Engineering": "#E45756",
    "Mathematics": "#72B7B2",
    "Physics": "#54A24B",
    "Quantitative Biology": "#EECA3B",
    "Quantitative Finance": "#B279A2",
    "Statistics": "#FF9DA6",
}
DOMAIN_SHORT = {
    "CS": "CS",
    "Economics": "Econ",
    "Electrical Engineering": "EE",
    "Mathematics": "Math",
    "Physics": "Phys",
    "Quantitative Biology": "Q-Bio",
    "Quantitative Finance": "Q-Fin",
    "Statistics": "Stats",
}

DARK = "#20242C"
MID = "#667085"
GRID = "#DDE2EA"
PANEL = "#F7F8FB"
WHITE = "#FFFFFF"
GOLD = "#C28A2C"

SEQUENCE_PASSAGE_SPECS = (
    {
        "passage": "Programming",
        "short": "CSV leading zeros",
        "source": "cross_feature_evidence",
        "feature_id": 63524,
        "evidence_index": 9,
    },
    {
        "passage": "Medical",
        "short": "cardiac care",
        "source": "cross_feature_evidence",
        "feature_id": 12074,
        "evidence_index": 12,
    },
    {
        "passage": "News",
        "short": "viral media reaction",
        "source": "cross_feature_evidence",
        "feature_id": 25322,
        "evidence_index": 0,
    },
    {
        "passage": "Math",
        "short": "calibrated geometry",
        "source": "arxiv_test",
        "record_id": "0710.3920",
    },
)
FEATURE_RANK_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
)
SEQUENCE_LABEL_MAX_WORDS = 6
SEQUENCE_LABEL_MAX_TOKENS = 12
HIGHLIGHT_DOMAIN_COUNT = 2

# Panel A is also published as a standalone feature-dynamics figure.  Keep its
# original four-method styling while extending the same visual grammar to all
# four Joint-Chunk alpha checkpoints.
SEQUENCE_METHODS = (
    "token",
    "temporal",
    "mean",
    "joint_alpha0p25",
    "joint_alpha0p5",
    "joint_alpha1",
    "joint_alpha1p5",
    "cross",
)
SEQUENCE_METHOD_LABELS = {
    **METHOD_LABELS,
    "joint_alpha0p25": "Joint-Chunk SAE (α=0.25)",
    "joint_alpha0p5": "Joint-Chunk SAE (α=0.5)",
    "joint_alpha1": "Joint-Chunk SAE (α=1)",
    "joint_alpha1p5": "Joint-Chunk SAE (α=1.5)",
}
SEQUENCE_METHOD_COLORS = {
    method: METHOD_COLORS[method] for method in SEQUENCE_METHODS
}
SEQUENCE_METHOD_PALE_COLORS = {
    method: METHOD_PALE_COLORS[method] for method in SEQUENCE_METHODS
}
SEQUENCE_METHOD_TEXT_COLORS = {
    method: "#252932" for method in SEQUENCE_METHODS
}
SEQUENCE_PASSAGE_FEATURE_LABELS = {
    "Programming": "Programming code with CSV leading zeros",
    "Medical": "Clinical text about cardiac care outcomes",
    "News": "News reports on viral media reactions",
    "Math": "Calibrated geometry and smooth manifold mathematics",
}
SEQUENCE_PASSAGE_CONTEXT_LABELS = {
    "Programming": "programming",
    "Medical": "medical",
    "News": "news",
    "Math": "mathematics",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Render a strict 2:1 paper figure with sequence activations, "
            "population alignment, and semantic manifolds."
        )
    )
    p.add_argument("--eval-root", required=True)
    p.add_argument(
        "--features",
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--output-name",
        default="paper_figure3_cross_high_level_features",
    )
    p.add_argument(
        "--manifold-output",
        default="semantic_geometry/figures/semantic_geometry",
        help="Standalone manifold output relative to --eval-root.",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--decoder-block-size", type=int, default=1024)
    p.add_argument(
        "--decoder-head",
        choices=("mean", "cross"),
        default=None,
        help=(
            "Read this head when a decoder checkpoint is Joint Chunk. "
            "Leave unset for legacy single-head checkpoints."
        ),
    )
    p.add_argument(
        "--decoder-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--encoder-block-size", type=int, default=2048)
    p.add_argument("--sequence-top-k", type=int, default=5)
    p.add_argument(
        "--sequence-chunk-lengths",
        "--sequence-windows",
        dest="sequence_windows",
        default="32,64,128",
        help=(
            "Training-supported independent chunk lengths to cache. The "
            "display length controls ranking and the main plot; other lengths "
            "are robustness checks only. Every length must evenly tile each "
            "displayed passage."
        ),
    )
    p.add_argument(
        "--sequence-display-chunk-length",
        type=int,
        default=32,
        help=(
            "Common chunk observation length used for feature ranking and "
            "Panel A. It must be included in --sequence-chunk-lengths."
        ),
    )
    p.add_argument(
        "--sequence-window",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--sequence-max-length", type=int, default=128)
    p.add_argument(
        "--sequence-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device used for the shared independent-chunk language-model "
            "forwards and SAE scans."
        ),
    )
    p.add_argument("--tsne-perplexity", type=float, default=75.0)
    p.add_argument("--tsne-iterations", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--refresh-sequence", action="store_true")
    p.add_argument("--refresh-embeddings", action="store_true")
    p.add_argument(
        "--skip-manifold-output",
        action="store_true",
        help="Do not overwrite the standalone semantic-manifold files.",
    )
    return p


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decoder_weight_name(
    checkpoint: Path,
    head: DecoderHead | None,
) -> str:
    """Resolve a decoder tensor without silently collapsing Joint heads."""

    config = _read_json(checkpoint / "config.json")
    with safe_open(
        str(checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        names = set(handle.keys())
    is_joint = (
        int(config.get("decoder_heads", 1)) == 2
        or "decoder_cross_weight" in names
        or "decoder_cross_bias" in names
    )
    if ("decoder_cross_weight" in names) != ("decoder_cross_bias" in names):
        raise ValueError(f"{checkpoint} stores an incomplete Cross decoder head")
    if int(config.get("decoder_heads", 1)) == 2 and not {
        "decoder_cross_weight",
        "decoder_cross_bias",
    }.issubset(names):
        raise ValueError(
            f"{checkpoint} declares two decoder heads but lacks the Cross head"
        )
    if is_joint:
        if head not in ("mean", "cross"):
            raise ValueError(
                f"{checkpoint} is a Joint checkpoint; pass --decoder-head"
            )
        name = "decoder_cross_weight" if head == "cross" else "decoder_weight"
        if name not in names:
            raise ValueError(f"{checkpoint} lacks {name}")
        return name
    if head not in (None, "mean"):
        raise ValueError(
            f"single-head checkpoint {checkpoint} cannot select decoder_head={head!r}"
        )
    return "decoder_weight"


def _feature_annotations(
    eval_root: Path,
) -> tuple[dict[str, dict[int, str]], Path | None]:
    path = eval_root / "feature_dynamics" / "sequence_top5_autointerp.json"
    if not path.is_file():
        return {}, None
    payload = _read_json(path)
    if payload.get("complete") is not True:
        raise ValueError(f"incomplete feature annotation artifact: {path}")
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise ValueError(f"feature annotation artifact lacks methods: {path}")
    result: dict[str, dict[int, str]] = {}
    for method in METHODS:
        rows = methods.get(method)
        if not isinstance(rows, list):
            raise ValueError(f"feature annotations lack method={method}")
        result[method] = {}
        for row in rows:
            label = str(row.get("short_label", "")).strip()
            token_count = int(row.get("short_label_token_count", -1))
            word_count = len(label.split())
            if (
                not label
                or not 0 <= token_count <= SEQUENCE_LABEL_MAX_TOKENS
                or word_count > SEQUENCE_LABEL_MAX_WORDS
            ):
                raise ValueError(
                    f"invalid short label for {method}/{row.get('feature_id')}"
                )
            result[method][int(row["feature_id"])] = label
    return result, path


def sequence_feature_annotations(
    eval_root: Path,
    metadata: Mapping[str, Any],
    *,
    methods: Sequence[str] | None = None,
) -> tuple[dict[str, dict[int, str]], Path | None]:
    """Load Panel-A labels and fill new methods from trace metadata.

    The curated AutoInterp artifact predates the four Joint-Chunk alpha runs,
    so it remains authoritative for the original methods.  Joint methods use
    a deterministic, approximately six-word summary of their passage
    activation profile; no additional model/API call is needed to render the
    figure.
    """

    annotations, source_path = _feature_annotations(eval_root)
    method_rows = metadata.get("methods")
    if not isinstance(method_rows, Mapping):
        raise ValueError("sequence metadata lacks a methods mapping")
    passages = metadata.get("passages")
    if not isinstance(passages, list):
        raise ValueError("sequence metadata lacks a passages list")
    selected = tuple(
        methods
        or metadata.get("method_order")
        or method_rows.keys()
    )
    for method in selected:
        row = method_rows.get(method)
        if not isinstance(row, Mapping):
            raise ValueError(f"sequence metadata lacks method={method}")
        per_feature = row.get("per_feature")
        if not isinstance(per_feature, list):
            raise ValueError(
                f"sequence metadata lacks per_feature rows for method={method}"
            )
        method_annotations = annotations.setdefault(method, {})
        for feature in per_feature:
            if not isinstance(feature, Mapping) or "feature_id" not in feature:
                raise ValueError(
                    f"invalid per_feature row for sequence method={method}"
                )
            feature_id = int(feature["feature_id"])
            fallback = (
                _auto_curve_explanation(dict(feature), passages)
                if "segment_means" in feature
                else str(feature.get("auto_explanation", "")).strip()
            )
            method_annotations.setdefault(
                feature_id,
                fallback or f"feature {feature_id}",
            )
    return annotations, source_path


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "font.weight": "medium",
            "axes.titlesize": 14.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 13.0,
            "axes.labelweight": "bold",
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#C8CFD9",
            "axes.linewidth": 0.9,
            "xtick.color": MID,
            "ytick.color": MID,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "text.color": DARK,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _interpretations(eval_root: Path) -> dict[int, dict[str, Any]]:
    evidence_dir = eval_root / "dictionary_utilization" / "feature_evidence"
    blind_map = _read_json(evidence_dir / "blind_map.json")
    evidence = {
        row["blind_id"]: row
        for row in _read_jsonl(evidence_dir / "evidence_blinded.jsonl")
    }
    judge = {
        row["blind_id"]: row
        for row in _read_jsonl(
            eval_root
            / "semantic_invariance"
            / "legacy_multilevel"
            / "judge_dpsk_v4_flash"
            / "judge_outputs.jsonl"
        )
    }
    output: dict[int, dict[str, Any]] = {}
    for blind_id, metadata in blind_map.items():
        if metadata["method"] != "cross" or blind_id not in judge:
            continue
        try:
            response = json.loads(judge[blind_id]["raw_response"])
        except json.JSONDecodeError:
            response = {}
        output[int(metadata["feature_id"])] = {
            "level": str(judge[blind_id]["level"]),
            "description": str(response.get("description", "")),
            "texts": list(evidence[blind_id]["activating_chunks"]),
        }
    return output


def _checkpoint_paths(eval_root: Path) -> dict[str, Path]:
    identity = _evaluation_identity(eval_root)
    modes = identity.get("sae_modes")
    if not isinstance(modes, dict):
        raise ValueError("feature-evidence manifest lacks SAE mode metadata")
    required = {"token", "mean", "cross"}
    missing = sorted(required - set(modes))
    if missing:
        raise ValueError(f"feature-evidence manifest lacks SAE modes: {missing}")
    return {
        mode: Path(str(modes[mode]["checkpoint_path"]))
        for mode in sorted(required)
    }


def _temporal_checkpoint(eval_root: Path) -> tuple[Path | None, bool]:
    identity = _evaluation_identity(eval_root)
    modes = identity.get("sae_modes")
    if not isinstance(modes, dict) or "temporal" not in modes:
        return None, False
    checkpoint = Path(str(modes["temporal"]["checkpoint_path"]))
    root = checkpoint.parent.parent
    completed = (root / "complete.json").is_file()
    return (checkpoint if checkpoint.is_dir() else None), completed


def _evaluation_identity(eval_root: Path) -> dict[str, Any]:
    manifest = _read_json(
        eval_root
        / "dictionary_utilization"
        / "feature_evidence"
        / "evidence_manifest.json"
    )
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("feature-evidence manifest lacks identity metadata")
    if not identity.get("model") or identity.get("layer") is None:
        raise ValueError("feature-evidence manifest lacks model/layer metadata")
    return identity


def _checkpoint_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    output: dict[str, Any] = {"path": str(path)}
    for name in ("sae.safetensors", "config.json", "checkpoint_manifest.json"):
        current = path / name
        if current.is_file():
            stat = current.stat()
            output[name] = {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return output


def _sequence_passages(eval_root: Path) -> list[dict[str, Any]]:
    evidence_dir = eval_root / "dictionary_utilization" / "feature_evidence"
    blind_map = _read_json(evidence_dir / "blind_map.json")
    evidence_rows = {
        row["blind_id"]: row
        for row in _read_jsonl(evidence_dir / "evidence_blinded.jsonl")
    }
    cross_evidence: dict[int, list[str]] = {}
    for blind_id, metadata in blind_map.items():
        if metadata.get("method") != "cross":
            continue
        row = evidence_rows.get(blind_id)
        if row is None:
            continue
        cross_evidence[int(metadata["feature_id"])] = [
            str(text) for text in row["activating_chunks"]
        ]
    benchmark_rows = {
        str(row["id"]): row
        for row in _read_jsonl(
            eval_root / "shared/downstream_transfer/arxiv_benchmark" / "test.jsonl"
        )
    }
    output: list[dict[str, Any]] = []
    for spec in SEQUENCE_PASSAGE_SPECS:
        row = dict(spec)
        source = str(row["source"])
        if source == "cross_feature_evidence":
            feature_id = int(row["feature_id"])
            if feature_id not in cross_evidence:
                raise ValueError(
                    f"missing Cross evidence for passage feature {feature_id}"
                )
            texts = cross_evidence[feature_id]
            evidence_index = int(row["evidence_index"])
            if not 0 <= evidence_index < len(texts):
                raise ValueError(
                    f"feature {feature_id} has no evidence index {evidence_index}"
                )
            text = str(texts[evidence_index])
            provenance = {
                "source": source,
                "feature_id": feature_id,
                "evidence_index": evidence_index,
            }
        elif source == "arxiv_test":
            record_id = str(row["record_id"])
            if record_id not in benchmark_rows:
                raise ValueError(f"missing ArXiv test record {record_id}")
            record = benchmark_rows[record_id]
            text = str(record["text"])
            provenance = {
                "source": source,
                "record_id": record_id,
                "label": str(record["label"]),
                "year": int(record["year"]),
                "categories": str(record["categories"]),
                "content_hash": str(record["content_hash"]),
            }
        else:
            raise ValueError(f"unknown sequence passage source: {source}")
        output.append(
            {
                "passage": str(row["passage"]),
                "short": str(row["short"]),
                "text": text,
                "provenance": provenance,
            }
        )
    return output


def _decoder_alignment(
    feature_ids: list[int],
    *,
    cross_checkpoint: Path,
    baselines: dict[str, tuple[Path, int | None]],
    block_size: int,
    device: torch.device,
    decoder_head: DecoderHead | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    cross_weight_name = _decoder_weight_name(cross_checkpoint, decoder_head)
    with safe_open(
        str(cross_checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        cross = (
            handle.get_slice(cross_weight_name)[:, feature_ids]
            .float()
            .to(device)
        )
    cross /= torch.linalg.vector_norm(
        cross,
        dim=0,
        keepdim=True,
    ).clamp_min(1e-12)
    output: dict[str, dict[str, np.ndarray]] = {}
    for mode, (checkpoint, width_limit) in baselines.items():
        best_values = torch.full(
            (len(feature_ids),),
            -1.0,
            device=device,
        )
        best_indices = torch.full(
            (len(feature_ids),),
            -1,
            dtype=torch.long,
            device=device,
        )
        with safe_open(
            str(checkpoint / "sae.safetensors"),
            framework="pt",
            device="cpu",
        ) as handle:
            decoder = handle.get_slice(
                # ``decoder_head`` selects only the target Cross/Joint
                # checkpoint. Baseline Token/Mean checkpoints retain their
                # legacy single decoder unless they independently declare a
                # Joint head.
                _decoder_weight_name(
                    checkpoint,
                    decoder_head
                    if checkpoint == cross_checkpoint
                    else None,
                )
            )
            width = int(decoder.get_shape()[1])
            if width_limit is not None:
                width = min(width, int(width_limit))
            for start in range(0, width, block_size):
                stop = min(width, start + block_size)
                block = decoder[:, start:stop].float().to(device)
                block /= torch.linalg.vector_norm(
                    block,
                    dim=0,
                    keepdim=True,
                ).clamp_min(1e-12)
                similarities = block.T @ cross
                values, indices = similarities.max(dim=0)
                replace = values > best_values
                best_values[replace] = values[replace]
                best_indices[replace] = indices[replace] + start
        output[mode] = {
            "feature_ids": best_indices.cpu().numpy().astype(np.int64),
            "cosine": best_values.cpu().numpy().astype(np.float64),
        }
    return output


def _load_or_build_population_alignment(
    *,
    eval_root: Path,
    feature_ids: list[int],
    baselines: dict[str, tuple[Path, int | None]],
    cross_checkpoint: Path,
    block_size: int,
    device: torch.device,
    decoder_head: DecoderHead | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    cache = eval_root / "feature_dynamics" / "cross_level4_decoder_alignment.npz"
    metadata_path = cache.with_suffix(".json")
    identity = {
        "feature_ids_sha256": hashlib.sha256(
            np.asarray(feature_ids, dtype=np.int64).tobytes()
        ).hexdigest(),
        "cross": _checkpoint_identity(cross_checkpoint),
        "baselines": {
            mode: {
                "checkpoint": _checkpoint_identity(checkpoint),
                "width_limit": width_limit,
            }
            for mode, (checkpoint, width_limit) in baselines.items()
        },
        "computation_device": str(device),
        "decoder_head": decoder_head,
    }
    if cache.is_file() and metadata_path.is_file():
        metadata = _read_json(metadata_path)
        if metadata.get("identity") == identity:
            with np.load(cache) as handle:
                return {
                    mode: {
                        "feature_ids": handle[f"{mode}_feature_ids"],
                        "cosine": handle[f"{mode}_cosine"],
                    }
                    for mode in metadata["methods"]
                }
    alignment = _decoder_alignment(
        feature_ids,
        cross_checkpoint=cross_checkpoint,
        baselines=baselines,
        block_size=block_size,
        device=device,
        decoder_head=decoder_head,
    )
    payload = {}
    for mode, row in alignment.items():
        payload[f"{mode}_feature_ids"] = row["feature_ids"]
        payload[f"{mode}_cosine"] = row["cosine"]
    np.savez_compressed(cache, **payload)
    metadata_path.write_text(
        json.dumps(
            {
                "format": "chunk-saes-cross-level4-alignment-v1",
                "identity": identity,
                "methods": list(alignment),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return alignment


def _patch_qwen_cpu_fallback() -> None:
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        if hasattr(qwen, name):
            setattr(qwen, name, None)
    if hasattr(qwen, "is_fast_path_available"):
        qwen.is_fast_path_available = False


@torch.inference_mode()
def _independent_sequence_views(
    extractor: Any,
    ids_by_passage: list[list[int]],
    windows: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not windows:
        raise ValueError("at least one sequence chunk length is required")
    if any(window <= 0 for window in windows):
        raise ValueError("sequence chunk lengths must be positive")
    passage_lengths = {len(ids) for ids in ids_by_passage}
    if len(passage_lengths) != 1:
        raise ValueError(
            "multiscale chunk visualization requires equal passage lengths"
        )
    passage_length = next(iter(passage_lengths))
    invalid = [
        window
        for window in windows
        if window > passage_length or passage_length % window != 0
    ]
    if invalid:
        raise ValueError(
            "displayed passage length must be divisible by every chunk "
            f"length; passage_length={passage_length}, invalid={invalid}"
        )
    token_views: dict[str, torch.Tensor] = {}
    chunk_views: dict[str, torch.Tensor] = {}
    for window in windows:
        token_hidden, chunk_hidden = _independent_sequence_view(
            extractor,
            ids_by_passage,
            window,
        )
        token_views[f"L{window}"] = token_hidden
        chunk_views[f"L{window}"] = chunk_hidden
    return token_views, chunk_views


@torch.inference_mode()
def _independent_sequence_view(
    extractor: Any,
    ids_by_passage: list[list[int]],
    chunk_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = [
        ids[start : start + chunk_length]
        for ids in ids_by_passage
        for start in range(0, len(ids), chunk_length)
    ]
    batch = extractor.forward_ids(chunks)
    token_hidden = batch.hidden.float().reshape(
        -1,
        batch.hidden.shape[-1],
    )
    means = batch.hidden.float().mean(dim=1)
    return token_hidden, means


def _normalize_feature_traces(traces: np.ndarray) -> np.ndarray:
    values = traces.astype(np.float64)
    return values / np.maximum(values.max(axis=1, keepdims=True), 1e-12)


@torch.inference_mode()
def _scan_top_features(
    checkpoint: Path,
    hidden_views: dict[str, torch.Tensor],
    *,
    token_level: bool,
    primary_view: str,
    feature_width: int,
    top_k: int,
    block_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if top_k <= 0:
        raise ValueError("sequence top-k must be positive")
    if block_size <= 0:
        raise ValueError("encoder block size must be positive")
    if not hidden_views:
        raise ValueError("feature scan requires at least one activation view")
    if primary_view not in hidden_views:
        raise ValueError(f"missing primary feature-selection view {primary_view}")
    config = _read_json(checkpoint / "config.json")
    with safe_open(
        str(checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        encoder = handle.get_slice("encoder_weight")
        dictionary_width = int(encoder.get_shape()[0])
        if int(feature_width) != dictionary_width:
            raise ValueError(
                f"{checkpoint} exposes dictionary width {dictionary_width}, "
                f"expected the complete width {feature_width}"
            )
        width = dictionary_width
        if width < top_k:
            raise ValueError(
                f"{checkpoint} exposes only {width} features for top-k={top_k}"
            )
        names = set(handle.keys())
        if "pre_bias" in names:
            pre_bias = handle.get_tensor("pre_bias").to(
                device=device,
                dtype=torch.float32,
            )
        elif (
            config.get("sae_parameter_schema_version")
            == SAE_PARAMETER_SCHEMA_VERSION
        ):
            raise ValueError(f"{checkpoint} lacks required pre_bias")
        else:
            pre_bias = handle.get_tensor("decoder_bias").to(
                device=device,
                dtype=torch.float32,
            )
        scale = float(handle.get_tensor("activation_scale"))
        threshold = float(handle.get_tensor("threshold"))
        centered_views = {
            name: (
                hidden.to(device=device, dtype=torch.float32) * scale
                - pre_bias
            )
            for name, hidden in hidden_views.items()
        }
        candidates: list[
            tuple[float, int, np.ndarray, dict[str, np.ndarray]]
        ] = []
        for start in range(0, width, block_size):
            stop = min(width, start + block_size)
            weight = encoder[start:stop, :].to(
                device=device,
                dtype=torch.float32,
            )
            bias = handle.get_slice("encoder_bias")[start:stop].to(
                device=device,
                dtype=torch.float32,
            )
            values_by_view = {}
            for name, centered in centered_views.items():
                values = F.relu(F.linear(centered, weight, bias))
                values *= values > threshold
                if token_level:
                    chunk_length = int(name.removeprefix("L"))
                    if values.shape[0] % chunk_length:
                        raise ValueError(
                            f"{name} token rows do not form complete chunks"
                        )
                    values = values.reshape(
                        -1,
                        chunk_length,
                        values.shape[-1],
                    ).mean(dim=1)
                values_by_view[name] = values
            scores = values_by_view[primary_view].mean(dim=0)
            local_k = min(top_k, stop - start)
            local_scores, local_indices = torch.topk(
                scores,
                local_k,
                sorted=False,
            )
            local_by_view = {
                name: values[:, local_indices].T.float().cpu().numpy()
                for name, values in values_by_view.items()
            }
            local_traces = local_by_view[primary_view]
            for candidate_index, (score, local_index, trace) in enumerate(zip(
                local_scores.float().cpu().tolist(),
                local_indices.cpu().tolist(),
                local_traces,
                strict=True,
            )):
                candidates.append(
                    (
                        float(score),
                        start + int(local_index),
                        trace,
                        {
                            name: traces[candidate_index]
                            for name, traces in local_by_view.items()
                        },
                    )
                )
            del weight, bias, values_by_view, scores
    candidates.sort(key=lambda row: (-row[0], row[1]))
    selected = candidates[:top_k]
    return (
        np.asarray([row[1] for row in selected], dtype=np.int64),
        np.asarray([row[0] for row in selected], dtype=np.float64),
        np.stack([row[2] for row in selected]).astype(np.float32),
        {
            name: np.stack(
                [row[3][name] for row in selected]
            ).astype(np.float32)
            for name in hidden_views
        },
    )


def _top_feature_statistics(
    traces: np.ndarray,
    bounds: np.ndarray,
) -> dict[str, Any]:
    normalized = np.clip(traces.astype(np.float64), 0.0, None)
    segment_means = np.asarray(
        [
            [
                float(trace[bounds[index] : bounds[index + 1]].mean())
                for index in range(len(bounds) - 1)
            ]
            for trace in normalized
        ],
        dtype=np.float64,
    )
    dominant = segment_means.argmax(axis=1)
    per_feature = []
    selective_passages: set[int] = set()
    all_roughness = []
    dominant_margins = []
    for feature_index, trace in enumerate(normalized):
        dominant_index = int(dominant[feature_index])
        means = segment_means[feature_index]
        other_max = float(np.delete(means, dominant_index).max())
        margin = float(means[dominant_index] - other_max)
        dominant_margins.append(margin)
        within_passage = []
        for passage_index in range(len(bounds) - 1):
            segment = trace[
                bounds[passage_index] : bounds[passage_index + 1]
            ]
            roughness = (
                float(np.abs(np.diff(segment)).mean())
                if segment.size > 1
                else 0.0
            )
            within_passage.append(roughness)
            all_roughness.append(roughness)
        dominant_segment = trace[
            bounds[dominant_index] : bounds[dominant_index + 1]
        ]
        active_fraction = float(np.mean(dominant_segment > 0))
        selective = bool(
            means[dominant_index] >= 0.10
            and means[dominant_index] >= 2.0 * max(other_max, 1e-8)
            and active_fraction >= 0.15
        )
        if selective:
            selective_passages.add(dominant_index)
        per_feature.append(
            {
                "rank": feature_index + 1,
                "dominant_passage_index": dominant_index,
                "segment_means": means.tolist(),
                "dominant_passage_margin": margin,
                "dominant_active_fraction": active_fraction,
                "within_passage_total_variation": within_passage,
                "selective": selective,
            }
        )
    return {
        "dominant_passages_covered": int(np.unique(dominant).size),
        "total_passages": int(len(bounds) - 1),
        "selective_passages_covered": int(len(selective_passages)),
        "selective_feature_count": int(
            sum(bool(row["selective"]) for row in per_feature)
        ),
        "mean_dominant_passage_margin": float(
            np.mean(dominant_margins)
        ),
        "mean_within_passage_total_variation": float(
            np.mean(all_roughness)
        ),
        "per_feature": per_feature,
    }


def _auto_curve_explanation(
    row: dict[str, Any],
    passages: list[dict[str, Any]],
) -> str:
    means = np.asarray(row["segment_means"], dtype=np.float64)
    order = np.argsort(means)[::-1]
    primary = int(order[0])
    secondary = int(order[1])
    primary_passage = str(passages[primary]["passage"])
    secondary_passage = str(passages[secondary]["passage"])
    primary_label = SEQUENCE_PASSAGE_FEATURE_LABELS.get(
        primary_passage,
        f"Feature concentrated on the displayed {primary_passage.lower()} passage",
    )
    primary_context = SEQUENCE_PASSAGE_CONTEXT_LABELS.get(
        primary_passage,
        f"{primary_passage.lower()} text",
    )
    secondary_context = SEQUENCE_PASSAGE_CONTEXT_LABELS.get(
        secondary_passage,
        f"{secondary_passage.lower()} text",
    )
    if means[primary] <= 1e-8:
        return "No meaningful activation across displayed passages"
    if (
        float(row["dominant_passage_margin"]) >= 0.18
        or means[primary] >= 1.8 * max(means[secondary], 1e-8)
    ):
        return primary_label
    if means[secondary] >= 0.65 * means[primary]:
        return (
            f"Shared {primary_context} and {secondary_context} text patterns"
        )
    return f"Broad activation across related {primary_context} text"


def _sequence_cache_identity(
    *,
    eval_root: Path,
    passages: list[dict[str, Any]],
    checkpoints: dict[str, Path],
    temporal_checkpoint: Path | None,
    top_k: int,
    windows: list[int],
    display_chunk_length: int,
    max_length: int,
    block_size: int,
    device: str,
) -> dict[str, Any]:
    identity = _evaluation_identity(eval_root)
    temporal_complete_path = (
        temporal_checkpoint.parent.parent / "complete.json"
        if temporal_checkpoint is not None
        else None
    )
    return {
        "protocol_version": "model-symmetric-own-top-features-v6",
        "model": identity["model"],
        "layer": identity["layer"],
        "passages_sha256": hashlib.sha256(
            json.dumps(passages, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "top_k": int(top_k),
        "chunk_lengths": [int(window) for window in windows],
        "display_chunk_length": int(display_chunk_length),
        "robustness_chunk_lengths": [
            int(window)
            for window in windows
            if int(window) != int(display_chunk_length)
        ],
        "max_length": int(max_length),
        "encoder_block_size": int(block_size),
        "device": str(device),
        "temporal_training_complete": bool(
            temporal_complete_path is not None
            and temporal_complete_path.is_file()
        ),
        "temporal_complete_marker": (
            {
                "bytes": temporal_complete_path.stat().st_size,
                "mtime_ns": temporal_complete_path.stat().st_mtime_ns,
            }
            if temporal_complete_path is not None
            and temporal_complete_path.is_file()
            else None
        ),
        "checkpoints": {
            **{
                mode: _checkpoint_identity(checkpoint)
                for mode, checkpoint in checkpoints.items()
            },
            "temporal": _checkpoint_identity(temporal_checkpoint),
        },
    }


def _load_or_build_sequence(
    *,
    eval_root: Path,
    passages: list[dict[str, Any]],
    checkpoints: dict[str, Path],
    temporal_checkpoint: Path | None,
    temporal_completed: bool,
    top_k: int,
    windows: list[int],
    display_chunk_length: int,
    max_length: int,
    block_size: int,
    device: str,
    refresh: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache = eval_root / "feature_dynamics" / "sequence_activation_traces.npz"
    metadata_path = cache.with_suffix(".json")
    identity = _sequence_cache_identity(
        eval_root=eval_root,
        passages=passages,
        checkpoints=checkpoints,
        temporal_checkpoint=temporal_checkpoint,
        top_k=top_k,
        windows=windows,
        display_chunk_length=display_chunk_length,
        max_length=max_length,
        block_size=block_size,
        device=device,
    )
    if (
        not refresh
        and cache.is_file()
        and metadata_path.is_file()
    ):
        metadata = _read_json(metadata_path)
        if metadata.get("identity") == identity:
            with np.load(cache, allow_pickle=True) as handle:
                arrays = {key: handle[key] for key in handle.files}
            return arrays, metadata

    sequence_device = torch.device(device)
    if sequence_device.type == "cpu":
        _patch_qwen_cpu_fallback()
    from chunk_saes.modeling import TargetLayerExtractor

    identity = _evaluation_identity(eval_root)
    extractor = TargetLayerExtractor(
        str(identity["model"]),
        int(identity["layer"]),
        str(sequence_device),
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    ids_by_passage = []
    for index, passage in enumerate(passages):
        text = str(passage["text"])
        if index:
            text = "\n\n" + text
        ids_by_passage.append(
            extractor.tokenizer.encode(
                text,
                add_special_tokens=False,
            )[:max_length]
        )
    full_ids = [
        token_id
        for passage_ids in ids_by_passage
        for token_id in passage_ids
    ]
    token_views, chunk_views = _independent_sequence_views(
        extractor,
        ids_by_passage,
        windows,
    )
    extractor.close()
    del extractor
    gc.collect()
    if sequence_device.type == "cuda":
        torch.cuda.empty_cache()

    token_bounds = np.cumsum(
        [0] + [len(ids) for ids in ids_by_passage]
    ).astype(np.int64)
    view_bounds = {
        f"L{window}": np.cumsum(
            [0]
            + [
                len(ids) // int(window)
                for ids in ids_by_passage
            ]
        ).astype(np.int64)
        for window in windows
    }
    primary_view = f"L{display_chunk_length}"
    if primary_view not in view_bounds:
        raise ValueError(
            "display chunk length must be included in sequence chunk lengths; "
            f"display={display_chunk_length}, lengths={windows}"
        )
    display_bounds = view_bounds[primary_view]
    qualitative_checkpoints: dict[str, Path] = {
        "token": checkpoints["token"],
    }
    if temporal_checkpoint is not None:
        qualitative_checkpoints["temporal"] = temporal_checkpoint
    qualitative_checkpoints["mean"] = checkpoints["mean"]
    qualitative_checkpoints["cross"] = checkpoints["cross"]

    arrays: dict[str, np.ndarray] = {
        "bounds": display_bounds,
        "token_bounds": token_bounds,
        "display_chunk_length": np.asarray(
            display_chunk_length,
            dtype=np.int64,
        ),
        "input_ids": np.asarray(full_ids, dtype=np.int64),
    }
    method_metadata: dict[str, Any] = {}
    for mode in METHODS:
        if mode not in qualitative_checkpoints:
            continue
        hidden_views = (
            token_views
            if mode in {"token", "temporal"}
            else chunk_views
        )
        config = _read_json(
            qualitative_checkpoints[mode] / "config.json"
        )
        feature_width = int(config["dict_size"])
        token_level = mode in {"token", "temporal"}
        (
            feature_ids,
            mean_activations,
            primary_raw_traces,
            traces_by_view,
        ) = _scan_top_features(
            qualitative_checkpoints[mode],
            hidden_views,
            token_level=token_level,
            primary_view=primary_view,
            feature_width=feature_width,
            top_k=top_k,
            block_size=block_size,
            device=sequence_device,
        )
        traces = _normalize_feature_traces(primary_raw_traces)
        arrays[f"{mode}_traces"] = traces
        arrays[f"{mode}_feature_ids"] = feature_ids
        arrays[f"{mode}_mean_activations"] = mean_activations
        for view_name, view_traces in traces_by_view.items():
            arrays[f"{mode}_{view_name}_traces"] = view_traces
        statistics = _top_feature_statistics(traces, display_bounds)
        for feature_id, row in zip(
            feature_ids.tolist(),
            statistics["per_feature"],
            strict=True,
        ):
            row["feature_id"] = int(feature_id)
            row["auto_explanation"] = _auto_curve_explanation(
                row,
                passages,
            )
        method_metadata[mode] = {
            "checkpoint": _checkpoint_identity(
                qualitative_checkpoints[mode]
            ),
            "dictionary_feature_width": feature_width,
            "selected_feature_ids": feature_ids.tolist(),
            "ranking_mean_activations": mean_activations.tolist(),
            "selection_view": primary_view,
            "display_observations": int(display_bounds[-1]),
            "token_level_before_chunk_aggregation": token_level,
            "activation_views": list(traces_by_view),
            "robustness_views": [
                view_name
                for view_name in traces_by_view
                if view_name != primary_view
            ],
            "per_view_statistics": {
                view_name: _top_feature_statistics(
                    _normalize_feature_traces(view_traces),
                    view_bounds[view_name],
                )
                for view_name, view_traces in traces_by_view.items()
            },
            "auto_explanation_basis": (
                "dominant and runner-up passage activation means on the "
                "displayed sequence; labels are descriptive summaries, not "
                "blind LLM interpretations"
            ),
            **statistics,
        }

    temporal_manifest = None
    if temporal_checkpoint is not None:
        manifest_path = temporal_checkpoint / "checkpoint_manifest.json"
        if manifest_path.is_file():
            temporal_manifest = _read_json(manifest_path)
    metadata = {
        "format": "chunk-saes-sequence-activation-traces-v6",
        "identity": identity,
        "passages": [
            {
                "passage": passage["passage"],
                "short": passage["short"],
                "text_sha256": hashlib.sha256(
                    str(passage["text"]).encode("utf-8")
                ).hexdigest(),
                "provenance": passage["provenance"],
                "tokens": int(
                    token_bounds[index + 1] - token_bounds[index]
                ),
                "display_chunks": int(
                    display_bounds[index + 1] - display_bounds[index]
                ),
            }
            for index, passage in enumerate(passages)
        ],
        "methods": method_metadata,
        "protocol": {
            "shared_text": True,
            "shared_feature_count": int(top_k),
            "independent_chunk_forwards": True,
            "position_policy": "reset to zero for every independent chunk",
            "feature_selection_anchor": None,
            "common_observation_unit": (
                "one independently forwarded "
                f"{display_chunk_length}-token chunk"
            ),
            "display_chunk_length": int(display_chunk_length),
            "display_chunks": int(display_bounds[-1]),
            "chunks_per_passage": int(
                display_bounds[1] - display_bounds[0]
            ),
            "feature_ranking": (
                "each SAE independently ranks its eligible dictionary by "
                f"mean thresholded activation over the {int(display_bounds[-1])} "
                f"aligned {primary_view} chunks"
            ),
            "eligible_dictionary": (
                "full trained dictionary for all four SAEs"
            ),
            "token_method_aggregation": (
                "for Token and Temporal, encode every token hidden state, "
                "apply the checkpoint threshold, then average each feature's "
                f"post-threshold activations within the same {primary_view} chunk"
            ),
            "chunk_method_aggregation": (
                "for Mean and Cross, average hidden states within the exact "
                f"same {primary_view} chunk, then encode and threshold once"
            ),
            "trace_normalization": (
                f"divide each selected feature by its maximum over the "
                f"{int(display_bounds[-1])} displayed {primary_view} chunks"
            ),
            "main_plot_resolution": (
                f"all methods have exactly {int(display_bounds[-1])} plotted "
                "chunk observations per feature; no chunk value is expanded "
                "or duplicated over token positions"
            ),
            "robustness_chunk_lengths": [
                int(window)
                for window in windows
                if int(window) != int(display_chunk_length)
            ],
            "robustness_role": (
                "cached with separately normalized traces and correctly sized "
                "passage bounds; excluded from main ranking and plotting"
            ),
            "reported_selective_feature_rule": (
                "dominant passage mean >= 0.10; dominant mean >= 2x every "
                "other passage; dominant active fraction >= 0.15"
            ),
        },
        "temporal": {
            "training_complete": temporal_completed,
            "qualitative_partial_checkpoint": (
                temporal_checkpoint is not None and not temporal_completed
            ),
            "checkpoint_step": (
                temporal_manifest.get("step")
                if temporal_manifest
                else None
            ),
            "checkpoint_samples_seen": (
                temporal_manifest.get("samples_seen")
                if temporal_manifest
                else None
            ),
        },
    }
    np.savez_compressed(cache, **arrays)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return arrays, metadata


def _sparse_matrix(
    arrays: dict[str, np.ndarray],
    prefix: str,
    width: int,
    *,
    max_nnz: int | None = None,
) -> sparse.csr_matrix:
    indices = arrays[f"{prefix}_indices"]
    values = arrays[f"{prefix}_values"].astype(np.float64, copy=False)
    nnz = arrays[f"{prefix}_nnz"]
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for row, count in enumerate(nnz):
        take = int(count)
        if max_nnz is not None:
            take = min(take, int(max_nnz))
        row_indices = indices[row, :take]
        row_values = values[row, :take]
        valid = (row_indices >= 0) & (row_values > 0)
        rows.extend([row] * int(valid.sum()))
        columns.extend(row_indices[valid].tolist())
        data.extend(row_values[valid].tolist())
    return sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(indices.shape[0], width),
    )


def _formal_domain_neighbor_purity(
    *,
    eval_root: Path,
    test_arrays: dict[str, np.ndarray],
    probes: dict[str, Any],
    geometry: dict[str, Any],
    feature_widths: dict[str, int],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    geometry_manifest = _read_json(
        eval_root
        / "semantic_geometry"
        / "representation_geometry"
        / "representation_geometry_manifest.json"
    )
    formal_identity = dict(geometry_manifest["identity"])
    formal_seed = int(formal_identity["seed"])
    formal_components = int(formal_identity["svd_components"])
    neighbors = int(formal_identity["neighbors"])
    token_eval_k = int(
        probes["metadata"]["sparsity_matching"]["token_eval_k"]
    )
    token_aggregation = str(probes["chosen_token_aggregation"])
    prefix_by_mode = {
        "token": f"token_{token_aggregation}",
        "temporal": "temporal_mean",
        "mean": "mean_direct",
        "cross": "cross_direct",
    }
    ood_path = (
        eval_root / "shared/downstream_transfer/probe_features" / "features-ood.npz"
    )
    with np.load(ood_path, allow_pickle=True) as handle:
        ood_arrays = {key: handle[key] for key in handle.files}
    labels = test_arrays["labels"].astype(str)
    available_modes = [
        mode
        for mode in METHODS
        if (
            f"{prefix_by_mode[mode]}_indices" in test_arrays
            and mode in geometry.get("methods", {})
        )
    ]
    output: dict[str, dict[str, float]] = {}
    for mode_index, mode in enumerate(available_modes):
        prefix = prefix_by_mode[mode]
        maximum = token_eval_k if mode in {"token", "temporal"} else None
        test_matrix = _sparse_matrix(
            test_arrays,
            prefix,
            feature_widths.get(mode, 65_536),
            max_nnz=maximum,
        )
        ood_matrix = _sparse_matrix(
            ood_arrays,
            prefix,
            feature_widths.get(mode, 65_536),
            max_nnz=maximum,
        )
        combined = sparse.vstack(
            (test_matrix, ood_matrix),
            format="csr",
        )
        combined = normalize(combined, norm="l2", copy=False)
        components = min(
            formal_components,
            combined.shape[0] - 1,
            combined.shape[1] - 1,
        )
        reduced = TruncatedSVD(
            n_components=components,
            n_iter=7,
            random_state=formal_seed + mode_index,
        ).fit_transform(combined)
        test_reduced = reduced[: labels.size]
        index = NearestNeighbors(
            n_neighbors=neighbors + 1,
            metric="cosine",
        ).fit(test_reduced)
        indices = index.kneighbors(return_distance=False)[:, 1:]
        neighbor_labels = labels[indices]
        overall = float((neighbor_labels == labels[:, None]).mean())
        published = float(
            geometry["methods"][mode]["geometry"]["neighbor_purity"]
        )
        if not np.isclose(overall, published, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"formal 10-NN reproduction mismatch for {mode}: "
                f"computed={overall}, published={published}"
            )
        output[mode] = {
            label: float(
                (
                    neighbor_labels[labels == label]
                    == label
                ).mean()
            )
            for label in DOMAIN_COLORS
        }
    return output, {
        "space": f"{formal_components}-D TruncatedSVD",
        "neighbors": neighbors,
        "metric": "cosine",
        "leave_one_out": True,
        "fit_scope": "joint test+OOD; purity evaluated on test",
        "seed": formal_seed,
        "source_artifact_digest": geometry_manifest["artifact_digest"],
    }


def _domain_comparisons(
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = []
    baselines = tuple(mode for mode in METHODS if mode != "cross")
    for domain in DOMAIN_COLORS:
        if not all(
            domain in metrics.get(mode, {}).get(
                "domain_neighbor_purity",
                {},
            )
            for mode in METHODS
        ):
            continue
        baseline_values = {
            mode: float(
                metrics[mode]["domain_neighbor_purity"][domain]
            )
            for mode in baselines
        }
        strongest_mode = max(
            baseline_values,
            key=baseline_values.__getitem__,
        )
        cross_value = float(
            metrics["cross"]["domain_neighbor_purity"][domain]
        )
        comparisons.append(
            {
                "domain": domain,
                "short": DOMAIN_SHORT[domain],
                "cross_neighbor_purity": cross_value,
                "strongest_baseline": strongest_mode,
                "strongest_baseline_neighbor_purity": baseline_values[
                    strongest_mode
                ],
                "cross_margin_over_strongest_baseline": (
                    cross_value - baseline_values[strongest_mode]
                ),
                "methods": {
                    mode: float(
                        metrics[mode]["domain_neighbor_purity"][domain]
                    )
                    for mode in METHODS
                },
            }
        )
    comparisons.sort(
        key=lambda row: (
            -float(row["cross_margin_over_strongest_baseline"]),
            str(row["domain"]),
        )
    )
    return comparisons


def _embedding_cache_identity(
    *,
    feature_path: Path,
    geometry_path: Path,
    perplexity: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "feature_mtime_ns": feature_path.stat().st_mtime_ns,
        "feature_bytes": feature_path.stat().st_size,
        "geometry_mtime_ns": geometry_path.stat().st_mtime_ns,
        "perplexity": float(perplexity),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def _load_or_build_embeddings(
    *,
    eval_root: Path,
    perplexity: float,
    iterations: int,
    seed: int,
    refresh: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    feature_path = eval_root / "shared/downstream_transfer/probe_features" / "features-test.npz"
    geometry_path = (
        eval_root
        / "semantic_geometry"
        / "representation_geometry"
        / "representation_geometry.json"
    )
    geometry = _read_json(geometry_path)
    cache_path = (
        eval_root
        / "semantic_geometry"
        / "representation_geometry"
        / "representation_display_embeddings.npz"
    )
    cache_meta_path = cache_path.with_suffix(".json")
    identity = _embedding_cache_identity(
        feature_path=feature_path,
        geometry_path=geometry_path,
        perplexity=perplexity,
        iterations=iterations,
        seed=seed,
    )
    with np.load(feature_path, allow_pickle=True) as handle:
        arrays = {key: handle[key] for key in handle.files}
    labels = arrays["labels"].astype(str)
    probes = _read_json(eval_root / "label_efficiency" / "linear_probe_results.json")
    token_eval_k = int(
        probes["metadata"]["sparsity_matching"]["token_eval_k"]
    )
    feature_widths = {
        str(mode): int(width)
        for mode, width in probes["metadata"].get(
            "feature_widths",
            {},
        ).items()
    }
    prefix_by_mode = {
        "token": "token_mean",
        "temporal": "temporal_mean",
        "mean": "mean_direct",
        "cross": "cross_direct",
    }
    matrices = {}
    for mode, prefix in prefix_by_mode.items():
        if f"{prefix}_indices" not in arrays:
            continue
        matrices[mode] = _sparse_matrix(
            arrays,
            prefix,
            feature_widths.get(mode, 65_536),
            max_nnz=token_eval_k,
        )

    cache_valid = (
        not refresh
        and cache_path.is_file()
        and cache_meta_path.is_file()
        and _read_json(cache_meta_path).get("identity") == identity
    )
    if cache_valid:
        with np.load(cache_path, allow_pickle=True) as handle:
            payload = {key: handle[key] for key in handle.files}
        payload["labels"] = labels
    else:
        payload: dict[str, np.ndarray] = {"labels": labels}
        for mode in METHODS:
            if mode not in matrices or mode not in geometry.get("methods", {}):
                continue
            normalized = normalize(matrices[mode], norm="l2", copy=True)
            reduced = TruncatedSVD(
                n_components=50,
                n_iter=7,
                random_state=seed,
            ).fit_transform(normalized)
            xy = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate="auto",
                init="pca",
                max_iter=iterations,
                random_state=seed,
            ).fit_transform(reduced)
            payload[f"{mode}_xy"] = xy.astype(np.float32)
        np.savez_compressed(cache_path, **payload)
        cache_meta_path.write_text(
            json.dumps(
                {
                    "format": "chunk-saes-display-tsne-v1",
                    "identity": identity,
                    "methods": [
                        mode
                        for mode in METHODS
                        if f"{mode}_xy" in payload
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    metrics = {
        mode: {
            "neighbor_purity": float(
                geometry["methods"][mode]["geometry"]["neighbor_purity"]
            )
        }
        for mode in METHODS
        if mode in geometry.get("methods", {})
        and f"{mode}_xy" in payload
    }
    domain_purity, domain_protocol = _formal_domain_neighbor_purity(
        eval_root=eval_root,
        test_arrays=arrays,
        probes=probes,
        geometry=geometry,
        feature_widths=feature_widths,
    )
    for mode, values in domain_purity.items():
        if mode in metrics:
            metrics[mode]["domain_neighbor_purity"] = values
            metrics[mode]["domain_neighbor_purity_protocol"] = domain_protocol
    return payload, metrics


def _robust_ellipse(
    points: np.ndarray,
    *,
    scale: float = 1.15,
) -> tuple[np.ndarray, float, float, float]:
    center = np.median(points, axis=0)
    centered = points - center
    distances = np.linalg.norm(centered, axis=1)
    keep = distances <= np.quantile(distances, 0.92)
    core = points[keep] if int(keep.sum()) >= 4 else points
    center = np.median(core, axis=0)
    covariance = np.cov(core.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 1e-8)
    vectors = vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    width = 2.0 * scale * math.sqrt(float(values[0]))
    height = 2.0 * scale * math.sqrt(float(values[1]))
    return center, width, height, angle


def _draw_sequence_panel(
    fig: plt.Figure,
    spec,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    feature_annotations: dict[str, dict[int, str]],
    methods: Sequence[str] = METHODS,
    method_labels: Mapping[str, str] = METHOD_LABELS,
    method_colors: Mapping[str, str] = METHOD_COLORS,
    method_pale_colors: Mapping[str, str] = METHOD_PALE_COLORS,
    method_text_colors: Mapping[str, str] = METHOD_TEXT_COLORS,
    panel_title: str = "A   Own top-5 features on identical text",
    x_axis_label: str = "Text position",
    line_width_scale: float = 1.0,
    feature_annotation_fontsize: float = 11.2,
    sequence_width_ratios: tuple[float, float] = (0.64, 0.36),
    panel_title_fontsize: float = 18.5,
    panel_title_y: float = 1.25,
    x_axis_label_fontsize: float = 17.0,
    y_axis_label_fontsize: float = 16.5,
    x_tick_fontsize: float = 15.5,
    feature_panel_title_fontsize: float = 15.5,
    rank_legend_fontsize: float = 11.5,
    rank_legend_x: float = 0.56,
    rank_legend_y: float = 1.34,
    rank_legend_loc: str = "center left",
    method_label_y: float = 0.78,
    method_label_vertical_alignment: str = "center",
) -> list[plt.Axes]:
    methods = tuple(methods)
    if not methods:
        raise ValueError("sequence panel requires at least one method")
    if line_width_scale <= 0:
        raise ValueError("line_width_scale must be positive")
    if feature_annotation_fontsize <= 0:
        raise ValueError("feature_annotation_fontsize must be positive")
    if len(sequence_width_ratios) != 2 or any(
        value <= 0 for value in sequence_width_ratios
    ):
        raise ValueError("sequence_width_ratios must contain two positive values")
    if panel_title_fontsize <= 0:
        raise ValueError("panel_title_fontsize must be positive")
    if x_axis_label_fontsize <= 0 or y_axis_label_fontsize <= 0:
        raise ValueError("axis label font sizes must be positive")
    if x_tick_fontsize <= 0:
        raise ValueError("x_tick_fontsize must be positive")
    if feature_panel_title_fontsize <= 0:
        raise ValueError("feature_panel_title_fontsize must be positive")
    if rank_legend_fontsize <= 0:
        raise ValueError("rank_legend_fontsize must be positive")
    grid = spec.subgridspec(
        len(methods),
        2,
        width_ratios=sequence_width_ratios,
        hspace=0.105,
        wspace=0.025,
    )
    axes = [
        fig.add_subplot(grid[index, 0])
        for index in range(len(methods))
    ]
    feature_axes = [
        fig.add_subplot(grid[index, 1])
        for index in range(len(methods))
    ]
    bounds = arrays["bounds"].astype(np.int64)
    total_chunks = int(bounds[-1])
    display_chunk_length = int(
        np.asarray(arrays["display_chunk_length"]).item()
    )
    annotation_font = FontProperties(
        family="DejaVu Sans",
        weight="bold",
    )

    def fitted_annotation_fontsize(
        text: str,
        feature_ax: plt.Axes,
    ) -> float:
        text_width = TextPath(
            (0.0, 0.0),
            text,
            size=1.0,
            prop=annotation_font,
        ).get_extents().width
        if text_width <= 0:
            return feature_annotation_fontsize
        available_width = (
            fig.get_figwidth()
            * 72.0
            * feature_ax.get_position().width
            * (0.94 - 0.125)
        )
        return min(
            feature_annotation_fontsize,
            0.95 * available_width / text_width,
        )

    x = np.arange(total_chunks, dtype=np.float64) + 0.5
    passage_faces = ("#EEF3F8", "#FAF7F0", "#F2F7F4", "#F7F2F8")
    for row_index, (ax, feature_ax, mode) in enumerate(
        zip(axes, feature_axes, methods, strict=True)
    ):
        feature_ax.set_axis_off()
        feature_ax.set_xlim(0, 1)
        feature_ax.set_ylim(0, 1)
        feature_ax.add_patch(
            FancyBboxPatch(
                (0.01, 0.035),
                0.98,
                0.93,
                transform=feature_ax.transAxes,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor=method_pale_colors[mode],
                edgecolor=method_colors[mode],
                linewidth=1.5 * line_width_scale,
                alpha=0.96,
                clip_on=False,
                zorder=0,
            )
        )
        feature_ax.add_patch(
            Rectangle(
                (0.025, 0.11),
                0.010,
                0.78,
                transform=feature_ax.transAxes,
                facecolor=method_colors[mode],
                edgecolor="none",
                alpha=0.95,
                zorder=1,
            )
        )
        for passage_index in range(len(bounds) - 1):
            ax.axvspan(
                bounds[passage_index],
                bounds[passage_index + 1],
                color=passage_faces[passage_index % len(passage_faces)],
                alpha=0.88,
                zorder=0,
            )
        for boundary in bounds[1:-1]:
            ax.axvline(
                boundary,
                color="#AAB2BF",
                lw=0.8 * line_width_scale,
                ls="--",
                zorder=1,
            )
        trace_key = f"{mode}_traces"
        if trace_key in arrays:
            traces = arrays[trace_key].astype(np.float64)
            feature_ids = arrays[f"{mode}_feature_ids"].astype(np.int64)
            method_stats = metadata["methods"][mode]
            feature_rows = method_stats["per_feature"]
            for rank, (trace, feature_id, feature_row) in enumerate(
                zip(traces, feature_ids, feature_rows, strict=True)
            ):
                if trace.size != total_chunks:
                    raise ValueError(
                        f"{mode} feature #{int(feature_id)} has "
                        f"{trace.size} observations; expected {total_chunks}"
                    )
                color = FEATURE_RANK_COLORS[rank % len(FEATURE_RANK_COLORS)]
                ax.plot(
                    x,
                    trace,
                    color=color,
                    lw=2.25 * line_width_scale,
                    alpha=0.96,
                    marker="o",
                    markersize=4.4,
                    markeredgewidth=0.45 * line_width_scale,
                    markeredgecolor=WHITE,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=3,
                )
                feature_ax.text(
                    0.075,
                    0.90 - 0.20 * rank,
                    str(rank + 1),
                    transform=feature_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    fontweight="heavy",
                    color=WHITE,
                    bbox={
                        "boxstyle": "circle,pad=0.22",
                        "facecolor": color,
                        "edgecolor": WHITE,
                        "linewidth": 1.0,
                        "alpha": 1.0,
                    },
                    zorder=3,
                )
                annotation = feature_annotations.get(mode, {}).get(
                    int(feature_id),
                    f"feature {int(feature_id)}",
                )
                feature_ax.text(
                    0.125,
                    0.90 - 0.20 * rank,
                    annotation,
                    transform=feature_ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=fitted_annotation_fontsize(
                        annotation,
                        feature_ax,
                    ),
                    fontweight="bold",
                    color=DARK,
                    clip_on=True,
                    zorder=3,
                )

            # Emphasize the feature with the highest mean activation in each
            # passage while preserving every base curve underneath.
            for start, end in zip(bounds[:-1], bounds[1:], strict=True):
                start = int(start)
                end = int(end)
                winner_rank = int(
                    np.argmax(traces[:, start:end].mean(axis=1))
                )
                winner_trace = traces[winner_rank, start:end]
                winner_color = FEATURE_RANK_COLORS[
                    winner_rank % len(FEATURE_RANK_COLORS)
                ]
                ax.plot(
                    x[start:end],
                    winner_trace,
                    color=winner_color,
                    lw=8.0 * line_width_scale,
                    alpha=0.18,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=4,
                )
                ax.plot(
                    x[start:end],
                    winner_trace,
                    color=winner_color,
                    lw=3.8 * line_width_scale,
                    alpha=1.0,
                    marker="o",
                    markersize=5.4,
                    markeredgewidth=0.75 * line_width_scale,
                    markeredgecolor=WHITE,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=4.1,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "checkpoint unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MID,
            )
        ax.text(
            0.012,
            method_label_y,
            method_labels[mode],
            transform=ax.transAxes,
            ha="left",
            va=method_label_vertical_alignment,
            fontsize=12.0,
            fontweight="bold",
            color=method_text_colors[mode],
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": method_pale_colors[mode],
                "edgecolor": "none",
                "alpha": 0.94,
            },
            zorder=5,
        )
        ax.set_xlim(0, total_chunks)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=12.0, length=3.2, pad=4)
        ax.set_xticks([])
        if row_index == (len(methods) - 1) // 2:
            ax.set_ylabel(
                "Feature activation (0–1)",
                fontsize=y_axis_label_fontsize,
                fontweight="heavy",
                labelpad=16,
            )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(method_colors[mode])
            spine.set_linewidth(0.75 * line_width_scale)
        if row_index == 0 and panel_title:
            ax.text(
                0.0,
                1.25,
                panel_title,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=panel_title_fontsize,
                fontweight="bold",
                color=DARK,
            )
            ax.texts[-1].set_y(panel_title_y)

    centers = (bounds[:-1] + bounds[1:]) / 2
    axes[-1].set_xticks(
        centers,
        [
            str(row["passage"])
            for index, row in enumerate(metadata["passages"])
        ],
    )
    axes[-1].tick_params(
        axis="x",
        length=0,
        pad=7,
        labelsize=x_tick_fontsize,
    )
    for label in axes[-1].get_xticklabels():
        label.set_fontweight("heavy")
    axes[-1].set_xlabel(
        x_axis_label,
        fontsize=x_axis_label_fontsize,
        fontweight="heavy",
        labelpad=10,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=FEATURE_RANK_COLORS[index],
            lw=3.0 * line_width_scale,
            marker="o",
            markersize=6.5,
            label=f"rank {index + 1}",
        )
        for index in range(5)
    ]
    axes[0].legend(
        handles=handles,
        ncol=5,
        loc=rank_legend_loc,
        bbox_to_anchor=(rank_legend_x, rank_legend_y),
        fontsize=rank_legend_fontsize,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=0.8,
        borderaxespad=0.0,
        frameon=False,
    )
    feature_axes[0].text(
        0.01,
        1.18,
        "Automatic feature interpretations",
        transform=feature_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=feature_panel_title_fontsize,
        fontweight="heavy",
        color=DARK,
    )
    feature_axes[0].plot(
        [0.01, 0.99],
        [1.11, 1.11],
        transform=feature_axes[0].transAxes,
        color="#AAB2BF",
        lw=1.5 * line_width_scale,
        clip_on=False,
    )
    return axes


def build_sequence_activation_figure(
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    feature_annotations: dict[str, dict[int, str]],
    methods: Sequence[str] = SEQUENCE_METHODS,
    figsize: tuple[float, float] | None = None,
    left: float = 0.055,
    right: float = 0.985,
    bottom: float = 0.065,
    top: float = 0.94,
    panel_title: str = "A   Own top-5 features on identical text",
    x_axis_label: str = "Text position",
    line_width_scale: float = 1.0,
    feature_annotation_fontsize: float = 11.2,
    sequence_width_ratios: tuple[float, float] = (0.64, 0.36),
    panel_title_fontsize: float = 18.5,
    panel_title_y: float = 1.25,
    x_axis_label_fontsize: float = 17.0,
    y_axis_label_fontsize: float = 16.5,
    x_tick_fontsize: float = 15.5,
    feature_panel_title_fontsize: float = 15.5,
    rank_legend_fontsize: float = 11.5,
    rank_legend_x: float = 0.56,
    rank_legend_y: float = 1.34,
    rank_legend_loc: str = "center left",
    method_label_y: float = 0.78,
    method_label_vertical_alignment: str = "center",
) -> plt.Figure:
    """Build the standalone publication-style Panel A.

    The row height matches the original four-method A panel; increasing the
    method count therefore increases the canvas height rather than compressing
    the curves or interpretation cards.
    """

    methods = tuple(methods)
    row_height = 1.24
    panel_height = row_height * (
        len(methods) + 0.105 * max(len(methods) - 1, 0)
    )
    figure_height = panel_height / (0.94 - 0.065)
    if figsize is None:
        figsize = (20, figure_height)
    with plt.rc_context():
        _style()
        fig = plt.figure(
            figsize=figsize,
            facecolor=WHITE,
        )
        spec = fig.add_gridspec(
            1,
            1,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
        )[0, 0]
        _draw_sequence_panel(
            fig,
            spec,
            arrays=arrays,
            metadata=metadata,
            feature_annotations=feature_annotations,
            methods=methods,
            method_labels=SEQUENCE_METHOD_LABELS,
            method_colors=SEQUENCE_METHOD_COLORS,
            method_pale_colors=SEQUENCE_METHOD_PALE_COLORS,
            method_text_colors=SEQUENCE_METHOD_TEXT_COLORS,
            panel_title=panel_title,
            x_axis_label=x_axis_label,
            line_width_scale=line_width_scale,
            feature_annotation_fontsize=feature_annotation_fontsize,
            sequence_width_ratios=sequence_width_ratios,
            panel_title_fontsize=panel_title_fontsize,
            panel_title_y=panel_title_y,
            x_axis_label_fontsize=x_axis_label_fontsize,
            y_axis_label_fontsize=y_axis_label_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            feature_panel_title_fontsize=feature_panel_title_fontsize,
            rank_legend_fontsize=rank_legend_fontsize,
            rank_legend_x=rank_legend_x,
            rank_legend_y=rank_legend_y,
            rank_legend_loc=rank_legend_loc,
            method_label_y=method_label_y,
            method_label_vertical_alignment=method_label_vertical_alignment,
        )
        style_figure_text(fig, minimum_tick_size=10.5)
    return fig


def _draw_population_panel(
    ax: plt.Axes,
    *,
    alignment: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    arrays = np.vstack(
        [alignment[mode]["cosine"] for mode in alignment]
    )
    best = arrays.max(axis=0)
    ordered = np.sort(best)
    cdf = np.arange(1, ordered.size + 1) / ordered.size
    median = float(np.median(best))
    below_half = float(np.mean(best < 0.5))
    ax.plot(
        ordered,
        cdf,
        color=METHOD_COLORS["cross"],
        lw=2.8,
        solid_capstyle="round",
    )
    ax.fill_between(
        ordered,
        0,
        cdf,
        color=METHOD_COLORS["cross"],
        alpha=0.10,
    )
    ax.axvline(0.5, color=GOLD, lw=1.1, ls="--")
    ax.axvline(median, color=DARK, lw=0.9, ls=":")
    ax.scatter(
        [median],
        [0.5],
        s=42,
        color=METHOD_COLORS["cross"],
        edgecolor=WHITE,
        linewidth=0.9,
        zorder=5,
    )
    ax.text(
        0.05,
        0.92,
        f"n = {best.size}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=MID,
    )
    ax.text(
        0.95,
        0.20,
        f"{below_half:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=METHOD_COLORS["cross"],
    )
    ax.text(
        0.95,
        0.10,
        "nearest baseline cosine < 0.50",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=DARK,
    )
    ax.annotate(
        f"median {median:.2f}",
        xy=(median, 0.5),
        xytext=(18, 18),
        textcoords="offset points",
        fontsize=7.0,
        fontweight="bold",
        color=DARK,
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": DARK},
    )
    ax.set_xlim(0.15, 0.70)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Nearest baseline decoder cosine")
    ax.set_ylabel("Fraction of level-4 Cross features")
    ax.grid(color=GRID, lw=0.65)
    ax.set_axisbelow(True)
    ax.set_title(
        "B   Dictionary-wide high-level novelty",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
        pad=19,
    )
    ax.text(
        0.0,
        1.035,
        "population complement to A · nearest single baseline atom",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color=MID,
    )
    return {
        "median_nearest_baseline_cosine": median,
        "fraction_below_0.5": below_half,
        "fraction_below_0.6": float(np.mean(best < 0.6)),
    }


def _draw_manifold(
    ax: plt.Axes,
    *,
    mode: str,
    labels: np.ndarray,
    coordinates: np.ndarray,
    neighbor_purity: float,
    domain_neighbor_purity: dict[str, float],
    highlighted_domains: list[str],
) -> None:
    centers: dict[str, np.ndarray] = {}
    ellipses: dict[str, tuple[float, float, float]] = {}
    for label, color in DOMAIN_COLORS.items():
        points = coordinates[labels == label]
        center, width, height, angle = _robust_ellipse(points)
        centers[label] = center
        ellipses[label] = (width, height, angle)
        ax.add_patch(
            Ellipse(
                center,
                width,
                height,
                angle=angle,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.045 if mode != "cross" else 0.085,
                zorder=0,
            )
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=4.3,
            color=color,
            alpha=0.50,
            linewidth=0,
            rasterized=True,
            zorder=2,
        )

    for highlight_index, label in enumerate(highlighted_domains):
        center = centers[label]
        width, height, angle = ellipses[label]
        ax.add_patch(
            Ellipse(
                center,
                width * 1.18,
                height * 1.18,
                angle=angle,
                facecolor="none",
                edgecolor=DOMAIN_COLORS[label],
                linewidth=2.2 if mode == "cross" else 1.7,
                alpha=0.98,
                zorder=4,
            )
        )
        ax.annotate(
            (
                f"{DOMAIN_SHORT[label]}  "
                f"{domain_neighbor_purity[label]:.2f}"
            ),
            xy=center,
            xytext=(
                (-42, 29)
                if highlight_index == 0
                else (43, -29)
            ),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10.2 if mode != "cross" else 11.2,
            fontweight="heavy",
            color=DARK,
            arrowprops={
                "arrowstyle": "<->",
                "mutation_scale": 8,
                "lw": 1.4 if mode != "cross" else 1.8,
                "color": DOMAIN_COLORS[label],
                "shrinkA": 2,
                "shrinkB": 2,
            },
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": WHITE,
                "edgecolor": DOMAIN_COLORS[label],
                "linewidth": 1.2 if mode != "cross" else 1.6,
                "alpha": 0.97,
            },
            zorder=8,
        )

    if mode == "cross":
        offsets = {
            "CS": (-28, -11),
            "Electrical Engineering": (-30, 13),
            "Economics": (18, -3),
            "Statistics": (-19, -13),
            "Quantitative Finance": (22, 8),
            "Mathematics": (14, -1),
            "Physics": (0, 13),
            "Quantitative Biology": (-8, 13),
        }
        for label, center in centers.items():
            if label in highlighted_domains:
                continue
                ax.annotate(
                DOMAIN_SHORT[label],
                xy=center,
                xytext=offsets[label],
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=10.2,
                fontweight="heavy",
                color=DARK,
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.85,
                    "color": DOMAIN_COLORS[label],
                },
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": WHITE,
                    "edgecolor": DOMAIN_COLORS[label],
                    "linewidth": 0.8,
                    "alpha": 0.96,
                },
                zorder=6,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(PANEL)
    ax.set_title(
        (
            f"{METHOD_LABELS[mode]}\n"
            f"10-NN purity  {neighbor_purity:.3f}"
        ),
        color=METHOD_TEXT_COLORS[mode],
        fontsize=13.5,
        fontweight="bold",
        pad=9,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(METHOD_COLORS[mode])
        spine.set_linewidth(1.8 if mode == "cross" else 0.75)


def _draw_temporal_placeholder(
    ax: plt.Axes,
    *,
    checkpoint: Path | None,
) -> None:
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (0.035, 0.055),
            0.93,
            0.89,
            transform=ax.transAxes,
            facecolor=METHOD_PALE_COLORS["temporal"],
            edgecolor=METHOD_COLORS["temporal"],
            linewidth=1.1,
            linestyle="--",
        )
    )
    ax.text(
        0.5,
        0.59,
        METHOD_LABELS["temporal"],
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=METHOD_TEXT_COLORS["temporal"],
    )
    ax.text(
        0.5,
        0.44,
        "geometry pending",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.0,
        color=MID,
    )
    ax.text(
        0.5,
        0.34,
        "training in progress" if checkpoint else "checkpoint unavailable",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.8,
        color=MID,
    )


def _add_purity_arrow(
    fig: plt.Figure,
    axes: list[plt.Axes],
    metrics: dict[str, dict[str, float]],
    *,
    y: float,
) -> None:
    if "token" not in metrics or "cross" not in metrics:
        return
    start = axes[0].get_position()
    end = axes[-1].get_position()
    x0 = start.x0 + 0.055
    x1 = end.x1 - 0.055
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
        mutation_scale=13,
        lw=1.5,
            color="#8B93A1",
        )
    )
    token = metrics["token"]["neighbor_purity"]
    cross = metrics["cross"]["neighbor_purity"]
    fig.text(
        (x0 + x1) / 2,
        y + 0.006,
        f"10-NN purity  +{100 * (cross - token):.1f} pp",
        ha="center",
        va="bottom",
        fontsize=11.2,
        fontweight="bold",
        color=METHOD_TEXT_COLORS["cross"],
        bbox={
            "boxstyle": "round,pad=0.23",
            "facecolor": WHITE,
            "edgecolor": "none",
            "alpha": 0.94,
        },
    )


def _draw_manifold_row(
    fig: plt.Figure,
    spec,
    *,
    embeddings: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    temporal_checkpoint: Path | None,
    include_panel_title: bool,
) -> list[plt.Axes]:
    labels = embeddings["labels"].astype(str)
    comparisons = _domain_comparisons(metrics)
    highlighted_domains = [
        str(row["domain"])
        for row in comparisons[:HIGHLIGHT_DOMAIN_COUNT]
    ]
    grid = spec.subgridspec(1, 4, wspace=0.065)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    for ax, mode in zip(axes, METHODS, strict=True):
        if f"{mode}_xy" in embeddings and mode in metrics:
            _draw_manifold(
                ax,
                mode=mode,
                labels=labels,
                coordinates=embeddings[f"{mode}_xy"],
                neighbor_purity=metrics[mode]["neighbor_purity"],
                domain_neighbor_purity=metrics[mode][
                    "domain_neighbor_purity"
                ],
                highlighted_domains=highlighted_domains,
            )
        else:
            _draw_temporal_placeholder(
                ax,
                checkpoint=temporal_checkpoint if mode == "temporal" else None,
            )
    if include_panel_title:
        axes[0].text(
            0.0,
            1.10,
            "B   Cross-Chunk forms cleaner semantic neighborhoods",
            transform=axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=13.2,
            fontweight="bold",
            color=DARK,
        )
        if len(comparisons) >= HIGHLIGHT_DOMAIN_COUNT:
            first, second = comparisons[:HIGHLIGHT_DOMAIN_COUNT]
            axes[-1].text(
                1.0,
                1.10,
                (
                    "largest Cross gains:  "
                    f"{first['short']} +"
                    f"{100 * first['cross_margin_over_strongest_baseline']:.1f}"
                    " pp   ·   "
                    f"{second['short']} +"
                    f"{100 * second['cross_margin_over_strongest_baseline']:.1f}"
                    " pp"
                ),
                transform=axes[-1].transAxes,
                ha="right",
                va="bottom",
                fontsize=8.4,
                fontweight="heavy",
                color=METHOD_COLORS["cross"],
            )
    return axes


def _domain_legend(
    fig: plt.Figure,
    *,
    y: float,
    x: float = 0.5,
    loc: str = "lower center",
    fontsize: float = 13.0,
    columnspacing: float = 1.05,
    handletextpad: float = 0.55,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DOMAIN_COLORS[label],
            markeredgecolor="none",
            markersize=9.5,
            label=DOMAIN_SHORT[label],
        )
        for label in DOMAIN_COLORS
    ]
    fig.legend(
        handles=handles,
        ncol=8,
        loc=loc,
        bbox_to_anchor=(x, y),
        fontsize=fontsize,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        borderaxespad=0.0,
        frameon=False,
    )


def _save(fig: plt.Figure, base: Path, dpi: int) -> list[Path]:
    # Standalone task plots live under ``*/figures`` while the combined paper
    # figure is intentionally stored at the evaluation root.  Both are
    # canonical outputs of this module.
    if (
        base.parent.name != "figures"
        and base.name
        not in {
            "sequence_activation_tsne_figure",
            "paper_figure3_cross_high_level_features",
        }
    ):
        raise ValueError(
            "figure output must be inside a figures/ directory or use a "
            f"canonical root-level paper-figure name: {base}"
        )
    base.parent.mkdir(parents=True, exist_ok=True)
    style_figure_text(fig, minimum_tick_size=10.5)
    outputs = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        kwargs: dict[str, Any] = {"facecolor": WHITE}
        if suffix == ".png":
            kwargs["dpi"] = dpi
        # Keep a published primary raster stable unless a caller explicitly
        # removes it; the PDF is always refreshed alongside it.
        if suffix != ".png" or not path.is_file():
            fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _render_standalone_manifold(
    *,
    base: Path,
    dpi: int,
    embeddings: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    temporal_checkpoint: Path | None,
) -> list[Path]:
    fig = plt.figure(figsize=(16, 4.25), facecolor=WHITE)
    spec = fig.add_gridspec(
        1,
        1,
        left=0.04,
        right=0.96,
        bottom=0.23,
        top=0.90,
    )[0, 0]
    axes = _draw_manifold_row(
        fig,
        spec,
        embeddings=embeddings,
        metrics=metrics,
        temporal_checkpoint=temporal_checkpoint,
        include_panel_title=False,
    )
    _domain_legend(fig, y=0.020)
    _add_purity_arrow(fig, axes, metrics, y=0.155)
    return _save(fig, base, dpi)


def render_semantic_manifold(
    *,
    eval_root: Path,
    base: Path,
    dpi: int = 300,
    perplexity: float = 75.0,
    iterations: int = 1500,
    seed: int = 20260817,
    refresh: bool = False,
) -> list[Path]:
    """Render the standalone semantic-manifold comparison."""

    _style()
    embeddings, metrics = _load_or_build_embeddings(
        eval_root=eval_root,
        perplexity=perplexity,
        iterations=iterations,
        seed=seed,
        refresh=refresh,
    )
    temporal_checkpoint, _ = _temporal_checkpoint(eval_root)
    return _render_standalone_manifold(
        base=base,
        dpi=dpi,
        embeddings=embeddings,
        metrics=metrics,
        temporal_checkpoint=temporal_checkpoint,
    )


def main() -> None:
    args = parser().parse_args()
    _style()
    eval_root = Path(args.eval_root).resolve()
    checkpoints = _checkpoint_paths(eval_root)
    temporal_checkpoint, temporal_completed = _temporal_checkpoint(eval_root)
    sequence_cache = eval_root / "feature_dynamics" / "sequence_activation_traces.npz"
    sequence_metadata_path = sequence_cache.with_suffix(".json")
    if (
        sequence_cache.is_file()
        and sequence_metadata_path.is_file()
        and not args.refresh_sequence
    ):
        sequence_metadata = _read_json(sequence_metadata_path)
        with np.load(sequence_cache, allow_pickle=True) as handle:
            sequence_arrays = {
                key: handle[key] for key in handle.files
            }
    else:
        passages = _sequence_passages(eval_root)
        sequence_windows = parse_int_csv(args.sequence_windows)
        if args.sequence_window is not None:
            sequence_windows = [int(args.sequence_window)]
        sequence_windows = list(dict.fromkeys(sequence_windows))
        if args.sequence_display_chunk_length <= 0:
            raise ValueError("--sequence-display-chunk-length must be positive")
        if args.sequence_display_chunk_length not in sequence_windows:
            raise ValueError(
                "--sequence-display-chunk-length must be included in "
                f"--sequence-chunk-lengths; display="
                f"{args.sequence_display_chunk_length}, lengths={sequence_windows}"
            )
        training_lengths = set(
            _read_json(
                eval_root
                / "dictionary_utilization"
                / "feature_evidence"
                / "evidence_manifest.json"
            )["identity"]["chunk_lengths"]
        )
        unsupported_windows = sorted(set(sequence_windows) - training_lengths)
        if unsupported_windows:
            raise ValueError(
                "Panel A chunk lengths must come from the formal training/evidence "
                f"support {sorted(training_lengths)}; found {unsupported_windows}"
            )
        sequence_arrays, sequence_metadata = _load_or_build_sequence(
            eval_root=eval_root,
            passages=passages,
            checkpoints=checkpoints,
            temporal_checkpoint=temporal_checkpoint,
            temporal_completed=temporal_completed,
            top_k=args.sequence_top_k,
            windows=sequence_windows,
            display_chunk_length=args.sequence_display_chunk_length,
            max_length=args.sequence_max_length,
            block_size=args.encoder_block_size,
            device=args.sequence_device,
            refresh=True,
        )

    embeddings, metrics = _load_or_build_embeddings(
        eval_root=eval_root,
        perplexity=args.tsne_perplexity,
        iterations=args.tsne_iterations,
        seed=args.seed,
        refresh=args.refresh_embeddings,
    )
    domain_comparisons = _domain_comparisons(metrics)
    feature_annotations, feature_annotation_path = (
        _feature_annotations(eval_root)
    )
    if not feature_annotations:
        raise FileNotFoundError(
            eval_root / "feature_dynamics" / "sequence_top5_autointerp.json"
        )

    fig = plt.figure(figsize=(20, 10), facecolor=WHITE)
    outer = fig.add_gridspec(
        2,
        1,
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.94,
        height_ratios=(1.45, 0.55),
        hspace=0.40,
    )
    sequence_axes = _draw_sequence_panel(
        fig,
        outer[0, 0],
        arrays=sequence_arrays,
        metadata=sequence_metadata,
        feature_annotations=feature_annotations,
    )
    manifold_axes = _draw_manifold_row(
        fig,
        outer[1, 0],
        embeddings=embeddings,
        metrics=metrics,
        temporal_checkpoint=temporal_checkpoint,
        include_panel_title=False,
    )
    fig.text(
        0.055,
        0.328,
        "B   Cross-Chunk forms cleaner semantic neighborhoods",
        ha="left",
        va="center",
        fontsize=18.5,
        fontweight="bold",
        color=DARK,
    )
    _domain_legend(
        fig,
        x=0.475,
        y=0.328,
        loc="center left",
        fontsize=12.0,
        columnspacing=0.8,
        handletextpad=0.45,
    )
    _add_purity_arrow(fig, manifold_axes, metrics, y=0.027)
    output_paths = _save(
        fig,
        eval_root / args.output_name,
        args.dpi,
    )

    manifold_base = eval_root / args.manifold_output
    if args.skip_manifold_output:
        manifold_paths = [
            manifold_base.with_suffix(suffix)
            for suffix in (".png", ".pdf")
            if manifold_base.with_suffix(suffix).is_file()
        ]
    else:
        manifold_paths = _render_standalone_manifold(
            base=manifold_base,
            dpi=args.dpi,
            embeddings=embeddings,
            metrics=metrics,
            temporal_checkpoint=temporal_checkpoint,
        )

    result = {
        "format": "chunk-saes-cross-concept-manifold-figure-v5",
        "complete": True,
        "sequence_activation": sequence_metadata,
        "temporal": {
            "checkpoint_available": temporal_checkpoint is not None,
            "training_complete": temporal_completed,
            "qualitative_sequence_available": (
                "temporal_traces" in sequence_arrays
            ),
            "formal_geometry_available": (
                "temporal_xy" in embeddings
                and "temporal" in metrics
            ),
        },
        "manifold_metrics": metrics,
        "domain_neighbor_purity_comparisons": domain_comparisons,
        "highlighted_domains": [
            row["domain"]
            for row in domain_comparisons[:HIGHLIGHT_DOMAIN_COUNT]
        ],
        "display_tsne": {
            "perplexity": args.tsne_perplexity,
            "iterations": args.tsne_iterations,
            "seed": args.seed,
            "displayed_metric": "formal 50-D 10-NN neighbor purity",
            "feature_id_annotations": False,
        },
        "sequence_curve_annotations": {
            "feature_ids": False,
            "automatic_feature_interpretations": True,
            "annotation_artifact": str(feature_annotation_path),
            "maximum_label_words": SEQUENCE_LABEL_MAX_WORDS,
            "maximum_label_tokens": SEQUENCE_LABEL_MAX_TOKENS,
        },
        "files": {
            "combined_figure": [str(path) for path in output_paths],
            "semantic_manifold": [str(path) for path in manifold_paths],
        },
    }
    result_path = (
        eval_root / "feature_dynamics" / "cross_concept_manifold_figure.json"
    )
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "combined_figure": [str(path) for path in output_paths],
                "semantic_manifold": [str(path) for path in manifold_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
