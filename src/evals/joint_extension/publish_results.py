#!/usr/bin/env python
"""Publish incremental Joint-Chunk results into the canonical evaluation tree.

This module is deliberately additive:

* non-Joint baseline rows are retained byte-for-byte at the value level;
* the legacy baseline key ``joint`` is normalized to ``joint_alpha0p25``;
* all four canonical Joint alpha rows are replaced idempotently from the
  current sidecars;
* all files are prepared and validated in a staging directory before any
  canonical file is replaced.

The ArXiv joint sidecar is shared by label efficiency, semantic geometry, and
temporal robustness.  Reasoning is allowed to be absent only when
``--allow-missing-reasoning`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from chunk_saes.plot_style import (
    METHOD_COLORS as SHARED_METHOD_COLORS,
    METHOD_MARKERS as SHARED_METHOD_MARKERS,
)


LEGACY_JOINT_KEY = "joint"
JOINT_ALPHAS = {
    "joint_alpha0p25": 0.25,
    "joint_alpha0p5": 0.5,
    "joint_alpha1": 1.0,
    "joint_alpha1p5": 1.5,
}
JOINT_KEYS = tuple(JOINT_ALPHAS)
JOINT_KEY_ALIASES = frozenset((*JOINT_KEYS, LEGACY_JOINT_KEY))
JOINT_LABELS = {
    "joint_alpha0p25": "Joint-Chunk SAE(alpha=0.25)",
    "joint_alpha0p5": "Joint-Chunk SAE(alpha=0.5)",
    "joint_alpha1": "Joint-Chunk SAE(alpha=1)",
    "joint_alpha1p5": "Joint-Chunk SAE(alpha=1.5)",
}
JOINT_SOURCE_LABEL_ALIASES = {
    key: frozenset(
        (
            label,
            *(
                ("Joint-Chunk SAE(alpha=1.0)",)
                if key == "joint_alpha1"
                else ()
            ),
        )
    )
    for key, label in JOINT_LABELS.items()
}
METHOD_ORDER = (
    "token",
    "temporal",
    "mean",
    "joint_alpha0p25",
    "joint_alpha0p5",
    "joint_alpha1",
    "joint_alpha1p5",
    "cross",
)
METHOD_LABELS = {
    "token": "BatchTopK SAE",
    "temporal": "Temporal SAE",
    "mean": "Mean-Chunk SAE",
    "joint_alpha0p25": JOINT_LABELS["joint_alpha0p25"],
    "joint_alpha0p5": JOINT_LABELS["joint_alpha0p5"],
    "joint_alpha1": JOINT_LABELS["joint_alpha1"],
    "joint_alpha1p5": JOINT_LABELS["joint_alpha1p5"],
    "cross": "Cross-Chunk SAE",
}
METHOD_COLORS = {
    method: SHARED_METHOD_COLORS[method] for method in METHOD_ORDER
}
METHOD_MARKERS = {
    method: SHARED_METHOD_MARKERS[method] for method in METHOD_ORDER
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
PUBLISH_FORMAT = "chunk-saes-joint-extension-publication-v1"
MANIFEST_FORMAT = "chunk-saes-joint-extension-publish-manifest-v1"

DICTIONARY_JSON = Path("dictionary_utilization/dictionary_utilization.json")
DICTIONARY_SIDECAR = Path("dictionary_utilization/joint_extension.json")
DOCUMENT_JSON = Path("document_linking/document_linking_results.json")
DOCUMENT_SUMMARY = Path("document_linking/summary_table.csv")
DOCUMENT_SIDECAR = Path("document_linking/joint_extension.json")
FEATURE_JSON = Path("feature_dynamics/sequence_activation_traces.json")
FEATURE_NPZ = Path("feature_dynamics/sequence_activation_traces.npz")
FEATURE_SIDECAR = Path("feature_dynamics/joint_extension.json")
FEATURE_SIDECAR_NPZ = Path("feature_dynamics/joint_extension_traces.npz")
PROBE_JSON = Path("label_efficiency/linear_probe_results.json")
ARXIV_SIDECAR = Path(
    "shared/downstream_transfer/probe_features/joint_extension/"
    "joint_extension.json"
)
ARXIV_EMBEDDINGS = Path(
    "shared/downstream_transfer/probe_features/joint_extension/"
    "representation_embeddings.npz"
)
ARXIV_FEATURE_MANIFEST = Path(
    "shared/downstream_transfer/probe_features/feature_manifest.json"
)
ARXIV_SPLITS = ("train", "validation", "test", "ood")
ARXIV_ENCODED_SPLITS = tuple(
    Path(
        "shared/downstream_transfer/probe_features/joint_extension/"
        f"{method}-{split}.npz"
    )
    for method in JOINT_KEYS
    for split in ARXIV_SPLITS
)
GEOMETRY_JSON = Path(
    "semantic_geometry/representation_geometry/representation_geometry.json"
)
GEOMETRY_NPZ = Path(
    "semantic_geometry/representation_geometry/representation_embeddings.npz"
)
TEMPORAL_JSON = Path("temporal_robustness/temporal_robustness.json")
RFVE_JSON = Path("rfve/training_fidelity.json")
RFVE_SIDECAR = Path("rfve/joint_extension.json")
REASONING_JSON = Path("reasoning/results_summary.json")
REASONING_SIDECAR = Path("reasoning/joint_extension.json")

PLOT_SPECS = {
    "dictionary_utilization": (
        Path("dictionary_utilization/figures/dictionary_utilization.png"),
        Path("dictionary_utilization/figures/dictionary_utilization.pdf"),
    ),
    "document_linking": (
        Path("document_linking/figures/lexical_controlled_document_linking.png"),
        Path("document_linking/figures/lexical_controlled_document_linking.pdf"),
    ),
    "feature_dynamics": (
        Path("feature_dynamics/figures/sequence_activation_tsne.png"),
        Path("feature_dynamics/figures/sequence_activation_tsne.pdf"),
    ),
    "label_efficiency": (
        Path("label_efficiency/figures/label_efficiency.png"),
        Path("label_efficiency/figures/label_efficiency.pdf"),
    ),
    "reasoning": (
        Path("reasoning/figures/reasoning_specificity.png"),
        Path("reasoning/figures/reasoning_specificity.pdf"),
    ),
    "rfve": (
        Path("rfve/figures/training_fidelity.png"),
        Path("rfve/figures/training_fidelity.pdf"),
    ),
    "semantic_geometry": (
        Path("semantic_geometry/figures/semantic_geometry.png"),
        Path("semantic_geometry/figures/semantic_geometry.pdf"),
    ),
    "temporal_robustness": (
        Path("temporal_robustness/figures/temporal_robustness.png"),
        Path("temporal_robustness/figures/temporal_robustness.pdf"),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        required=True,
        help="Canonical layer-specific evaluation directory.",
    )
    parser.add_argument(
        "--allow-missing-reasoning",
        action="store_true",
        help=(
            "Publish/test the other seven tasks when reasoning/joint_extension.json "
            "has not arrived yet.  The canonical reasoning files are left untouched."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and validate all outputs in staging without replacing canonical "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--dry-run-dir",
        default=None,
        help=(
            "Optional empty directory in which to retain dry-run artifacts for "
            "inspection.  Implies --dry-run."
        ),
    )
    parser.add_argument("--dpi", type=int, default=260)
    return parser


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _canonical_method_key(method: str) -> str:
    return JOINT_KEYS[0] if method == LEGACY_JOINT_KEY else method


def _available_methods(keys: Iterable[str]) -> list[str]:
    key_set = {_canonical_method_key(str(key)) for key in keys}
    ordered = [method for method in METHOD_ORDER if method in key_set]
    ordered.extend(sorted(key_set - set(ordered)))
    return ordered


def _label(method: str, row: Mapping[str, Any] | None = None) -> str:
    method = _canonical_method_key(method)
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if row is not None and isinstance(row.get("label"), str):
        return str(row["label"])
    return method


def _color(method: str) -> str:
    method = _canonical_method_key(method)
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    digest = hashlib.sha256(method.encode("utf-8")).digest()
    return f"#{digest[0]:02x}{digest[1]:02x}{digest[2]:02x}"


def _marker(method: str) -> str:
    method = _canonical_method_key(method)
    return METHOD_MARKERS.get(method, "o")


def _normalize_legacy_joint_mapping(
    mapping: MutableMapping[str, Any],
    *,
    path: str,
) -> None:
    """Move a legacy alpha=0.25 ``joint`` entry to its canonical key.

    When the canonical entry is already present it is authoritative.  This is
    important for idempotent reruns over files produced by older publishers.
    """

    if LEGACY_JOINT_KEY not in mapping:
        return
    legacy = mapping[LEGACY_JOINT_KEY]
    canonical_key = JOINT_KEYS[0]
    mapping.setdefault(canonical_key, legacy)
    del mapping[LEGACY_JOINT_KEY]


def _normalize_legacy_joint_sequence(values: Iterable[Any]) -> list[Any]:
    normalized: list[Any] = []
    for value in values:
        candidate = JOINT_KEYS[0] if value == LEGACY_JOINT_KEY else value
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


METHOD_KEYED_METADATA_FIELDS = frozenset(
    {
        "checkpoints",
        "feature_widths",
        "full_dictionary_widths",
        "mean_nnz",
        "method_labels",
        "methods",
        "qualified_feature_count_by_method",
        "rank_distributions",
        "reference_fves",
        "representation",
        "sampled_alive_features",
    }
)


def _normalize_legacy_joint_metadata(value: Any) -> None:
    if not isinstance(value, dict):
        return
    for field_name, child in list(value.items()):
        if isinstance(child, dict):
            if field_name in METHOD_KEYED_METADATA_FIELDS:
                _normalize_legacy_joint_mapping(
                    child,
                    path=field_name,
                )
            if field_name == "comparisons":
                _normalize_legacy_joint_comparisons(child)
            _normalize_legacy_joint_metadata(child)
        elif (
            isinstance(child, list)
            and field_name in {"method_order", "methods"}
        ):
            value[field_name] = _normalize_legacy_joint_sequence(child)


def _normalize_legacy_joint_comparisons(
    mapping: MutableMapping[str, Any],
) -> None:
    for key in list(mapping):
        if not isinstance(key, str):
            continue
        parts = key.split("_minus_")
        canonical = "_minus_".join(
            JOINT_KEYS[0] if part == LEGACY_JOINT_KEY else part
            for part in parts
        )
        if canonical == key:
            continue
        mapping.setdefault(canonical, mapping[key])
        del mapping[key]


def _is_joint_scoped_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    return any(part in JOINT_KEY_ALIASES for part in key.split("_minus_"))


def _normalize_legacy_joint_arrays(
    arrays: MutableMapping[str, np.ndarray],
    *,
    suffixes: Iterable[str],
) -> None:
    for suffix in suffixes:
        legacy_key = f"{LEGACY_JOINT_KEY}_{suffix}"
        canonical_key = f"{JOINT_KEYS[0]}_{suffix}"
        if legacy_key not in arrays:
            continue
        arrays.setdefault(canonical_key, arrays[legacy_key])
        del arrays[legacy_key]


def _is_joint_array_key(key: str) -> bool:
    return any(
        key == method or key.startswith(f"{method}_")
        for method in JOINT_KEY_ALIASES
    )


def _published_joint_checkpoint(
    sidecar: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    checkpoint = copy.deepcopy(sidecar["joint_checkpoints"][key])
    checkpoint["key"] = key
    checkpoint["label"] = JOINT_LABELS[key]
    checkpoint["alpha"] = JOINT_ALPHAS[key]
    return checkpoint


def _validate_joint_sidecar(
    path: Path,
    *,
    expected_task: str,
) -> dict[str, Any]:
    sidecar = _load_json(path)
    if sidecar.get("complete") is not True:
        raise ValueError(f"incomplete joint sidecar: {path}")
    if sidecar.get("format") != "chunk-saes-joint-extension-v1":
        raise ValueError(f"unexpected joint sidecar format: {path}")
    if sidecar.get("task") != expected_task:
        raise ValueError(
            f"{path}: expected task={expected_task!r}, "
            f"found {sidecar.get('task')!r}"
        )
    methods = sidecar.get("methods")
    checkpoints = sidecar.get("joint_checkpoints")
    if not isinstance(methods, dict) or not isinstance(checkpoints, dict):
        raise ValueError(f"{path}: missing methods/checkpoint mappings")
    if LEGACY_JOINT_KEY in methods or LEGACY_JOINT_KEY in checkpoints:
        raise ValueError(
            f"{path}: sidecar contains legacy key {LEGACY_JOINT_KEY!r}"
        )
    method_order = sidecar.get("method_order")
    if isinstance(method_order, list):
        if LEGACY_JOINT_KEY in method_order:
            raise ValueError(
                f"{path}: method_order contains legacy key "
                f"{LEGACY_JOINT_KEY!r}"
            )
        ordered_joint = [
            key for key in method_order if key in JOINT_KEYS
        ]
        if ordered_joint != list(JOINT_KEYS):
            raise ValueError(
                f"{path}: Joint methods are not in canonical alpha order: "
                f"{ordered_joint}"
            )
    for key in JOINT_KEYS:
        if key not in methods or key not in checkpoints:
            raise ValueError(f"{path}: missing {key}")
        actual_label = methods[key].get("label")
        if actual_label not in JOINT_SOURCE_LABEL_ALIASES[key]:
            raise ValueError(
                f"{path}: {key} label {actual_label!r} is not one of "
                f"{sorted(JOINT_SOURCE_LABEL_ALIASES[key])!r}"
            )
        checkpoint_label = checkpoints[key].get("label")
        if checkpoint_label not in JOINT_SOURCE_LABEL_ALIASES[key]:
            raise ValueError(
                f"{path}: {key} checkpoint label {checkpoint_label!r} does "
                f"not match an accepted source label "
                f"{sorted(JOINT_SOURCE_LABEL_ALIASES[key])!r}"
            )
        method_alpha = _finite(methods[key].get("alpha"))
        checkpoint_alpha = _finite(checkpoints[key].get("alpha"))
        expected_alpha = JOINT_ALPHAS[key]
        if not math.isclose(
            method_alpha,
            expected_alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{path}: {key} alpha {methods[key].get('alpha')!r} does "
                f"not equal {expected_alpha}"
            )
        if not math.isclose(
            checkpoint_alpha,
            expected_alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{path}: {key} checkpoint alpha "
                f"{checkpoints[key].get('alpha')!r} does not equal "
                f"{expected_alpha}"
            )
    return sidecar


def _extension_metadata(
    *,
    sidecar: Mapping[str, Any],
    sidecar_rel: Path,
) -> dict[str, Any]:
    return {
        "format": PUBLISH_FORMAT,
        "method_keys": list(JOINT_KEYS),
        "method_labels": dict(JOINT_LABELS),
        "source_sidecar": sidecar_rel.as_posix(),
        "source_artifact_digest": sidecar.get("artifact_digest"),
        "joint_checkpoints": {
            key: _published_joint_checkpoint(sidecar, key)
            for key in JOINT_KEYS
        },
        "protocol": copy.deepcopy(sidecar.get("protocol", {})),
    }


def _attach_method_metadata(
    payload: dict[str, Any],
    *,
    method_keys: Iterable[str],
    sidecar: Mapping[str, Any],
    sidecar_rel: Path,
) -> None:
    _normalize_legacy_joint_metadata(payload)
    methods = _available_methods(method_keys)
    payload["method_order"] = methods
    labels = payload.setdefault("method_labels", {})
    if not isinstance(labels, dict):
        raise ValueError("method_labels must be a mapping when present")
    _normalize_legacy_joint_mapping(labels, path="method_labels")
    for method in methods:
        labels[method] = _label(method)
    payload["joint_extension"] = _extension_metadata(
        sidecar=sidecar,
        sidecar_rel=sidecar_rel,
    )


def _assert_preserved_methods(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    path: str,
) -> None:
    for key, value in before.items():
        if _is_joint_scoped_key(key):
            continue
        if key not in after or after[key] != value:
            raise AssertionError(f"publisher changed pre-existing {path}.{key}")


def _finite(value: Any, *, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mean(values: Iterable[Any]) -> float | None:
    numbers = np.asarray(
        [
            number
            for value in values
            if math.isfinite(number := _finite(value))
        ],
        dtype=np.float64,
    )
    return float(numbers.mean()) if numbers.size else None


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "xtick.color": "#303744",
            "ytick.color": "#303744",
            "text.color": "#202733",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _bold_figure_text(fig: plt.Figure) -> None:
    for ax in fig.axes:
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        ax.title.set_fontweight("bold")
        for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick.set_fontweight("bold")
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontweight("bold")
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontweight("bold")


@dataclass
class PublishContext:
    eval_root: Path
    stage_root: Path
    dry_run: bool
    dpi: int
    outputs: set[Path] = field(default_factory=set)
    task_outputs: dict[str, list[Path]] = field(default_factory=dict)
    skipped_tasks: list[str] = field(default_factory=list)

    def source(self, relative: Path | str) -> Path:
        path = self.eval_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def staged(self, relative: Path | str) -> Path:
        return self.stage_root / Path(relative)

    def register(self, task: str, relative: Path) -> Path:
        path = self.staged(relative)
        self.outputs.add(relative)
        self.task_outputs.setdefault(task, []).append(relative)
        return path

    def write_json(
        self,
        task: str,
        relative: Path | str,
        payload: Mapping[str, Any],
    ) -> Path:
        rel = Path(relative)
        path = self.register(task, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json_bytes(dict(payload)))
        return path

    def write_csv(
        self,
        task: str,
        relative: Path | str,
        fieldnames: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
    ) -> Path:
        rel = Path(relative)
        path = self.register(task, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_npz(
        self,
        task: str,
        relative: Path | str,
        arrays: Mapping[str, np.ndarray],
    ) -> Path:
        rel = Path(relative)
        path = self.register(task, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def save_figure(self, task: str, fig: plt.Figure) -> tuple[Path, Path]:
        png_rel, pdf_rel = PLOT_SPECS[task]
        paths = tuple(
            self.register(task, rel) for rel in (png_rel, pdf_rel)
        )
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        _bold_figure_text(fig)
        fig.savefig(paths[0], dpi=self.dpi, bbox_inches="tight")
        fig.savefig(paths[1], bbox_inches="tight")
        plt.close(fig)
        return paths

    def path_for_validation(self, relative: Path) -> Path:
        staged = self.staged(relative)
        return staged if staged.is_file() else self.eval_root / relative


def _record_for(
    ctx: PublishContext,
    *,
    target_rel: Path,
    relative_to: Path,
) -> dict[str, Any]:
    path = ctx.path_for_validation(target_rel)
    if not path.is_file():
        raise FileNotFoundError(path)
    target = ctx.eval_root / target_rel
    recorded = Path(os.path.relpath(target, ctx.eval_root / relative_to))
    return {
        "path": recorded.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _stage_manifest(
    ctx: PublishContext,
    *,
    task: str,
    manifest_rel: Path,
    tracked: Mapping[str, Path],
    sidecar_rel: Path,
    sidecar: Mapping[str, Any],
) -> None:
    live_path = ctx.eval_root / manifest_rel
    if live_path.is_file():
        manifest = _load_json(live_path)
    else:
        manifest = {
            "format": MANIFEST_FORMAT,
            "complete": True,
            "identity": {},
            "files": {},
        }
    manifest.pop("artifact_digest", None)
    files = manifest.setdefault("files", {})
    if not isinstance(files, dict):
        raise ValueError(f"{manifest_rel}: files must be a mapping")
    # The canonical figure contract stores only PNG/PDF pairs in figures/.
    # Drop obsolete SVG records from earlier layouts.
    for name, record in list(files.items()):
        record_path = (
            str(record.get("path", ""))
            if isinstance(record, Mapping)
            else ""
        )
        if Path(record_path).suffix.lower() == ".svg":
            files.pop(name, None)
            continue
        if not record_path:
            files.pop(name, None)
            continue
        target = (ctx.eval_root / manifest_rel.parent / record_path).resolve()
        try:
            target_rel = target.relative_to(ctx.eval_root)
        except ValueError:
            current = target
        else:
            current = ctx.path_for_validation(target_rel)
        if not current.is_file():
            files.pop(name, None)
            continue
        files[name] = {
            "path": record_path,
            "bytes": current.stat().st_size,
            "sha256": _file_sha256(current),
        }
    manifest_parent = manifest_rel.parent
    for name, relative in tracked.items():
        files[name] = _record_for(
            ctx,
            target_rel=relative,
            relative_to=manifest_parent,
        )
    identity = manifest.setdefault("identity", {})
    if not isinstance(identity, dict):
        identity = {"base_identity": copy.deepcopy(identity)}
        manifest["identity"] = identity
    _normalize_legacy_joint_metadata(identity)
    identity["joint_extension"] = _extension_metadata(
        sidecar=sidecar,
        sidecar_rel=sidecar_rel,
    )
    manifest["complete"] = True
    manifest["artifact_digest"] = _json_digest(manifest)
    ctx.write_json(task, manifest_rel, manifest)


def _plot_legend(
    methods: Sequence[str],
    *,
    linewidth: float = 2.6,
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=_color(method),
            marker=_marker(method),
            lw=linewidth,
            markersize=6,
            label=_label(method),
        )
        for method in methods
    ]


def _publication_note(ax: plt.Axes) -> None:
    ax.text(
        1.0,
        -0.16,
        "Frozen SAE checkpoints · shared evaluation examples",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#667085",
    )


def _publish_dictionary(
    ctx: PublishContext,
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(_load_json(ctx.source(DICTIONARY_JSON)))
    methods = payload.setdefault("methods", {})
    before = copy.deepcopy(methods)
    identity = payload.setdefault("identity", {})
    widths = identity.setdefault("full_dictionary_widths", {})
    sampled = identity.setdefault("sampled_alive_features", {})
    _normalize_legacy_joint_mapping(methods, path="dictionary.methods")
    _normalize_legacy_joint_mapping(
        widths,
        path="dictionary.identity.full_dictionary_widths",
    )
    _normalize_legacy_joint_mapping(
        sampled,
        path="dictionary.identity.sampled_alive_features",
    )
    for key in JOINT_KEYS:
        source = sidecar["methods"][key]
        row = copy.deepcopy(source["dictionary_utilization"])
        checkpoint = sidecar["joint_checkpoints"][key]
        sampled_features = int(
            source.get(
                "sampled_features",
                row.get("sampled_features", row.get("active_features", 0)),
            )
        )
        row.update(
            {
                "label": JOINT_LABELS[key],
                "alpha": float(source["alpha"]),
                "representation": source.get("representation"),
                # Match the canonical Eval-2 per-method schema while retaining
                # the richer Joint-only diagnostics below.
                "sampled_alive_features": sampled_features,
                "full_dictionary_width": int(
                    checkpoint["dictionary_width"]
                ),
                "k128_equivalent_slots": 128.0
                * float(row["effective_feature_fraction"]),
                "adjacent_pair_metrics": copy.deepcopy(
                    source.get("pair_metrics", {})
                ),
                "feature_persistence_metrics": copy.deepcopy(
                    source.get("feature_metrics", {})
                ),
            }
        )
        methods[key] = row
        widths[key] = int(checkpoint["dictionary_width"])
        sampled[key] = sampled_features
    _assert_preserved_methods(before, methods, path="dictionary.methods")
    _attach_method_metadata(
        payload,
        method_keys=methods,
        sidecar=sidecar,
        sidecar_rel=DICTIONARY_SIDECAR,
    )
    ctx.write_json("dictionary_utilization", DICTIONARY_JSON, payload)

    plot_methods = _available_methods(methods)
    fractions = np.asarray(
        [float(methods[key]["effective_feature_fraction"]) for key in plot_methods]
    )
    slots = 128.0 * fractions
    y = np.arange(len(plot_methods))
    fig, ax = plt.subplots(figsize=(12.8, 6.0))
    ax.barh(y, np.full_like(slots, 128.0), height=0.62, color="#E9EDF3")
    ax.barh(
        y,
        slots,
        height=0.62,
        color=[_color(key) for key in plot_methods],
        edgecolor="white",
        linewidth=1.0,
    )
    ax.set_yticks(y, [_label(key, methods[key]) for key in plot_methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    ax.set_xlabel("Entropy-equivalent content slots in the K=128 budget")
    ax.set_title(
        "Dictionary utilization: effective share of distinct content features",
        loc="left",
        pad=16,
    )
    for index, (fraction, slot) in enumerate(zip(fractions, slots, strict=True)):
        ax.text(
            slot + 1.2,
            index,
            f"{slot:.1f} slots  ({fraction:.1%})",
            va="center",
            fontweight="bold",
            fontsize=9.2,
        )
    ax.grid(axis="x", color="#DDE2EA", lw=0.75, zorder=0)
    ax.legend(
        handles=[
            Patch(color="#E9EDF3", label="Remaining K=128 capacity"),
            Patch(color="#52606D", label="Entropy-equivalent content capacity"),
        ],
        loc="lower right",
    )
    _publication_note(ax)
    fig.tight_layout()
    ctx.save_figure("dictionary_utilization", fig)

    tracked = {
        "results": DICTIONARY_JSON,
        "plot_png": PLOT_SPECS["dictionary_utilization"][0],
        "plot_pdf": PLOT_SPECS["dictionary_utilization"][1],
        "joint_extension_sidecar": DICTIONARY_SIDECAR,
    }
    _stage_manifest(
        ctx,
        task="dictionary_utilization",
        manifest_rel=Path(
            "dictionary_utilization/dictionary_utilization_manifest.json"
        ),
        tracked=tracked,
        sidecar_rel=DICTIONARY_SIDECAR,
        sidecar=sidecar,
    )
    return payload


def _merge_csv_rows(
    base_fields: Sequence[str],
    base_rows: Sequence[Mapping[str, Any]],
    extension_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    rows = [
        dict(row)
        for row in base_rows
        if str(row.get("method", "")) not in JOINT_KEY_ALIASES
    ]
    rows.extend(dict(row) for row in extension_rows)
    fields = list(base_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields, rows


def _publish_document_linking(
    ctx: PublishContext,
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(_load_json(ctx.source(DOCUMENT_JSON)))
    methods = payload.setdefault("methods", {})
    before = copy.deepcopy(methods)
    ranks = payload.setdefault("rank_distributions", {})
    before_ranks = copy.deepcopy(ranks)
    comparisons = payload.setdefault("comparisons", {})
    before_comparisons = copy.deepcopy(comparisons)
    identity = payload.setdefault("identity", {})
    widths = identity.setdefault("feature_widths", {})
    representations = identity.setdefault("representation", {})
    _normalize_legacy_joint_mapping(methods, path="document_linking.methods")
    _normalize_legacy_joint_mapping(
        ranks,
        path="document_linking.rank_distributions",
    )
    _normalize_legacy_joint_mapping(
        widths,
        path="document_linking.identity.feature_widths",
    )
    _normalize_legacy_joint_mapping(
        representations,
        path="document_linking.identity.representation",
    )
    _normalize_legacy_joint_comparisons(comparisons)
    for key in JOINT_KEYS:
        source = copy.deepcopy(sidecar["methods"][key])
        rank_distribution = source.pop("rank_distribution", None)
        method_comparisons = source.pop("comparisons", {})
        source["label"] = JOINT_LABELS[key]
        methods[key] = source
        if rank_distribution is not None:
            ranks[key] = rank_distribution
        comparisons.update(method_comparisons)
        widths[key] = int(
            sidecar["joint_checkpoints"][key]["dictionary_width"]
        )
        representations[key] = source.get("representation")
    _assert_preserved_methods(before, methods, path="document_linking.methods")
    _assert_preserved_methods(
        before_ranks,
        ranks,
        path="document_linking.rank_distributions",
    )
    _assert_preserved_methods(
        before_comparisons,
        comparisons,
        path="document_linking.comparisons",
    )
    _attach_method_metadata(
        payload,
        method_keys=methods,
        sidecar=sidecar,
        sidecar_rel=DOCUMENT_SIDECAR,
    )
    ctx.write_json("document_linking", DOCUMENT_JSON, payload)

    existing_fields, existing_rows = _load_csv(ctx.source(DOCUMENT_SUMMARY))
    extension_rows = []
    for key in JOINT_KEYS:
        row = methods[key]
        extension_rows.append(
            {
                "mode": key,
                "method": JOINT_LABELS[key],
                "recall_at_1": row["recall_at_1"],
                "recall_at_1_ci_low": row["recall_at_1_95ci"][0],
                "recall_at_1_ci_high": row["recall_at_1_95ci"][1],
                "recall_at_5": row["recall_at_5"],
                "mrr": row["mrr"],
                "mrr_ci_low": row["mrr_95ci"][0],
                "mrr_ci_high": row["mrr_95ci"][1],
                "median_rank": row["median_rank"],
            }
        )
    base_rows = [
        row
        for row in existing_rows
        if row.get("mode") not in JOINT_KEY_ALIASES
    ]
    fields = list(existing_fields)
    for row in extension_rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    ctx.write_csv(
        "document_linking",
        DOCUMENT_SUMMARY,
        fields,
        [*base_rows, *extension_rows],
    )

    plot_methods = _available_methods(methods)
    metrics = (
        ("recall_at_1", "Recall@1"),
        ("recall_at_5", "Recall@5"),
        ("mrr", "MRR"),
    )
    x = np.arange(len(metrics), dtype=np.float64)
    width = min(0.13, 0.78 / max(1, len(plot_methods)))
    offsets = (
        np.arange(len(plot_methods), dtype=np.float64)
        - (len(plot_methods) - 1) / 2
    ) * width
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.6, 5.8),
        gridspec_kw={"width_ratios": (1.65, 1.0)},
    )
    for method, offset in zip(plot_methods, offsets, strict=True):
        values = [float(methods[method][metric]) for metric, _ in metrics]
        axes[0].bar(
            x + offset,
            values,
            width=width * 0.92,
            color=_color(method),
            edgecolor="white",
            linewidth=0.7,
        )
    axes[0].set_xticks(x, [name for _, name in metrics])
    axes[0].set_ylim(0.0, 0.86)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Retrieval score")
    axes[0].set_title("(a) Lexically controlled same-document retrieval")
    axes[0].grid(axis="y", color="#DDE2EA", lw=0.75)
    controls = payload.get("controls", {})
    if "raw_mean_hidden" in controls:
        axes[0].axhline(
            float(controls["raw_mean_hidden"]["recall_at_1"]),
            color="#687386",
            ls=(0, (5, 3)),
            lw=1.2,
            alpha=0.85,
        )
        axes[0].text(
            -0.48,
            float(controls["raw_mean_hidden"]["recall_at_1"]) + 0.012,
            "raw hidden Recall@1",
            color="#596273",
            fontsize=8,
        )

    mean_ranks = np.asarray(
        [float(methods[method]["mean_rank"]) for method in plot_methods]
    )
    y = np.arange(len(plot_methods))
    axes[1].barh(
        y,
        mean_ranks,
        color=[_color(method) for method in plot_methods],
        height=0.62,
    )
    axes[1].set_yticks(y, [_label(method) for method in plot_methods])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean retrieval rank (lower is better)")
    axes[1].set_title("(b) Ranking error")
    axes[1].grid(axis="x", color="#DDE2EA", lw=0.75)
    for index, value in enumerate(mean_ranks):
        axes[1].text(
            value + max(mean_ranks) * 0.015,
            index,
            f"{value:.1f}",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    fig.legend(
        handles=_plot_legend(plot_methods),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        "Document linking with exact-length galleries and low lexical overlap",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    ctx.save_figure("document_linking", fig)

    tracked = {
        "results": DOCUMENT_JSON,
        "summary_csv": DOCUMENT_SUMMARY,
        "plot_png": PLOT_SPECS["document_linking"][0],
        "plot_pdf": PLOT_SPECS["document_linking"][1],
        "joint_extension_sidecar": DOCUMENT_SIDECAR,
        "joint_extension_features": Path(
            "document_linking/joint_extension_features.npz"
        ),
    }
    _stage_manifest(
        ctx,
        task="document_linking",
        manifest_rel=Path("document_linking/manifest.json"),
        tracked=tracked,
        sidecar_rel=DOCUMENT_SIDECAR,
        sidecar=sidecar,
    )
    return payload


def _publish_feature_dynamics(
    ctx: PublishContext,
    sidecar: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = copy.deepcopy(_load_json(ctx.source(FEATURE_JSON)))
    methods = payload.setdefault("methods", {})
    before = copy.deepcopy(methods)
    _normalize_legacy_joint_mapping(methods, path="feature_dynamics.methods")
    for key in JOINT_KEYS:
        methods[key] = copy.deepcopy(sidecar["methods"][key])
        methods[key]["label"] = JOINT_LABELS[key]
        methods[key]["checkpoint"] = _published_joint_checkpoint(
            sidecar,
            key,
        )
        methods[key].setdefault("robustness_views", [])
        if "per_view_statistics" not in methods[key]:
            row = methods[key]
            methods[key]["per_view_statistics"] = {
                "L32": {
                    "dominant_passages_covered": int(
                        row["dominant_passages_covered"]
                    ),
                    "mean_dominant_passage_margin": float(
                        row["mean_dominant_passage_margin"]
                    ),
                    "mean_within_passage_total_variation": float(
                        row["mean_within_passage_total_variation"]
                    ),
                    "per_feature": [
                        {
                            field: copy.deepcopy(feature[field])
                            for field in (
                                "dominant_active_fraction",
                                "dominant_passage_index",
                                "dominant_passage_margin",
                                "rank",
                                "segment_means",
                                "selective",
                                "within_passage_total_variation",
                            )
                        }
                        for feature in row["per_feature"]
                    ],
                    "selective_feature_count": int(
                        row["selective_feature_count"]
                    ),
                    "selective_passages_covered": int(
                        row["selective_passages_covered"]
                    ),
                    "total_passages": int(row["total_passages"]),
                }
            }
    _assert_preserved_methods(before, methods, path="feature_dynamics.methods")
    identity = payload.setdefault("identity", {})
    checkpoints = identity.setdefault("checkpoints", {})
    _normalize_legacy_joint_mapping(
        checkpoints,
        path="feature_dynamics.identity.checkpoints",
    )
    for key in JOINT_KEYS:
        checkpoints[key] = _published_joint_checkpoint(sidecar, key)
    _attach_method_metadata(
        payload,
        method_keys=methods,
        sidecar=sidecar,
        sidecar_rel=FEATURE_SIDECAR,
    )
    ctx.write_json("feature_dynamics", FEATURE_JSON, payload)

    arrays = _load_npz(ctx.source(FEATURE_NPZ))
    before_arrays = {key: value.copy() for key, value in arrays.items()}
    _normalize_legacy_joint_arrays(
        arrays,
        suffixes=("traces", "feature_ids", "mean_activations", "L32_traces"),
    )
    extension_arrays = _load_npz(ctx.source(FEATURE_SIDECAR_NPZ))
    arrays.update(extension_arrays)
    for key, value in before_arrays.items():
        if _is_joint_array_key(key):
            continue
        if not np.array_equal(arrays[key], value):
            raise AssertionError(
                f"publisher changed pre-existing feature trace array {key}"
            )
    ctx.write_npz("feature_dynamics", FEATURE_NPZ, arrays)

    plot_methods = _available_methods(methods)
    from evals.feature_dynamics.plot_cross_feature_semantic_manifold import (
        build_sequence_activation_figure,
        sequence_feature_annotations,
    )
    feature_annotations, _ = sequence_feature_annotations(
        ctx.eval_root,
        payload,
        methods=plot_methods,
    )
    fig = build_sequence_activation_figure(
        arrays=arrays,
        metadata=payload,
        feature_annotations=feature_annotations,
        methods=plot_methods,
    )
    ctx.save_figure("feature_dynamics", fig)

    tracked = {
        "results": FEATURE_JSON,
        "traces": FEATURE_NPZ,
        "plot_png": PLOT_SPECS["feature_dynamics"][0],
        "plot_pdf": PLOT_SPECS["feature_dynamics"][1],
        "joint_extension_sidecar": FEATURE_SIDECAR,
        "joint_extension_traces": FEATURE_SIDECAR_NPZ,
    }
    _stage_manifest(
        ctx,
        task="feature_dynamics",
        manifest_rel=Path(
            "feature_dynamics/sequence_activation_traces_manifest.json"
        ),
        tracked=tracked,
        sidecar_rel=FEATURE_SIDECAR,
        sidecar=sidecar,
    )
    return payload, arrays


def _synthesized_probe_representation(
    *,
    key: str,
    source: Mapping[str, Any],
    budgets: Sequence[int],
) -> dict[str, Any]:
    detailed = source.get("probe")
    if (
        isinstance(detailed, Mapping)
        and isinstance(detailed.get("low_label"), Mapping)
        and isinstance(detailed.get("full"), Mapping)
        and isinstance(detailed.get("ood"), Mapping)
    ):
        result = copy.deepcopy(dict(detailed))
        result["representation"] = key
        result["label"] = JOINT_LABELS[key]
        result["alpha"] = float(source["alpha"])
        return result

    summary = _arxiv_probe_summary(source)
    geometry = source["geometry"]
    test_balance = copy.deepcopy(geometry["test_class_balance"])
    ood_balance = copy.deepcopy(geometry["ood_class_balance"])
    return {
        "representation": key,
        "label": JOINT_LABELS[key],
        "alpha": float(source["alpha"]),
        "low_label": {
            str(budget): {
                "mean_accuracy": float(summary[f"low_label_{budget}"]),
                "uncertainty_available_in_sidecar": False,
            }
            for budget in budgets
        },
        "full": {
            "accuracy": float(summary["full_accuracy"]),
            **test_balance,
        },
        "ood": {
            "accuracy": float(summary["ood_accuracy"]),
            **ood_balance,
        },
        "high_level_transfer": summary,
        "measurement_availability": {
            "low_label_means": True,
            "low_label_seed_values": False,
            "low_label_standard_deviations": False,
            "feature_budget_curve": False,
            "full_and_ood_classwise_accuracy": True,
            "source": "ArXiv joint_extension sidecar",
        },
    }


def _arxiv_probe_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact transfer summary across sidecar schema revisions."""

    candidate = source.get("probe_summary")
    if isinstance(candidate, Mapping) and "low_label_auc" in candidate:
        return copy.deepcopy(dict(candidate))
    candidate = source.get("probe")
    if isinstance(candidate, Mapping) and "low_label_auc" in candidate:
        return copy.deepcopy(dict(candidate))
    if not isinstance(candidate, Mapping):
        raise ValueError("ArXiv joint method has no probe payload")
    low_label = candidate.get("low_label")
    full = candidate.get("full")
    ood = candidate.get("ood")
    if not all(isinstance(value, Mapping) for value in (low_label, full, ood)):
        raise ValueError("ArXiv joint probe lacks detailed or summary metrics")
    budgets = sorted(int(value) for value in low_label)
    values = np.asarray(
        [
            float(low_label[str(budget)]["mean_accuracy"])
            for budget in budgets
        ],
        dtype=np.float64,
    )
    log_budgets = np.log2(np.asarray(budgets, dtype=np.float64))
    integrate = (
        np.trapezoid
        if hasattr(np, "trapezoid")
        else np.trapz
    )
    low_label_auc = float(
        integrate(values, log_budgets)
        / (log_budgets[-1] - log_budgets[0])
    )
    chance = 1.0 / len(full["per_class_accuracy"])
    accuracies = {
        "full_accuracy": float(full["accuracy"]),
        "ood_accuracy": float(ood["accuracy"]),
        **{
            f"low_label_{budget}": float(
                low_label[str(budget)]["mean_accuracy"]
            )
            for budget in budgets
        },
    }
    normalized = {
        name: (value - chance) / (1.0 - chance)
        for name, value in accuracies.items()
    }
    return {
        "representation": source.get("representation"),
        **accuracies,
        "chance_normalized": normalized,
        "high_level_transfer_score": float(np.mean(list(normalized.values()))),
        "low_label_auc": low_label_auc,
        "chance_normalized_low_label_auc": (
            low_label_auc - chance
        )
        / (1.0 - chance),
        "ood_retention": float(ood["accuracy"])
        / max(float(full["accuracy"]), 1e-12),
        "worst_class_ood_accuracy": float(
            ood["worst_class_accuracy"]
        ),
        "ood_class_accuracy_std": float(ood["class_accuracy_std"]),
    }


def _publish_label_efficiency(
    ctx: PublishContext,
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(_load_json(ctx.source(PROBE_JSON)))
    representations = payload.setdefault("representations", {})
    before_representations = copy.deepcopy(representations)
    high_level = payload.setdefault("high_level_transfer", {})
    high_methods = high_level.setdefault("methods", {})
    before_high_methods = copy.deepcopy(high_methods)
    budgets = [
        int(value)
        for value in payload.get("metadata", {}).get(
            "low_label_budgets",
            sidecar.get("protocol", {}).get("low_label_budgets", []),
        )
    ]
    if not budgets:
        raise ValueError("label efficiency has no low-label budgets")
    metadata = payload.setdefault("metadata", {})
    widths = metadata.setdefault("feature_widths", {})
    mean_nnz = metadata.setdefault("mean_nnz", {})
    _normalize_legacy_joint_mapping(
        representations,
        path="label_efficiency.representations",
    )
    _normalize_legacy_joint_mapping(
        high_methods,
        path="label_efficiency.high_level_transfer.methods",
    )
    _normalize_legacy_joint_mapping(
        widths,
        path="label_efficiency.metadata.feature_widths",
    )
    _normalize_legacy_joint_mapping(
        mean_nnz,
        path="label_efficiency.metadata.mean_nnz",
    )
    for key in JOINT_KEYS:
        source = sidecar["methods"][key]
        summary = _arxiv_probe_summary(source)
        summary["representation"] = key
        summary["label"] = JOINT_LABELS[key]
        summary["alpha"] = float(source["alpha"])
        high_methods[key] = summary
        representations[key] = _synthesized_probe_representation(
            key=key,
            source=source,
            budgets=budgets,
        )
        widths[key] = int(
            sidecar["joint_checkpoints"][key]["dictionary_width"]
        )
        mean_nnz[key] = {
            split: float(row["mean_nnz"])
            for split, row in source.get("encoding", {}).items()
        }
    _assert_preserved_methods(
        before_representations,
        representations,
        path="label_efficiency.representations",
    )
    _assert_preserved_methods(
        before_high_methods,
        high_methods,
        path="label_efficiency.high_level_transfer.methods",
    )
    _attach_method_metadata(
        payload,
        method_keys=high_methods,
        sidecar=sidecar,
        sidecar_rel=ARXIV_SIDECAR,
    )
    ctx.write_json("label_efficiency", PROBE_JSON, payload)

    plot_methods = _available_methods(high_methods)
    x = np.asarray(budgets, dtype=np.float64)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.5),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
    )
    for method in plot_methods:
        row = high_methods[method]
        values = np.asarray(
            [float(row[f"low_label_{budget}"]) for budget in budgets]
        )
        axes[0].plot(
            x,
            values,
            color=_color(method),
            marker=_marker(method),
            markersize=5.5,
            lw=2.2,
            label=_label(method, row),
        )
        representation_name = str(row.get("representation", ""))
        detail = representations.get(representation_name)
        if detail is None:
            detail = representations.get(method)
        if isinstance(detail, dict):
            stds = []
            for budget in budgets:
                value = (
                    detail.get("low_label", {})
                    .get(str(budget), {})
                    .get("std_accuracy")
                )
                stds.append(_finite(value))
            stds_array = np.asarray(stds)
            mask = np.isfinite(stds_array)
            if mask.any():
                axes[0].fill_between(
                    x[mask],
                    values[mask] - stds_array[mask],
                    values[mask] + stds_array[mask],
                    color=_color(method),
                    alpha=0.10,
                    linewidth=0,
                )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(x, [str(value) for value in budgets])
    axes[0].set_ylim(0.38, 0.88)
    axes[0].set_xlabel("Training examples per class")
    axes[0].set_ylabel("8-way test accuracy")
    axes[0].set_title("(a) Frozen-feature low-label scaling")
    axes[0].grid(color="#DDE2EA", lw=0.75)

    summary_metrics = (
        ("low_label_auc", "Low-label AUC"),
        ("full_accuracy", "Full-data accuracy"),
    )
    metric_x = np.arange(len(summary_metrics), dtype=np.float64)
    width = min(0.13, 0.78 / max(1, len(plot_methods)))
    offsets = (
        np.arange(len(plot_methods)) - (len(plot_methods) - 1) / 2
    ) * width
    for method, offset in zip(plot_methods, offsets, strict=True):
        axes[1].bar(
            metric_x + offset,
            [
                float(high_methods[method][name])
                for name, _ in summary_metrics
            ],
            width=width * 0.92,
            color=_color(method),
            edgecolor="white",
            linewidth=0.6,
        )
    axes[1].set_xticks(metric_x, [label for _, label in summary_metrics])
    axes[1].set_ylim(0.65, 0.87)
    axes[1].set_ylabel("Accuracy / normalized curve area")
    axes[1].set_title("(b) Aggregate label efficiency")
    axes[1].grid(axis="y", color="#DDE2EA", lw=0.75)
    fig.legend(
        handles=_plot_legend(plot_methods),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
    )
    fig.suptitle(
        "Label efficiency on frozen ArXiv representations",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    ctx.save_figure("label_efficiency", fig)

    tracked = {
        "results": PROBE_JSON,
        "plot_png": PLOT_SPECS["label_efficiency"][0],
        "plot_pdf": PLOT_SPECS["label_efficiency"][1],
        "joint_extension_sidecar": ARXIV_SIDECAR,
        "base_feature_manifest": ARXIV_FEATURE_MANIFEST,
        "joint_extension_embeddings": ARXIV_EMBEDDINGS,
        **{
            f"joint_encoded_{index:02d}": path
            for index, path in enumerate(ARXIV_ENCODED_SPLITS)
        },
    }
    _stage_manifest(
        ctx,
        task="label_efficiency",
        manifest_rel=Path("label_efficiency/linear_probe_manifest.json"),
        tracked=tracked,
        sidecar_rel=ARXIV_SIDECAR,
        sidecar=sidecar,
    )
    return payload


def _publish_semantic_geometry(
    ctx: PublishContext,
    sidecar: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = copy.deepcopy(_load_json(ctx.source(GEOMETRY_JSON)))
    methods = payload.setdefault("methods", {})
    before = copy.deepcopy(methods)
    identity = payload.setdefault("identity", {})
    widths = identity.setdefault("feature_widths", {})
    _normalize_legacy_joint_mapping(methods, path="semantic_geometry.methods")
    _normalize_legacy_joint_mapping(
        widths,
        path="semantic_geometry.identity.feature_widths",
    )
    comparisons = payload.get("comparisons")
    if isinstance(comparisons, dict):
        _normalize_legacy_joint_comparisons(comparisons)
    for key in JOINT_KEYS:
        methods[key] = copy.deepcopy(sidecar["methods"][key]["geometry"])
        methods[key]["label"] = JOINT_LABELS[key]
        widths[key] = int(
            sidecar["joint_checkpoints"][key]["dictionary_width"]
        )
    _assert_preserved_methods(before, methods, path="semantic_geometry.methods")
    _attach_method_metadata(
        payload,
        method_keys=methods,
        sidecar=sidecar,
        sidecar_rel=ARXIV_SIDECAR,
    )
    ctx.write_json("semantic_geometry", GEOMETRY_JSON, payload)

    arrays = _load_npz(ctx.source(GEOMETRY_NPZ))
    before_arrays = {key: value.copy() for key, value in arrays.items()}
    _normalize_legacy_joint_arrays(arrays, suffixes=("xy",))
    joint_arrays = _load_npz(ctx.source(ARXIV_EMBEDDINGS))
    if not np.array_equal(arrays["labels"].astype(str), joint_arrays["labels"].astype(str)):
        raise ValueError("joint/base semantic embedding labels are not aligned")
    for key in JOINT_KEYS:
        arrays[f"{key}_xy"] = joint_arrays[f"{key}_xy"]
    for key, value in before_arrays.items():
        if _is_joint_array_key(key):
            continue
        if not np.array_equal(arrays[key], value):
            raise AssertionError(
                f"publisher changed pre-existing semantic embedding {key}"
            )
    ctx.write_npz("semantic_geometry", GEOMETRY_NPZ, arrays)

    labels = arrays["labels"].astype(str)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    for label in unique_labels:
        if label not in DOMAIN_COLORS:
            DOMAIN_COLORS[label] = _color(f"domain:{label}")
            DOMAIN_SHORT[label] = label
    plot_methods = [
        method
        for method in _available_methods(methods)
        if f"{method}_xy" in arrays
    ]
    expected_plot_count = len(METHOD_ORDER)
    if len(plot_methods) != expected_plot_count:
        raise ValueError(
            "semantic geometry must contain all baseline and Joint embeddings; "
            f"found {plot_methods}"
        )
    columns = 4
    rows = math.ceil(expected_plot_count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(20.5, 5.0 * rows),
        squeeze=False,
    )
    for ax, method in zip(axes.reshape(-1), plot_methods, strict=True):
        coordinates = np.asarray(arrays[f"{method}_xy"], dtype=np.float64)
        for domain in unique_labels:
            mask = labels == domain
            points = coordinates[mask]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=7,
                color=DOMAIN_COLORS[domain],
                alpha=0.52,
                linewidth=0,
                rasterized=True,
            )
        metrics = methods[method]["geometry"]
        ax.set_title(
            f"{_label(method, methods[method])}\n"
            f"silhouette {float(metrics['silhouette']):.3f} · "
            f"neighbor purity {float(metrics['neighbor_purity']):.3f}",
            color=_color(method),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(_color(method))
            spine.set_linewidth(1.6)
    for ax in axes.reshape(-1)[len(plot_methods) :]:
        ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DOMAIN_COLORS[domain],
            markeredgecolor="none",
            label=DOMAIN_SHORT[domain],
        )
        for domain in unique_labels
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(8, len(handles)),
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        "Semantic geometry of held-out ArXiv sparse codes",
        x=0.055,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    ctx.save_figure("semantic_geometry", fig)

    tracked = {
        "results": GEOMETRY_JSON,
        "embeddings": GEOMETRY_NPZ,
        "plot_png": PLOT_SPECS["semantic_geometry"][0],
        "plot_pdf": PLOT_SPECS["semantic_geometry"][1],
        "joint_extension_sidecar": ARXIV_SIDECAR,
        "joint_extension_embeddings": ARXIV_EMBEDDINGS,
        "base_feature_manifest": ARXIV_FEATURE_MANIFEST,
        **{
            f"joint_encoded_{index:02d}": path
            for index, path in enumerate(ARXIV_ENCODED_SPLITS)
        },
    }
    _stage_manifest(
        ctx,
        task="semantic_geometry",
        manifest_rel=Path(
            "semantic_geometry/representation_geometry/"
            "representation_geometry_manifest.json"
        ),
        tracked=tracked,
        sidecar_rel=ARXIV_SIDECAR,
        sidecar=sidecar,
    )
    return payload, arrays


def _probe_method_summary(
    probe: Mapping[str, Any],
    method: str,
) -> Mapping[str, Any]:
    methods = probe.get("high_level_transfer", {}).get("methods", {})
    if method not in methods:
        raise KeyError(f"missing probe summary for {method}")
    return methods[method]


def _publish_temporal_robustness(
    ctx: PublishContext,
    *,
    probe: Mapping[str, Any],
    geometry: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    live = ctx.eval_root / TEMPORAL_JSON
    payload = copy.deepcopy(_load_json(live)) if live.is_file() else {}
    existing_methods = copy.deepcopy(payload.get("methods", {}))
    _normalize_legacy_joint_mapping(
        existing_methods,
        path="temporal_robustness.methods",
    )
    method_keys = _available_methods(geometry.get("methods", {}))
    methods: dict[str, Any] = {}
    for method in method_keys:
        probe_row = _probe_method_summary(probe, method)
        geometry_row = geometry["methods"][method]
        ood_balance = geometry_row["ood_class_balance"]
        methods[method] = {
            "label": _label(method, geometry_row),
            "representation": probe_row.get("representation", method),
            "full_accuracy": float(probe_row["full_accuracy"]),
            "ood_accuracy": float(probe_row["ood_accuracy"]),
            "ood_retention": float(probe_row["ood_retention"]),
            "worst_class_ood_accuracy": float(
                probe_row["worst_class_ood_accuracy"]
            ),
            "ood_class_accuracy_std": float(
                probe_row["ood_class_accuracy_std"]
            ),
            "per_class_ood_accuracy": copy.deepcopy(
                ood_balance["per_class_accuracy"]
            ),
            "cross_time_knn_accuracy": float(
                geometry_row["cross_time_neighbors"]["knn_accuracy"]
            ),
            "cross_time_neighbor_purity": float(
                geometry_row["cross_time_neighbors"]["neighbor_purity"]
            ),
        }
        if method in JOINT_KEYS:
            methods[method]["alpha"] = float(
                sidecar["methods"][method]["alpha"]
            )
    for key, row in existing_methods.items():
        if key not in JOINT_KEYS and key in methods and methods[key] != row:
            # A prior publisher version may already have produced this derived
            # file.  The current authoritative probe/geometry values win, but
            # only if its metric payload is exactly derivable from those inputs.
            for metric, value in row.items():
                if metric in methods[key] and methods[key][metric] != value:
                    raise AssertionError(
                        f"existing temporal result disagrees for {key}.{metric}"
                    )
    payload.update(
        {
            "format": "chunk-saes-temporal-robustness-v1",
            "complete": True,
            "definition": (
                "Frozen linear-probe transfer from the in-distribution ArXiv "
                "split to the held-out publication-year split, with domain-wise "
                "accuracy and cross-time nearest-neighbor stability."
            ),
            "methods": methods,
        }
    )
    _attach_method_metadata(
        payload,
        method_keys=methods,
        sidecar=sidecar,
        sidecar_rel=ARXIV_SIDECAR,
    )
    payload["sources"] = {
        "label_efficiency": PROBE_JSON.as_posix(),
        "semantic_geometry": GEOMETRY_JSON.as_posix(),
    }
    ctx.write_json("temporal_robustness", TEMPORAL_JSON, payload)

    labels = list(
        next(iter(methods.values()))["per_class_ood_accuracy"].keys()
    )
    matrix = np.asarray(
        [
            [
                float(methods[method]["per_class_ood_accuracy"][label])
                for label in labels
            ]
            for method in method_keys
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.4),
        gridspec_kw={"width_ratios": (1.55, 1.0)},
    )
    vmin = max(0.0, math.floor(float(matrix.min()) * 20) / 20)
    vmax = min(1.0, math.ceil(float(matrix.max()) * 20) / 20)
    image = axes[0].imshow(
        matrix,
        aspect="auto",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=max(vmin + 0.05, vmax),
    )
    axes[0].set_xticks(
        np.arange(len(labels)),
        [DOMAIN_SHORT.get(label, label) for label in labels],
        rotation=35,
        ha="right",
    )
    axes[0].set_yticks(
        np.arange(len(method_keys)),
        [_label(method, methods[method]) for method in method_keys],
    )
    axes[0].set_title("(a) Held-out-year accuracy by scientific domain")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axes[0].text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if value > (vmin + vmax) / 2 else "#1F2937",
            )
    fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.025)

    y = np.arange(len(method_keys))
    height = 0.24
    axes[1].barh(
        y - height,
        [methods[method]["ood_accuracy"] for method in method_keys],
        height=height,
        color=[_color(method) for method in method_keys],
        alpha=0.95,
        label="OOD accuracy",
    )
    axes[1].barh(
        y,
        [methods[method]["ood_retention"] for method in method_keys],
        height=height,
        color=[_color(method) for method in method_keys],
        alpha=0.55,
        hatch="//",
        label="OOD / in-distribution retention",
    )
    axes[1].barh(
        y + height,
        [
            methods[method]["worst_class_ood_accuracy"]
            for method in method_keys
        ],
        height=height,
        color=[_color(method) for method in method_keys],
        alpha=0.30,
        label="Worst-domain OOD accuracy",
    )
    axes[1].set_yticks(y, [_label(method) for method in method_keys])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.55, 1.01)
    axes[1].set_xlabel("Score")
    axes[1].set_title("(b) Transfer stability summary")
    axes[1].grid(axis="x", color="#DDE2EA", lw=0.75)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        fontsize=8.2,
        columnspacing=1.0,
        handlelength=2.0,
    )
    fig.suptitle(
        "Temporal robustness of frozen sparse representations",
        x=0.055,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    ctx.save_figure("temporal_robustness", fig)

    tracked = {
        "results": TEMPORAL_JSON,
        "plot_png": PLOT_SPECS["temporal_robustness"][0],
        "plot_pdf": PLOT_SPECS["temporal_robustness"][1],
        "joint_extension_sidecar": ARXIV_SIDECAR,
        "base_feature_manifest": ARXIV_FEATURE_MANIFEST,
        "joint_extension_embeddings": ARXIV_EMBEDDINGS,
        **{
            f"joint_encoded_{index:02d}": path
            for index, path in enumerate(ARXIV_ENCODED_SPLITS)
        },
    }
    _stage_manifest(
        ctx,
        task="temporal_robustness",
        manifest_rel=Path(
            "temporal_robustness/temporal_robustness_manifest.json"
        ),
        tracked=tracked,
        sidecar_rel=ARXIV_SIDECAR,
        sidecar=sidecar,
    )
    return payload


def _smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    width = min(window, values.size if values.size % 2 else values.size - 1)
    width = max(3, width)
    pad = width // 2
    return np.convolve(
        np.pad(values, (pad, pad), mode="edge"),
        np.full(width, 1.0 / width),
        mode="valid",
    )


def _publish_rfve(
    ctx: PublishContext,
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(_load_json(ctx.source(RFVE_JSON)))
    methods = payload.setdefault("methods", {})
    before = copy.deepcopy(methods)
    references = payload.setdefault("references", {})
    before_references = copy.deepcopy(references)
    identity = payload.setdefault("identity", {})
    reference_fves = identity.setdefault("reference_fves", {})
    _normalize_legacy_joint_mapping(methods, path="rfve.methods")
    _normalize_legacy_joint_mapping(references, path="rfve.references")
    _normalize_legacy_joint_mapping(
        reference_fves,
        path="rfve.identity.reference_fves",
    )
    for key in JOINT_KEYS:
        methods[key] = copy.deepcopy(sidecar["methods"][key])
        methods[key]["label"] = JOINT_LABELS[key]
        references[key] = copy.deepcopy(methods[key].get("reference", {}))
        reference_fves[key] = float(methods[key]["joint_reference_fve"])
    _assert_preserved_methods(before, methods, path="rfve.methods")
    _assert_preserved_methods(
        before_references,
        references,
        path="rfve.references",
    )
    _attach_method_metadata(
        payload,
        method_keys=methods,
        sidecar=sidecar,
        sidecar_rel=RFVE_SIDECAR,
    )
    ctx.write_json("rfve", RFVE_JSON, payload)

    plot_methods = _available_methods(methods)
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    selected_points: dict[str, float] = {}
    for method in plot_methods:
        row = methods[method]
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"RFVE method {method} has no trajectory")
        occurrences = np.asarray(
            [
                float(point["samples_seen"]) / 1_000_000_000
                for point in trajectory
            ]
        )
        values = np.asarray([float(point["rfve"]) for point in trajectory])
        selected_value = float(row["rfve"])
        selected_step = float(row["selected_step"])
        selected_occurrences = selected_step * 32_000 / 1_000_000_000
        selected_points[method] = selected_value
        line_style = (0, (5, 2.2)) if method in JOINT_KEYS else "-"
        ax.plot(
            occurrences,
            values,
            color=_color(method),
            lw=0.8,
            alpha=0.13,
            ls=line_style,
        )
        ax.plot(
            occurrences,
            _smooth(values),
            color=_color(method),
            lw=2.6,
            ls=line_style,
            label=f"{_label(method, row)}  {selected_value:.3f}",
        )
        ax.scatter(
            [selected_occurrences],
            [selected_value],
            marker=_marker(method),
            s=72,
            color=_color(method),
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
    ax.axhline(
        1.0,
        color="#667085",
        lw=1.1,
        ls=(0, (4, 4)),
        label="Dense / identity reference",
    )
    ax.set_xlim(
        0,
        max(
            float(point["samples_seen"]) / 1_000_000_000
            for row in methods.values()
            for point in row["trajectory"]
        )
        * 1.015,
    )
    all_values = [
        float(point["rfve"])
        for row in methods.values()
        for point in row["trajectory"]
    ]
    ax.set_ylim(max(-0.05, min(all_values) - 0.02), 1.035)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Training occurrences seen (billions)")
    ax.set_ylabel("Reference FVE (RFVE)")
    ax.set_title(
        "Reference-normalized reconstruction across 1B training occurrences",
        loc="left",
        pad=16,
    )
    ax.grid(color="#DDE2EA", lw=0.75)
    ax.legend(
        loc="lower right",
        ncol=2,
        fontsize=8.3,
        title="Validation-selected checkpoint",
    )
    _publication_note(ax)
    fig.tight_layout()
    ctx.save_figure("rfve", fig)

    tracked = {
        "results": RFVE_JSON,
        "plot_png": PLOT_SPECS["rfve"][0],
        "plot_pdf": PLOT_SPECS["rfve"][1],
        "joint_extension_sidecar": RFVE_SIDECAR,
    }
    _stage_manifest(
        ctx,
        task="rfve",
        manifest_rel=Path("rfve/training_fidelity_manifest.json"),
        tracked=tracked,
        sidecar_rel=RFVE_SIDECAR,
        sidecar=sidecar,
    )
    return payload


def _reasoning_sidecar_csv(
    ctx: PublishContext,
    sidecar: Mapping[str, Any],
    name: str,
) -> Path:
    files = sidecar.get("files", {})
    relative = files.get(name)
    if not isinstance(relative, str):
        raise ValueError(f"reasoning sidecar has no files.{name}")
    return ctx.source(Path("reasoning") / relative)


def _enrich_reasoning_rows(
    rows_by_view: Mapping[str, list[dict[str, str]]],
) -> None:
    native = {
        (row["method"], row["category"]): row
        for row in rows_by_view["native"]
    }
    background = {
        (row["method"], row["category"]): row
        for row in rows_by_view["background"]
    }
    for row in rows_by_view["native"]:
        bg = background.get((row["method"], row["category"]))
        if bg is None:
            continue
        row["pile_background_calibrated_count"] = bg.get("active_count", "")
        row["pile_background_calibrated_rate"] = bg.get("rate", "")
        row["pile_background_checkpoint_count"] = bg.get(
            "checkpoint_active_count", ""
        )
        row["pile_background_checkpoint_rate"] = bg.get(
            "checkpoint_activation_rate", ""
        )
        row["pile_background_legacy_topk_rate"] = bg.get(
            "legacy_topk_activation_rate", ""
        )
    for row in rows_by_view["cue_free"]:
        base = native.get((row["method"], row["category"]))
        if base is None:
            continue
        native_rate = _finite(base.get("rate"))
        rate = _finite(row.get("rate"))
        row["native_recall"] = native_rate
        row["transfer_ratio"] = (
            rate / native_rate
            if math.isfinite(rate)
            and math.isfinite(native_rate)
            and native_rate > 0
            else ""
        )
    for row in rows_by_view["cue_only"]:
        row["cue_only_false_activation_rate"] = row.get("rate", "")
    for row in rows_by_view["other_scenario"]:
        row["other_scenario_false_activation_rate"] = row.get("rate", "")


def _publish_reasoning(
    ctx: PublishContext,
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(_load_json(ctx.source(REASONING_JSON)))
    _normalize_legacy_joint_metadata(payload)
    summary_methods = _normalize_legacy_joint_sequence(
        payload.get("methods", [])
    )
    summary_methods = [
        method for method in summary_methods if method not in JOINT_KEYS
    ]
    summary_methods.extend(JOINT_KEYS)
    payload["methods"] = _available_methods(summary_methods)
    payload["method_labels"] = {
        method: _label(method) for method in payload["methods"]
    }
    metric_fields = (
        "native_recall_mean",
        "cue_free_recall_mean",
        "cue_only_false_activation_mean",
        "other_scenario_false_activation_mean",
    )
    for field_name in metric_fields:
        mapping = payload.setdefault(field_name, {})
        _normalize_legacy_joint_mapping(
            mapping,
            path=f"reasoning.{field_name}",
        )
        for key in JOINT_KEYS:
            mapping[key] = sidecar["methods"][key].get(field_name)
    payload["joint_extension"] = _extension_metadata(
        sidecar=sidecar,
        sidecar_rel=REASONING_SIDECAR,
    )

    side_csv_names = {
        "native": "native",
        "cue_free": "cue_free",
        "cue_only": "cue_only",
        "other_scenario": "other_scenario",
        "background": "background",
    }
    rows_by_view: dict[str, list[dict[str, str]]] = {}
    for view, side_name in side_csv_names.items():
        _, rows = _load_csv(
            _reasoning_sidecar_csv(ctx, sidecar, side_name)
        )
        for row in rows:
            method = row.get("method")
            if method in JOINT_KEYS:
                row["method_label"] = JOINT_LABELS[method]
        rows_by_view[view] = rows
    _enrich_reasoning_rows(rows_by_view)

    csv_mapping = {
        "native": Path("reasoning/native_recall.csv"),
        "cue_free": Path("reasoning/cue_free_recall.csv"),
        "cue_only": Path("reasoning/cue_only_false_activation.csv"),
        "other_scenario": Path(
            "reasoning/cross_scenario_false_activation.csv"
        ),
        "background": Path("reasoning/pile_background_summary.csv"),
    }
    merged_rows: dict[str, list[dict[str, Any]]] = {}
    for view, target in csv_mapping.items():
        fields, base_rows = _load_csv(ctx.source(target))
        merged_fields, rows = _merge_csv_rows(
            fields,
            base_rows,
            rows_by_view[view],
        )
        ctx.write_csv("reasoning", target, merged_fields, rows)
        merged_rows[view] = rows

    calibrated = payload.setdefault(
        "pile_background_calibrated_activation_mean",
        {},
    )
    checkpoint = payload.setdefault(
        "pile_background_checkpoint_activation_mean",
        {},
    )
    _normalize_legacy_joint_mapping(
        calibrated,
        path="reasoning.pile_background_calibrated_activation_mean",
    )
    _normalize_legacy_joint_mapping(
        checkpoint,
        path="reasoning.pile_background_checkpoint_activation_mean",
    )
    for method in payload["methods"]:
        method_rows = [
            row
            for row in merged_rows["background"]
            if row.get("method") == method
        ]
        calibrated[method] = _mean(row.get("rate") for row in method_rows)
        checkpoint[method] = _mean(
            row.get("checkpoint_activation_rate") for row in method_rows
        )
    payload["qualified_feature_count_by_method"] = {
        method: sum(
            str(row.get("feature_status", "qualified")) == "qualified"
            for row in merged_rows["native"]
            if row.get("method") == method
        )
        for method in payload["methods"]
    }
    payload["qualified_feature_count"] = int(
        sum(payload["qualified_feature_count_by_method"].values())
    )
    ctx.write_json("reasoning", REASONING_JSON, payload)

    feature_top1_rel = Path("reasoning/feature_top1.json")
    feature_top1 = copy.deepcopy(_load_json(ctx.source(feature_top1_rel)))
    _normalize_legacy_joint_metadata(feature_top1)
    frozen_methods = feature_top1.setdefault("methods", {})
    before = copy.deepcopy(frozen_methods)
    _normalize_legacy_joint_mapping(
        frozen_methods,
        path="reasoning.feature_top1.methods",
    )
    for key in JOINT_KEYS:
        frozen_methods[key] = copy.deepcopy(
            sidecar["methods"][key].get("selected", {})
        )
        for record in frozen_methods[key].values():
            if isinstance(record, dict):
                record["method"] = key
                record["method_label"] = JOINT_LABELS[key]
    _assert_preserved_methods(
        before,
        frozen_methods,
        path="reasoning.feature_top1.methods",
    )
    feature_top1["method_order"] = payload["methods"]
    feature_top1["method_labels"] = payload["method_labels"]
    feature_top1["joint_extension"] = payload["joint_extension"]
    ctx.write_json("reasoning", feature_top1_rel, feature_top1)

    calibration_rel = Path("reasoning/feature_calibration.csv")
    calibration_fields, calibration_base = _load_csv(
        ctx.source(calibration_rel)
    )
    calibration_extension: list[dict[str, Any]] = []
    for key in JOINT_KEYS:
        for category, selected in sidecar["methods"][key].get(
            "selected", {}
        ).items():
            row = copy.deepcopy(selected)
            row.update(
                {
                    "method": key,
                    "method_label": JOINT_LABELS[key],
                    "category": category,
                }
            )
            calibration_extension.append(row)
    calibration_fields, calibration_rows = _merge_csv_rows(
        calibration_fields,
        calibration_base,
        calibration_extension,
    )
    ctx.write_csv(
        "reasoning",
        calibration_rel,
        calibration_fields,
        calibration_rows,
    )

    scenarios = list(payload.get("scenarios", []))
    if not scenarios:
        scenarios = list(
            dict.fromkeys(row["category"] for row in merged_rows["native"])
        )
    display = {
        "causal_mechanism": "Causal\nchain",
        "math_derivation": "Math\nderivation",
        "planning": "Planning",
        "backtracking": "Backtracking",
        "conditional_assumption": "Conditional",
        "induction": "Induction",
    }
    methods = payload["methods"]
    x = np.arange(len(scenarios), dtype=np.float64)
    width = min(0.13, 0.80 / max(1, len(methods)))
    offsets = (
        np.arange(len(methods), dtype=np.float64)
        - (len(methods) - 1) / 2
    ) * width
    fig, axes = plt.subplots(1, 3, figsize=(23.5, 7.8), sharey=True)
    panels = (
        ("native", "A  Target relation recall", True),
        ("cue_free", "B  Recall after removing surface cues", False),
        ("cue_only", "C  Cue kept, relation removed", False),
    )
    for panel_index, (view, title, include_pile) in enumerate(panels):
        ax = axes[panel_index]
        lookup = {
            (row["method"], row["category"]): row
            for row in merged_rows[view]
        }
        categories = [*scenarios, "pile"] if include_pile else scenarios
        panel_x = np.arange(len(categories), dtype=np.float64)
        background_lookup = {
            (row["method"], row["category"]): row
            for row in merged_rows["background"]
        }
        for method, offset in zip(methods, offsets, strict=True):
            values = []
            statuses = []
            lows = []
            highs = []
            for category in categories:
                if category == "pile":
                    candidates = [
                        background_lookup.get((method, scenario))
                        for scenario in scenarios
                    ]
                    candidates = [
                        row for row in candidates if row is not None
                    ]
                    value = _mean(row.get("rate") for row in candidates)
                    values.append(0.0 if value is None else value)
                    statuses.append("qualified" if candidates else "unavailable")
                    lows.append(float("nan"))
                    highs.append(float("nan"))
                    continue
                row = lookup.get((method, category))
                if row is None:
                    values.append(0.0)
                    statuses.append("unavailable")
                    lows.append(float("nan"))
                    highs.append(float("nan"))
                    continue
                status = str(row.get("feature_status", "qualified"))
                statuses.append(status)
                value = _finite(row.get("rate"), default=0.0)
                values.append(value if status == "qualified" else 0.0)
                lows.append(_finite(row.get("ci_low")))
                highs.append(_finite(row.get("ci_high")))
            bars = ax.bar(
                panel_x + offset,
                values,
                width=width * 0.92,
                color=_color(method),
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, category, value, status in zip(
                bars, categories, values, statuses, strict=True
            ):
                if category == "pile":
                    bar.set_hatch("//")
                if status != "qualified":
                    bar.set_facecolor("#E4E7EC")
                    bar.set_edgecolor("#667085")
                    bar.set_hatch("///")
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.015,
                        "NQ",
                        ha="center",
                        va="bottom",
                        fontsize=5.7,
                    )
                elif value >= 0.08:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(0.98, value + 0.012),
                        f"{value:.0%}",
                        ha="center",
                        va="bottom",
                        fontsize=6.1,
                        rotation=90,
                    )
            lows_array = np.asarray(lows)
            highs_array = np.asarray(highs)
            values_array = np.asarray(values)
            mask = np.isfinite(lows_array) & np.isfinite(highs_array)
            if mask.any():
                ax.errorbar(
                    (panel_x + offset)[mask],
                    values_array[mask],
                    yerr=np.vstack(
                        (
                            np.maximum(
                                0.0, values_array[mask] - lows_array[mask]
                            ),
                            np.maximum(
                                0.0, highs_array[mask] - values_array[mask]
                            ),
                        )
                    ),
                    fmt="none",
                    ecolor="#344054",
                    elinewidth=0.65,
                    capsize=1.5,
                )
        tick_labels = [display.get(category, category) for category in categories]
        if include_pile:
            tick_labels[-1] = "Unrelated\nPile"
        ax.set_xticks(panel_x, tick_labels)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Reasoning scenario")
        ax.grid(axis="y", color="#DDE2EA", lw=0.75)
        ax.axhline(0.05, color="#98A2B3", ls=(0, (4, 4)), lw=0.8)
    axes[0].set_ylabel("Document-level feature activation rate")
    fig.legend(
        handles=_plot_legend(methods),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
    )
    fig.suptitle(
        "Reasoning-feature specificity after discovery-time freezing",
        x=0.04,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    ctx.save_figure("reasoning", fig)

    manifest_tracked = {
        "results_summary": REASONING_JSON,
        "feature_top1": feature_top1_rel,
        "feature_calibration": calibration_rel,
        "native_recall": csv_mapping["native"],
        "cue_free_recall": csv_mapping["cue_free"],
        "cue_only_false_activation": csv_mapping["cue_only"],
        "cross_scenario_false_activation": csv_mapping["other_scenario"],
        "pile_background_summary": csv_mapping["background"],
        "primary_plot": PLOT_SPECS["reasoning"][0],
        "plot_pdf": PLOT_SPECS["reasoning"][1],
        "joint_extension_sidecar": REASONING_SIDECAR,
        **{
            f"joint_extension_{name}": Path("reasoning") / str(relative)
            for name, relative in sidecar.get("files", {}).items()
        },
    }
    _stage_manifest(
        ctx,
        task="reasoning",
        manifest_rel=Path("reasoning/manifest.json"),
        tracked=manifest_tracked,
        sidecar_rel=REASONING_SIDECAR,
        sidecar=sidecar,
    )
    return payload


def _validate_json_methods(
    path: Path,
    *,
    method_path: Sequence[str],
) -> None:
    value: Any = _load_json(path)
    for key in method_path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{path}: missing {'.'.join(method_path)}")
        value = value[key]
    if isinstance(value, dict):
        keys = set(value)
    elif isinstance(value, list):
        keys = set(value)
    else:
        raise ValueError(f"{path}: method collection is not a mapping/list")
    missing = set(JOINT_KEYS) - keys
    if missing:
        raise ValueError(f"{path}: missing published methods {sorted(missing)}")
    root = _load_json(path)
    labels = root.get("method_labels", {})
    for key in JOINT_KEYS:
        if labels.get(key) != JOINT_LABELS[key]:
            raise ValueError(f"{path}: incorrect published label for {key}")


def _validate_manifest(ctx: PublishContext, relative: Path) -> None:
    path = ctx.staged(relative)
    manifest = _load_json(path)
    expected = manifest.get("artifact_digest")
    body = dict(manifest)
    body.pop("artifact_digest", None)
    actual = _json_digest(body)
    if expected != actual:
        raise ValueError(
            f"{relative}: manifest digest mismatch {actual} != {expected}"
        )
    for record in manifest.get("files", {}).values():
        if not isinstance(record, dict):
            raise ValueError(f"{relative}: invalid file record")
        target_from_manifest = (
            ctx.eval_root / relative.parent / str(record["path"])
        ).resolve()
        try:
            target_rel = target_from_manifest.relative_to(ctx.eval_root)
        except ValueError:
            target_path = target_from_manifest
        else:
            target_path = ctx.path_for_validation(target_rel)
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        if int(record["bytes"]) != target_path.stat().st_size:
            raise ValueError(f"{relative}: size mismatch for {target_path}")
        if record.get("sha256") != _file_sha256(target_path):
            raise ValueError(f"{relative}: hash mismatch for {target_path}")


def _validate_staged(
    ctx: PublishContext,
    *,
    reasoning_published: bool,
) -> None:
    json_requirements = {
        DICTIONARY_JSON: ("methods",),
        DOCUMENT_JSON: ("methods",),
        FEATURE_JSON: ("methods",),
        PROBE_JSON: ("high_level_transfer", "methods"),
        RFVE_JSON: ("methods",),
        GEOMETRY_JSON: ("methods",),
        TEMPORAL_JSON: ("methods",),
    }
    if reasoning_published:
        json_requirements[REASONING_JSON] = ("methods",)
    for relative, method_path in json_requirements.items():
        _validate_json_methods(
            ctx.path_for_validation(relative),
            method_path=method_path,
        )

    feature_arrays = _load_npz(ctx.path_for_validation(FEATURE_NPZ))
    geometry_arrays = _load_npz(ctx.path_for_validation(GEOMETRY_NPZ))
    for key in JOINT_KEYS:
        for suffix in ("traces", "feature_ids", "mean_activations", "L32_traces"):
            if f"{key}_{suffix}" not in feature_arrays:
                raise ValueError(f"feature dynamics NPZ misses {key}_{suffix}")
        if f"{key}_xy" not in geometry_arrays:
            raise ValueError(f"semantic geometry NPZ misses {key}_xy")

    joint_embeddings = _load_npz(ctx.source(ARXIV_EMBEDDINGS))
    if "labels" not in joint_embeddings:
        raise ValueError(f"{ARXIV_EMBEDDINGS}: missing labels")
    for key in JOINT_KEYS:
        if f"{key}_xy" not in joint_embeddings:
            raise ValueError(f"{ARXIV_EMBEDDINGS}: missing {key}_xy")
        for split in ARXIV_SPLITS:
            relative = Path(
                "shared/downstream_transfer/probe_features/joint_extension/"
                f"{key}-{split}.npz"
            )
            encoded = ctx.source(relative)
            try:
                with np.load(encoded, allow_pickle=False) as handle:
                    required = {"labels", "years", "indices", "values", "nnz"}
                    missing = required - set(handle.files)
                    if missing:
                        raise ValueError(
                            f"{relative}: missing arrays {sorted(missing)}"
                        )
                    rows = int(handle["labels"].shape[0])
                    if handle["years"].shape[0] != rows:
                        raise ValueError(f"{relative}: years row mismatch")
                    if handle["indices"].shape[0] != rows:
                        raise ValueError(f"{relative}: indices row mismatch")
                    if handle["values"].shape != handle["indices"].shape:
                        raise ValueError(f"{relative}: value/index shape mismatch")
                    if handle["nnz"].shape != (rows,):
                        raise ValueError(f"{relative}: nnz row mismatch")
            except (OSError, ValueError) as error:
                if isinstance(error, ValueError) and str(error).startswith(
                    str(relative)
                ):
                    raise
                raise ValueError(f"invalid encoded split {relative}") from error

    tasks = [
        task
        for task in PLOT_SPECS
        if task != "reasoning" or reasoning_published
    ]
    for task in tasks:
        png_rel, pdf_rel = PLOT_SPECS[task]
        png = ctx.path_for_validation(png_rel)
        pdf = ctx.path_for_validation(pdf_rel)
        if png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"invalid PNG: {png}")
        if pdf.read_bytes()[:4] != b"%PDF":
            raise ValueError(f"invalid PDF: {pdf}")
        image = plt.imread(png)
        if image.ndim < 2 or min(image.shape[:2]) < 200:
            raise ValueError(f"implausibly small raster figure: {png}")

    for relative in sorted(
        path
        for path in ctx.outputs
        if path.name.endswith("manifest.json")
    ):
        _validate_manifest(ctx, relative)

    if reasoning_published:
        for relative in (
            Path("reasoning/native_recall.csv"),
            Path("reasoning/cue_free_recall.csv"),
            Path("reasoning/cue_only_false_activation.csv"),
            Path("reasoning/cross_scenario_false_activation.csv"),
            Path("reasoning/pile_background_summary.csv"),
        ):
            _, rows = _load_csv(ctx.path_for_validation(relative))
            methods = {row.get("method") for row in rows}
            if not set(JOINT_KEYS).issubset(methods):
                raise ValueError(f"{relative}: missing Joint reasoning rows")


def _commit(ctx: PublishContext) -> None:
    manifests = {
        relative
        for relative in ctx.outputs
        if relative.name.endswith("manifest.json")
    }
    ordered = sorted(ctx.outputs - manifests)
    ordered.extend(sorted(manifests))
    for relative in ordered:
        source = ctx.staged(relative)
        target = ctx.eval_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.joint-publish-{os.getpid()}"
        )
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def _prepare_stage(args: argparse.Namespace, eval_root: Path) -> tuple[Path, bool]:
    if args.dry_run_dir:
        stage_root = Path(args.dry_run_dir).expanduser().resolve()
        if stage_root.exists() and any(stage_root.iterdir()):
            raise ValueError(f"--dry-run-dir must be empty: {stage_root}")
        stage_root.mkdir(parents=True, exist_ok=True)
        return stage_root, False
    if args.dry_run:
        return Path(tempfile.mkdtemp(prefix="joint-publish-dry-run-")), True
    return Path(
        tempfile.mkdtemp(prefix=".joint-extension-publish-", dir=eval_root)
    ), True


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run_dir:
        args.dry_run = True
    if args.dpi < 100:
        raise ValueError("--dpi must be at least 100")
    eval_root = Path(args.eval_root).expanduser().resolve()
    if not eval_root.is_dir():
        raise NotADirectoryError(eval_root)
    stage_root, clean_stage = _prepare_stage(args, eval_root)
    _style()
    ctx = PublishContext(
        eval_root=eval_root,
        stage_root=stage_root,
        dry_run=bool(args.dry_run),
        dpi=int(args.dpi),
    )
    reasoning_published = False
    try:
        dictionary_sidecar = _validate_joint_sidecar(
            ctx.source(DICTIONARY_SIDECAR),
            expected_task="dictionary_utilization",
        )
        document_sidecar = _validate_joint_sidecar(
            ctx.source(DOCUMENT_SIDECAR),
            expected_task="document_linking",
        )
        feature_sidecar = _validate_joint_sidecar(
            ctx.source(FEATURE_SIDECAR),
            expected_task="feature_dynamics",
        )
        arxiv_sidecar = _validate_joint_sidecar(
            ctx.source(ARXIV_SIDECAR),
            expected_task="arxiv_transfer",
        )
        rfve_sidecar = _validate_joint_sidecar(
            ctx.source(RFVE_SIDECAR),
            expected_task="rfve",
        )
        reasoning_path = eval_root / REASONING_SIDECAR
        if reasoning_path.is_file():
            reasoning_sidecar = _validate_joint_sidecar(
                reasoning_path,
                expected_task="reasoning",
            )
        elif args.allow_missing_reasoning:
            reasoning_sidecar = None
            ctx.skipped_tasks.append("reasoning")
        else:
            raise FileNotFoundError(
                f"{reasoning_path} is not ready; rerun after reasoning finishes "
                "or pass --allow-missing-reasoning"
            )

        print("[publish] preparing dictionary_utilization", flush=True)
        _publish_dictionary(ctx, dictionary_sidecar)
        print("[publish] preparing document_linking", flush=True)
        _publish_document_linking(ctx, document_sidecar)
        print("[publish] preparing feature_dynamics", flush=True)
        _publish_feature_dynamics(ctx, feature_sidecar)
        print("[publish] preparing label_efficiency", flush=True)
        probe = _publish_label_efficiency(ctx, arxiv_sidecar)
        print("[publish] preparing semantic_geometry", flush=True)
        geometry, _ = _publish_semantic_geometry(ctx, arxiv_sidecar)
        print("[publish] preparing temporal_robustness", flush=True)
        _publish_temporal_robustness(
            ctx,
            probe=probe,
            geometry=geometry,
            sidecar=arxiv_sidecar,
        )
        print("[publish] preparing rfve", flush=True)
        _publish_rfve(ctx, rfve_sidecar)
        if reasoning_sidecar is not None:
            print("[publish] preparing reasoning", flush=True)
            _publish_reasoning(ctx, reasoning_sidecar)
            reasoning_published = True
        else:
            print(
                "[publish] reasoning sidecar absent; task intentionally skipped",
                flush=True,
            )

        print("[publish] validating staged artifacts", flush=True)
        _validate_staged(ctx, reasoning_published=reasoning_published)
        if args.dry_run:
            status = "dry-run-ok"
        else:
            print("[publish] committing validated artifacts", flush=True)
            _commit(ctx)
            status = "published"
        report = {
            "status": status,
            "eval_root": str(eval_root),
            "stage_root": (
                str(stage_root)
                if args.dry_run and not clean_stage
                else None
            ),
            "published_tasks": sorted(ctx.task_outputs),
            "skipped_tasks": ctx.skipped_tasks,
            "output_count": len(ctx.outputs),
            "methods": list(JOINT_KEYS),
            "method_labels": dict(JOINT_LABELS),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finally:
        if clean_stage:
            shutil.rmtree(stage_root, ignore_errors=True)


if __name__ == "__main__":
    main()
