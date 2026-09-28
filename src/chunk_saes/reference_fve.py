"""Provenance helpers for task-specific Reference FVE (RFVE).

RFVE is the fraction of a task-specific reference predictor's explained
variance captured by a sparse SAE:

    RFVE = sparse_sae_fve / reference_predictor_fve

Self-reconstruction tasks use the exact identity oracle (reference FVE = 1).
Cross-Chunk uses a train-fitted, direction-blind dense predictor evaluated
under the same A/B occurrence weighting as the sparse SAE.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import load_artifact_manifest


REFERENCE_MANIFEST_FORMAT = "chunk-saes-reference-fve-v1"
SELF_REFERENCE_MODES = ("token", "temporal", "mean")


def identity_reference(mode: str) -> dict[str, Any]:
    if mode not in SELF_REFERENCE_MODES:
        raise ValueError(f"identity RFVE reference is not defined for mode={mode}")
    return {
        "mode": mode,
        "kind": "exact_identity_oracle",
        "reference_fve": 1.0,
        "provenance_status": "exact",
        "description": (
            "The target equals the input, so the identity map is an exact "
            "same-information oracle with FVE 1."
        ),
    }


def load_cross_reference_manifest(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = load_artifact_manifest(
        manifest_path,
        expected_format=REFERENCE_MANIFEST_FORMAT,
        verify_files=verify_files,
    )
    if manifest.get("complete") is not True:
        raise ValueError(f"Cross RFVE artifact is incomplete: {manifest_path}")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"Cross RFVE artifact lacks identity: {manifest_path}")
    if identity.get("mode") != "cross":
        raise ValueError(f"RFVE artifact is not for Cross-Chunk: {manifest_path}")
    if identity.get("direction_policy") != "shared_direction_blind_bidirectional":
        raise ValueError(
            "Cross RFVE reference must use one shared predictor without an A/B "
            "direction input"
        )
    results_record = (manifest.get("files") or {}).get("results")
    if not isinstance(results_record, Mapping):
        raise ValueError(f"Cross RFVE artifact lacks a results file: {manifest_path}")
    from json import load

    results_path = manifest_path.parent / str(results_record["path"])
    with results_path.open(encoding="utf-8") as handle:
        results = load(handle)
    if not isinstance(results, dict) or results.get("complete") is not True:
        raise ValueError(f"invalid Cross RFVE results: {results_path}")
    test = results.get("test")
    if not isinstance(test, Mapping):
        raise ValueError(f"Cross RFVE results lack independent test metrics: {results_path}")
    reference_fve = float(test.get("pooled_fve", float("nan")))
    if not math.isfinite(reference_fve) or reference_fve <= 0.0:
        raise ValueError("Cross RFVE pooled test FVE must be finite and positive")
    return manifest, results


def validate_cross_reference_against_sae(
    *,
    manifest: Mapping[str, Any],
    results: Mapping[str, Any],
    sae_common: Mapping[str, Any],
) -> None:
    identity = manifest["identity"]
    if int(identity.get("activation_dim", -1)) != int(
        sae_common.get("activation_dim", -2)
    ):
        raise ValueError("Cross RFVE activation dimension does not match the SAE")
    if int(identity.get("layer", -1)) != int(sae_common.get("layer", -2)):
        raise ValueError("Cross RFVE layer does not match the SAE")
    train_digest = identity.get("train_cache_digest")
    if train_digest != sae_common.get("train_cache_digest"):
        raise ValueError("Cross RFVE train cache does not match the SAE train cache")
    validation_digest = identity.get("validation_cache_digest")
    if validation_digest != sae_common.get("validation_cache_digest"):
        raise ValueError(
            "Cross RFVE validation cache does not match the SAE validation cache"
        )
    if identity.get("test_cache_digest") in {None, ""}:
        raise ValueError("Cross RFVE artifact must bind an independent test cache")
    sweep = results.get("capacity_sweep")
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("Cross RFVE results must contain a non-empty capacity sweep")
    selected = results.get("selected_model")
    if not isinstance(selected, Mapping):
        raise ValueError("Cross RFVE results must identify the selected dense model")
    if selected.get("selection_split") != "validation":
        raise ValueError("Cross RFVE model selection must use validation only")
    if results.get("test_evaluation_count") != 1:
        raise ValueError("Cross RFVE test split must be evaluated exactly once")


def audit_scalar_cross_reference(value: float) -> dict[str, Any]:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Cross audit RFVE reference must be finite and positive")
    return {
        "mode": "cross",
        "kind": "legacy_audit_scalar",
        "reference_fve": float(value),
        "provenance_status": "audit_only",
        "description": (
            "Legacy scalar retained only to reproduce the existing 9B audit. "
            "It is not publication-grade until replaced by a verified "
            "chunk-saes-reference-fve-v1 artifact."
        ),
    }
