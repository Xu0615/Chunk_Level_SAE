from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .utils import atomic_json_dump


ARTIFACT_DIGEST_KEY = "artifact_digest"
SAE_SET_FORMAT = "chunk-saes-evaluation-sae-set-v1"
DEFAULT_SAE_MODES = ("token", "temporal", "mean", "cross")
REQUIRED_SAE_MODES = ("token", "mean", "cross")


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: str | Path, *, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
    hash_content: bool = True,
) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    recorded_path = (
        Path(os.path.relpath(resolved, Path(relative_to))).as_posix()
        if relative_to is not None
        else resolved.name
    )
    record = {
        "path": recorded_path,
        "bytes": resolved.stat().st_size,
    }
    if hash_content:
        record["sha256"] = file_sha256(resolved)
    return record


def verify_file_record(root: str | Path, record: Mapping[str, Any]) -> Path:
    path = Path(root) / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = int(record["bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"artifact file size mismatch for {path}: {actual_size} != {expected_size}"
        )
    expected_digest = record.get("sha256")
    if expected_digest is not None:
        actual_digest = file_sha256(path)
        if actual_digest != str(expected_digest):
            raise ValueError(
                f"artifact file digest mismatch for {path}: "
                f"{actual_digest} != {expected_digest}"
            )
    return path


def add_artifact_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop(ARTIFACT_DIGEST_KEY, None)
    result[ARTIFACT_DIGEST_KEY] = json_digest(result)
    return result


def write_artifact_manifest(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    result = add_artifact_digest(payload)
    atomic_json_dump(result, path)
    return result


def load_artifact_manifest(
    path: str | Path,
    *,
    expected_format: str | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if expected_format is not None and manifest.get("format") != expected_format:
        raise ValueError(
            f"{manifest_path}: expected format={expected_format!r}, "
            f"found {manifest.get('format')!r}"
        )
    expected_digest = manifest.get(ARTIFACT_DIGEST_KEY)
    if not isinstance(expected_digest, str):
        raise ValueError(f"{manifest_path}: missing {ARTIFACT_DIGEST_KEY}")
    without_digest = dict(manifest)
    without_digest.pop(ARTIFACT_DIGEST_KEY, None)
    actual_digest = json_digest(without_digest)
    if actual_digest != expected_digest:
        raise ValueError(
            f"{manifest_path}: artifact digest mismatch: "
            f"{actual_digest} != {expected_digest}"
        )
    if verify_files:
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            raise ValueError(f"{manifest_path}: files must be a mapping")
        for record in files.values():
            if not isinstance(record, Mapping):
                raise ValueError(f"{manifest_path}: invalid file record {record!r}")
            verify_file_record(manifest_path.parent, record)
    return manifest


def ensure_reusable_artifact(
    manifest_path: str | Path,
    *,
    expected_format: str,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(manifest_path)
    if not path.exists():
        return None
    manifest = load_artifact_manifest(
        path,
        expected_format=expected_format,
        verify_files=True,
    )
    if manifest.get("complete") is not True:
        raise ValueError(f"artifact is not complete: {path}")
    if manifest.get("identity") != dict(expected_identity):
        raise ValueError(
            f"existing artifact identity does not match this run: {path}. "
            "Use the explicit overwrite option or a new output directory."
        )
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _checkpoint_step(path: Path, checkpoint_manifest: Mapping[str, Any] | None) -> int:
    if checkpoint_manifest is not None and checkpoint_manifest.get("step") is not None:
        return int(checkpoint_manifest["step"])
    state_path = path / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"checkpoint has neither a step manifest nor training state: {path}"
        )
    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("step") is None:
        raise ValueError(f"invalid training state: {state_path}")
    return int(state["step"])


def available_sae_modes(
    sae_root: str | Path,
    *,
    modes: Sequence[str] = DEFAULT_SAE_MODES,
    required_modes: Sequence[str] = REQUIRED_SAE_MODES,
) -> tuple[str, ...]:
    """Return completed modes while allowing an unfinished optional Temporal SAE.

    Token, Mean, and Cross are required for this evaluation. Temporal is
    optional until its ``complete.json`` marker is published. A mode directory
    that exists but lacks a complete marker is treated as training-in-progress,
    never as an evaluable checkpoint.
    """

    root = Path(sae_root).expanduser().resolve()
    required = set(required_modes)
    selected = []
    for mode in modes:
        complete_path = root / mode / "complete.json"
        if not complete_path.is_file():
            if mode in required:
                raise FileNotFoundError(complete_path)
            continue
        complete = _read_json(complete_path)
        if complete.get("complete") is not True or complete.get("mode") != mode:
            if mode in required:
                raise ValueError(
                    f"incomplete or mismatched required SAE mode: {root / mode}"
                )
            continue
        selected.append(mode)
    return tuple(selected)


def resolve_sae_artifact_set(
    sae_root: str | Path,
    *,
    selection: str = "best",
    modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    if selection not in {"best", "final"}:
        raise ValueError("SAE checkpoint selection must be 'best' or 'final'")
    root = Path(sae_root).expanduser().resolve()
    if modes is None:
        modes = available_sae_modes(root)
    mode_entries: dict[str, dict[str, Any]] = {}
    common: dict[str, Any] | None = None
    common_keys = (
        "project",
        "training_format",
        "model",
        "layer",
        "activation_dim",
        "dict_size",
        "k",
        "global_batch_size",
        "steps",
        "exact_coverage_required",
        "seed",
        "run_id",
        "train_cache_digest",
        "validation_cache_digest",
        "best_metric",
    )
    for mode in modes:
        mode_root = root / mode
        complete_path = mode_root / "complete.json"
        config_path = mode_root / "config.json"
        complete = _read_json(complete_path)
        final_config = _read_json(config_path)
        if complete.get("complete") is not True or complete.get("mode") != mode:
            raise ValueError(f"incomplete or mismatched SAE mode: {mode_root}")
        if complete.get("exact_coverage") is not True:
            raise ValueError(f"SAE mode is not exact-coverage v2: {mode_root}")
        if float(complete.get("coverage_fraction", -1.0)) != 1.0:
            raise ValueError(f"SAE mode lacks full occurrence coverage: {mode_root}")
        if int(complete.get("steps", -1)) != int(final_config.get("steps", -2)):
            raise ValueError(f"SAE step mismatch between config and complete marker: {mode_root}")
        if int(complete.get("samples_seen", -1)) != int(
            complete.get("unique_occurrences_seen", -2)
        ):
            raise ValueError(f"SAE exposures are not one exact occurrence epoch: {mode_root}")
        if not isinstance(complete.get("final_full_validation_metrics"), Mapping):
            raise ValueError(f"SAE lacks complete held-out validation metrics: {mode_root}")

        checkpoint_root = (
            mode_root / "checkpoints" / "best" if selection == "best" else mode_root
        )
        checkpoint_config = _read_json(checkpoint_root / "config.json")
        checkpoint_manifest_path = checkpoint_root / "checkpoint_manifest.json"
        checkpoint_manifest = (
            load_artifact_manifest(
                checkpoint_manifest_path,
                expected_format="chunk-saes-sae-checkpoint-v2",
                verify_files=False,
            )
            if checkpoint_manifest_path.is_file()
            else None
        )
        step = (
            _checkpoint_step(checkpoint_root, checkpoint_manifest)
            if selection == "best"
            else int(complete["steps"])
        )
        if selection == "best" and step != int(complete.get("best_step", -1)):
            raise ValueError(
                f"validation-best checkpoint step {step} does not match "
                f"complete marker {complete.get('best_step')}: {checkpoint_root}"
            )
        for key in common_keys:
            if checkpoint_config.get(key) != final_config.get(key):
                raise ValueError(
                    f"checkpoint/final config mismatch for {mode}.{key}: "
                    f"{checkpoint_config.get(key)!r} != {final_config.get(key)!r}"
                )
        if checkpoint_config.get("mode") != mode:
            raise ValueError(f"checkpoint mode mismatch: {checkpoint_root}")
        if checkpoint_config.get("project") != "chunk-saes":
            raise ValueError(f"checkpoint is not a chunk-saes artifact: {checkpoint_root}")
        if checkpoint_config.get("training_format") != "chunk-saes-activation-cache-v2":
            raise ValueError(f"checkpoint does not derive from activation cache v2: {checkpoint_root}")

        weights_path = checkpoint_root / "sae.safetensors"
        weights = file_record(weights_path, relative_to=checkpoint_root)
        entry = {
            "mode": mode,
            "checkpoint_path": str(checkpoint_root),
            "step": step,
            "best_step": int(complete.get("best_step", -1)),
            "best_metric_value": complete.get("best_metric_value"),
            "weights": weights,
            "config_digest": json_digest(checkpoint_config),
            "complete_digest": json_digest(complete),
        }
        mode_entries[mode] = entry
        values = {key: checkpoint_config.get(key) for key in common_keys}
        if common is None:
            common = values
        elif common != values:
            raise ValueError(
                "Token/Temporal/Mean/Cross checkpoint configs do not share "
                f"one run: {root}"
            )

    assert common is not None
    payload = {
        "format": SAE_SET_FORMAT,
        "complete": True,
        "sae_root": str(root),
        "selection": selection,
        "common": common,
        "modes": mode_entries,
    }
    return add_artifact_digest(payload)


def model_metadata_fingerprint(model_root: str | Path) -> dict[str, Any]:
    """Identify a local model without rereading every multi-GB weight payload."""

    root = Path(model_root).expanduser().resolve()
    metadata_names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    metadata = {}
    for name in metadata_names:
        path = root / name
        if path.is_file():
            metadata[name] = file_record(path, relative_to=root)
    weights = [
        {"path": path.name, "bytes": path.stat().st_size}
        for path in sorted(root.glob("*.safetensors"))
    ]
    if "config.json" not in metadata or not weights:
        raise FileNotFoundError(f"incomplete local model artifact: {root}")
    identity = {
        "algorithm": "sha256(metadata-content+weight-file-names-and-sizes)",
        "metadata": metadata,
        "weights": weights,
    }
    return {
        "path": str(root),
        "fingerprint": json_digest(identity),
        **identity,
    }
