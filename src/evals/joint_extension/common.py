from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from chunk_saes.plot_style import (
    METHOD_COLORS as SHARED_METHOD_COLORS,
    METHOD_MARKERS as SHARED_METHOD_MARKERS,
)


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

JOINT_ALPHAS = {
    "joint_alpha0p25": 0.25,
    "joint_alpha0p5": 0.5,
    "joint_alpha1": 1.0,
    "joint_alpha1p5": 1.5,
}

JOINT_KEYS = tuple(JOINT_ALPHAS)

JOINT_ROOT_ARGUMENTS = {
    "joint_alpha0p25": "joint_alpha0p25_root",
    "joint_alpha0p5": "joint_alpha0p5_root",
    "joint_alpha1": "joint_alpha1_root",
    "joint_alpha1p5": "joint_alpha1p5_root",
}

METHOD_LABELS = {
    "token": "BatchTopK SAE",
    "temporal": "Temporal SAE",
    "mean": "Mean-Chunk SAE",
    "joint_alpha0p25": "Joint-Chunk SAE(alpha=0.25)",
    "joint_alpha0p5": "Joint-Chunk SAE(alpha=0.5)",
    "joint_alpha1": "Joint-Chunk SAE(alpha=1.0)",
    "joint_alpha1p5": "Joint-Chunk SAE(alpha=1.5)",
    "cross": "Cross-Chunk SAE",
}

METHOD_SHORT_LABELS = {
    "token": "BatchTopK",
    "temporal": "Temporal",
    "mean": "Mean-Chunk",
    "joint_alpha0p25": "Joint α=0.25",
    "joint_alpha0p5": "Joint α=0.5",
    "joint_alpha1": "Joint α=1.0",
    "joint_alpha1p5": "Joint α=1.5",
    "cross": "Cross-Chunk",
}

METHOD_COLORS = {
    method: SHARED_METHOD_COLORS[method] for method in METHOD_ORDER
}

METHOD_MARKERS = {
    method: SHARED_METHOD_MARKERS[method] for method in METHOD_ORDER
}


@dataclass(frozen=True)
class JointSpec:
    key: str
    label: str
    alpha: float
    root: Path
    checkpoint: Path
    cross_prefix: int
    dictionary_width: int
    activation_dim: int
    validation_cache_digest: str
    best_step: int


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def add_joint_root_arguments(parser: Any) -> None:
    """Register the complete Joint alpha sweep required by evaluation runs."""

    for key in JOINT_KEYS:
        parser.add_argument(
            f"--{JOINT_ROOT_ARGUMENTS[key].replace('_', '-')}",
            required=True,
        )


def joint_specs_from_args(args: Any) -> list[JointSpec]:
    """Load all four validation-selected Joint checkpoints in alpha order."""

    return [
        load_joint_spec(
            getattr(args, JOINT_ROOT_ARGUMENTS[key]),
            key=key,
            expected_alpha=JOINT_ALPHAS[key],
        )
        for key in JOINT_KEYS
    ]


def joint_root_cli_args(args: Any) -> list[str]:
    """Serialize all Joint root options for a child CLI invocation."""

    result: list[str] = []
    for key in JOINT_KEYS:
        argument = JOINT_ROOT_ARGUMENTS[key]
        result.extend(
            (
                f"--{argument.replace('_', '-')}",
                str(getattr(args, argument)),
            )
        )
    return result


def load_joint_spec(root: str | Path, *, key: str, expected_alpha: float) -> JointSpec:
    root = Path(root).expanduser().resolve()
    mode_root = root / "joint_chunk"
    complete = load_json(mode_root / "complete.json")
    checkpoint = mode_root / "checkpoints" / "best"
    config = load_json(checkpoint / "config.json")
    if complete.get("complete") is not True:
        raise ValueError(f"incomplete Joint checkpoint: {mode_root}")
    if complete.get("mode") != "joint_chunk" or config.get("mode") != "joint_chunk":
        raise ValueError(f"not a joint_chunk checkpoint: {mode_root}")
    if config.get("joint_chunk_layout") != "nested_prefix":
        raise ValueError(f"Joint checkpoint is not nested-prefix: {mode_root}")
    alpha = float(config.get("joint_chunk_alpha", -1.0))
    if not np.isclose(alpha, float(expected_alpha), rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{root} has alpha={alpha}, expected alpha={expected_alpha}"
        )
    cross_prefix = int(config.get("joint_cross_prefix", 0))
    width = int(config.get("dict_size", 0))
    if not 0 < cross_prefix < width:
        raise ValueError(f"invalid nested Cross prefix: {cross_prefix}/{width}")
    if int(complete.get("best_step", -1)) <= 0:
        raise ValueError(f"Joint checkpoint lacks a validation-selected step: {root}")
    weights = checkpoint / "sae.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        required = {
            "encoder_weight",
            "encoder_bias",
            "pre_bias",
            "threshold",
            "activation_scale",
            "decoder_weight",
            "decoder_cross_bias",
            "feature_counts",
        }
        missing = required - names
        if missing:
            raise ValueError(f"{weights} lacks tensors: {sorted(missing)}")
        shape = handle.get_slice("encoder_weight").get_shape()
        if [int(value) for value in shape] != [
            width,
            int(config["activation_dim"]),
        ]:
            raise ValueError(f"unexpected encoder shape in {weights}: {shape}")
    return JointSpec(
        key=key,
        label=METHOD_LABELS[key],
        alpha=alpha,
        root=root,
        checkpoint=checkpoint,
        cross_prefix=cross_prefix,
        dictionary_width=width,
        activation_dim=int(config["activation_dim"]),
        validation_cache_digest=str(config["validation_cache_digest"]),
        best_step=int(complete["best_step"]),
    )


def spec_identity(spec: JointSpec) -> dict[str, Any]:
    weights = spec.checkpoint / "sae.safetensors"
    config = spec.checkpoint / "config.json"
    complete = spec.root / "joint_chunk" / "complete.json"
    return {
        "key": spec.key,
        "label": spec.label,
        "alpha": spec.alpha,
        "root": str(spec.root),
        "checkpoint": str(spec.checkpoint),
        "best_step": spec.best_step,
        "cross_prefix": spec.cross_prefix,
        "dictionary_width": spec.dictionary_width,
        "activation_dim": spec.activation_dim,
        "validation_cache_digest": spec.validation_cache_digest,
        "files": {
            "weights": {
                "bytes": weights.stat().st_size,
                "sha256": file_sha256(weights),
            },
            "config": {
                "bytes": config.stat().st_size,
                "sha256": file_sha256(config),
            },
            "complete": {
                "bytes": complete.stat().st_size,
                "sha256": file_sha256(complete),
            },
        },
    }


class FrozenJointEncoder:
    """Encoder-only thresholded view of a nested Joint-Chunk checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device,
        feature_ids: Sequence[int] | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        config = load_json(self.checkpoint / "config.json")
        with safe_open(
            str(self.checkpoint / "sae.safetensors"),
            framework="pt",
            device="cpu",
        ) as handle:
            width = int(handle.get_slice("encoder_weight").get_shape()[0])
            if feature_ids is None:
                ids = torch.arange(width, dtype=torch.long)
            else:
                ids = torch.as_tensor(feature_ids, dtype=torch.long)
                if ids.numel() == 0:
                    raise ValueError("feature_ids must not be empty")
                if int(ids.min()) < 0 or int(ids.max()) >= width:
                    raise ValueError("feature ID outside Joint dictionary")
            self.feature_ids = ids
            self.weight = handle.get_tensor("encoder_weight")[ids].to(self.device)
            self.bias = handle.get_tensor("encoder_bias")[ids].to(self.device)
            self.pre_bias = handle.get_tensor("pre_bias").to(self.device)
            self.threshold = float(handle.get_tensor("threshold"))
            self.scale = float(handle.get_tensor("activation_scale"))
            self.feature_counts = handle.get_tensor("feature_counts")
        self.width = width
        self.cross_prefix = int(config["joint_cross_prefix"])
        self.alpha = float(config["joint_chunk_alpha"])

    @torch.inference_mode()
    def dense(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.device, dtype=self.weight.dtype)
        pre = F.relu(
            F.linear(
                hidden * self.scale - self.pre_bias.to(self.weight.dtype),
                self.weight,
                self.bias,
            )
        )
        return pre * (pre > self.threshold)

    @torch.inference_mode()
    def topk(
        self,
        hidden: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dense = self.dense(hidden)
        values, indices = dense.topk(min(int(k), dense.shape[-1]), dim=-1)
        valid = values > 0
        return indices, values, valid.sum(dim=-1)

    def close(self) -> None:
        del self.weight, self.bias, self.pre_bias
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def sparse_fixed(
    indices: torch.Tensor,
    values: torch.Tensor,
    nnz: torch.Tensor,
    storage_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = int(indices.shape[0])
    out_i = np.full((batch, storage_k), -1, dtype=np.int32)
    out_v = np.zeros((batch, storage_k), dtype=np.float16)
    take = min(storage_k, int(indices.shape[1]))
    valid = values[:, :take] > 0
    out_i[:, :take] = indices[:, :take].detach().cpu().numpy().astype(np.int32)
    out_v[:, :take] = values[:, :take].detach().cpu().numpy().astype(np.float16)
    valid_np = valid.detach().cpu().numpy()
    out_i[:, :take][~valid_np] = -1
    out_n = np.minimum(
        nnz.detach().cpu().numpy().astype(np.int32),
        storage_k,
    ).astype(np.int16)
    return out_i, out_v, out_n


def merge_sidecar(
    path: str | Path,
    *,
    task: str,
    methods: Mapping[str, Any],
    specs: Sequence[JointSpec],
    protocol: Mapping[str, Any],
    files: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "format": "chunk-saes-joint-extension-v1",
        "complete": True,
        "task": task,
        "method_order": list(METHOD_ORDER),
        "methods": dict(methods),
        "joint_checkpoints": {
            spec.key: spec_identity(spec)
            for spec in specs
        },
        "protocol": dict(protocol),
    }
    if files:
        payload["files"] = dict(files)
    payload["artifact_digest"] = json_digest(payload)
    atomic_json(path, payload)
