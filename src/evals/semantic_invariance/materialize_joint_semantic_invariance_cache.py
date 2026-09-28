#!/usr/bin/env python
"""Materialize native, variable-length Joint-Chunk activations.

The high-level-feature census must compare every chunk SAE on the same native
32/64/128/256/512-token chunk population.  The baseline cache already contains
the chunk plan plus Mean/Cross activations; this program runs the four frozen
Joint encoders on every row of that plan and writes only the sampled feature
coordinates.

The cache is resumable at shard granularity and contains no LLM-derived data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from chunk_saes.modeling import TargetLayerExtractor


METHOD_ALPHAS = {
    "joint_alpha0p25": 0.25,
    "joint_alpha0p5": 0.5,
    "joint_alpha1": 1.0,
    "joint_alpha1p5": 1.5,
}
METHODS = tuple(METHOD_ALPHAS)
ROOT_ARGUMENTS = {
    "joint_alpha0p25": "joint_alpha0p25_root",
    "joint_alpha0p5": "joint_alpha0p5_root",
    "joint_alpha1": "joint_alpha1_root",
    "joint_alpha1p5": "joint_alpha1p5_root",
}
WIDTH = 65_536
SAMPLE_SIZE = 1_000
FORMAT = "native-high-level-joint-cache-v1"
ACTIVATION_DTYPE = torch.bfloat16


def _sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve_checkpoint(value: str, method: str) -> Path:
    root = Path(value).expanduser().resolve()
    for candidate in (
        root,
        root / "checkpoints" / "best",
        root / "joint_chunk" / "checkpoints" / "best",
    ):
        if (
            (candidate / "config.json").is_file()
            and (candidate / "sae.safetensors").is_file()
        ):
            return candidate
    raise FileNotFoundError(f"could not resolve {method} checkpoint from {root}")


def _checkpoint_identity(checkpoint: Path, method: str) -> dict[str, Any]:
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "sae.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_alpha = METHOD_ALPHAS[method]
    if config.get("mode") != "joint_chunk":
        raise ValueError(f"{method} checkpoint is not joint_chunk")
    if config.get("joint_chunk_layout") != "nested_prefix":
        raise ValueError(f"{method} checkpoint is not nested_prefix")
    if not math.isclose(
        float(config.get("joint_chunk_alpha", -1.0)),
        expected_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{method} alpha={config.get('joint_chunk_alpha')}, "
            f"expected {expected_alpha}"
        )
    if int(config.get("dict_size", 0)) != WIDTH:
        raise ValueError(f"{method} dictionary width is not {WIDTH}")
    cross_prefix = int(config.get("joint_cross_prefix", 0))
    if not 0 < cross_prefix < WIDTH:
        raise ValueError(f"{method} has invalid cross prefix {cross_prefix}")
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        if int(handle.get_slice("encoder_weight").get_shape()[0]) != WIDTH:
            raise ValueError(f"{method} encoder width is not {WIDTH}")
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    checkpoint_manifest = (
        json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        if checkpoint_manifest_path.is_file()
        else {}
    )
    return {
        "checkpoint": str(checkpoint),
        "alpha": expected_alpha,
        "layout": "nested_prefix",
        "cross_prefix": cross_prefix,
        "dictionary_width": WIDTH,
        "activation_dim": int(config["activation_dim"]),
        "config_sha256": _sha256(config_path),
        "checkpoint_artifact_digest": str(
            checkpoint_manifest.get("artifact_digest", "")
        ),
        "weights_bytes": weights_path.stat().st_size,
    }


def _source_identity(
    source_dir: Path,
    feature_limit: int,
) -> tuple[dict[str, Any], list[int], list[dict[str, Any]]]:
    manifest_path = source_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("source activation cache is incomplete")
    raw_ids = manifest.get("feature_ids") or {}
    reference = list(map(int, raw_ids.get("token", ())))
    if len(reference) != SAMPLE_SIZE:
        raise ValueError("source cache does not contain the frozen 1,000 features")
    for method in ("temporal", "mean", "cross"):
        if list(map(int, raw_ids.get(method, ()))) != reference:
            raise ValueError("baseline methods do not share one feature sample")
    feature_ids = reference[:feature_limit]

    plan_paths = sorted((source_dir / "chunk_pool" / "plan").glob("shard-*.safetensors"))
    if not plan_paths:
        raise FileNotFoundError("source chunk plan is missing")
    tasks: list[dict[str, Any]] = []
    expected_chunk_id = 0
    length_counts: dict[str, int] = {}
    for shard_index, path in enumerate(plan_paths):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            chunk_ids = handle.get_tensor("chunk_ids")
            lengths = handle.get_tensor("lengths")
        rows = int(chunk_ids.numel())
        if rows <= 0:
            raise ValueError(f"empty chunk-plan shard: {path}")
        expected = torch.arange(
            expected_chunk_id,
            expected_chunk_id + rows,
            dtype=chunk_ids.dtype,
        )
        if not torch.equal(chunk_ids, expected):
            raise ValueError(f"non-canonical chunk IDs in {path}")
        for length, count in zip(*torch.unique(lengths, return_counts=True)):
            key = str(int(length))
            length_counts[key] = length_counts.get(key, 0) + int(count)
        tasks.append(
            {
                "shard_index": shard_index,
                "plan_path": str(path),
                "rows": rows,
                "tokens": int(lengths.sum().item()),
                "chunk_id_start": expected_chunk_id,
            }
        )
        expected_chunk_id += rows
    if expected_chunk_id != int(manifest.get("chunk_count", -1)):
        raise ValueError(
            f"plan rows={expected_chunk_id}, manifest chunk_count="
            f"{manifest.get('chunk_count')}"
        )
    expected_lengths = {"32", "64", "128", "256", "512"}
    if set(length_counts) != expected_lengths:
        raise ValueError(f"unexpected native chunk lengths: {length_counts}")
    return manifest, feature_ids, tasks


def _load_encoder(
    checkpoint: Path,
    feature_ids: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    ids = torch.tensor(feature_ids, dtype=torch.long)
    with safe_open(
        str(checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        names = set(handle.keys())
        pre_bias_name = "pre_bias" if "pre_bias" in names else "decoder_bias"
        return {
            "weight": handle.get_tensor("encoder_weight")
            .index_select(0, ids)
            .to(device=device, dtype=dtype),
            "bias": handle.get_tensor("encoder_bias")
            .index_select(0, ids)
            .to(device=device, dtype=dtype),
            "pre_bias": handle.get_tensor(pre_bias_name).to(
                device=device,
                dtype=dtype,
            ),
            "threshold": float(handle.get_tensor("threshold")),
            "scale": float(handle.get_tensor("activation_scale")),
        }


@torch.inference_mode()
def _encode(hidden: torch.Tensor, parameters: Mapping[str, Any]) -> torch.Tensor:
    hidden = hidden.to(
        device=parameters["weight"].device,
        dtype=parameters["weight"].dtype,
    )
    pre = F.relu(
        F.linear(
            hidden * float(parameters["scale"]) - parameters["pre_bias"],
            parameters["weight"],
            parameters["bias"],
        )
    )
    return pre * (pre > float(parameters["threshold"]))


def _expected_chunk_ids(plan_path: str | Path) -> torch.Tensor:
    with safe_open(str(plan_path), framework="pt", device="cpu") as handle:
        return handle.get_tensor("chunk_ids")


def _valid_shard(
    path: Path,
    *,
    feature_ids: Sequence[int],
    expected_chunk_ids: torch.Tensor,
) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if not torch.equal(handle.get_tensor("chunk_ids"), expected_chunk_ids):
                return False
            rows = int(expected_chunk_ids.numel())
            for method in METHODS:
                if list(map(int, handle.get_tensor(f"{method}_feature_ids").tolist())) != list(
                    map(int, feature_ids)
                ):
                    return False
                values = handle.get_slice(f"{method}_activations")
                if list(values.get_shape()) != [rows, len(feature_ids)]:
                    return False
                if values.get_dtype() != "BF16":
                    return False
        return True
    except Exception:
        return False


def _existing_manifest_matches(
    output_dir: Path,
    *,
    source_manifest: Mapping[str, Any],
    feature_ids: Sequence[int],
    checkpoint_identity: Mapping[str, Mapping[str, Any]],
    model_path: str,
    layer: int,
) -> bool:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            existing.get("complete") is True
            and existing.get("format") == FORMAT
            and existing.get("source_data_artifact_digest")
            == source_manifest.get("artifact_digest")
            and existing.get("model") == model_path
            and int(existing.get("layer", -1)) == int(layer)
            and list(map(int, existing.get("feature_ids", ())))
            == list(map(int, feature_ids))
            and existing.get("methods") == checkpoint_identity
        )
    except Exception:
        return False


def _forward_means_with_backoff(
    extractor: TargetLayerExtractor,
    sequences: Sequence[Sequence[int]],
    *,
    initial_batch_size: int,
) -> torch.Tensor:
    """Run equal-length sequences, reducing the batch on CUDA OOM."""

    outputs: list[torch.Tensor] = []
    cursor = 0
    batch_size = max(1, int(initial_batch_size))
    while cursor < len(sequences):
        current = list(sequences[cursor : cursor + batch_size])
        try:
            outputs.append(extractor.forward_ids(current).means())
            cursor += len(current)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(
                f"[joint-native-cache] CUDA OOM; retrying with batch={batch_size}",
                flush=True,
            )
    return torch.cat(outputs, dim=0)


def _worker(
    worker_index: int,
    device_index: int,
    tasks: Sequence[Mapping[str, Any]],
    *,
    model_path: str,
    layer: int,
    checkpoints: Mapping[str, str],
    feature_ids: Sequence[int],
    output_dir: str,
    max_batch_size: int,
    forward_token_budget: int,
    model_dtype: str,
    attn_implementation: str,
) -> None:
    if not tasks:
        return
    time.sleep(1.5 * worker_index)
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    extractor = TargetLayerExtractor(
        model_path,
        layer,
        str(device),
        dtype=model_dtype,
        attn_implementation=attn_implementation,
        truncated_load=True,
    )
    dtype = torch.bfloat16 if model_dtype == "bfloat16" else torch.float16
    encoders = {
        method: _load_encoder(Path(checkpoints[method]), feature_ids, device, dtype)
        for method in METHODS
    }
    output_root = Path(output_dir)
    try:
        for task_number, task in enumerate(tasks, start=1):
            shard_index = int(task["shard_index"])
            plan_path = Path(str(task["plan_path"]))
            target = (
                output_root
                / "activations"
                / f"shard-{shard_index:05d}.safetensors"
            )
            rows = int(task["rows"])
            expected_ids = _expected_chunk_ids(plan_path)[:rows].contiguous()
            if _valid_shard(
                target,
                feature_ids=feature_ids,
                expected_chunk_ids=expected_ids,
            ):
                print(
                    f"[joint-native-cache gpu={device_index}] "
                    f"reuse shard={shard_index}",
                    flush=True,
                )
                continue
            with safe_open(str(plan_path), framework="pt", device="cpu") as handle:
                token_ids = handle.get_tensor("token_ids")
                offsets = handle.get_tensor("offsets")
                lengths = handle.get_tensor("lengths")
                chunk_ids = handle.get_tensor("chunk_ids")
            lengths = lengths[:rows]
            chunk_ids = chunk_ids[:rows].contiguous()
            selected = {
                method: torch.empty(
                    (rows, len(feature_ids)),
                    dtype=ACTIVATION_DTYPE,
                )
                for method in METHODS
            }
            for length_value in sorted(set(map(int, lengths.tolist()))):
                length_indices = torch.nonzero(
                    lengths == length_value,
                    as_tuple=False,
                ).flatten()
                batch_size = max(
                    1,
                    min(
                        int(max_batch_size),
                        int(forward_token_budget) // length_value,
                    ),
                )
                cursor = 0
                while cursor < len(length_indices):
                    current = length_indices[cursor : cursor + batch_size]
                    sequences = [
                        token_ids[
                            int(offsets[index]) : int(offsets[index + 1])
                        ].tolist()
                        for index in current.tolist()
                    ]
                    means = _forward_means_with_backoff(
                        extractor,
                        sequences,
                        initial_batch_size=batch_size,
                    )
                    for method in METHODS:
                        selected[method].index_copy_(
                            0,
                            current,
                            _encode(means, encoders[method])
                            .to("cpu", dtype=ACTIVATION_DTYPE),
                        )
                    cursor += len(current)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            save_file(
                {
                    "chunk_ids": chunk_ids.contiguous(),
                    **{
                        f"{method}_activations": selected[method].contiguous()
                        for method in METHODS
                    },
                    **{
                        f"{method}_feature_ids": torch.tensor(
                            feature_ids,
                            dtype=torch.long,
                        )
                        for method in METHODS
                    },
                },
                str(temporary),
            )
            os.replace(temporary, target)
            print(
                f"[joint-native-cache gpu={device_index}] "
                f"shard={shard_index} rows={rows} tokens={task['tokens']} "
                f"task={task_number}/{len(tasks)}",
                flush=True,
            )
    finally:
        extractor.close()
        del encoders
        torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Materialize all-length native Joint-Chunk activations"
    )
    value.add_argument("--source-data-dir", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--model", default="")
    value.add_argument("--layer", type=int, default=21)
    value.add_argument("--joint-alpha0p25-root", required=True)
    value.add_argument("--joint-alpha0p5-root", required=True)
    value.add_argument("--joint-alpha1-root", required=True)
    value.add_argument("--joint-alpha1p5-root", required=True)
    value.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    value.add_argument(
        "--shard-indices",
        default="",
        help="Optional comma-separated subset for a resumable smoke/worker run.",
    )
    value.add_argument("--feature-limit", type=int, default=SAMPLE_SIZE)
    value.add_argument("--max-batch-size", type=int, default=2048)
    value.add_argument("--forward-token-budget", type=int, default=98_304)
    value.add_argument(
        "--max-shard-tokens",
        type=int,
        default=0,
        help=(
            "Debug only: keep a complete-pair prefix of each requested shard "
            "within this token budget. Zero means the full shard."
        ),
    )
    value.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    value.add_argument("--attn-implementation", default="sdpa")
    value.add_argument("--overwrite", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if not 1 <= int(args.feature_limit) <= SAMPLE_SIZE:
        raise ValueError(f"--feature-limit must be in [1, {SAMPLE_SIZE}]")
    source_dir = Path(args.source_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest, feature_ids, all_tasks = _source_identity(
        source_dir,
        int(args.feature_limit),
    )
    checkpoints = {
        method: _resolve_checkpoint(str(getattr(args, argument)), method)
        for method, argument in ROOT_ARGUMENTS.items()
    }
    checkpoint_identity = {
        method: _checkpoint_identity(checkpoints[method], method)
        for method in METHODS
    }
    model_path = str(
        Path(args.model or source_manifest["identity"]["model"])
        .expanduser()
        .resolve()
    )
    if _existing_manifest_matches(
        output_dir,
        source_manifest=source_manifest,
        feature_ids=feature_ids,
        checkpoint_identity=checkpoint_identity,
        model_path=model_path,
        layer=int(args.layer),
    ):
        print(output_dir / "manifest.json", flush=True)
        return

    requested = {
        int(value.strip())
        for value in str(args.shard_indices).split(",")
        if value.strip()
    }
    available = {int(task["shard_index"]) for task in all_tasks}
    if requested - available:
        raise ValueError(
            f"requested unavailable shards: {sorted(requested - available)}"
        )
    tasks = (
        [task for task in all_tasks if int(task["shard_index"]) in requested]
        if requested
        else all_tasks
    )
    debug_partial = int(args.max_shard_tokens) > 0
    if debug_partial:
        truncated: list[dict[str, Any]] = []
        for task in tasks:
            with safe_open(
                str(task["plan_path"]),
                framework="pt",
                device="cpu",
            ) as handle:
                lengths = handle.get_tensor("lengths").to(torch.int64)
            cumulative = torch.cumsum(lengths, dim=0)
            rows = int(
                torch.searchsorted(
                    cumulative,
                    torch.tensor(int(args.max_shard_tokens)),
                    right=True,
                ).item()
            )
            rows = max(2, rows - rows % 2)
            rows = min(rows, int(task["rows"]))
            truncated.append(
                {
                    **task,
                    "rows": rows,
                    "tokens": int(lengths[:rows].sum().item()),
                }
            )
        tasks = truncated
    pending = []
    for task in tasks:
        target = (
            output_dir
            / "activations"
            / f"shard-{int(task['shard_index']):05d}.safetensors"
        )
        if not _valid_shard(
            target,
            feature_ids=feature_ids,
            expected_chunk_ids=_expected_chunk_ids(task["plan_path"])[
                : int(task["rows"])
            ].contiguous(),
        ):
            pending.append(task)

    devices = [
        int(value.strip())
        for value in str(args.devices).split(",")
        if value.strip()
    ]
    if not devices:
        raise ValueError("--devices must name at least one CUDA device")
    # Each source shard contains approximately the same token count. Round-robin
    # assignment therefore balances model work while keeping resumability simple.
    assigned = [pending[index:: len(devices)] for index in range(len(devices))]
    worker_kwargs = {
        "model_path": model_path,
        "layer": int(args.layer),
        "checkpoints": {method: str(checkpoints[method]) for method in METHODS},
        "feature_ids": feature_ids,
        "output_dir": str(output_dir),
        "max_batch_size": int(args.max_batch_size),
        "forward_token_budget": int(args.forward_token_budget),
        "model_dtype": str(args.model_dtype),
        "attn_implementation": str(args.attn_implementation),
    }
    jobs = [
        (worker_index, device, worker_tasks)
        for worker_index, (device, worker_tasks) in enumerate(
            zip(devices, assigned, strict=True)
        )
        if worker_tasks
    ]
    if len(jobs) == 1:
        worker_index, device, worker_tasks = jobs[0]
        _worker(worker_index, device, worker_tasks, **worker_kwargs)
    elif jobs:
        context = mp.get_context("spawn")
        processes: list[mp.Process] = []
        for worker_index, device, worker_tasks in jobs:
            process = context.Process(
                target=_worker,
                args=(worker_index, device, worker_tasks),
                kwargs=worker_kwargs,
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        failed = [process.pid for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"Joint cache workers failed: {failed}")

    if debug_partial:
        print(
            "[joint-native-cache] debug partial run complete; no full manifest "
            "was published",
            flush=True,
        )
        return

    files: list[dict[str, Any]] = []
    missing: list[int] = []
    length_counts: dict[str, int] = {}
    for task in all_tasks:
        shard_index = int(task["shard_index"])
        path = output_dir / "activations" / f"shard-{shard_index:05d}.safetensors"
        expected_ids = _expected_chunk_ids(task["plan_path"])
        if not _valid_shard(
            path,
            feature_ids=feature_ids,
            expected_chunk_ids=expected_ids,
        ):
            missing.append(shard_index)
            continue
        with safe_open(str(task["plan_path"]), framework="pt", device="cpu") as handle:
            lengths = handle.get_tensor("lengths")
        for length, count in zip(*torch.unique(lengths, return_counts=True)):
            key = str(int(length))
            length_counts[key] = length_counts.get(key, 0) + int(count)
        files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "rows": int(task["rows"]),
                "tokens": int(task["tokens"]),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if missing:
        print(
            f"[joint-native-cache] partial run complete; remaining shards={missing}",
            flush=True,
        )
        return

    manifest: dict[str, Any] = {
        "format": FORMAT,
        "complete": True,
        "source_data_dir": str(source_dir),
        "source_data_artifact_digest": source_manifest.get("artifact_digest"),
        "source_data_manifest_sha256": _sha256(source_dir / "data_manifest.json"),
        "model": model_path,
        "layer": int(args.layer),
        "native_chunk_lengths": sorted(map(int, length_counts)),
        "length_counts": length_counts,
        "contexts": sum(int(task["rows"]) for task in all_tasks),
        "tokens": sum(int(task["tokens"]) for task in all_tasks),
        "feature_ids": list(map(int, feature_ids)),
        "feature_count": len(feature_ids),
        "feature_sample_rule": (
            "same frozen uniform dictionary-coordinate sample as all baseline SAEs"
        ),
        "methods": checkpoint_identity,
        "activation_dtype": "bfloat16",
        "inference_parameter_dtype": str(args.model_dtype),
        "max_forward_batch_size": int(args.max_batch_size),
        "forward_token_budget": int(args.forward_token_budget),
        "attention_implementation": str(args.attn_implementation),
        "activation_computation": (
            "checkpoint encoder applied to the masked mean of layer-21 hidden "
            "states over each complete native 32/64/128/256/512-token source chunk"
        ),
        "files": files,
        "document_linking_inputs_used": False,
    }
    manifest["artifact_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output_dir / "manifest.json", manifest)
    print(output_dir / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
