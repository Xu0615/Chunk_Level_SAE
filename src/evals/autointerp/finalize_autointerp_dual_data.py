#!/usr/bin/env python
"""Finalize already-materialized dual-granularity AutoInterp forward data.

This utility is only a recovery path for a prepare run that completed model
forwards but stopped before coverage/manifest publication. New runs should use
``prepare_autointerp_data.py`` directly.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from chunk_saes.artifacts import resolve_sae_artifact_set
from chunk_saes.autointerp import AUTOINTERP_PROTOCOL, AutoInterpConfig
from chunk_saes.evaluation_protocol import full_dictionary_feature_widths
from chunk_saes.runtime import (
    bind_local_rank_cpu_affinity,
    broadcast_object,
    configure_cpu_threads,
    initialize_distributed,
)
from chunk_saes.utils import atomic_json_dump
from evals.autointerp.prepare_autointerp_data import (
    CHUNK_METHODS,
    METHODS,
    TOKEN_METHODS,
    _coverage_scan,
    _write_manifest,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Finish exact full-dictionary coverage and publish a clean "
            "dual-granularity AutoInterp data manifest."
        )
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--checkpoint-selection", choices=("best", "final"), default="best")
    p.add_argument("--feature-block-size", type=int, default=256)
    p.add_argument("--sequence-batch-size", type=int, default=64)
    p.add_argument("--chunk-batch-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delete-work", action="store_true")
    p.add_argument("--delete-stale-matched-pool", action="store_true")
    return p


def _feature_ids(data_dir: Path) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for method in TOKEN_METHODS:
        paths = sorted(
            (data_dir / "token_pool" / "activations" / method).glob(
                "rank-*.safetensors"
            )
        )
        if not paths:
            raise FileNotFoundError(f"no {method} activation shards")
        with safe_open(str(paths[0]), framework="pt", device="cpu") as handle:
            result[method] = list(
                map(int, handle.get_tensor("feature_ids").tolist())
            )
    paths = sorted(
        (data_dir / "chunk_pool" / "activations").glob(
            "shard-*.safetensors"
        )
    )
    if not paths:
        raise FileNotFoundError("no chunk activation shards")
    with safe_open(str(paths[0]), framework="pt", device="cpu") as handle:
        for method in CHUNK_METHODS:
            result[method] = list(
                map(
                    int,
                    handle.get_tensor(f"{method}_feature_ids").tolist(),
                )
            )
    for method, ids in result.items():
        if len(ids) != 1000 or len(set(ids)) != 1000:
            raise ValueError(f"{method} feature IDs are not exact-1000")
    return result


def _base_identity(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "data_manifest.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return (json.load(handle).get("identity") or {})


def main() -> None:
    args = parser().parse_args()
    configure_cpu_threads(default=4)
    rank, world_size, local_rank = initialize_distributed()
    bind_local_rank_cpu_affinity(
        local_rank=local_rank,
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
    )
    data_dir = Path(args.data_dir)
    if rank == 0:
        sae_set = resolve_sae_artifact_set(
            args.sae_root,
            selection=args.checkpoint_selection,
            modes=METHODS,
        )
        feature_ids = _feature_ids(data_dir)
        widths = full_dictionary_feature_widths(sae_set, modes=METHODS)
        if set(widths.values()) != {65_536}:
            raise ValueError(f"expected four complete 65,536 dictionaries: {widths}")
        with (data_dir / "chunk_pool" / "plan" / "manifest.json").open(
            encoding="utf-8"
        ) as handle:
            chunk_plan = json.load(handle)
        with (data_dir / "budget_analysis.json").open(encoding="utf-8") as handle:
            budget = json.load(handle)
        base = _base_identity(data_dir)
    else:
        sae_set = feature_ids = widths = chunk_plan = budget = base = None
    sae_set = broadcast_object(sae_set, rank=rank)
    feature_ids = broadcast_object(feature_ids, rank=rank)
    widths = broadcast_object(widths, rank=rank)
    chunk_plan = broadcast_object(chunk_plan, rank=rank)
    budget = broadcast_object(budget, rank=rank)
    base = broadcast_object(base, rank=rank)

    token_tensor = torch.load(
        data_dir / "token_pool" / "tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    context_size = int(token_tensor.shape[1])
    protocol = AutoInterpConfig(context_size=context_size, buffer=10, seed=args.seed)
    coverage_dir = data_dir / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    token_paths = sorted(
        (data_dir / "work" / "token_hidden").glob("rank-*.safetensors")
    )
    if len(token_paths) != world_size:
        raise ValueError(
            f"expected {world_size} token hidden shards, found {len(token_paths)}"
        )
    for method in TOKEN_METHODS:
        report, tensors = _coverage_scan(
            method=method,
            checkpoint=Path(sae_set["modes"][method]["checkpoint_path"]),
            shard_paths=token_paths,
            tensor_name="hidden",
            unit="token",
            feature_block_size=args.feature_block_size,
            unit_batch_size=args.sequence_batch_size,
            protocol=protocol,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
        reports[method] = report
        if rank == 0:
            save_file(tensors, str(coverage_dir / f"{method}.safetensors"))

    chunk_paths = sorted(
        (data_dir / "work" / "chunk_means").glob("shard-*.safetensors")
    )
    if not chunk_paths:
        raise ValueError(
            "chunk means are absent; rerun prepare with "
            "--recompute-chunk-means-for-coverage"
        )
    for method in CHUNK_METHODS:
        report, tensors = _coverage_scan(
            method=method,
            checkpoint=Path(sae_set["modes"][method]["checkpoint_path"]),
            shard_paths=chunk_paths,
            tensor_name="means",
            unit="chunk",
            feature_block_size=args.feature_block_size,
            unit_batch_size=args.chunk_batch_size,
            protocol=protocol,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
        reports[method] = report
        if rank == 0:
            save_file(tensors, str(coverage_dir / f"{method}.safetensors"))

    complete = all(reports[m]["coverage_complete"] for m in METHODS)
    if not complete:
        failures = {
            m: reports[m]["unscorable_features"]
            for m in METHODS
            if not reports[m]["coverage_complete"]
        }
        raise RuntimeError(f"incomplete full-dictionary coverage: {failures}")
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        coverage = {
            "complete_dictionary_coverage": True,
            "methods": reports,
        }
        atomic_json_dump(coverage, data_dir / "coverage_report.json")
        identity = {
            "protocol": AUTOINTERP_PROTOCOL,
            "protocol_config": protocol.to_dict(),
            "model": base.get("model"),
            "tokenizer_fingerprint": base.get("tokenizer_fingerprint"),
            "layer": base.get("layer"),
            "dataset": base.get("dataset"),
            "token_pool": {
                "requested_total_tokens": int(budget["chosen_token_budget"]),
                "total_tokens": int(token_tensor.numel()),
                "context_size": context_size,
                "granularity": "saebench-token-centered-window",
            },
            "chunk_pool": {
                "total_tokens": int(chunk_plan["total_tokens"]),
                "chunk_lengths": list(map(int, chunk_plan["chunk_lengths"])),
                "granularity": "complete-variable-length-chunk",
                "length_distribution":
                    "exactly balanced over ordered training length pairs",
                "sample_seed": int(chunk_plan["sample_seed"]),
            },
            "feature_selection": {
                "sample_size_per_method": 1000,
                "population": "complete training-alive dictionary",
                "algorithm": "python random.Random(seed).sample",
                "seed": args.seed,
                "temporal_prefix_restriction": False,
            },
            "feature_ids": feature_ids,
            "feature_widths": widths,
            "granularity_protocol": {
                "token": "token-centered-window",
                "temporal": "token-centered-window",
                "mean": "complete-variable-length-chunk",
                "cross": "complete-variable-length-chunk",
            },
            "sae_set_digest": sae_set["artifact_digest"],
            "checkpoint_selection": args.checkpoint_selection,
            "world_size": world_size,
            "activation_dtype": base.get("activation_dtype", "bfloat16"),
            "model_dtype": base.get("model_dtype", "bfloat16"),
            "budget_analysis": budget,
        }
        if args.delete_stale_matched_pool:
            shutil.rmtree(data_dir / "matched_chunk_pool", ignore_errors=True)
        if args.delete_work:
            shutil.rmtree(data_dir / "work", ignore_errors=True)
        _write_manifest(
            output_dir=data_dir,
            identity=identity,
            feature_ids=feature_ids,
            coverage=coverage,
            chunk_plan=chunk_plan,
        )
        print(json.dumps(coverage, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
