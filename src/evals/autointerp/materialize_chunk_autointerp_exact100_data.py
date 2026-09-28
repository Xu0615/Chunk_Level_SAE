#!/usr/bin/env python
"""Materialize a true random exact-100 chunk-only view of an exact-1000 cache."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from chunk_saes.artifacts import (
    file_record,
    load_artifact_manifest,
    write_artifact_manifest,
)
from chunk_saes.utils import atomic_json_dump


METHODS = ("mean", "cross")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract the first 100 draws of the original Random(42) sample "
            "from a cached exact-1000 chunk activation artifact."
        )
    )
    p.add_argument("--source-data-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--features-per-method", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = parser().parse_args()
    source_manifest_path = Path(args.source_data_manifest)
    source_root = source_manifest_path.parent
    output_root = Path(args.output_dir)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_root}; use --overwrite"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    source = load_artifact_manifest(
        source_manifest_path,
        expected_format="chunk-saes-autointerp-data-v2",
        verify_files=False,
    )
    if args.features_per_method <= 0:
        raise ValueError("--features-per-method must be positive")
    dictionary_size = int(
        source["identity"]["feature_widths"]["mean"]
    )
    selected = random.Random(args.seed).sample(
        range(dictionary_size),
        args.features_per_method,
    )
    positions_by_method: dict[str, list[int]] = {}
    for method in METHODS:
        cached = list(map(int, source["feature_ids"][method]))
        missing = sorted(set(selected) - set(cached))
        if missing:
            raise ValueError(
                f"source cache does not contain exact random sample for "
                f"{method}: {missing[:10]}"
            )
        positions_by_method[method] = [
            cached.index(feature_id) for feature_id in selected
        ]

    plan_source = source_root / "chunk_pool/plan"
    plan_target = output_root / "chunk_pool/plan"
    for path in sorted(plan_source.glob("*")):
        if path.is_file():
            _link_or_copy(path, plan_target / path.name)

    activation_source = source_root / "chunk_pool/activations"
    activation_target = output_root / "chunk_pool/activations"
    activation_target.mkdir(parents=True, exist_ok=True)
    for path in sorted(activation_source.glob("shard-*.safetensors")):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors["chunk_ids"] = handle.get_tensor("chunk_ids")
            for method in METHODS:
                stored_ids = list(
                    map(
                        int,
                        handle.get_tensor(
                            f"{method}_feature_ids"
                        ).tolist(),
                    )
                )
                if stored_ids != list(
                    map(int, source["feature_ids"][method])
                ):
                    raise ValueError(
                        f"{method} feature IDs disagree in {path}"
                    )
                positions = positions_by_method[method]
                tensors[f"{method}_activations"] = (
                    handle.get_tensor(f"{method}_activations")[
                        :,
                        positions,
                    ].contiguous()
                )
                tensors[f"{method}_feature_ids"] = torch.tensor(
                    selected,
                    dtype=torch.long,
                )
        save_file(tensors, activation_target / path.name)
        print(
            f"[exact100-data] wrote {path.name}",
            flush=True,
        )

    coverage_target = output_root / "coverage"
    for method in METHODS:
        _link_or_copy(
            source_root / f"coverage/{method}.safetensors",
            coverage_target / f"{method}.safetensors",
        )

    identity = dict(source["identity"])
    identity["feature_selection"] = {
        "algorithm": "python random.Random(seed).sample",
        "population": "complete training-alive dictionary",
        "sample_size_per_method": args.features_per_method,
        "seed": args.seed,
        "same_feature_ids_for_mean_and_cross": True,
        "selection_order_preserved": True,
        "source_exact1000_artifact_digest": source["artifact_digest"],
    }
    identity["feature_ids"] = {
        method: selected for method in METHODS
    }
    identity["granularity_protocol"] = {
        "mean": "complete-variable-length-chunk",
        "cross": "complete-variable-length-chunk",
    }

    coverage = {
        "complete_dictionary_coverage": True,
        "note": (
            "Full 65,536-feature coverage was established by the source "
            "artifact; this view stores activation columns only for the "
            "true random exact-100 Mean/Cross sample."
        ),
        "source_data_artifact_digest": source["artifact_digest"],
    }
    files: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(
        sorted(plan_target.glob("shard-*.safetensors"))
    ):
        files[f"chunk_plan_{index:04d}"] = file_record(
            path,
            relative_to=output_root,
            hash_content=False,
        )
    manifest_path = plan_target / "manifest.json"
    if manifest_path.is_file():
        files["chunk_plan_manifest"] = file_record(
            manifest_path,
            relative_to=output_root,
            hash_content=False,
        )
    for index, path in enumerate(
        sorted(activation_target.glob("shard-*.safetensors"))
    ):
        files[f"chunk_activation_{index:04d}"] = file_record(
            path,
            relative_to=output_root,
            hash_content=False,
        )
    for method in METHODS:
        files[f"coverage_{method}"] = file_record(
            coverage_target / f"{method}.safetensors",
            relative_to=output_root,
            hash_content=False,
        )

    payload = {
        "format": "chunk-saes-autointerp-data-v2",
        "complete": True,
        "chunk_count": int(source["chunk_count"]),
        "features_per_method": args.features_per_method,
        "feature_ids": {
            method: selected for method in METHODS
        },
        "coverage": coverage,
        "identity": identity,
        "files": files,
    }
    manifest = write_artifact_manifest(
        payload,
        output_root / "data_manifest.json",
    )
    atomic_json_dump(
        {
            "complete": True,
            "features_per_method": args.features_per_method,
            "seed": args.seed,
            "feature_ids": selected,
            "minimum_feature_id": min(selected),
            "maximum_feature_id": max(selected),
            "source_data_artifact_digest": source["artifact_digest"],
            "artifact_digest": manifest["artifact_digest"],
        },
        output_root / "selection_summary.json",
    )
    print(
        json.dumps(
            {
                "feature_ids": selected,
                "artifact_digest": manifest["artifact_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
