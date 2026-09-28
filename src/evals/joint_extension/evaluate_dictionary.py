#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chunk_saes.evaluation_protocol import fixed_chunk_protocol_metadata
from evals.dictionary_utilization import analyze_adjacent_feature_consistency as adjacent
from evals.joint_extension.common import (
    FrozenJointEncoder,
    JointSpec,
    add_joint_root_arguments,
    joint_specs_from_args,
    merge_sidecar,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate four nested Joint-Chunk SAEs on dictionary utilization."
    )
    p.add_argument("--eval-root", required=True)
    p.add_argument("--validation-cache-dir", required=True)
    add_joint_root_arguments(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample-pairs", type=int, default=2048)
    p.add_argument("--feature-sample-size", type=int, default=8192)
    p.add_argument("--min-feature-support", type=int, default=8)
    p.add_argument("--utilization-k", type=int, default=8)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=72)
    return p


def _specs(args: argparse.Namespace) -> list[JointSpec]:
    return joint_specs_from_args(args)


def _sampled_encoder(
    spec: JointSpec,
    *,
    feature_sample_size: int,
    seed: int,
    device: torch.device,
) -> FrozenJointEncoder:
    from safetensors import safe_open

    with safe_open(
        str(spec.checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        counts = handle.get_tensor("feature_counts")
    alive = torch.nonzero(counts > 0, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(alive.numel(), generator=generator)
    feature_ids = (
        alive[order[: min(int(feature_sample_size), int(alive.numel()))]]
        .sort()
        .values
        .tolist()
    )
    return FrozenJointEncoder(spec.checkpoint, device, feature_ids)


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root).resolve()
    cache_dir = Path(args.validation_cache_dir).resolve()
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = adjacent._read(cache_manifest_path)
    cache_identity = adjacent._cache_identity(cache_dir, cache_manifest)
    specs = _specs(args)
    for spec in specs:
        if spec.validation_cache_digest != cache_identity["activation_digest"]:
            raise ValueError(
                f"{spec.key} validation cache does not match the supplied cache"
            )

    shard_paths = adjacent._shard_paths(cache_dir, cache_manifest)
    metadata, document_hashes = adjacent._scan_pair_metadata(shard_paths)
    selected = adjacent._stratified_sample(
        metadata,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
    )
    (
        means_a,
        means_b,
        _tokens_a,
        _tokens_b,
        pair_ids,
        lengths_a,
        lengths_b,
    ) = adjacent._load_selected_pairs(
        shard_paths,
        metadata,
        selected,
    )
    shuffled = adjacent._matched_derangement(
        lengths_a,
        lengths_b,
        document_hashes[selected],
        seed=args.seed + 1,
    )

    device = torch.device(args.device)
    methods: dict[str, dict] = {}
    for spec_index, spec in enumerate(specs):
        print(f"[joint-dictionary] encoding {spec.label}", flush=True)
        encoder = _sampled_encoder(
            spec,
            feature_sample_size=args.feature_sample_size,
            seed=args.seed + 104 + spec_index,
            device=device,
        )
        try:
            codes_a = []
            codes_b = []
            batch_size = 64
            for start in range(0, int(means_a.shape[0]), batch_size):
                stop = min(int(means_a.shape[0]), start + batch_size)
                codes_a.append(
                    encoder.dense(means_a[start:stop]).float().cpu().numpy()
                )
                codes_b.append(
                    encoder.dense(means_b[start:stop]).float().cpu().numpy()
                )
            array_a = np.concatenate(codes_a, axis=0)
            array_b = np.concatenate(codes_b, axis=0)
            metrics = adjacent._analyze_codes(
                array_a,
                array_b,
                shuffled,
                min_feature_support=args.min_feature_support,
                utilization_k=args.utilization_k,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 5_000 + spec_index,
            )
            methods[spec.key] = {
                "label": spec.label,
                "alpha": spec.alpha,
                "representation": (
                    "thresholded Joint-Chunk shared encoder activation of the "
                    "owning chunk mean over the complete 65,536 dictionary"
                ),
                "sampled_feature_ids": encoder.feature_ids.tolist(),
                "sampled_features": int(encoder.feature_ids.numel()),
                "mean_active_features_per_chunk": float(
                    np.mean((array_a > 0).sum(axis=1))
                ),
                **metrics,
            }
        finally:
            encoder.close()

    output = eval_root / "dictionary_utilization" / "joint_extension.json"
    merge_sidecar(
        output,
        task="dictionary_utilization",
        methods=methods,
        specs=specs,
        protocol={
            "source": "shared validation activation-cache-v2",
            "sample_pairs": int(args.sample_pairs),
            "pair_ids_digest": adjacent.json_digest(pair_ids.tolist()),
            "length_matched_different_document_shuffle": True,
            "feature_sample_size": int(args.feature_sample_size),
            "feature_sample_seeds": {
                spec.key: int(args.seed + 104 + index)
                for index, spec in enumerate(specs)
            },
            "utilization_k": int(args.utilization_k),
            "min_feature_support": int(args.min_feature_support),
            "representation_protocol": fixed_chunk_protocol_metadata(
                feature_widths={spec.key: spec.dictionary_width for spec in specs},
                chunk_lengths=sorted(
                    set(lengths_a.tolist()) | set(lengths_b.tolist())
                ),
            ),
            "joint_primary_view": "complete_dictionary",
            "nested_cross_prefix": 32768,
        },
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
