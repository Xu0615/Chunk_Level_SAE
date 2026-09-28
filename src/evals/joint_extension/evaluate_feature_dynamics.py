#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evals.feature_dynamics import (
    plot_cross_feature_semantic_manifold as dynamics,
)
from evals.joint_extension.common import (
    JointSpec,
    add_joint_root_arguments,
    joint_specs_from_args,
    merge_sidecar,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate four nested Joint-Chunk SAEs on the frozen sequence "
            "passages used by the feature-dynamics task."
        )
    )
    p.add_argument("--eval-root", required=True)
    add_joint_root_arguments(p)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--display-chunk-length", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--encoder-block-size", type=int, default=2048)
    return p


def _specs(args: argparse.Namespace) -> list[JointSpec]:
    return joint_specs_from_args(args)


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root).resolve()
    specs = _specs(args)
    passages = dynamics._sequence_passages(eval_root)
    device = torch.device(args.device)
    from chunk_saes.modeling import TargetLayerExtractor

    evaluation_identity = dynamics._evaluation_identity(eval_root)
    extractor = TargetLayerExtractor(
        str(evaluation_identity["model"]),
        int(evaluation_identity["layer"]),
        str(device),
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
            )[: int(args.max_length)]
        )
    _token_views, chunk_views = dynamics._independent_sequence_views(
        extractor,
        ids_by_passage,
        [int(args.display_chunk_length)],
    )
    extractor.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bounds = np.cumsum(
        [0]
        + [
            len(ids) // int(args.display_chunk_length)
            for ids in ids_by_passage
        ]
    ).astype(np.int64)
    methods: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {
        "bounds": bounds,
        "display_chunk_length": np.asarray(
            int(args.display_chunk_length),
            dtype=np.int64,
        ),
    }
    for spec in specs:
        print(f"[joint-dynamics] scanning {spec.label}", flush=True)
        feature_ids, means, raw_traces, traces_by_view = dynamics._scan_top_features(
            spec.checkpoint,
            chunk_views,
            token_level=False,
            primary_view=f"L{int(args.display_chunk_length)}",
            feature_width=spec.dictionary_width,
            top_k=int(args.top_k),
            block_size=int(args.encoder_block_size),
            device=device,
        )
        traces = dynamics._normalize_feature_traces(raw_traces)
        stats = dynamics._top_feature_statistics(traces, bounds)
        for feature_id, row in zip(
            feature_ids.tolist(),
            stats["per_feature"],
            strict=True,
        ):
            row["feature_id"] = int(feature_id)
            row["auto_explanation"] = dynamics._auto_curve_explanation(
                row,
                passages,
            )
        arrays[f"{spec.key}_traces"] = traces
        arrays[f"{spec.key}_feature_ids"] = feature_ids
        arrays[f"{spec.key}_mean_activations"] = means
        for view_name, values in traces_by_view.items():
            arrays[f"{spec.key}_{view_name}_traces"] = values
        methods[spec.key] = {
            "label": spec.label,
            "alpha": spec.alpha,
            "dictionary_feature_width": spec.dictionary_width,
            "selected_feature_ids": feature_ids.tolist(),
            "ranking_mean_activations": means.tolist(),
            "selection_view": f"L{int(args.display_chunk_length)}",
            "display_observations": int(bounds[-1]),
            "token_level_before_chunk_aggregation": False,
            "activation_views": list(traces_by_view),
            "auto_explanation_basis": (
                "dominant and runner-up passage activation means on the "
                "displayed sequence; no external LLM call"
            ),
            **stats,
        }

    output_root = eval_root / "feature_dynamics"
    npz_path = output_root / "joint_extension_traces.npz"
    np.savez_compressed(npz_path, **arrays)
    output = output_root / "joint_extension.json"
    merge_sidecar(
        output,
        task="feature_dynamics",
        methods=methods,
        specs=specs,
        protocol={
            "shared_text": True,
            "independent_chunk_forwards": True,
            "display_chunk_length": int(args.display_chunk_length),
            "display_chunks": int(bounds[-1]),
            "feature_count": int(args.top_k),
            "feature_ranking": (
                "each Joint SAE independently ranks the complete dictionary "
                "by mean thresholded activation over aligned chunks"
            ),
            "trace_normalization": "per-feature maximum over displayed chunks",
            "labels": "deterministic passage-activation summaries",
            "llm_api_used": False,
        },
        files={"traces": npz_path.name},
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
