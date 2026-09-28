#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evals.document_linking import (
    evaluate_lexical_controlled_document_linking as linking,
)
from evals.joint_extension.common import (
    FrozenJointEncoder,
    JointSpec,
    add_joint_root_arguments,
    joint_specs_from_args,
    merge_sidecar,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Incrementally evaluate nested Joint-Chunk SAEs on document linking."
    )
    p.add_argument("--eval-root", required=True)
    add_joint_root_arguments(p)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260817)
    return p


def _specs(args: argparse.Namespace) -> list[JointSpec]:
    return joint_specs_from_args(args)


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root).resolve()
    task_dir = eval_root / "document_linking"
    with np.load(task_dir / "features.npz") as handle:
        arrays = {key: handle[key] for key in handle.files}
    base_results = json.loads(
        (task_dir / "document_linking_results.json").read_text(encoding="utf-8")
    )
    pair_rows = [
        json.loads(line)
        for line in (task_dir / "selected_pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    global_to_row = {
        int(value): index
        for index, value in enumerate(arrays["global_chunk_ids"])
    }
    left_rows = np.asarray(
        [global_to_row[int(row["left_global_chunk_id"])] for row in pair_rows],
        dtype=np.int64,
    )
    right_rows = np.asarray(
        [global_to_row[int(row["right_global_chunk_id"])] for row in pair_rows],
        dtype=np.int64,
    )
    lengths = arrays["retokenized_lengths"].astype(np.int16)
    left_lengths = lengths[left_rows]
    right_lengths = lengths[right_rows]
    pair_ids = np.concatenate(
        [
            np.arange(len(pair_rows), dtype=np.int32),
            np.arange(len(pair_rows), dtype=np.int32),
        ]
    )
    hidden = torch.from_numpy(arrays["raw_mean_hidden"].astype(np.float32))
    specs = _specs(args)
    device = torch.device(args.device)
    methods: dict[str, dict] = {}
    feature_arrays: dict[str, np.ndarray] = {}
    for spec_index, spec in enumerate(specs):
        print(f"[joint-linking] encoding {spec.label}", flush=True)
        encoder = FrozenJointEncoder(spec.checkpoint, device)
        all_indices: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        all_nnz: list[np.ndarray] = []
        try:
            for start in range(0, int(hidden.shape[0]), int(args.batch_size)):
                stop = min(int(hidden.shape[0]), start + int(args.batch_size))
                indices, values, nnz = encoder.topk(
                    hidden[start:stop],
                    args.top_k,
                )
                all_indices.append(indices.cpu().numpy().astype(np.int32))
                all_values.append(values.float().cpu().numpy().astype(np.float16))
                all_nnz.append(nnz.cpu().numpy().astype(np.int16))
        finally:
            encoder.close()
        indices = np.concatenate(all_indices)
        values = np.concatenate(all_values)
        nnz = np.concatenate(all_nnz)
        invalid = values <= 0
        indices[invalid] = -1
        representation = linking._csr_from_fixed_topk(
            indices,
            values,
            width=spec.dictionary_width,
        )
        forward = linking.retrieval_ranks(
            representation,
            query_indices=left_rows,
            gallery_indices=right_rows,
            gallery_lengths=right_lengths,
        )
        backward = linking.retrieval_ranks(
            representation,
            query_indices=right_rows,
            gallery_indices=left_rows,
            gallery_lengths=left_lengths,
        )
        ranks = np.concatenate((forward, backward))
        methods[spec.key] = {
            "label": spec.label,
            "alpha": spec.alpha,
            **linking._metrics_from_ranks(ranks),
            **linking._bootstrap_method(
                ranks,
                pair_ids,
                samples=args.bootstrap_samples,
                seed=args.seed + 50 + spec_index,
            ),
            "mean_nonzero_features": float(nnz.mean()),
            "representation": (
                "thresholded Joint-Chunk shared encoder activation of "
                "chunk-mean hidden state, then top-k"
            ),
            "rank_distribution": ranks.tolist(),
        }
        feature_arrays[f"{spec.key}_indices"] = indices
        feature_arrays[f"{spec.key}_values"] = values
        feature_arrays[f"{spec.key}_nnz"] = nnz

        comparisons = {}
        for reference_index, (reference, reference_ranks) in enumerate(
            (base_results.get("rank_distributions") or {}).items()
        ):
            reference_array = np.asarray(reference_ranks, dtype=np.float64)
            if reference_array.shape != ranks.shape:
                continue
            comparisons[f"{spec.key}_minus_{reference}"] = (
                linking._paired_bootstrap_difference(
                    ranks,
                    reference_array,
                    pair_ids,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 500 + spec_index * 50 + reference_index,
                )
            )
        methods[spec.key]["comparisons"] = comparisons

    feature_path = task_dir / "joint_extension_features.npz"
    np.savez_compressed(
        feature_path,
        global_chunk_ids=arrays["global_chunk_ids"],
        **feature_arrays,
    )
    output = task_dir / "joint_extension.json"
    merge_sidecar(
        output,
        task="document_linking",
        methods=methods,
        specs=specs,
        protocol={
            "base_pair_archive": "selected_pairs.jsonl",
            "pairs": len(pair_rows),
            "directions": int(2 * len(pair_rows)),
            "representation": "complete Joint encoder dictionary",
            "top_k": int(args.top_k),
            "exact_target_length_gallery": True,
            "word_set_jaccard_max": 0.10,
            "qwen_forward_reused": True,
            "raw_mean_hidden_source": "document_linking/features.npz",
        },
        files={
            "joint_features": str(feature_path.name),
        },
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
