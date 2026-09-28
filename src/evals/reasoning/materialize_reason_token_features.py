#!/usr/bin/env python
"""Materialize complete Token-SAE mean-pooled codes for Eval-6 chunks.

Eval 6 stores only the largest 128 coordinates per chunk.  Counterpart search
for Eval 7 is stronger if feature screening sees every thresholded Token-SAE
coordinate, including weak coordinates outside that retrieval-oriented cache.
This script replays exactly the Eval-6 chunk texts through the frozen base
model and writes a memory-mappable [chunk, dictionary] float16 matrix.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from chunk_saes.artifacts import (
    file_record,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.modeling import TargetLayerExtractor
from evals.reasoning.evaluate_reason_feature import SparseEncoder, _pool_token_features


FORMAT = "chunk-saes-eval7-full-token-features-v1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, default=21)
    p.add_argument("--cross-feature-id", type=int, default=20232)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--overwrite", action="store_true")
    return p


def _load_chunks(evidence_dir: Path) -> dict[int, dict]:
    output: dict[int, dict] = {}
    for path in sorted((evidence_dir / "partials").glob("chunks-rank*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    output[int(row["global_chunk_id"])] = row
    return output


@torch.inference_mode()
def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(eval_root / "document_linking/features.npz") as data:
        global_ids = data["global_chunk_ids"].copy()
        expected_lengths = data["retokenized_lengths"].copy()
    chunks = _load_chunks(eval_root / "dictionary_utilization/feature_evidence")

    device = torch.device(args.device)
    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        str(device),
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
    )
    sae_set = resolve_sae_artifact_set(
        args.sae_root,
        selection="best",
        modes=("token", "cross"),
    )
    token_encoder = SparseEncoder(
        Path(sae_set["modes"]["token"]["checkpoint_path"]),
        device,
    )
    cross_encoder = SparseEncoder(
        Path(sae_set["modes"]["cross"]["checkpoint_path"]),
        device,
    )
    sequences = []
    for chunk_id in global_ids.tolist():
        row = chunks[int(chunk_id)]
        ids = extractor.tokenizer(
            str(row["text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        ).input_ids
        if not ids:
            raise RuntimeError(f"empty retokenized chunk {chunk_id}")
        sequences.append(list(map(int, ids)))
    actual_lengths = np.asarray(
        [len(row) for row in sequences],
        dtype=np.int16,
    )
    if not np.array_equal(actual_lengths, expected_lengths):
        mismatches = np.flatnonzero(actual_lengths != expected_lengths)
        raise RuntimeError(
            f"retokenized lengths differ from Eval-6 for "
            f"{len(mismatches)} rows"
        )

    matrix_path = output_dir / "token_mean_full.npy"
    # Fill in RAM and perform one sequential write at the end.  Random indexed
    # writes into a network-filesystem memmap are dramatically slower.
    matrix = np.empty(
        (len(sequences), token_encoder.width),
        dtype=np.float16,
    )
    cross_feature = np.empty(len(sequences), dtype=np.float32)
    order = sorted(
        range(len(sequences)),
        key=lambda index: (len(sequences[index]), index),
    )
    try:
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start : start + args.batch_size]
            batch = extractor.forward_ids(
                [sequences[index] for index in batch_indices]
            )
            means = batch.means()
            pooled = _pool_token_features(
                token_encoder.dense(batch.hidden),
                batch.mask,
                "mean",
            )
            matrix[np.asarray(batch_indices)] = (
                pooled.float().cpu().numpy().astype(np.float16)
            )
            cross_feature[np.asarray(batch_indices)] = (
                cross_encoder.dense(means)[:, args.cross_feature_id]
                .float()
                .cpu()
                .numpy()
            )
            if (start + len(batch_indices)) % 128 == 0 or (
                start + len(batch_indices) == len(order)
            ):
                print(
                    f"[eval7/full-token] "
                    f"{start + len(batch_indices)}/{len(order)}",
                    flush=True,
                )
        np.save(matrix_path, matrix)
    finally:
        del matrix
        extractor.close()
        token_encoder.close()
        cross_encoder.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ids_path = output_dir / "global_chunk_ids.npy"
    lengths_path = output_dir / "retokenized_lengths.npy"
    cross_path = (
        output_dir / f"cross_feature_{args.cross_feature_id}.npy"
    )
    np.save(ids_path, global_ids)
    np.save(lengths_path, actual_lengths)
    np.save(cross_path, cross_feature)
    manifest = write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": {
                "model": str(Path(args.model).resolve()),
                "sae_root": str(Path(args.sae_root).resolve()),
                "sae_set_digest": sae_set["artifact_digest"],
                "layer": args.layer,
                "pooling": "mean_after_threshold",
                "dictionary_coordinates": "all_thresholded_coordinates",
                "dtype": "float16",
                "shape": [len(sequences), token_encoder.width],
                "cross_feature_id": args.cross_feature_id,
            },
            "files": {
                "token_mean_full": file_record(
                    matrix_path,
                    relative_to=output_dir,
                ),
                "global_chunk_ids": file_record(
                    ids_path,
                    relative_to=output_dir,
                ),
                "retokenized_lengths": file_record(
                    lengths_path,
                    relative_to=output_dir,
                ),
                "cross_feature": file_record(
                    cross_path,
                    relative_to=output_dir,
                ),
            },
        },
        output_dir / "manifest.json",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "rows": len(sequences),
                "width": token_encoder.width,
                "artifact_digest": manifest["artifact_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
