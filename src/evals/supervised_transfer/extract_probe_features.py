#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from chunk_saes.artifacts import (
    ensure_reusable_artifact,
    file_record,
    json_digest,
    load_artifact_manifest,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.evaluation_protocol import (
    fixed_chunk_protocol_metadata,
    full_dictionary_feature_widths,
    mean_after_threshold,
    validate_full_dictionary_width,
)
from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.runtime import (
    bind_local_rank_cpu_affinity,
    broadcast_object,
    configure_cpu_threads,
    initialize_distributed,
)
from chunk_saes.sae import SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.utils import log


class EncoderOnly:
    def __init__(self, path: Path, device: str) -> None:
        with (path / "config.json").open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        with safe_open(str(path / "sae.safetensors"), framework="pt", device="cpu") as handle:
            tensor_names = set(handle.keys())
            self.weight = handle.get_tensor("encoder_weight").to(device)
            self.bias = handle.get_tensor("encoder_bias").to(device)
            self.decoder_bias = handle.get_tensor("decoder_bias").to(device)
            if "pre_bias" in tensor_names:
                self.pre_bias = handle.get_tensor("pre_bias").to(device)
            elif (
                self.config.get("sae_parameter_schema_version")
                == SAE_PARAMETER_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{path} declares {SAE_PARAMETER_SCHEMA_VERSION} "
                    "but sae.safetensors lacks pre_bias"
                )
            else:
                # Legacy checkpoints used decoder_bias for both input
                # centering and decoder output bias.
                self.pre_bias = self.decoder_bias
            self.threshold = float(handle.get_tensor("threshold"))
            self.scale = float(handle.get_tensor("activation_scale"))
        self.dictionary_width = int(self.weight.shape[0])
        if int(self.weight.shape[0]) != int(
            self.config.get("dict_size", self.weight.shape[0])
        ):
            raise ValueError(
                f"{path} encoder width {self.weight.shape[0]} does not match "
                "the checkpoint dictionary width"
            )
        self.device = device

    @torch.inference_mode()
    def code(self, hidden: torch.Tensor, *, top_k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden.to(self.weight.dtype)
        pre = F.relu(
            F.linear(hidden * self.scale - self.pre_bias, self.weight, self.bias)
        )
        if top_k is not None:
            values, indices = pre.topk(min(top_k, pre.shape[-1]), dim=-1)
            values = values * (values > self.threshold)
            return indices, values
        return (pre > self.threshold).nonzero(as_tuple=False), pre

    @torch.inference_mode()
    def dense_code(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.weight.dtype)
        pre = F.relu(
            F.linear(hidden * self.scale - self.pre_bias, self.weight, self.bias)
        )
        return pre * (pre > self.threshold)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract frozen SAE sparse features for ArXiv probes.")
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark-dir", required=True)
    p.add_argument("--benchmark-manifest", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--storage-k", type=int, default=2048)
    p.add_argument("--token-candidate-k", type=int, default=2048)
    p.add_argument(
        "--chunk-length",
        type=int,
        default=128,
        help="Use exactly this many tokenizer tokens for every probe example.",
    )
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--checkpoint-selection", choices=("best", "final"), default="best")
    p.add_argument("--overwrite", action="store_true")
    return p


def pad_ids(tokenizer, rows, max_length, device):
    encoded = tokenizer(
        rows,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def fixed_chunk_ids(tokenizer, text: str, chunk_length: int) -> list[int] | None:
    """Return one exact-L token chunk, or None when the text is too short."""

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < chunk_length:
        return None
    return [int(value) for value in ids[:chunk_length]]


def sparse_fixed(indices: torch.Tensor, values: torch.Tensor, storage_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Inputs are [batch, feature] and already top-k ordered.
    batch = indices.shape[0]
    out_i = np.full((batch, storage_k), -1, dtype=np.int32)
    out_v = np.zeros((batch, storage_k), dtype=np.float16)
    out_n = np.zeros(batch, dtype=np.int16)
    take = min(storage_k, indices.shape[1])
    valid = values[:, :take] > 0
    out_i[:, :take] = indices[:, :take].detach().cpu().numpy().astype(np.int32)
    out_v[:, :take] = values[:, :take].detach().cpu().numpy().astype(np.float16)
    out_n[:] = valid.sum(dim=1).detach().cpu().numpy().astype(np.int16)
    out_i[:, :take][~valid.detach().cpu().numpy()] = -1
    return out_i, out_v, out_n


@torch.inference_mode()
def extract_split(args, split: str, rows: list[dict], rank: int, world_size: int, extractor, encoders, out_dir: Path) -> None:
    device = str(extractor.device)
    selected = [row for index, row in enumerate(rows) if index % world_size == rank]
    hidden_size = extractor.hidden_size
    raw, labels, years, example_ids = [], [], [], []
    codes: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {mode: [] for mode in encoders}
    for start in range(0, len(selected), args.batch_size):
        batch_rows = selected[start : start + args.batch_size]
        sequences = []
        kept_rows = []
        for row in batch_rows:
            chunk_ids = fixed_chunk_ids(
                extractor.tokenizer,
                str(row["text"]),
                args.chunk_length,
            )
            if chunk_ids is None:
                continue
            sequences.append(chunk_ids)
            kept_rows.append(row)
        if not sequences:
            continue
        layer_batch = extractor.forward_ids(sequences)
        means = layer_batch.means()
        raw.append(means.float().cpu().numpy().astype(np.float16))
        labels.extend(row["label"] for row in kept_rows)
        years.extend(row["year"] for row in kept_rows)
        example_ids.extend(row["id"] for row in kept_rows)
        for mode in (
            mode for mode in ("mean", "cross") if mode in encoders
        ):
            indices, values = encoders[mode].code(means, top_k=args.storage_k)
            codes[mode].append(sparse_fixed(indices, values, args.storage_k))
        # Store only the predeclared mean-after-threshold chunk code for the
        # formal probe comparison.
        hidden, token_mask = layer_batch.hidden, layer_batch.mask.bool()
        mean_agg = torch.zeros(
            (hidden.shape[0], encoders["token"].weight.shape[0]),
            device=device,
        )
        for row_index in range(hidden.shape[0]):
            valid_hidden = hidden[row_index, token_mask[row_index]]
            token_codes = encoders["token"].dense_code(valid_hidden)
            mean_agg[row_index] = mean_after_threshold(token_codes)
        values, indices = mean_agg.topk(
            min(args.token_candidate_k, mean_agg.shape[1]),
            dim=1,
        )
        codes["token"].append(sparse_fixed(indices, values, args.storage_k))
        if "temporal" in encoders:
            temporal_mean = torch.zeros(
                (hidden.shape[0], encoders["temporal"].dictionary_width),
                device=device,
            )
            for row_index in range(hidden.shape[0]):
                valid_hidden = hidden[row_index, token_mask[row_index]]
                temporal_codes = encoders["temporal"].dense_code(valid_hidden)
                temporal_mean[row_index] = mean_after_threshold(temporal_codes)
            values, indices = temporal_mean.topk(
                min(args.token_candidate_k, temporal_mean.shape[1]),
                dim=1,
            )
            codes["temporal"].append(
                sparse_fixed(indices, values, args.storage_k)
            )
        del layer_batch, hidden

    rank_dir = out_dir / "partials"
    rank_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw": np.concatenate(raw, axis=0) if raw else np.empty((0, hidden_size), dtype=np.float16),
        "labels": np.asarray(labels, dtype="U32"),
        "years": np.asarray(years, dtype=np.int32),
        "ids": np.asarray(example_ids, dtype=object),
    }
    for mode in codes:
        suffixes = ("mean",) if mode in {"token", "temporal"} else ("direct",)
        for suffix in suffixes:
            if not codes[mode]:
                payload[f"{mode}_{suffix}_indices"] = np.empty((0, args.storage_k), dtype=np.int32)
                payload[f"{mode}_{suffix}_values"] = np.empty((0, args.storage_k), dtype=np.float16)
                payload[f"{mode}_{suffix}_nnz"] = np.empty((0,), dtype=np.int16)
                continue
            tuples = codes[mode]
            indices = np.concatenate([item[0] for item in tuples], axis=0)
            values = np.concatenate([item[1] for item in tuples], axis=0)
            nnz = np.concatenate([item[2] for item in tuples], axis=0)
            payload[f"{mode}_{suffix}_indices"] = indices
            payload[f"{mode}_{suffix}_values"] = values
            payload[f"{mode}_{suffix}_nnz"] = nnz
    np.savez_compressed(rank_dir / f"features-{split}-rank{rank:03d}.npz", **payload)
    log(f"probe features split={split} rank samples={len(labels)}", rank=rank)


def merge_split(out_dir: Path, split: str, world_size: int) -> None:
    paths = [out_dir / "partials" / f"features-{split}-rank{rank:03d}.npz" for rank in range(world_size)]
    arrays = [dict(np.load(path, allow_pickle=True)) for path in paths]
    keys = arrays[0].keys()
    merged = {}
    for key in keys:
        if key == "ids":
            merged[key] = np.concatenate([array[key].astype(object) for array in arrays])
        else:
            merged[key] = np.concatenate([array[key] for array in arrays], axis=0)
    row_count = len(merged["ids"])
    for key, value in merged.items():
        if key in {"ids"} or value.ndim == 0:
            continue
        if value.shape[0] != row_count:
            raise ValueError(
                f"{split} representation {key} has {value.shape[0]} rows, "
                f"expected shared ID count {row_count}"
            )
    if len(set(merged["ids"].astype(str).tolist())) != row_count:
        raise ValueError(f"{split} contains duplicate benchmark IDs")
    np.savez_compressed(out_dir / f"features-{split}.npz", **merged)


def _broadcast(value, rank: int):
    return broadcast_object(value, rank=rank)


def main() -> None:
    args = parser().parse_args()
    if not 0 < args.token_candidate_k <= args.storage_k:
        raise ValueError("token-candidate-k must be positive and no larger than storage-k")
    if args.chunk_length <= 0:
        raise ValueError("chunk-length must be positive")
    if args.chunk_length > args.max_length:
        raise ValueError("chunk-length cannot exceed max-length")
    configure_cpu_threads(default=4)
    rank, world_size, local_rank = initialize_distributed()
    bind_local_rank_cpu_affinity(
        local_rank=local_rank,
        local_world_size=int(
            os.environ.get("LOCAL_WORLD_SIZE", world_size)
        ),
    )
    out_dir = Path(args.output_dir)
    if rank == 0:
        benchmark_manifest = load_artifact_manifest(
            args.benchmark_manifest,
            expected_format="chunk-saes-arxiv-benchmark-v2",
            verify_files=True,
        )
        if Path(args.benchmark_manifest).parent.resolve() != Path(args.benchmark_dir).resolve():
            raise ValueError("--benchmark-manifest must belong to --benchmark-dir")
        sae_set = resolve_sae_artifact_set(
            args.sae_root, selection=args.checkpoint_selection
        )
        if int(sae_set["common"]["layer"]) != args.layer:
            raise ValueError("SAE checkpoint layer does not match --layer")
        if Path(str(sae_set["common"]["model"])).resolve() != Path(args.model).resolve():
            raise ValueError("SAE checkpoint model does not match --model")
        feature_widths = full_dictionary_feature_widths(sae_set)
        identity = {
            "benchmark_artifact_digest": benchmark_manifest["artifact_digest"],
            "sae_set_digest": sae_set["artifact_digest"],
            "sae_selection": args.checkpoint_selection,
            "sae_modes": sae_set["modes"],
            "model": str(Path(args.model).resolve()),
            "layer": args.layer,
            "max_length": args.max_length,
            "chunk_length": args.chunk_length,
            "storage_k": args.storage_k,
            "token_candidate_k": args.token_candidate_k,
            "feature_widths": feature_widths,
            "representation": "frozen_target_layer_hidden_then_frozen_sae_encoder",
            "representation_protocol": fixed_chunk_protocol_metadata(
                feature_widths=feature_widths,
                chunk_lengths=[args.chunk_length],
            ),
        }
        marker = out_dir / "feature_manifest.json"
        existing = None
        if marker.exists() and not args.overwrite:
            existing = ensure_reusable_artifact(
                marker,
                expected_format="chunk-saes-probe-features-v2",
                expected_identity=identity,
            )
        if args.overwrite and out_dir.exists():
            shutil.rmtree(out_dir)
        elif existing is None and out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError(
                f"probe feature directory is incomplete or unidentified: {out_dir}; "
                "use --overwrite"
            )
        skip = existing is not None
    else:
        benchmark_manifest = None
        sae_set = None
        identity = None
        skip = None
    benchmark_manifest = _broadcast(benchmark_manifest, rank)
    sae_set = _broadcast(sae_set, rank)
    identity = _broadcast(identity, rank)
    skip = bool(_broadcast(skip, rank))
    if skip:
        log(f"reusing verified probe features at {out_dir}", rank=rank, main_only=True)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return
    if dist.is_initialized():
        dist.barrier()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{local_rank}"
    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        device,
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
    )
    encoders = {
        mode: EncoderOnly(Path(sae_set["modes"][mode]["checkpoint_path"]), device)
        for mode in sae_set["modes"]
    }
    expected_widths = full_dictionary_feature_widths(sae_set)
    for mode, encoder in encoders.items():
        validate_full_dictionary_width(
            mode,
            encoder.dictionary_width,
            expected_widths[mode],
        )
    for split in ("train", "validation", "test", "ood"):
        record = benchmark_manifest["files"].get(split)
        if not isinstance(record, Mapping):
            raise ValueError(f"benchmark manifest lacks split={split}")
        path = Path(args.benchmark_dir) / str(record["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        extract_split(args, split, rows, rank, world_size, extractor, encoders, out_dir)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            merge_split(out_dir, split, world_size)
        if dist.is_initialized():
            dist.barrier()
    extractor.close()
    if rank == 0:
        final_files = {
            split: file_record(
                out_dir / f"features-{split}.npz", relative_to=out_dir
            )
            for split in ("train", "validation", "test", "ood")
        }
        sample_audit = {}
        for split in ("train", "validation", "test", "ood"):
            with np.load(out_dir / f"features-{split}.npz", allow_pickle=True) as data:
                split_ids = data["ids"].astype(str).tolist()
            sample_audit[split] = {
                "count": len(split_ids),
                "ids_digest": json_digest(split_ids),
            }
        write_artifact_manifest(
            {
                "format": "chunk-saes-probe-features-v2",
                "complete": True,
                "identity": identity,
                "model": args.model,
                "layer": args.layer,
                "storage_k": args.storage_k,
                "token_candidate_k": args.token_candidate_k,
                "chunk_length": identity["chunk_length"],
                "feature_widths": identity["feature_widths"],
                "representation_protocol": identity["representation_protocol"],
                "representation": "frozen_target_layer_hidden_then_frozen_sae_encoder",
                "sample_protocol": {
                    "fixed_chunk_length": identity["chunk_length"],
                    "shared_ids_across_methods": True,
                    "independent_forward": True,
                    "splits": sample_audit,
                },
                "classifier": "linear_only",
                "files": final_files,
            },
            out_dir / "feature_manifest.json",
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
